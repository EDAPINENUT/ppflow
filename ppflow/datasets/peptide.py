from torch.utils.data import Dataset
import os 
import torch 
from tqdm.auto import tqdm
from .parse import *
from Bio import SeqRecord, SeqIO, Seq
import subprocess
from ._base import register_dataset
import random
import math 
from joblib import Parallel, delayed, cpu_count

@register_dataset('pair')
def get_pair_dataset(cfg, transform):
    return PairDataset(data_dir = cfg.data_dir,
                       split_path = cfg.split_path,
                       processed_dir = cfg.processed_dir,
                       include_pdb_path = cfg.get('include_pdb_path', None),
                       exclude_pdb_path = cfg.get('exclude_pdb_path', None),
                       split = cfg.split,
                       split_seed = cfg.get('split_seed', 2024),
                       transform = transform,
                       benchmark_test = cfg.get('benchmark_test', False))

class PairDataset(Dataset):

    def __init__(
        self, 
        data_dir = './dataset/ppbench2024/',
        split_path = './processed/split.pt',
        processed_dir = './processed',
        include_pdb_path = None,
        exclude_pdb_path = None,
        split = 'train',
        split_seed = 2022,
        transform = None,
        reset = False,
        num_preprocess_jobs = math.floor(cpu_count() * 0.8),
        cluster_patch_size = 32,
        benchmark_test = False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split_path = split_path
        self.processed_dir = processed_dir
        self.include_pdb_path = include_pdb_path
        self.exclude_pdb_path = exclude_pdb_path
        self.num_preprocess_jobs = num_preprocess_jobs
        self.benchmark_test = benchmark_test

        os.makedirs(processed_dir, exist_ok=True)
        self.split = split

        self.cluster_patch_size = cluster_patch_size

        self.structures = None
        self._load_structures(reset)

        self.clusters = None
        self.id_to_cluster = None
        self._load_clusters(reset)

        self.ids_in_split = None
        self._load_split(split, split_seed, reset)

        self.transform = transform

    def _load_structures(self, reset):
        if not os.path.exists(self.structure_cache_path) or reset:
            if os.path.exists(self.structure_cache_path):
                os.unlink(self.structure_cache_path)
            self._preprocess_structures()
        print('Loading Structures for {} set...'.format(self.split))
        self.structures = torch.load(self.structure_cache_path)

    @property
    def structure_cache_path(self):
        return os.path.join(self.processed_dir, 'parsed_pair.pt')
        
    def _preprocess_structures(self):
        print('Preprocessing Structures...')
        protein_names = os.listdir(self.data_dir)
        if self.include_pdb_path is not None:
            protein_names_filtered = torch.load(self.include_pdb_path)
            protein_names = [protein_name for protein_name in protein_names 
                             if protein_name in protein_names_filtered]
        if self.exclude_pdb_path is not None:
            protein_names_excluded = torch.load((self.exclude_pdb_path))
            protein_names = [protein_name for protein_name in protein_names 
                             if protein_name[:4] not in protein_names_excluded]
            
        tasks = []
        for protein_name in protein_names:
            task = {'receptor_path': self.data_dir + '/{}/receptor_repaired.pdb'.format(protein_name), 
                    'peptide_path': self.data_dir + '/{}/peptide_repaired.pdb'.format(protein_name),
                    'pdb_name': protein_name}
            tasks.append(task)  

        data_list = []

        def parse_pair(task):
            paired_data = {}
            data_receptor = parse_biopython_structure(task['receptor_path'])
            data_peptide = parse_biopython_structure(task['peptide_path'])
            paired_data['receptor'] = data_receptor
            paired_data['peptide'] = data_peptide
            paired_data['pdb_name'] = task['pdb_name']
            return paired_data
        
        data_list = Parallel(n_jobs=self.num_preprocess_jobs)(
                    delayed(parse_pair)(task)
                    for task in tqdm(tasks)
                )
        
        data_list_filter = []
        for data in data_list:
            data = filter_none(data)
            if data is not None:
                data_list_filter.append(data)

        torch.save(data_list_filter, self.structure_cache_path)

    @property
    def _cluster_path(self):
        return os.path.join(self.processed_dir, 'cluster_result_cluster.tsv')

    def _load_clusters(self, reset):
        if not os.path.exists(self._cluster_path) or reset:
            self._create_clusters()

        clusters, id_to_cluster = {}, {}
        with open(self._cluster_path, 'r') as f:
            for line in f.readlines():
                cluster_name, data_id = line.split()
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(data_id)
                id_to_cluster[data_id] = cluster_name
        self.clusters = clusters
        self.id_to_cluster = id_to_cluster

    def _create_clusters(self):
        seq_records = []
        for id in tqdm(range(len(self.structures))):
            structure = self.structures[id]['receptor']
            peptide_pos = self.structures[id]['peptide']['pos_heavyatom'][:,BBHeavyAtom.CA,:]
            chain_dist = []
            chain_name = []
            for chain_nb in structure['chain_nb'].unique():
                chain_msk = (structure['chain_nb'] == chain_nb)
                chain_pos = structure['pos_heavyatom'][chain_msk][:,BBHeavyAtom.CA,:]
                dist_from_pep = torch.cdist(chain_pos, peptide_pos).min(dim=1)[0]
                patch_idx = torch.argsort(dist_from_pep)[:self.cluster_patch_size]
                dist_patch_mean = dist_from_pep[patch_idx].mean()
                chain_dist.append(dist_patch_mean)
                chain_name.append(chain_nb)

            target_chain = chain_name[np.argmin(chain_dist)]
            chain_msk = (structure['chain_nb'] == target_chain)
            aa_seq = ''.join([AA.map_index_to_type(aa) for aa in structure['aa'][chain_msk].tolist()])
            
            seq_records.append(SeqRecord.SeqRecord(
                Seq.Seq(aa_seq),
                id = self.structures[id]['pdb_name'],
                name = '',
                description = '',
            ))
        fasta_path = os.path.join(self.processed_dir, 'receptor_sequences.fasta')
        SeqIO.write(seq_records, fasta_path, 'fasta')

        cmd = ' '.join([
            'mmseqs', 'easy-cluster',
            os.path.realpath(fasta_path),
            'cluster_result', 'cluster_tmp',
            '--min-seq-id', '0.5',
            '-c', '0.8',
            '--cov-mode', '1',
        ])
        subprocess.run(cmd, cwd=self.processed_dir, shell=True, check=True)

    def _load_split(self, split, split_seed, reset):
        assert split in ('train', 'val', 'test')
        
        if not os.path.exists(self.split_path) or reset:
            self._create_splits(split_seed)

        complex_splits = torch.load(self.split_path)

        complexes_this = complex_splits[split]

        entries = []
        for cplx in complexes_this:
            entries += self.clusters[cplx]
        self.entries = entries
        
        self.ids_in_split = []
        for id, struct in enumerate(self.structures):
            if struct['pdb_name'] in self.entries:
                self.ids_in_split.append(id)

        self.ids_in_split = list(set(self.ids_in_split))

    def _create_splits(self, split_seed):
        complex_list = sorted(self.clusters.keys())
        random.Random(split_seed).shuffle(complex_list)
        split_sizes = [math.ceil(len(complex_list) * 0.9), 
                       math.ceil(len(complex_list) * 1.0)]
        
        if self.benchmark_test:
            split_sizes = [math.ceil(len(complex_list) * 0.0), 
                           math.ceil(len(complex_list) * 0.0)]
            
        complex_splits = {
            'train': complex_list[0 : split_sizes[0]],
            'val': complex_list[split_sizes[0] : split_sizes[1]],
            'test': complex_list[split_sizes[1] : ],
        }
        torch.save(complex_splits, self.split_path)


    def __len__(self):
        return len(self.ids_in_split)

    def __getitem__(self, index):
        id = self.ids_in_split[index]
        data = self.structures[id]
        if self.transform is not None:
            data = self.transform(data)
        return data


    def get_raw(self, index):
        id = self.ids_in_split[index]
        data = self.structures[id]
        return data
