import shutil
import pickle
import xml.etree.ElementTree as ET
#from plip.structure.preparation import PDBComplex
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
import os.path as osp
from tqdm import tqdm
import numpy as np
from glob import glob
import os
import sys
sys.path.append("..")
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from ..base import merge_protein_ligand

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BondType.names.keys())}
import tempfile
TMPDIR = tempfile.TemporaryDirectory().name

class PDBProtein(object):

    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto'):
        super().__init__()
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break   # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass
        
        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name   # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=np.int_),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=bool),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int_)
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.int_),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, ligand, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        for center in ligand['pos']:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected

    def residues_to_pdb_block(self, residues, name='POCKET'):
        block =  "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block


def parse_pdbbind_index_file(path):
    pdb_id = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#'): continue
        pdb_id.append(line.split()[0])
    return pdb_id


def parse_sdf_file(path):
    mol = Chem.MolFromMolFile(path, sanitize=True)
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=True)))
    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.int_)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    with open(path, 'r') as f:
        sdf = f.read()

    sdf = sdf.splitlines()
    num_atoms, num_bonds = map(int, [sdf[3][0:3], sdf[3][3:6]])
    assert num_atoms == rd_num_atoms

    ptable = Chem.GetPeriodicTable()
    element, pos = [], []
    accum_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    accum_mass = 0.0
    for atom_line in map(lambda x:x.split(), sdf[4:4+num_atoms]):
        x, y, z = map(float, atom_line[:3])
        symb = atom_line[3]
        atomic_number = ptable.GetAtomicNumber(symb.capitalize())
        element.append(atomic_number)
        pos.append([x, y, z])
        
        atomic_weight = ptable.GetAtomicWeight(atomic_number)
        accum_pos += np.array([x, y, z]) * atomic_weight
        accum_mass += atomic_weight

    center_of_mass = np.array(accum_pos / accum_mass, dtype=np.float32)

    element = np.array(element, dtype=np.int_)
    pos = np.array(pos, dtype=np.float32)

    BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
    bond_type_map = {
        1: BOND_TYPES[BondType.SINGLE],
        2: BOND_TYPES[BondType.DOUBLE],
        3: BOND_TYPES[BondType.TRIPLE],
        4: BOND_TYPES[BondType.AROMATIC],
    }
    row, col, edge_type = [], [], []
    for bond_line in sdf[4+num_atoms:4+num_atoms+num_bonds]:
        start, end = int(bond_line[0:3])-1, int(bond_line[3:6])-1
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond_type_map[int(bond_line[6:9])]]

    edge_index = np.array([row, col], dtype=np.int_)
    edge_type = np.array(edge_type, dtype=np.int_)

    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    neighbor_dict = {}

    #used in rotation angle prediction
    for i, atom in enumerate(mol.GetAtoms()):
        neighbor_dict[i] = [n.GetIdx() for n in atom.GetNeighbors()]

    data = {
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'neighbors': neighbor_dict
    }
    return data


def sdf2centroid(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file, sanitize=False)
    lig_xyz = supp[0].GetConformer().GetPositions()
    centroid_x = lig_xyz[:,0].mean()
    centroid_y = lig_xyz[:,1].mean()
    centroid_z = lig_xyz[:,2].mean()
    return centroid_x, centroid_y, centroid_z

def pocket_trunction(pdb_file, threshold=10, outname=None, sdf_file=None, centroid=None):
    pdb_parser = PDBProtein(pdb_file)
    if centroid is None:
        centroid = sdf2centroid(sdf_file)
    else:
        centroid = centroid
    residues = pdb_parser.query_residues_radius(centroid,threshold)
    residue_block = pdb_parser.residues_to_pdb_block(residues)
    if outname is None:
        outname = pdb_file[:-4]+f'_pocket{threshold}.pdb'
    f = open(outname,'w')
    f.write(residue_block)
    f.close()
    return outname

def clear_plip_file(dir):
    files = glob(dir+'/plip*')
    for i in range(len(files)):
        os.remove(files[i])

def read_pkl(pkl_file):
    with open(pkl_file,'rb') as f:
        data = pickle.load(f)
    return data

def write_pkl(data_list, pkl_file):
    with open(pkl_file, 'wb') as f:
        pickle.dump(data_list, f)

def plip_parser(xml_file):
    xml_tree = ET.parse(xml_file)
    report = xml_tree.getroot()
    interaction_ele = report.findall('bindingsite/interactions')
    if len(interaction_ele) == 0:
        return None 
    else:
        interaction_ele = interaction_ele[0]
    result = {}
    for interaction in interaction_ele:
        result['num_hydrophobic'] = len(interaction_ele.findall('hydrophobic_interactions/*'))
        result['num_hydrogen'] = len(interaction_ele.findall('hydrogen_bonds/*'))
        result['num_wb'] = len(interaction_ele.findall('water_bridges/*'))
        result['num_pi_stack'] = len(interaction_ele.findall('pi_stacks/*'))
        result['num_pi_cation'] = len(interaction_ele.findall('pi_cation_interactions/*'))
        result['num_halogen'] = len(interaction_ele.findall('halogen_bonds/*'))
        result['num_metal'] = len(interaction_ele.findall('metal_complexes/*'))
    return result

def patter_analysis(ori_report, gen_report):
    compare = {}
    num_ori = 0
    num_gen = 0
    patterns = ['num_hydrophobic','num_hydrogen','num_wb','num_pi_stack','num_pi_cation','num_halogen','num_metal']
    for pattern in patterns:
        if (ori_report[pattern] == 0)&(gen_report[pattern]==0):
            continue
        num_ori += ori_report[pattern]
        num_gen += gen_report[pattern]
        #compare[pattern] = max(ori_report[pattern] - gen_report[pattern],0)
        try:
            compare[pattern] = min(gen_report[pattern]/ori_report[pattern],1)
        except:
            compare[pattern] = None

    return compare, num_ori, num_gen


def read_sdf(file):
    supp = Chem.SDMolSupplier(file)
    return [i for i in supp]

def merge_lig_pkt(pdb_file, sdf_file, out_name, mol=None):
    '''
    pdb_file = './1A1C_MALDO_2_433_0/1m4n_A_rec_1m7y_ppg_lig_tt_min_0_pocket10.pdb'
    sdf_file = './1A1C_MALDO_2_433_0/1m4n_A_rec_1m7y_ppg_lig_tt_min_0.sdf'
    '''
    protein = Chem.MolFromPDBFile(pdb_file)
    if mol == None:
        ligand = read_sdf(sdf_file)[0]
    else:
        ligand = mol
    complex = Chem.CombineMols(protein,ligand)
    Chem.MolToPDBFile(complex, out_name)


def plip_analysis(pdb_file,out_dir):
    '''
    out_dir 
    '''
    command = 'plip -f {pdb_file} -o {out_dir} -x'.format(pdb_file=pdb_file,
                                                            out_dir = out_dir)
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    proc.communicate()
    return out_dir + '/report.xml'

def plip_analysis_visual(pdb_file,out_dir):
    '''
    out_dir 
    '''
    command = 'plip -f {pdb_file} -o {out_dir} -tpy'.format(pdb_file=pdb_file,
                                                            out_dir = out_dir)
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    proc.communicate()
    return out_dir + '/report.xml'

def interact_analysis(results_pkl, pkt_file, sdf_file, k=10):
    '''
    Designed for a bunch of interaction analysis performed on results file
    results_pkl contained the score and docked poses
    pkt_file contained the .pdb file 
    sdf_file contained the original ligand
    '''
    results = read_pkl(results_pkl)
    scores = []
    mols = []
    for i in range(len(results)):
        try:
            scores.append(results[i][0]['affinity'])
            mols.append(results[i][0]['rdmol'])
        except:
            scores.append(0)
            mols.append(0)
    scores_zip = zip(np.sort(scores),np.argsort(scores))
    scores = np.sort(scores)
    scores_idx = np.argsort(scores)
    sorted_mols = [mols[i] for i in scores_idx]
    truncted_file = pkt_file.split('/')[-1][:-4] + '_pocket10.pdb'
    truncted_file = pocket_trunction(pkt_file, outname=f'./tmp/{truncted_file}',sdf_file=sdf_file)
    if k == 'all':
        k = len(sorted_mols)
    
    gen_report = []
    for i in range(min(k,len(sorted_mols))):
        try:
            merge_lig_pkt(truncted_file, None, f'./tmp/{i}.pdb',mol=sorted_mols[i])
            report = plip_parser(plip_analysis(f'./tmp/{i}.pdb','./tmp'))
            gen_report.append(report)
        except:
            #print(i,'failed')
            ...
    clear_plip_file('./tmp/')
    return gen_report, sdf_file.split('/')[-1]



if __name__ == '__main__':
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_dir', type=str, default='./dataset/ppbench2024')
    parser.add_argument('--ref_dir', type=str, default='./dataset/ppbench2024')
    parser.add_argument('--save_path', type=str, default='./results/ppbench2024')
    args = parser.parse_args()

    logger = get_logger('PLIP_ANALYSIS', args.save_path)
    reports = {'num_hydrophobic': 0, 'num_hydrogen': 0, 'num_wb': 0, 'num_pi_stack': 0, 'num_pi_cation': 0, 'num_halogen': 0, 'num_metal': 0}
    interaction_detected = 0

    for ref_name in ref_dir:
        protein_file = os.path.join('./dataset/ppbench2024', ref_name, 'receptor.pdb')
        ligand_file = os.path.join('./dataset/ppbench2024', ref_name, 'peptide.pdb')

        merged_pdb_file = osp.join(TMPDIR, 'merged.pdb')
        merge_protein_ligand(protein_file, ligand_file, merged_pdb_file)

        report = plip_parser(plip_analysis(merged_pdb_file, 'interaction'))
        if report is not None:
            print(report)
            for k, v in report.items():
                if v >0:
                    reports[k] += 1
            interaction_detected += 1
            for k, v in reports.items():
                logger.info(f'{k}, {v/interaction_detected}')