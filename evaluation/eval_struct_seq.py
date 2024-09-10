import argparse
from tools.score import rosetta_energy, similarity, check_validity, foldx_energy
import os 
import numpy as np
import logging

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_dir', type=str, default='./results/diffpp/codesign_diffpp/0000_2qbx_2024_01_18__18_37_25')
    parser.add_argument('--ref_dir', type=str, default='./PPDbench/2qbx/')
    parser.add_argument('--save_path', type=str, default='./results/diffpp/codesign_diffpp/0000_2qbx_2024_01_18__18_37_25')
    args = parser.parse_args()
    
    logger.info('evaluating the samples in {}'.format(args.gen_dir))
    pdb_names = sorted([f for f in os.listdir(args.gen_dir) if f.endswith('.pdb')])[:20]
    valid_pdb_names = []

    for pdb_name in pdb_names:
        try:
            valid_bool = check_validity.bond_length_validation(os.path.join(args.gen_dir, pdb_name))
            if valid_bool:
                valid_pdb_names.append(pdb_name)
        except:
            pass
    
    metrics = {'dg':[], 'novel':[], 'seq_div':[], 'str_div':[]}
    metrics['validity'] = len(valid_pdb_names) / len(pdb_names)
    ref_file = os.path.join(args.ref_dir, 'peptide.pdb')
    protein_file = os.path.join(args.ref_dir, 'receptor.pdb')
    
    score_calculator = foldx_energy.FoldXGibbsEnergy()
    score_calculator.set_ligand(ref_file)
    score_calculator.set_receptor(protein_file)
    score_calculator.side_chain_packing('ligand')
    dg = score_calculator.cal_interface_energy()
    
    metrics['dg_ref'] = dg
    metrics['dg'] = []
    for pdb_name in valid_pdb_names:
        try:
            score_calculator = foldx_energy.FoldXGibbsEnergy()
            gen_bb_file = os.path.join(args.gen_dir, pdb_name)
            score_calculator.set_ligand(gen_bb_file)
            score_calculator.set_receptor(protein_file)
            score_calculator.side_chain_packing('ligand')
            dg = score_calculator.cal_interface_energy()
            metrics['dg'].append(dg)

            novel_bool = similarity.check_novelty(gen_bb_file, ref_file)
            metrics['novel'].append(novel_bool)

            for pdb_name_vs in valid_pdb_names:
                vs_pdb_file = os.path.join(args.gen_dir, pdb_name_vs)

                seq_sim = similarity.seq_similarity(vs_pdb_file, gen_bb_file)
                metrics['seq_div'].append(1 - seq_sim)

                str_sim = similarity.tm_score(vs_pdb_file, gen_bb_file)
                metrics['str_div'].append(1 - str_sim)
                
        except:
            pass
    
    np.save(os.path.join(args.save_path, 'metrics_meta'), metrics)
    print('Save file to {}'.format(os.path.join(args.save_path, 'metrics_meta.npy')))
        



        



        
        