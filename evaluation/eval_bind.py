import argparse
import os 
import numpy as np
from tools.dock.adcpdock import ADCPDock
from ppflow.utils.misc import get_logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_dir', type=str, default='./results/ppflow/codesign_ppflow/0008_3tzy_2024_01_19__19_16_21')
    parser.add_argument('--ref_dir', type=str, default='./PPDbench/3tzy/')
    parser.add_argument('--save_path', type=str, default='./results/ppflow/codesign_ppflow/0008_3tzy_2024_01_19__19_16_21')
    args = parser.parse_args()

    logger = get_logger('ADCPBinding', args.save_path)

    logger.info('evaluating the samples in {}'.format(args.gen_dir))
    pdb_names = sorted([f for f in os.listdir(args.gen_dir) if f.endswith('bb4.pdb')])[4:20]

    docking_energy = {}
    for pdb_name in pdb_names:
        try:
            pdb_id = pdb_name.split('.')[0]
            dock = ADCPDock(save_dir=args.save_path)
            protein_file = os.path.join(args.ref_dir, 'receptor.pdb')
            ligand_file = os.path.join(args.gen_dir, pdb_name)

            logger.info('docking {} to its receptor...'.format(ligand_file) )
            
            dock.set_receptor(protein_file)
            dock.set_ligand(ligand_file)

            dock.side_chain_packing('ligand')
            docking_energy[pdb_id] = np.sum(dock.dock(save_name='{}_docked.pdb'.format(pdb_id), 
                                                      n_save=1, auto_box=True))
            logger.info('The binding energy:{}'.format(docking_energy[pdb_id]))
        
        except:
            pass
    
    np.save(os.path.join(args.save_path, 'binding_meta'), docking_energy)
        



        



        
        