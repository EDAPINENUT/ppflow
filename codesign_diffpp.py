import os
import argparse
import copy
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch

from ppflow.datasets import get_dataset
from ppflow.models import get_model
from ppflow.modules.common.geometry import reconstruct_backbone_partially
from ppflow.modules.common.so3 import so3vec_to_rotation
from ppflow.utils.protein.writers import save_pdb
from ppflow.utils.train import recursive_to
from ppflow.utils.misc import *
from ppflow.utils.data import *
from ppflow.utils.transforms import *
from ppflow.utils.inference import *
from ppflow.utils.transforms import _index_select_data
import os 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('-c', '--config', type=str, default='./configs/test/codesign_diffpp.yml')
    parser.add_argument('-o', '--out_root', type=str, default='./results/diffpp')
    parser.add_argument('-t', '--tag', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)

    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None)

    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args)
    if args.checkpoint is not None:
        config.model.checkpoint = args.checkpoint
    tag_postfix = '_%s' % args.tag if args.tag else ''
    seed_all(args.seed if args.seed is not None else config.sampling.seed)

    # Testset 
    dataset = get_dataset(config.dataset, split='test')

    dr = os.path.join(args.out_root, config_name + tag_postfix)

    if not os.path.exists(dr):

        os.makedirs(dr, exist_ok=True)

    mark = 0

    for i in range(mark, len(dataset)):

        args.index = i
        get_structure = lambda: dataset[args.index]
        get_raw_structure = lambda: dataset.get_raw(args.index)

        # Logging
        structure_ = get_structure()
        raw_strcuture_ = get_raw_structure()

        structure_id = structure_['pdb_name']
        
        log_dir = get_new_log_dir(dr, prefix='%04d_%s' % (args.index, structure_id))
        
        logger = get_logger('sample', log_dir)
        logger.info('Data ID: %s' % structure_['pdb_name'])
        data_native = raw_strcuture_['peptide']
        save_pdb(data_native, os.path.join(log_dir, 'reference.pdb'))

        # Load checkpoint and model
        logger.info('Loading model config and checkpoints: %s' % (config.model.checkpoint))
        ckpt = torch.load(config.model.checkpoint, map_location='cpu')
        cfg_ckpt = ckpt['config']
        model = get_model(cfg_ckpt.model).to(args.device)
        lsd = model.load_state_dict(ckpt['model'])
        logger.info(str(lsd))


        # Start sampling
        collate_fn = PaddingCollate(eight=False)

        data_list_repeat = [ structure_ ] * config.sampling.num_samples
        loader = DataLoader(data_list_repeat, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        count = 0
        
        for batch in tqdm(loader, desc=structure_id, dynamic_ncols=True):
            torch.set_grad_enabled(False)
            model.eval()
            batch = recursive_to(batch, args.device)
            traj_batch = model.sample(batch, sample_opt={
                'pbar': True,
                'sample_structure': config.sampling.sample_structure,
                'sample_sequence': config.sampling.sample_sequence,
            })

            aa_new = traj_batch[0][2]   # 0: Last sampling step. 2: Amino acid.
            pos_atom_new, mask_atom_new = reconstruct_backbone_partially(
                pos_ctx = batch['pos_heavyatom'],
                R_new = so3vec_to_rotation(traj_batch[0][0]),
                t_new = traj_batch[0][1],
                aa = aa_new,
                chain_nb = batch['chain_nb'],
                res_nb = batch['res_nb'],
                mask_atoms = batch['mask_heavyatom'],
                mask_recons = batch['mask_gen_pos'],
            )
            aa_new = aa_new.cpu()
            pos_atom_new = pos_atom_new.cpu()[:,:,:4]
            mask_atom_new = mask_atom_new.cpu()[:,:,:4]

            for i in range(aa_new.size(0)):
                
                peptide_patch_idx = torch.where(batch['mask_gen_pos'][i])[0].cpu()
                data_tmpl = _index_select_data(structure_, peptide_patch_idx)
                aa = aa_new[i][peptide_patch_idx]
                mask_ha = mask_atom_new[i][peptide_patch_idx]
                pos_ha = pos_atom_new[i][peptide_patch_idx]
                pos_ha_translate = pos_ha + structure_['pos_center_org']
                save_path = os.path.join(log_dir, '%04d.pdb' % (count, ))
                
                save_pdb({
                    'chain_nb': data_tmpl['chain_nb'],
                    'chain_id': data_tmpl['chain_id'],
                    'resseq': data_tmpl['resseq'],
                    'icode': data_tmpl['icode'],
                    # Generated
                    'aa': aa,
                    'mask_heavyatom': mask_ha,
                    'pos_heavyatom': pos_ha_translate,
                }, path=save_path)

                count += 1

        logger.info('Finished.\n')

        

if __name__ == '__main__':
    main()
