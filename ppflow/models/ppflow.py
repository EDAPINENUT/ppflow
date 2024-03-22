
import torch
import torch.nn as nn

from ppflow.modules.common.geometry import construct_3d_basis, global_frame
from ppflow.modules.common.so3 import rotation_to_so3vec
from ppflow.modules.encoders.residue import ResidueEmbedding
from ppflow.modules.encoders.pair import ResiduePairEncoder
from ppflow.modules.flows.torusflow import TorusFlow
from ppflow.datasets.constants import max_num_heavyatoms, BBHeavyAtom
from ._base import register_model
from torch_scatter import scatter_min, scatter_max

resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
 }


    

@register_model('ppflow')
class PPFlowMatching(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]
        self.residue_embed = ResidueEmbedding(cfg.res_feat_dim, num_atoms)
        self.pair_embed = ResiduePairEncoder(cfg.pair_feat_dim, num_atoms)

        self.flow = TorusFlow(
            cfg.res_feat_dim,
            cfg.pair_feat_dim,
            **cfg.flow,
        )

    def encode(self, batch, remove_structure, remove_sequence):
        """
        Returns:
            res_feat:   (N, L, res_feat_dim)
            pair_feat:  (N, L, L, pair_feat_dim)
        """
        # This is used throughout embedding and encoding layers
        #   to avoid data leakage.
        structure_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], 
            ~batch['mask_gen_pos']     # Context means ``not generated''
        ) if remove_structure else None
        sequence_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], 
            ~batch['mask_gen_aa']     # Context means ``not generated''
        ) if remove_sequence else None

        res_feat = self.residue_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            fragment_type = batch['fragment_type'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
        )

        pair_feat = self.pair_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
        )

        R = construct_3d_basis(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )
        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        p_global, R_global = global_frame(batch['pos_heavyatom'][:,:,:3,:],
                                          batch['mask_gen_pos'])
        
        d_local = torch.stack([batch['phi'], batch['psi'], batch['omega']], dim=-1)
        mask_d = torch.stack([batch['phi_mask'], batch['psi_mask'], batch['omega_mask']], dim=-1)
        mask_gen_d = torch.logical_and(mask_d, batch['mask_gen_pos'].unsqueeze(-1))
        
        return res_feat, pair_feat, R, p, R_global, p_global, d_local, mask_gen_d
    
    def forward(self, batch):
        mask_gen_pos = batch['mask_gen_pos']
        mask_gen_aa = batch['mask_gen_aa']

        mask_res = batch['mask']
        res_feat, pair_feat, R_local, p_local, R_1, p_1, d_1, mask_gen_d = self.encode(
            batch,
            remove_structure = self.cfg.get('train_structure', True),
            remove_sequence = self.cfg.get('train_sequence', True)
        )
        s_1 = batch['aa']
        X_1 = batch['pos_heavyatom'][:,:,:4,:]

        loss_dict = self.flow(
            R_1, p_1, d_1, s_1, X_1, res_feat, pair_feat, 
            mask_gen_d, mask_gen_aa, mask_gen_pos, mask_res,
            denoise_structure = self.cfg.get('train_structure', True),
            denoise_sequence  = self.cfg.get('train_sequence', True),
        )
        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        batch, 
        sample_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_gen_pos = batch['mask_gen_pos']
        mask_gen_aa = batch['mask_gen_aa']
        mask_res = batch['mask']
        res_feat, pair_feat, R_local, p_local, R_1, p_1, d_1, mask_gen_d = self.encode(
            batch,
            remove_structure = sample_opt.get('sample_structure', True),
            remove_sequence = sample_opt.get('sample_sequence', True)
        )
        s_1 = batch['aa']
        X_1 = batch['pos_heavyatom'][:,:,:4,:]
        traj = self.flow.sample(s_1, R_1, p_1, d_1, X_1, res_feat, pair_feat, mask_gen_d,
                                mask_gen_pos, mask_gen_aa, mask_res, **sample_opt)
        return traj

    @torch.no_grad()
    def optimize(
        self, 
        batch, 
        interm_t, 
        optimize_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_gen_pos = batch['mask_gen_pos']
        mask_gen_aa = batch['mask_gen_aa']

        mask_res = batch['mask']
        res_feat, pair_feat, R_local, p_local, R_1, p_1, d_1, mask_gen_d = self.encode(
            batch,
            remove_structure = self.cfg.get('train_structure', True),
            remove_sequence = self.cfg.get('train_sequence', True)
        )
        s_1 = batch['aa']
        X_1 = batch['pos_heavyatom'][:,:,:4,:]

        traj = self.flow.optimize(s_1, R_1, p_1, d_1, X_1, interm_t, res_feat, pair_feat, mask_gen_d,
                                  mask_gen_pos, mask_gen_aa, mask_res, **optimize_opt)
        return traj
