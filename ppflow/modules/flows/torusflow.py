from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm

from ppflow.modules.common.geometry import *
from ppflow.modules.encoders.ga import GAEncoder
from .flow_sampler import SO3FlowSampler, R3FlowSampler, TorusFlowSampler, TypeFlowSampler
from ..common.so3 import *
from ..common.geometry import construct_3d_basis, manifold_to_euclid
from ..common.so2 import regularize
from ppflow.datasets.constants import max_num_heavyatoms, BBHeavyAtom
from scipy.spatial.transform import Rotation
# from ..common.nerf import nerf_build_batch_bb4

def loss_rot_func(v, u):
    size = list(v.shape[:-2])
    ncol = v.numel() // 3

    RT_pred = v.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)
    RT_true = u.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)

    ones = torch.ones([ncol, ], dtype=torch.long, device=v.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  # (ncol*3, )
    loss = loss.reshape(size + [3]).sum(dim=-1)  
    return loss

def loss_seq_func(p_true, p_est):
    return F.cross_entropy(
        input=torch.log(p_est + 1e-8).transpose(-1,1), 
        target=p_true.argmax(-1), 
        reduction='none'
    )


class VectorFieldNet(nn.Module):

    def __init__(self, res_feat_dim, pair_feat_dim, num_layers, encoder_opt={}):
        super().__init__()
        self.current_sequence_embedding = nn.Embedding(25, res_feat_dim)  # 22 is padding
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 2, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
        )
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, **encoder_opt)

        self.eps_crd_net = nn.Sequential(
            nn.Linear(res_feat_dim+12, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_rot_net = nn.Sequential(
            nn.Linear(res_feat_dim+12, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_seq_net = nn.Sequential(
            nn.Linear(res_feat_dim+12, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 20), nn.Softmax(dim=-1) 
        )

        self.eps_dihed_net = nn.Sequential(
            nn.Linear(res_feat_dim+12, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

    def forward(self, d_t, s_t, X_t, R_t, res_feat, pair_feat, t, 
                mask_gen_d, mask_gen_aa, mask_gen_pos, mask_res):
        """
        Args:
            v_t:    (N, L, 3).
            p_t:    (N, L, 3).
            s_t:    (N, L).
            res_feat:   (N, L, res_dim).
            pair_feat:  (N, L, L, pair_dim).
            beta:   (N,).
            mask_gen_pos:    (N, L).
            mask_res:       (N, L).
        Returns:
            v_next: UPDATED (not epsilon) SO3-vector of orietnations, (N, L, 3).
            eps_pos: (N, L, 3).
        """
        N, L = mask_res.size()

        # s_t = s_t.clamp(min=0, max=19)  # TODO: clamping is good but ugly.
        res_feat = self.res_feat_mixer(torch.cat([res_feat, self.current_sequence_embedding(s_t)], dim=-1)) # [Important] Incorporate sequence at the current step.
        res_feat = self.encoder(R_t, X_t, res_feat, pair_feat, mask_res)

        t_embed = torch.stack([t, torch.sin(t), torch.cos(t)], dim=-1)[:, None, :].expand(N, L, 3)
        d_embed = torch.cat([d_t, torch.sin(d_t), torch.cos(d_t)], dim=-1)
        in_feat = torch.cat([res_feat, t_embed, d_embed], dim=-1)

        # Position changes
        eps_crd = self.eps_crd_net(in_feat)    # (N, L, 3)
        eps_pos = apply_rotation_to_vector(R_t, eps_crd)  # (N, L, 3)
        mask_gen = mask_gen_pos.clone()
        mask_gen_pos = mask_gen[:, :, None].expand_as(eps_pos)
        eps_pos = torch.where(mask_gen_pos, eps_pos, torch.zeros_like(eps_pos))
        vp_t = (eps_pos * mask_gen[:, :, None]).sum(1) / mask_gen[:, :, None].sum(1)

        # New orientation
        eps_rot = self.eps_rot_net(in_feat)    # (N, L, 3)
        mask_gen_rot = mask_gen[:, :, None].expand_as(eps_rot)
        eps_rot = torch.where(mask_gen_rot, eps_rot, torch.zeros_like(eps_rot))
        eps_rot = (eps_rot * mask_gen[:, :, None]).sum(1) / mask_gen[:, :, None].sum(1)
        vr_t = quaternion_1ijk_to_rotation_matrix(eps_rot) # (N, L, 3, 3)
        
        eps_dihed = self.eps_dihed_net(in_feat)
        vd_t = torch.where(mask_gen_d, eps_dihed, torch.zeros_like(eps_dihed))

        # New sequence categorical distributions
        eps_c = self.eps_seq_net(in_feat)
        vc_t = torch.where(mask_gen_aa[:, :, None], eps_c, torch.zeros_like(eps_c))

        return vp_t, vr_t, vd_t, vc_t


class TorusFlow(nn.Module):

    def __init__(
        self, 
        res_feat_dim, 
        pair_feat_dim, 
        eps_net_opt={}, 
        rot_sampler_opt={}, 
        tra_sampler_opt={}, 
        local_tor_sampler_opt={},
        seq_type_sampler_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
    ):
        super().__init__()
        self.eps_net = VectorFieldNet(res_feat_dim, pair_feat_dim, **eps_net_opt)
        self.rot_sampler = SO3FlowSampler(**rot_sampler_opt)
        self.tra_sampler = R3FlowSampler(**tra_sampler_opt)
        self.local_tor_sampler = TorusFlowSampler(**local_tor_sampler_opt)
        self.seq_type_sampler = TypeFlowSampler(**seq_type_sampler_opt)

        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, -1))
        self.register_buffer('_dummy', torch.empty([0, ]))
        
        self.num_class = self.seq_type_sampler.num_classes

    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p


    def forward(self, R_1, p_1, d_1, s_1, X_1, res_feat, pair_feat, 
                mask_gen_d, mask_gen_aa, mask_gen_pos, mask_res, 
                denoise_structure, denoise_sequence, t=None):
        N, L = res_feat.shape[:2]
        if t == None:
            t = torch.rand((N,)).type_as(p_1).to(p_1.device)

        if denoise_structure:
            ur_t, dr_t = self.rot_sampler.sample_field(R_1, t)
            up_t, p_t = self.tra_sampler.sample_field(p_1, t)
            ud_t, d_t = self.local_tor_sampler.sample_field(d_1, t, mask_gen_d)
            
        else:
            dr_t = torch.eye(3).to(R_1.device).repeat(N, 1, 1)
            p_t = p_1.clone()
            d_t = d_1.clone()
            ur_t = torch.eye(3).to(R_1.device).repeat(N, 1, 1)
            up_t = torch.zeros_like(p_t)
            ud_t = torch.zeros_like(d_t)
        
        X_t = manifold_to_euclid(dr_t, p_t, d_t, X_1, mask_gen_pos, dr=True)
                
        X_t, R_t = X_t[:, :, BBHeavyAtom.CA], construct_3d_basis(X_t[:, :, BBHeavyAtom.CA],
                                                                 X_t[:, :, BBHeavyAtom.C],
                                                                 X_t[:, :, BBHeavyAtom.N],)
        

        if denoise_sequence:
            # Add noise to sequence
            uc_t, s_t, ct = self.seq_type_sampler.sample_field(s_1, t, mask_gen_aa)
        else:
            s_t = s_1.clone()
            uc_t = torch.zeros_like(s_t.unsqueeze(-1)).float().repeat(1, 1, self.num_class)

        vp_t, vr_t, vd_t, vc_t = self.eps_net(
            d_t, s_t, X_t, R_t, res_feat, pair_feat, t, 
            mask_gen_d, mask_gen_aa, mask_gen_pos, mask_res
        )   # (N, L, 3), (N, L, 3, 3), (N, L, 3), (N, L, self.num_class), (N, L)

        loss_dict = {}

        # Rotation loss
        loss_rot = loss_rot_func(vr_t, ur_t).mean(dim=-1) # (N, )
        # loss_rot = (loss_rot * mask_gen_pos).sum() / (mask_gen_pos.sum().float() + 1e-8)
        loss_dict['rot'] = loss_rot

        # Position loss
        loss_pos = F.mse_loss(vp_t, up_t, reduction='none').sum(dim=-1).mean(dim=-1)  # (N, L)
        loss_dict['pos'] = loss_pos

        # Dihedral loss
        loss_dihed = F.mse_loss(vd_t, ud_t, reduction='none')
        loss_dihed = (loss_dihed * mask_gen_d).sum() / (mask_gen_d.sum().float() + 1e-8)
        loss_dict['dihed'] = loss_dihed

        # Sequence categorical loss
        post_true = self.seq_type_sampler.posterior(ct=ct, uc_t=uc_t, t=t)
        post_esti = self.seq_type_sampler.posterior(ct=ct, uc_t=vc_t, t=t)
        loss_seq = loss_seq_func(post_true, post_esti)
        loss_seq = (loss_seq * mask_gen_aa).sum() / (mask_gen_aa.sum().float() + 1e-8)
        loss_dict['seq'] = loss_seq

        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        s, r, p, d, X_ctx,
        res_feat, pair_feat, mask_gen_d,
        mask_gen_pos, mask_gen_aa, mask_res, 
        sample_structure=True, sample_sequence=True,
        pbar=False, num_steps=500,
    ):
        """
        Args:
            v:  Orientations of contextual residues, (N, L, 3).
            p:  Positions of contextual residues, (N, L, 3).
            s:  Sequence of contextual residues, (N, L).
        """
        N, L = d.shape[:2]

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            r_init = torch.tensor(Rotation.random(N).as_matrix()).to(r).reshape(N, 3, 3)
            p_init = torch.randn_like(p)
            d_init = torch.where(mask_gen_d, regularize(torch.randn_like(d)), d)
        else:
            r_init, p_init, d_init = r, p, d

        if sample_sequence:
            s_rand = torch.randint_like(s, low=0, high=19)
            s_init = torch.where(mask_gen_aa, s_rand, s)
            c_init = torch.ones_like(F.one_hot(s_rand, self.num_class)) * 1/self.num_class
        else:
            s_init = s
            c_init = torch.zeros_like(F.one_hot(s_init, self.num_class)) * 1/self.num_class

        traj = {0: (r_init, p_init, d_init, s_init, c_init)}

        if pbar:
            pbar = functools.partial(tqdm, total=num_steps, desc='Sampling')
        else:
            pbar = lambda x: x

        for t in pbar(range(0, num_steps)):
            r_t, p_t, d_t, s_t, c_t = traj[t]
            
            dt = torch.full([N, ], fill_value=1.0/num_steps, dtype=torch.float32, device=self._dummy.device)
            t_tensor = torch.full([N, ], fill_value=(t+1)/num_steps, dtype=torch.float32, device=self._dummy.device)

            X_t = manifold_to_euclid(r_t, p_t, d_t, X_ctx, mask_gen_pos)
            _, R_t_global = global_frame(X_t, mask_gen_pos)
            X_t, R_t = X_t[:, :, BBHeavyAtom.CA], construct_3d_basis(X_t[:, :, BBHeavyAtom.CA],
                                                                     X_t[:, :, BBHeavyAtom.C],
                                                                     X_t[:, :, BBHeavyAtom.N],)

            vp_t, vr_t, vd_t, vc_t = self.eps_net(
                    d_t, s_t, X_t, R_t, res_feat, pair_feat, t_tensor, 
                    mask_gen_d, mask_gen_aa, mask_gen_pos, mask_res
                )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            r_next = self.rot_sampler.inference(r_t, vr_t, dt)
            p_next = self.tra_sampler.inference(p_t, vp_t, dt)
            d_next = self.local_tor_sampler.inference(d_t, vd_t, dt, mask_gen_d)
            s_next, c_next = self.seq_type_sampler.inference(s_t, c_t, vc_t, dt, mask_gen_aa)

            if not sample_structure:
                r_next, p_next, d_next = r_t, p_t, d_t
            if not sample_sequence:
                s_next = s_t

            traj[t+1] = (r_next, p_next, regularize(d_next), s_next, c_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj

    
    @torch.no_grad()
    def optimize(
        self, 
        s, r, p, d, X_ctx, start_t,
        res_feat, pair_feat, mask_gen_d,
        mask_gen_pos, mask_gen_aa, mask_res, 
        sample_structure=True, sample_sequence=True,
        pbar=False, num_steps=500,
    ):
        N, L = d.shape[:2]
        t0 = start_t
        t0 = torch.ones((N,)).to(p) * t0

        if sample_structure:
            ur_t, r_init = self.rot_sampler.sample_field(r, t0)
            up_t, p_init = self.tra_sampler.sample_field(p, t0)
            ud_t, d_init = self.local_tor_sampler.sample_field(d, t0, mask_gen_d)
        else:
            r_init, p_init, d_init = r, p, d
        
        if sample_sequence:
            uc_t, s_init, c_init = self.seq_type_sampler.sample_field(s, t0, mask_gen_aa)
        else:
            s_init = s.clone()
            uc_t = torch.zeros_like(s_t.unsqueeze(-1)).float().repeat(1, 1, self.num_class)
        
        
        traj = {int(num_steps * start_t): (r_init, p_init, d_init, s_init, c_init)}

        t_range = [t for t in range(int(num_steps * start_t), num_steps)]

        if pbar:
            pbar = functools.partial(tqdm, total=len(t_range), desc='Optimizing')
        else:
            pbar = lambda x: x
        

        for t in pbar(t_range):
            r_t, p_t, d_t, s_t, c_t = traj[t]
            
            dt = torch.full([N, ], fill_value=1.0/num_steps, dtype=torch.float32, device=self._dummy.device)
            t_tensor = torch.full([N, ], fill_value=t/num_steps, dtype=torch.float32, device=self._dummy.device)

            X_t = manifold_to_euclid(r_t, p_t, d_t, X_ctx, mask_gen_pos)
        
            X_t, R_t = X_t[:, :, BBHeavyAtom.CA], construct_3d_basis(X_t[:, :, BBHeavyAtom.CA],
                                                                     X_t[:, :, BBHeavyAtom.C],
                                                                     X_t[:, :, BBHeavyAtom.N],)
                                                                     

            vp_t, vr_t, vd_t, vc_t = self.eps_net(
                    d_t, s_t, X_t, R_t, res_feat, pair_feat, t_tensor, 
                    mask_gen_d, mask_gen_aa, mask_gen_pos, mask_res
                )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            r_next = self.rot_sampler.inference(r_t, vr_t, dt)
            p_next = self.tra_sampler.inference(p_t, vp_t, dt)
            d_next = self.local_tor_sampler.inference(d_t, vd_t, dt, mask_gen_d)
            s_next, c_next = self.seq_type_sampler.inference(s_t, c_t, vc_t, dt, mask_gen_aa)

            if not sample_structure:
                r_next, p_next, d_next = r_t, p_t, d_t
            if not sample_sequence:
                s_next = s_t

            traj[t+1] = (r_next, p_next, regularize(d_next), s_next, c_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj
