import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ppflow.modules.encoders.residue import PerResidueEncoder
from ppflow.modules.encoders.pair import ResiduePairEncoder
from ppflow.modules.encoders.ga import GAEncoder
from ppflow.datasets.constants import max_num_heavyatoms, BBHeavyAtom, num_aa_types, chi_angles_atoms
from ppflow.modules.distributions.vonmises import VonMisesMix
from ppflow.modules.common.layers import LeakyMLP
from ppflow.modules.common.geometry import construct_3d_basis
from ._base import register_model

resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}

def _latent_log_prob(z, num_chis):
    assert z.size(-1) == num_chis
    volume = (2*np.pi) ** num_chis
    logp = np.log(1 / volume)
    shape = list(z.size())[:-1]
    return torch.full(shape, logp, device=z.device, dtype=torch.float)


def sample_latent(shape, num_chis, device):
    shape = [*shape, num_chis]
    z = torch.rand(shape, device=device) * (2*np.pi) - np.pi
    return z


def _get_latent_log_prob_fn(num_chis):
    return functools.partial(_latent_log_prob, num_chis=num_chis)


def _build_flow(num_chis, n_context_dims, cfg_flow):
    blocks = [LeakyMLP(
        dim_start = n_context_dims,
        dim_hidden = cfg_flow.num_hidden_dims,
        dim_end = 3 * cfg_flow.num_mixtures,
        num_layer = cfg_flow.num_hidden_layers,
    ) for chi in range(num_chis)] # 
    # blocks.append(LeakyMLP(
    #     dim_start = n_context_dims,
    #     dim_hidden = cfg_flow.num_hidden_dims,
    #     dim_end = 3 * cfg_flow.num_mixtures,
    #     num_layer = cfg_flow.num_hidden_layers,
    # ) for chi in range(num_chis))
    sequential_flow = nn.ModuleList(blocks)
    return sequential_flow

@register_model('rdevm')
class VonMisesRotamerDensityEstimator(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'backbone+CB')]
        
        # Encoding
        self.single_encoder = PerResidueEncoder(
            feat_dim=cfg.res_feat_dim,
            max_num_atoms=num_atoms,    # N, CA, C, O, CB,
        )
        self.masked_bias = nn.Embedding(
            num_embeddings = 2,
            embedding_dim = cfg.res_feat_dim,
            padding_idx = 0,
        )
        self.pair_encoder = ResiduePairEncoder(
            feat_dim=cfg.pair_feat_dim,
            max_num_atoms=num_atoms,    # N, CA, C, O, CB,
        )
        self.attn_encoder = GAEncoder(node_feat_dim=cfg.res_feat_dim,
                                      pair_feat_dim=cfg.pair_feat_dim,
                                      num_layers=cfg.num_layers,
                                      **cfg.get('encoder_opt', {}))

        # Flows
        n_context_dim = cfg.res_feat_dim

        self.vonmise_heads = nn.ModuleList([
            _build_flow(n_chis, n_context_dim, cfg.flow)
            for n_chis in range(1, 4+1)
        ])

        self.register_buffer('num_chis_of_aa', torch.tensor(
            data=[
                len(chi_angles_atoms[i]) if i < len(chi_angles_atoms) else 0
                for i in range(num_aa_types+1)
            ],
            dtype=torch.long,
        ))

    def encode(self, batch):
        res_mask = batch['mask_heavyatom'][:, :, BBHeavyAtom.CA]
        chi = batch['chi_corrupt']
        pos_atoms = batch['pos_heavyatom']

        x = self.single_encoder(
            aa = batch['aa'],
            phi = batch['phi'], phi_mask = batch['phi_mask'],
            psi = batch['psi'], psi_mask = batch['psi_mask'],
            chi = chi, chi_mask = batch['chi_mask'],
            residue_mask = res_mask,
        )
        b = self.masked_bias(batch['chi_masked_flag'].long())
        x = x + b
        z = self.pair_encoder(
            aa = batch['aa'], 
            res_nb = batch['res_nb'], 
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'], 
            mask_atoms = batch['mask_heavyatom'],
        )

        R = construct_3d_basis(
            pos_atoms[:, :, BBHeavyAtom.CA], 
            pos_atoms[:, :, BBHeavyAtom.C], 
            pos_atoms[:, :, BBHeavyAtom.N]
        )
        t = pos_atoms[:, :, BBHeavyAtom.CA]

        x = self.attn_encoder(
            R = R, t = t,
            res_feat = x, pair_feat = z, 
            mask = res_mask
        )
        return x

    def _mle_loss(self, batch, logprobs):
        """
        Args:
            logprobs: [(N, L), ...].
        """
        n_chis_data = batch['chi_mask'].sum(-1) # (N, L)
        chi_complete = batch['chi_complete']    # (N, L)
        loss_dict = {}
        for n_chis in range(1, 4+1):
            logprob = logprobs[n_chis-1]    # (N, L)
            supervise_mask = torch.logical_and(
                torch.logical_and(chi_complete, n_chis_data == n_chis),
                batch['chi_corrupt_flag']
            )   # (N, L)
            sum_mask = supervise_mask.sum().float()
            loss = ((-logprob) * supervise_mask).sum() / (sum_mask + 1e-8)
            loss_dict['mle_%dchis' % n_chis] = loss
        return loss_dict
    
    def _pll(self, angles, params):
        """Negative log likelihood of von Mises distribution.
        
        Args:
            angles: (N, L, 1)
            params: (N, L, 3 * num_mixtures)
        """
        loc_pred, conc_pred, logits = params.split(self.cfg.flow.num_mixtures, dim=-1)
        distribution = VonMisesMix(loc_pred, conc_pred, logits)
        loss = distribution.log_prob(angles.unsqueeze(-1))
        return loss

    def forward(self, batch):
        loss_dict = {}
        c = self.encode(batch)
        
        log_prob_list = []
        for n_chis in range(1, 4+1):
            log_prob_n_chis = []
            for chis in range(n_chis):
                params = self.vonmise_heads[n_chis-1][chis](
                    c,
                )
                angles = batch['chi_native'][:, :, chis]
                log_prob_n_chis.append(self._pll(angles, params))
            log_prob_list.append(torch.stack(log_prob_n_chis, dim=-1).sum(-1))

        loss_dict.update(self._mle_loss(batch, log_prob_list))

        return loss_dict

    def sample_angles(self, params, n_samples):
        """Sample angles from von Mises distribution.
        
        Args:
            params: (N, L, 3 * num_mixtures)
        """
        loc_pred, conc_pred, logits = params.split(self.cfg.flow.num_mixtures, dim=-1)
        distribution = VonMisesMix(loc_pred, conc_pred, logits)
        angles = distribution.sample(n_samples)

        log_probs = distribution.log_prob_samples(angles)
        return angles, log_probs

    def sample(self, batch, n_samples=1, residue_subset=None):
        c = self.encode(batch)

        if residue_subset is not None:
            c = c[:, residue_subset, :]
        
        if n_samples > 1:
            c_shape = [n_samples] + list(c.shape)   # (n_samples, N, L, d)
            c = c.unsqueeze(0).expand(c_shape)

        xs = []
        log_ps = []
        for n_chis in range(1, 4+1):
            angle_n_chis = []
            log_probs_n_chis = []
            for chis in range(n_chis):
                params = self.vonmise_heads[n_chis-1][chis](
                        c,
                    )
                angles, log_probs = self.sample_angles(params, n_samples)
                angle_n_chis.append(angles)
                log_probs_n_chis.append(log_probs)

            angle_n_chis = torch.stack(angle_n_chis, dim=-1)
            angle_n_chis = F.pad(angle_n_chis, pad=(0, 4-n_chis), value=0)
            log_probs_n_chis = torch.stack(log_probs_n_chis, dim=-1).sum(-1)
            # log_probs_n_chis = F.pad(log_probs_n_chis, pad=(0, 4-n_chis), value=0)
                
            xs.append(angle_n_chis)
            log_ps.append(log_probs_n_chis)
        
        xs = F.pad(
            torch.stack(xs, dim=-1), 
            pad=(1, 0), value=0
        )   # (N, L, 4, 5)
        log_ps = F.pad(
            torch.stack(log_ps, dim=-1),
            pad=(1, 0), value=0
        )   # (N, L, 5)


        n_chis_all = self.num_chis_of_aa[batch['aa'].flatten()].reshape_as(batch['aa']) # (N, L)
        xs = torch.gather(
            xs, dim=-1, 
            index=n_chis_all[:, :, None, None].expand_as(xs)
        )[..., 0]   # (N, L, 4)
        log_ps = torch.gather(
            log_ps, dim=-1,
            index=n_chis_all[:, :, None].expand_as(log_ps)
        )[..., 0] 

        return xs, log_ps

    def pack(self, batch, n_samples=100):
        xs, logprobs = self.sample(batch, n_samples=n_samples)  # (s, N, L, 4), (s, N, L)
        logprobs_max, smp_idx = logprobs.max(dim=0)    # (N, L)
        smp_idx = smp_idx[None, :, :, None].repeat(1, 1, 1, 4)  # (1, N, L, 4)
        xs = torch.gather(xs, dim=0, index=smp_idx).squeeze(0)
        return xs, logprobs_max

    def entropy(self, batch, n_samples=200):
        _, logprobs = self.sample(batch, n_samples=n_samples)
        entropy = -logprobs.mean(dim=0)    # (B, L)
        return entropy
