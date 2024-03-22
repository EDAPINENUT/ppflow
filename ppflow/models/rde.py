import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ppflow.modules.encoders.residue import PerResidueEncoder
from ppflow.modules.encoders.pair import ResiduePairEncoder
from ppflow.datasets.constants import max_num_heavyatoms, BBHeavyAtom, num_aa_types, chi_angles_atoms
from ppflow.modules.encoders.ga import GAEncoder
from ppflow.modules.common.geometry import construct_3d_basis
from ppflow.modules.flows.spline import ContextualCircularSplineFlow
from ppflow.modules.flows.coupling import ContextualCircularSplineCouplingLayer
from ppflow.modules.flows.container import ContextualSequentialFlow
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
    assert cfg_flow.num_blocks >= num_chis
    blocks = []
    if num_chis == 1:
        # Only 1 dimension, cannot do coupling.
        for i in range(cfg_flow.num_blocks):
            blocks.append(ContextualCircularSplineFlow(
                n_context_dims = n_context_dims,
                n_hidden_dims = cfg_flow.num_hidden_dims,
                n_spline_bins = cfg_flow.num_spline_bins,
            ))
    else:
        for i in range(cfg_flow.num_blocks):
            update_dim = [i%num_chis, ]
            blocks.append(ContextualCircularSplineCouplingLayer(
                n_dims = num_chis,
                mapping_dims = update_dim,
                n_context_dims = n_context_dims,
                n_hidden_dims = cfg_flow.num_hidden_dims,
                n_spline_bins = cfg_flow.num_spline_bins,
            ))
    sequential_flow = ContextualSequentialFlow(blocks)
    return sequential_flow

@register_model('rde')
class CircularSplineRotamerDensityEstimator(nn.Module):

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
        self.flows = nn.ModuleList([
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

    def forward(self, batch):
        loss_dict = {}
        c = self.encode(batch)

        zs, logabsdets, logprobs = [], [], []
        for n_chis in range(1, 4+1):
            z, logabsdet = self.flows[n_chis-1](
                batch['chi_native'][:, :, :n_chis],
                c, inverse=False
            )
            logp = _latent_log_prob(z, n_chis) + logabsdet
            zs.append(z)
            logabsdets.append(logabsdet)
            logprobs.append(logp)
        loss_dict.update(self._mle_loss(batch, logprobs))

        return loss_dict

    def sample(self, batch, n_samples=1, residue_subset=None):
        c = self.encode(batch)

        if residue_subset is not None:
            c = c[:, residue_subset, :]
        
        if n_samples > 1:
            c_shape = [n_samples] + list(c.shape)   # (n_samples, N, L, d)
            c = c.unsqueeze(0).expand(c_shape)

        zs = [
            sample_latent(c.shape[:-1], n_chis, device=c.device)
            for n_chis in range(1, 4+1)
        ]   # [(N, L, 1), ..., (N, L, 4)]
        logpzs = [
            _latent_log_prob(zs[n_chis-1], n_chis)
            for n_chis in range(1, 4+1)
        ]

        xs, logprobs = [], []
        for n_chis in range(1, 4+1):
            x, logabsdet = self.flows[n_chis-1](
                zs[n_chis-1], c, inverse=True
            )   # (N, L, n_chis), (N, L)
            x = F.pad(x, pad=(0, 4-n_chis), value=0) # (N, L, 4)
            logpx = logpzs[n_chis-1] - logabsdet     # (N, L)
            xs.append(x)
            logprobs.append(logpx)
        
        xs = F.pad(
            torch.stack(xs, dim=-1), 
            pad=(1, 0), value=0
        )   # (N, L, 4, 5)
        logprobs = F.pad(
            torch.stack(logprobs, dim=-1),
            pad=(1, 0), value=0
        )   # (N, L, 5)

        n_chis_all = self.num_chis_of_aa[batch['aa'].flatten()].reshape_as(batch['aa']) # (N, L)
        xs = torch.gather(
            xs, dim=-1, 
            index=n_chis_all[:, :, None, None].expand_as(xs)
        )[..., 0]   # (N, L, 4)
        logprobs = torch.gather(
            logprobs, dim=-1,
            index=n_chis_all[:, :, None].expand_as(logprobs)
        )[..., 0]   # (N, L)

        return xs, logprobs

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
