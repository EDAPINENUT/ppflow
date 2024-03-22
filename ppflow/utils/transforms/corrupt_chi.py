from typing import Any
import torch
import numpy as np
from ._base import register_transform, _get_CB_positions

    
@register_transform('corrupt_chi_angle')
class CorruptChiAngle(object):

    def __init__(self, ratio_mask=0.1, add_noise=True, maskable_flag_attr=None):
        super().__init__()
        self.ratio_mask = ratio_mask
        self.add_noise = add_noise
        self.maskable_flag_attr = maskable_flag_attr

    def _normalize_angles(self, angles):
        angles = angles % (2*np.pi)
        return torch.where(angles > np.pi, angles - 2*np.pi, angles)

    def _get_min_dist(self, data, center_idx):
        pos_beta_all = _get_CB_positions(data['pos_heavyatom'], data['mask_heavyatom'])
        pos_beta_center = pos_beta_all[center_idx]
        cdist = torch.cdist(pos_beta_all, pos_beta_center)  # (L, K)
        if len(center_idx) >= 1:
            min_dist = cdist.min(dim=1)[0]
        else:
            min_dist = torch.ones_like(data['pos_heavyatom'][:, 0, 0]) * 0.1
        return min_dist

    def _get_noise_std(self, min_dist):
        return torch.clamp_min((-1/16) * min_dist + 1, 0)

    def _get_flip_prob(self, min_dist):
        return torch.where(
            min_dist <= 8.0,
            torch.full_like(min_dist, 0.25),
            torch.zeros_like(min_dist,),
        )

    def _add_chi_gaussian_noise(self, chi, noise_std, chi_mask):
        """
        Args:
            chi: (L, 4)
            noise_std: (L, )
            chi_mask: (L, 4)
        """
        noise = torch.randn_like(chi) * noise_std[:, None] * chi_mask
        return self._normalize_angles(chi + noise)

    def _random_flip_chi(self, chi, flip_prob, chi_mask):
        """
        Args:
            chi: (L, 4)
            flip_prob: (L, )
            chi_mask: (L, 4)
        """
        delta = torch.where(
            torch.rand_like(chi) <= flip_prob[:, None],
            torch.full_like(chi, np.pi),
            torch.zeros_like(chi),
        ) * chi_mask
        return self._normalize_angles(chi + delta)

    def __call__(self, data):
        L = data['aa'].size(0)
        idx = torch.arange(0, L)
        num_mask = max(int(self.ratio_mask * L), 1)
        if self.maskable_flag_attr is not None:
            flag = data[self.maskable_flag_attr]
            idx = idx[flag]

        idx = idx.tolist()
        np.random.shuffle(idx)
        try:
            assert len(idx) > 0
        except:
            print(idx)
        idx_mask = idx[:num_mask]
        min_dist = self._get_min_dist(data, idx_mask)
        noise_std = self._get_noise_std(min_dist)
        flip_prob = self._get_flip_prob(min_dist)

        chi_native = torch.where(
            torch.randn_like(data['chi']) > 0,
            data['chi'],
            data['chi_alt'],
        )   # (L, 4), randomly pick from chi and chi_alt
        chi = chi_native.clone()
        chi_mask = torch.logical_and(data['chi_mask'].clone(), 
                                     data[self.maskable_flag_attr].unsqueeze(-1))
        
        if self.add_noise:
            chi = self._add_chi_gaussian_noise(chi, noise_std, chi_mask)
            chi = self._random_flip_chi(chi, flip_prob, chi_mask)
        chi[idx_mask] = 0.0     # Mask chi angles

        corrupt_flag = torch.zeros(L, dtype=torch.bool)
        corrupt_flag[idx_mask] = True
        corrupt_flag[chi_mask[:,0]] = True

        masked_flag = torch.zeros(L, dtype=torch.bool)
        masked_flag[idx_mask] = True

        data['chi_native'] = chi_native
        data['chi_corrupt'] = chi
        data['chi_corrupt_flag'] = corrupt_flag
        data['chi_masked_flag'] = masked_flag
        return data
