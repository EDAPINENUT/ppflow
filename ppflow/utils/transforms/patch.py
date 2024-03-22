import torch
import random
from typing import List, Optional

from ppflow.datasets import constants
from ._base import register_transform, _index_select_data

@register_transform('fix_patch')
class FixPatch(object):
    def __init__(self, patch_target, anchor_target, 
                 anchor_flag, patch_size=64, contin_chain=True):
        super().__init__()
        self.patch_target = patch_target
        self.anchor_target = anchor_target
        self.anchor_flag = anchor_flag
        self.patch_size = patch_size
        self.contin_chain = contin_chain

    def __call__(self, structure):
        target_data = structure[self.patch_target]
        anchor_data = structure[self.anchor_target]
        anchor_flag = anchor_data[self.anchor_flag]
        anchor_pos = anchor_data['pos_heavyatom'][anchor_flag][:,constants.BBHeavyAtom.CA]
        target_pos = target_data['pos_heavyatom'][:,constants.BBHeavyAtom.CA]
        dist_from_anchor = torch.cdist(target_pos, anchor_pos).min(dim=1)[0]
        patch_idx = torch.argsort(dist_from_anchor)[:self.patch_size]
        patch_mask = torch.zeros_like(target_data['aa']).bool()
        patch_mask[patch_idx] = True

        if self.contin_chain:
            for chain_nb in target_data['chain_nb'].unique():
                chain_mask = target_data['chain_nb'] == chain_nb
                seqid_on_chain = torch.where(torch.logical_and(chain_mask, patch_mask))[0]
                if len(seqid_on_chain) > 0:
                    seqid_min, seqid_max = seqid_on_chain.min(), seqid_on_chain.max()
                    patch_mask[seqid_min:seqid_max + 1] = True
                    
        patch_idx = torch.where(patch_mask)[0]
        target_data_patched = _index_select_data(target_data, patch_idx)
        structure[self.patch_target] = target_data_patched
        return structure