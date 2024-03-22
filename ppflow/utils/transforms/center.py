import torch
import random
from typing import List, Optional

from ppflow.datasets import constants
from ._base import register_transform

@register_transform('center_pos')
class CenterPos(object):

    def __init__(self, center_flag, select_atoms='bb3'):
        super().__init__()
        self.center_flag = center_flag
        self.select_atoms = select_atoms

    def __call__(self, structure):
        pos_heavyatom = structure['pos_heavyatom'].clone()
        mask_heavyatom = structure['mask_heavyatom']
        pos_context_mask = torch.logical_and(structure[self.center_flag].unsqueeze(-1), mask_heavyatom)
        pos_center = pos_heavyatom[pos_context_mask].view(-1, 3).mean(0, keepdim=True)
        
        if self.select_atoms == 'bb3':
            pos_center = pos_heavyatom[:,:3][pos_context_mask[:,:3]].view(-1, 3).mean(0, keepdim=True)
            pos_center_lig = pos_heavyatom[~structure[self.center_flag]][:,:3].reshape(-1, 3).mean(0, keepdim=True)
        elif self.select_atoms == 'bb4':
            pos_center = pos_heavyatom[:,:4][pos_context_mask[:,:4]].view(-1, 3).mean(0, keepdim=True)
            pos_center_lig = pos_heavyatom[~structure[self.center_flag]][:,:4].reshape(-1, 3).mean(0, keepdim=True)

        pos_heavyatom_centered = pos_heavyatom - pos_center.unsqueeze(0)
        pos_heavyatom = torch.where(mask_heavyatom[:,:,None], pos_heavyatom_centered, pos_heavyatom)
        structure['pos_heavyatom'] = pos_heavyatom
        structure['pos_center_org'] = pos_center
        structure['pos_center_lig'] = pos_center_lig
        return structure