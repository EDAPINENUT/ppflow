import torch
import random
from typing import List, Optional

from ppflow.datasets import constants
from ._base import register_transform

@register_transform('mask_peptide')
class MaskPeptide(object):

    def __init__(self, mask_type=True, mask_pos=True):
        super().__init__()
        self.mask_type = mask_type
        self.mask_pos = mask_pos

    def __call__(self, structure):
        peptide = structure['peptide']
        receptor = structure['receptor']
        if self.mask_type:
            peptide['mask_gen_aa'] = torch.ones_like(peptide['aa']).bool()
        else:
            peptide['mask_gen_aa'] = torch.zeros_like(peptide['aa']).bool()
            
        if self.mask_pos:
            peptide['mask_gen_pos'] = torch.ones_like(peptide['aa']).bool()
        else:
            peptide['mask_gen_pos'] = torch.zeros_like(peptide['aa']).bool()

        receptor['mask_gen_aa'] = torch.zeros_like(receptor['aa']).bool()
        receptor['mask_gen_pos'] = torch.zeros_like(receptor['aa']).bool()

        return structure