import torch
from ..datasets import constants


class RemoveNative(object):

    def __init__(self, remove_structure, remove_sequence):
        super().__init__()
        self.remove_structure = remove_structure
        self.remove_sequence = remove_sequence

    def __call__(self, data):
        seq_generate_flag = data['mask_gen_aa'].clone()
        pos_generate_flag = data['mask_gen_pos'].clone()
        if self.remove_sequence:
            data['aa'] = torch.where(
                seq_generate_flag, 
                torch.full_like(data['aa'], fill_value=int(constants.AA.UNK)),    # Is loop
                data['aa']
            )

        if self.remove_structure:
            data['pos_heavyatom'] = torch.where(
                pos_generate_flag[:, None, None].expand(data['pos_heavyatom'].shape),
                torch.randn_like(data['pos_heavyatom']) * 10,
                data['pos_heavyatom']
            )

        return data