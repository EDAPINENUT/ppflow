import torch

from ppflow.datasets import constants
from ._base import register_transform


@register_transform('merge_chains')
class MergeChains(object):

    def __init__(self):
        super().__init__()

    def assign_chain_number_(self, data_list):
        chains = set()
        for data in data_list:
            chains.update(data['chain_id'])
        chains = {c: i for i, c in enumerate(chains)}

        for data in data_list:
            data['chain_nb'] = torch.LongTensor([
                chains[c] for c in data['chain_id']
            ])

    def __call__(self, structure):
        data_list = []
        if structure['peptide'] is not None:
            structure['peptide']['fragment_type'] = torch.full_like(
                structure['peptide']['aa'],
                fill_value = constants.Fragment.Peptide,
            )
            data_list.append(structure['peptide'])

        if structure['receptor'] is not None:
            structure['receptor']['fragment_type'] = torch.full_like(
                structure['receptor']['aa'],
                fill_value = constants.Fragment.Receptor,
            )
            data_list.append(structure['receptor'])

        self.assign_chain_number_(data_list)

        list_props = {
            'chain_id': [],
            'icode': []
        }
        tensor_props = {
            'chain_nb': [],
            'resseq': [],
            'res_nb': [],
            'aa': [],
            'pos_heavyatom': [],
            'mask_heavyatom': [],
            'mask_gen_pos': [],
            'mask_gen_aa': [],
            'bfactor_heavyatom': [],
            'fragment_type': [],
            'phi': [],
            'phi_mask': [],
            'psi': [],
            'psi_mask': [],
            'omega': [],
            'omega_mask': [],
            'chi': [],
            'chi_alt': [],
            'chi_mask': [],
            'chi_complete': []
            }

        for data in data_list:
            for k in list_props.keys():
                list_props[k].append(data[k])
            for k in tensor_props.keys():
                tensor_props[k].append(data[k])

        list_props = {k: sum(v, start=[]) for k, v in list_props.items()}
        tensor_props = {k: torch.cat(v, dim=0) for k, v in tensor_props.items()}
        data_out = {
            **list_props,
            **tensor_props,
        }
        data_out['pdb_name'] = structure['pdb_name']
        data_out['peptide_flag'] = (data_out['fragment_type'] == constants.Fragment.Peptide)
        data_out['receptor_flag'] = (data_out['fragment_type'] == constants.Fragment.Receptor)
        return data_out

