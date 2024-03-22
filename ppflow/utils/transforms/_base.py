import copy
import torch

from ppflow.datasets.constants import BBHeavyAtom

_TRANSFORM_DICT = {}

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

def register_transform(name):
    def decorator(cls):
        _TRANSFORM_DICT[name] = cls
        return cls
    return decorator


def get_transform(cfg):
    if cfg is None or len(cfg) == 0:
        return None
    tfms = []
    for t_dict in cfg:
        t_dict = copy.deepcopy(t_dict)
        cls = _TRANSFORM_DICT[t_dict.pop('type')]
        tfms.append(cls(**t_dict))
    return Compose(tfms)


def _index_select(v, index, n):
    if isinstance(v, torch.Tensor) and v.size(0) == n:
        return v[index]
    elif isinstance(v, list) and len(v) == n:
        return [v[i] for i in index]
    else:
        return v


def _index_select_data(data, index):
    return {
        k: _index_select(v, index, data['aa'].size(0))
        for k, v in data.items()
    }


def _mask_select(v, mask):
    if isinstance(v, torch.Tensor) and v.size(0) == mask.size(0):
        return v[mask]
    elif isinstance(v, list) and len(v) == mask.size(0):
        return [v[i] for i, b in enumerate(mask) if b]
    else:
        return v


def _mask_select_data(data, mask):
    return {
        k: _mask_select(v, mask)
        for k, v in data.items()
    }


def _get_CB_positions(pos_atoms, mask_atoms):
    """
    Args:
        pos_atoms:  (L, A, 3)
        mask_atoms: (L, A)
    """
    L = pos_atoms.size(0)
    pos_CA = pos_atoms[:, BBHeavyAtom.CA]   # (L, 3)
    if pos_atoms.size(1) < 5:
        return pos_CA
    pos_CB = pos_atoms[:, BBHeavyAtom.CB]
    mask_CB = mask_atoms[:, BBHeavyAtom.CB, None].expand(L, 3)
    return torch.where(mask_CB, pos_CB, pos_CA)
