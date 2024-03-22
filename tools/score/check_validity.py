import torch
from Bio.PDB import Selection
from Bio.PDB.Residue import Residue
import Bio.PDB
from ppflow.datasets.constants import (
    AA, max_num_heavyatoms,
    restype_to_heavyatom_names, 
    BBHeavyAtom, chi_angles_atoms, 
    chi_pi_periodic
)

N_CA_LENGTH = 1.46  # Check, approxiamtely right
CA_C_LENGTH = 1.53  # Check, approximately right
C_N_LENGTH = 1.34  # Check, approximately right

def _check_validity(bond_dict):
    nca = torch.tensor(bond_dict['nca'])
    cac = torch.tensor(bond_dict['cac'])
    cn = torch.tensor(bond_dict['cn'])
    nca_wrong = torch.logical_and(nca > N_CA_LENGTH + 0.5, nca < N_CA_LENGTH - 0.5).sum()
    cac_wrong = torch.logical_and(cac > CA_C_LENGTH + 0.5, cac < CA_C_LENGTH - 0.5).sum()
    cn_wrong = torch.logical_and(cn > C_N_LENGTH + 0.5, cn < C_N_LENGTH - 0.5).sum()
    if nca_wrong + cac_wrong + cn_wrong > 0:
        return False
    else:
        return True

pdb_parser = Bio.PDB.PDBParser(QUIET = True)

def _get_residue_heavyatom_info(res: Residue):
    pos_heavyatom = torch.zeros([max_num_heavyatoms, 3], dtype=torch.float)
    mask_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.bool)
    bfactor_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.float)
    restype = AA(res.get_resname())
    for idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):
        if atom_name == '': continue
        if atom_name in res:
            pos_heavyatom[idx] = torch.tensor(res[atom_name].get_coord().tolist(), dtype=pos_heavyatom.dtype)
            mask_heavyatom[idx] = True
            bfactor_heavyatom[idx] = res[atom_name].get_bfactor()
    return pos_heavyatom, mask_heavyatom, bfactor_heavyatom

def _get_bond_length(x_n, x_ca, x_c):
    bl_nca = torch.norm(x_n - x_ca, dim=-1)
    bl_cac = torch.norm(x_ca - x_c, dim=-1)
    bl_cn = torch.norm(x_n[1:] - x_c[:-1], dim=-1)
    return bl_nca, bl_cac, bl_cn


def bond_length_validation(pdb_file):
    entity = pdb_parser.get_structure(pdb_file, pdb_file)[0]
    chains = Selection.unfold_entities(entity, 'C')
    bond_dict = {'nca':[], 'cac':[], 'cn':[]}
    for i, chain in enumerate(chains):
        chain.atom_to_internal_coordinates()
        residues = Selection.unfold_entities(chain, 'R')
        residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))
        pos_heavyatoms = []

        for residue in residues:
            pos_heavyatom, mask_heavyatom, bfactor_heavyatom = _get_residue_heavyatom_info(residue)
            pos_heavyatoms.append(pos_heavyatom)
        
        pos_heavyatoms = torch.stack(pos_heavyatoms, dim=0)
        x_n = pos_heavyatoms[:,0]
        x_ca = pos_heavyatoms[:,1]
        x_c = pos_heavyatoms[:,2]
        bl_nca, bl_cac, bl_cn = _get_bond_length(x_n, x_ca, x_c)
        bond_dict['nca'].extend(bl_nca.tolist())
        bond_dict['cac'].extend(bl_cac.tolist())
        bond_dict['cn'].extend(bl_cn.tolist())

    return _check_validity(bond_dict)

if __name__ == '__main__':
    pdb_file = '/linhaitao/peptidesign/results/diffpp/dock_diffpp/0000_2qbx_2024_01_16__15_35_53/0000.pdb'

    if_valid = bond_length_validation(pdb_file)