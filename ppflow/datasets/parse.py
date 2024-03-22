
import torch
from Bio.PDB import Selection
from Bio.PDB.Residue import Residue
import Bio.PDB
from easydict import EasyDict
from .constants import (
    AA, max_num_heavyatoms,
    restype_to_heavyatom_names, 
    BBHeavyAtom, chi_angles_atoms, 
    chi_pi_periodic
)
import os
from tqdm.auto import tqdm
import numpy as np 
from Bio.PDB.internal_coords import *
import logging 

pdb_parser = Bio.PDB.PDBParser(QUIET = True)

def filter_none(data):
    for key, val in data.items():
        if val is None:
            return None
    return data

def get_chi_angles(restype: AA, res: Residue):
    ic = res.internal_coord
    chi_angles = torch.zeros([4, ])
    chi_angles_alt = torch.zeros([4, ],)
    chi_angles_mask = torch.zeros([4, ], dtype=torch.bool)
    count_chi_angles = len(chi_angles_atoms[restype])
    if ic is not None:
        for i in range(count_chi_angles):
            angle_name = 'chi%d' % (i+1)
            if ic.get_angle(angle_name) is not None:
                angle = np.deg2rad(ic.get_angle(angle_name))
                chi_angles[i] = angle
                chi_angles_mask[i] = True

                if chi_pi_periodic[restype][i]:
                    if angle >= 0:
                        angle_alt = angle - np.pi
                    else:
                        angle_alt = angle + np.pi
                    chi_angles_alt[i] = angle_alt
                else:
                    chi_angles_alt[i] = angle
            
    chi_complete = (count_chi_angles == chi_angles_mask.sum().item())
    return chi_angles, chi_angles_alt, chi_angles_mask, chi_complete


def get_backbone_torsions(res: Residue):
    ic = res.internal_coord
    if ic is None:
        return None, None, None
    phi, psi, omega = ic.get_angle('phi'), ic.get_angle('psi'), ic.get_angle('omega')
    if phi is not None: phi = np.deg2rad(phi)
    if psi is not None: psi = np.deg2rad(psi)
    if omega is not None: omega = np.deg2rad(omega)
    return phi, psi, omega


class ParsingException(Exception):
    pass

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



def parse_biopython_structure(pdb_path, unknown_threshold=1.0):
    try:
        entity = pdb_parser.get_structure(id, pdb_path)[0]
        chains = Selection.unfold_entities(entity, 'C')
        chains.sort(key=lambda c: c.get_id())
        data = EasyDict({
            'chain_id': [], 'chain_nb': [],
            'resseq': [], 'icode': [], 'res_nb': [],
            'aa': [],
            'pos_heavyatom': [], 'mask_heavyatom': [],
            'bfactor_heavyatom': [],
            'phi': [], 'phi_mask': [],
            'psi': [], 'psi_mask': [],
            'omega': [], 'omega_mask': [],
            'chi': [], 'chi_alt': [], 'chi_mask': [], 'chi_complete': [],
        })
        tensor_types = {
            'chain_nb': torch.LongTensor,
            'resseq': torch.LongTensor,
            'res_nb': torch.LongTensor,
            'aa': torch.LongTensor,
            'pos_heavyatom': torch.stack,
            'mask_heavyatom': torch.stack,
            'bfactor_heavyatom': torch.stack,

            'phi': torch.FloatTensor,
            'phi_mask': torch.BoolTensor,
            'psi': torch.FloatTensor,
            'psi_mask': torch.BoolTensor,
            'omega': torch.FloatTensor,
            'omega_mask': torch.BoolTensor,

            'chi': torch.stack,
            'chi_alt': torch.stack,
            'chi_mask': torch.stack,
            'chi_complete': torch.BoolTensor,
        }

        count_aa, count_unk = 0, 0

        for i, chain in enumerate(chains):
            chain.atom_to_internal_coordinates()
            seq_this = 0   # Renumbering residues
            residues = Selection.unfold_entities(chain, 'R')
            residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))   # Sort residues by resseq-icode
            
            for res_id, res in enumerate(residues):
                resname = res.get_resname()
                if not AA.is_aa(resname): 
                    if res_id == 0 or res_id == len(residues) - 1:
                        continue 
                    logging.warning(f"Unknown AA type in {res_id}-th residue from PDB: {pdb_path}. Skip the pair.")
                    return None
                if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): 
                    logging.warning(f"Incomplete backbone atom in {res_id}-th residue from PDB: {pdb_path}. Skip the pair")
                    return None
                
                restype = AA(resname)
                count_aa += 1
                if restype == AA.UNK: 
                    count_unk += 1
                    return None

                # Chain info
                data.chain_id.append(chain.get_id())
                data.chain_nb.append(i)

                # Residue types
                data.aa.append(restype) # Will be automatically cast to torch.long

                # Heavy atoms
                pos_heavyatom, mask_heavyatom, bfactor_heavyatom = _get_residue_heavyatom_info(res)
                data.pos_heavyatom.append(pos_heavyatom)
                data.mask_heavyatom.append(mask_heavyatom)
                data.bfactor_heavyatom.append(bfactor_heavyatom)

                # Backbone torsions
                phi, psi, omega = get_backbone_torsions(res)
                if phi is None:
                    data.phi.append(0.0)
                    data.phi_mask.append(False)
                else:
                    data.phi.append(phi)
                    data.phi_mask.append(True)
                if psi is None:
                    data.psi.append(0.0)
                    data.psi_mask.append(False)
                else:
                    data.psi.append(psi)
                    data.psi_mask.append(True)
                if omega is None:
                    data.omega.append(0.0)
                    data.omega_mask.append(False)
                else:
                    data.omega.append(omega)
                    data.omega_mask.append(True)

                # Chi
                chi, chi_alt, chi_mask, chi_complete = get_chi_angles(restype, res)
                data.chi.append(chi)
                data.chi_alt.append(chi_alt)
                data.chi_mask.append(chi_mask)
                data.chi_complete.append(chi_complete)

                # Sequential number
                resseq_this = int(res.get_id()[1])
                icode_this = res.get_id()[2]
                if seq_this == 0:
                    seq_this = 1
                else:
                    d_CA_CA = torch.linalg.norm(data.pos_heavyatom[-2][BBHeavyAtom.CA] - data.pos_heavyatom[-1][BBHeavyAtom.CA], ord=2).item()
                    if d_CA_CA <= 4.0:
                        seq_this += 1
                    else:
                        d_resseq = resseq_this - data.resseq[-1]
                        seq_this += max(2, d_resseq)

                data.resseq.append(resseq_this)
                data.icode.append(icode_this)
                data.res_nb.append(seq_this)

        if len(data.aa) == 0:
            return None

        if (count_unk / count_aa) >= unknown_threshold:
            return None
  
        seq_map = {}
        for i, (chain_id, resseq, icode) in enumerate(zip(data.chain_id, data.resseq, data.icode)):
            seq_map[(chain_id, resseq, icode)] = i

        for key, convert_fn in tensor_types.items():
            data[key] = convert_fn(data[key])

        return data
    
    except:
        return None

    



if __name__ == '__main__':
    protein_names = os.listdir('./dataset/ppbench2024/')
    tasks = []
    for protein_name in protein_names:
        task = {'receptor_path': './dataset/ppbench2024/{}/receptor.pdb'.format(protein_name), 
                'peptide_path': './dataset/ppbench2024/{}/peptide.pdb'.format(protein_name),
                'pdb_name': protein_name}
        tasks.append(task)

    data_list = []

    for task in tqdm(tasks):
        paired_data = {}
        data_receptor = parse_biopython_structure(task['receptor_path'])
        data_peptide = parse_biopython_structure(task['peptide_path'])
        paired_data['receptor'] = data_receptor
        paired_data['peptide'] = data_peptide
        paired_data['pdb_name'] = task['pdb_name']
        data_list.append(paired_data)

    data_list_filter = []
    for data in data_list:
        data = filter_none(data)
        if data is not None:
            data_list_filter.append(data)
    torch.save(data_list_filter, './processed/parsed_pair.pt')
