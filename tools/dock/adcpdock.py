import os
import shutil
import tempfile
import subprocess
import dataclasses as dc
from typing import List, Optional
from Bio import PDB
from Bio.PDB import Model as PDBModel
from ppflow.datasets.constants import *
from tools.relax.rosetta_packing import side_chain_packing
from Bio.PDB import PDBParser, PDBIO
from tqdm import tqdm
import numpy as np


parser = PDBParser(QUIET=True)


import abc
from typing import List


FilePath = str


class DockingEngine(abc.ABC):

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, typ, value, traceback):
        pass

    @abc.abstractmethod
    def set_receptor(self, pdb_path: FilePath):
        pass

    @abc.abstractmethod
    def set_ligand(self, pdb_path: FilePath):
        pass

    @abc.abstractmethod
    def dock(self) -> List[FilePath]:
        pass

def fix_docked_pdb(pdb_path):
    fixed = []
    with open(pdb_path, 'r') as f:
        for ln in f.readlines():
            if (ln.startswith('ATOM') or ln.startswith('HETATM')) and len(ln) == 56:
                fixed.append( ln[:-1] + ' 1.00  0.00              \n' )
            else:
                fixed.append(ln)
    with open(pdb_path, 'w') as f:
        f.write(''.join(fixed))




class ADCPDock(DockingEngine):

    def __init__(
            self,
            save_dir,
            reduce_bin = './bin/bin/reduce',
            adcp_bin = './bin/bin/adcp',
            prepare_ligand_bin = './bin/bin/prepare_ligand',
            prepare_receptor_bin = './bin/bin/prepare_receptor',
            agfr_bin = './bin/bin/agfr'
        ):
        super().__init__()
        self.adcp_bin = os.path.realpath(adcp_bin)
        self.reduce_bin = os.path.realpath(reduce_bin)
        self.prepare_ligand_bin = os.path.realpath(prepare_ligand_bin)
        self.prepare_receptor_bin = os.path.realpath(prepare_receptor_bin)
        self.agfr_bin = os.path.realpath(agfr_bin)

        self.tmpdir = tempfile.TemporaryDirectory()
        self.save_dir = save_dir

        self._has_receptor = False
        self._has_ligand = False
        self._ligand_packed = False
        self._receptor_packed = False

    def __enter__(self):

        return self
    
    def __exit__(self, typ, value, traceback):

        self.tmpdir.cleanup()

    def set_receptor(self, pdb_path):

        shutil.copyfile(pdb_path, os.path.join(self.tmpdir.name, 'receptor.pdb'))
        self._receptor_path = os.path.join(self.tmpdir.name,'receptor.pdb')

        self._has_receptor = True

    def set_ligand(self, pdb_path):

        shutil.copyfile(pdb_path, os.path.join(self.tmpdir.name, 'ligand.pdb'))
        self._ligand_path = os.path.join(self.tmpdir.name, 'ligand.pdb')
        self._has_ligand = True

    def side_chain_packing(self, type):
        out_path = side_chain_packing(pdb_file=os.path.join(self.tmpdir.name, '{}.pdb'.format(type)),
                                      output_file=os.path.join(self.tmpdir.name, '{}_packed.pdb'.format(type)))
        
        if type == 'ligand':
            self._ligand_path = out_path
            self._ligand_packed = True
        elif type == 'receptor':
            self._receptor_path = out_path
            self._receptor_packed = True

        else:
            raise ValueError('No such type')

        return out_path


    def _dump_complex_pdb(self, save_name, n_save):
        
        shutil.copyfile(os.path.join(self.tmpdir.name,  "rl_redocking_{}.pdb".format(n_save)),
                        os.path.join(self.save_dir, "{}_{}.pdb".format(save_name, n_save)))
        
        shutil.copyfile(os.path.join(self.tmpdir.name, "terminal_record.txt"),
                        os.path.join(self.save_dir, "energy_record.txt"))
        
        energies = []
        
        with open(os.path.join(self.tmpdir.name, "terminal_record.txt"), 'r') as f:

            data = f.read().splitlines()
            
            energs = []

            for i, line in enumerate(data):

                if line[0] == "-":

                    energs = data[i+1 : -1]

            for e in energs:

                energies.append(float(e.split()[1]))

        if len(energies) == 0:
            print('Failed!')
            return np.array([np.nan])

        return np.array(energies)

        

    def load_pep_seq_and_struc(self):
        structure_ligand = parser.get_structure(self._ligand_path, self._ligand_path)[0]

        for chain in structure_ligand:
            seq = []
            struc = []
            for residue in chain:
                seq.append(resindex_to_ressymb[AA(residue.get_resname())])
                atom_names = ['N', 'CA', 'C']
                
                for idx, atom_name in enumerate(atom_names):
                    if atom_name == '': continue
                    if atom_name in residue:
                        struc.append(np.array(residue[atom_name].get_coord().tolist()))   

        seq = ''.join(seq)
        seq = seq.lower()

        return seq, np.array(struc)

    def _reduce_receptor(self, receptor_h_path):
        reduce_receptor_list = [self.reduce_bin, os.path.basename(self._receptor_path), '>',
                        os.path.basename(receptor_h_path)]
        cmdline_cd = 'cd {}'.format(self.tmpdir.name)
        cmdline = cmdline_cd + ' && ' + ' '.join(reduce_receptor_list)
        os.system(cmdline)

    def _reduce_ligand(self, ligand_h_path):
        reduce_ligand_list = [self.reduce_bin, os.path.basename(self._ligand_path), '>',
                        os.path.basename(ligand_h_path)]
        cmdline_cd = 'cd {}'.format(self.tmpdir.name)
        cmdline = cmdline_cd + ' && ' + ' '.join(reduce_ligand_list)
        os.system(cmdline)

    def _bio_reduce_ligand(self, ligand_h_path):
        pdb_parser = PDB.PDBParser(QUIET=True)
        structure = pdb_parser.get_structure('protein', self._ligand_path)

        hydrogen_pdb = PDB.PDBIO()
        hydrogen_pdb.set_structure(structure)

        hydrogen_pdb.save(ligand_h_path, select=None)

    def auto_dock_box(self, lig_struc):
        docking_center = lig_struc.mean(0)
        box_size = np.abs(lig_struc - docking_center[None,:]).max(0) * 1.5
        return docking_center, box_size

    def dock(self, save_name, n_save, n_search=20, n_steps=100000, auto_box=False):

        if not (self._has_receptor and self._has_ligand):

            raise ValueError('Missing receptor or peptide.')

        if auto_box:
            lig_seq, lig_struc = self.load_pep_seq_and_struc()
            docking_center, box_size = self.auto_dock_box(lig_struc)

        receptor_h_path = self._receptor_path.split('/')[-1].split('.')[0] + 'H.pdb'
        receptor_h_path = os.path.join(self.tmpdir.name, receptor_h_path)

        receptor_pt_path = self._receptor_path.split('/')[-1].split('.')[0] + '.pdbqt'
        receptor_pt_path = os.path.join(self.tmpdir.name, receptor_pt_path)

        ligand_h_path = self._ligand_path.split('/')[-1].split('.')[0] + 'H.pdb'
        ligand_h_path = os.path.join(self.tmpdir.name, ligand_h_path)

        cmdline_cd = 'cd {}'.format(self.tmpdir.name)

        if not self._receptor_packed: # the packed side chain does not need to add Hs
            self._reduce_receptor(receptor_h_path)
        else:
            receptor_h_path = self._receptor_path

        if not self._ligand_packed:
            self._reduce_ligand(ligand_h_path)
        else:
            ligand_h_path = self._ligand_path

        prep_rec_list = [self.prepare_receptor_bin, '-r', os.path.basename(receptor_h_path)]

        prep_lig_list = [self.prepare_ligand_bin, '-l', os.path.basename(ligand_h_path)]

        cmdline = cmdline_cd + ' && ' + ' '.join(prep_rec_list)
        os.system(cmdline)

        cmdline = cmdline_cd + ' && ' + ' '.join(prep_lig_list)
        os.system(cmdline)

        if auto_box:
            agfr_list = [self.agfr_bin, 
                        '-r',  receptor_h_path.split('.')[0] + '.pdbqt', 
                        '-b', 'user {} {}'.format(' '.join(str(x) for x in docking_center), 
                                                ' '.join(str(x) for x in box_size)),
                        '-l',  ligand_h_path.split('.')[0] + '.pdbqt',
                        '-asv', '1.1',
                        '-o', 'prepared']
        else:
            agfr_list = [self.agfr_bin, 
            '-r',  receptor_h_path.split('.')[0] + '.pdbqt', 
            '-l',  ligand_h_path.split('.')[0] + '.pdbqt',
            '-asv', '1.1',
            '-o', 'prepared']

        cmdline = cmdline_cd + ' && ' + ' '.join(agfr_list)
        os.system(cmdline)

        prepared_file = [name for name in os.listdir(self.tmpdir.name) if name[:8] == 'prepared' and name.endswith('.trg')][0]
        adcp_list = [self.adcp_bin, 
                     '-t', prepared_file,
                     '-s', lig_seq,
                     '-N', str(n_search),
                     '-n', str(n_steps),
                     '-o', 'rl_redocking',
                     '-ref', os.path.basename(self._ligand_path),
                     '>', os.path.join(self.tmpdir.name, 'terminal_record.txt')]
        
        cmdline = cmdline_cd + ' && ' + ' '.join(adcp_list)
        
        os.system(cmdline)

        return self._dump_complex_pdb(save_name, n_save) # click 进去
    
if __name__ == '__main__':
    protein_file = './PPDbench/2qbx/receptor.pdb'
    ligand_file = './scripts/examples/0000_bb3.pdb'
    dock = ADCPDock(save_dir='./scripts/examples/')
    dock.set_ligand(ligand_file)
    dock.set_receptor(protein_file) 
    dock.side_chain_packing('ligand')
    np.sum(dock.dock(save_name='peptide_docked.pdb', n_save=1, auto_box=True))
