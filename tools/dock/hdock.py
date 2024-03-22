import os
import shutil
import tempfile
import subprocess
import dataclasses as dc
from typing import List, Optional
from Bio import PDB
from Bio.PDB import Model as PDBModel
from tools.relax.rosetta_packing import side_chain_packing

import abc
from typing import List


FilePath = str


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



class HDock(DockingEngine):

    def __init__(
        self, 
        save_dir,
        hdock_bin='./bin/hdock/hdock',
        createpl_bin='./bin/hdock/createpl',
    ):
        super().__init__()
        self.hdock_bin = os.path.realpath(hdock_bin)
        self.createpl_bin = os.path.realpath(createpl_bin)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self._has_receptor = False
        self._has_ligand = False

        self._receptor_chains = []
        self._ligand_chains = []

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        self.tmpdir.cleanup()

    def set_receptor(self, pdb_path):
        shutil.copyfile(pdb_path, os.path.join(self.tmpdir.name, 'receptor.pdb'))
        self._has_receptor = True
        self._receptor_path = os.path.join(self.tmpdir.name, 'receptor.pdb')

    def set_ligand(self, pdb_path):
        shutil.copyfile(pdb_path, os.path.join(self.tmpdir.name, 'ligand.pdb'))
        self._has_ligand = True
        self._ligand_path = os.path.join(self.tmpdir.name, 'ligand.pdb')

    def side_chain_packing(self, type):
        out_path = side_chain_packing(pdb_file=os.path.join(self.tmpdir.name, '{}.pdb'.format(type)),
                                      output_file=os.path.join(self.tmpdir.name, '{}.pdb'.format(type)))
        
        if type == 'ligand':
            self._ligand_path = out_path
        elif type == 'receptor':
            self._receptor_path = out_path
        else:
            raise ValueError('No such type')

        return out_path

    def _dump_complex_pdb(self, save_all):
        parser = PDB.PDBParser(QUIET=True)
        model_receptor = parser.get_structure(None, os.path.join(self.tmpdir.name, 'receptor.pdb'))[0]
        docked_pdb_path = os.path.join(self.tmpdir.name, 'ligand_docked.pdb')
        fix_docked_pdb(docked_pdb_path)
        structure_ligdocked = parser.get_structure(None, docked_pdb_path)

        pdb_io = PDB.PDBIO()
        paths = []
        for i, model_ligdocked in enumerate(structure_ligdocked):
            model_complex = PDBModel.Model(0)
            for chain in model_receptor:
                model_complex.add(chain.copy())
            for chain in model_ligdocked:
                model_complex.add(chain.copy())
            pdb_io.set_structure(model_complex)
            save_path = os.path.join(self.tmpdir.name, f"complex_{i}.pdb")
            pdb_io.save(save_path)
            paths.append(save_path)

        if save_all == True: 
            for count, path in enumerate(paths):
                shutil.copyfile(path, os.path.join(self.save_dir, '%04d.pdb' % (count, )))
        else:
            shutil.copyfile(paths[-1], os.path.join(self.save_dir, save_all))

        return paths

    def dock(self, save_all=True):
        if not (self._has_receptor and self._has_ligand):
            raise ValueError('Missing receptor or ligand.')
        subprocess.run(
            [self.hdock_bin, "receptor.pdb", "ligand.pdb"],
            cwd=self.tmpdir.name, check=True
        )
        subprocess.run(
            [self.createpl_bin, "Hdock.out", "ligand_docked.pdb"], 
            cwd=self.tmpdir.name, check=True
        )
        return self._dump_complex_pdb(save_all)

if __name__ == "__main__":
    pdb_id = '2qbx'
    gen_bb_file = './PPDbench/{}/peptide.pdb'.format(pdb_id)
    protein_file = './PPDbench/{}/receptor.pdb'.format(pdb_id)

    dock_method = HDock(save_dir='results/hdock/{}/'.format(pdb_id))
    dock_method.set_ligand(gen_bb_file)
    dock_method.set_receptor(protein_file)
    dock_method.side_chain_packing('ligand')
    dock_method.dock()