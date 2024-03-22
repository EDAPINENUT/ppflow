import os
import shutil
import tempfile
import subprocess
import dataclasses as dc
from typing import List, Optional
from Bio import PDB
from Bio.PDB import Model as PDBModel
from tools.relax.rosetta_packing import side_chain_packing
import AutoDockTools
from meeko import MoleculePreparation
from openbabel import openbabel
from openbabel import pybel
from vina import Vina
import abc
import contextlib
from typing import List
FilePath = str

from rdkit import Chem
from rdkit.Chem import AllChem

def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper


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



class VinaDock(DockingEngine):

    def __init__(
        self, 
        save_dir
    ):
        super().__init__()
        self.vina_task = Vina(sf_name='vina', seed=0, verbosity=0)
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

    def addH(self, pdb_file, prot_pqr):  # call pdb2pqr

        subprocess.Popen(['pdb2pqr30','--ff=AMBER', pdb_file, prot_pqr],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()
        return prot_pqr

    def get_receptor_pdbqt(self, prot_pqr, prot_pdbqt):
        prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
        subprocess.Popen(['python3', prepare_receptor, '-r', prot_pqr, '-o', prot_pdbqt],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()
        return prot_pdbqt

    @supress_stdout
    def get_ligand_pdbqt(self, sdf_file, pdbqt_file):
        mol = Chem.SDMolSupplier(sdf_file, removeHs=False)[0]

        preparator = MoleculePreparation()
        preparator.prepare(mol)

        if pdbqt_file is not None: 
            preparator.write_pdbqt_file(pdbqt_file)
            return pdbqt_file
        else: 
            return preparator.write_pdbqt_string()
            

    def prepare_peptide_pdbqt(self, pdb_file, sdf_file, pdbqt_file):

        with open(pdb_file, 'r') as file:
            pdb_content = file.read()

        mol = Chem.MolFromPDBBlock(pdb_content)

        AllChem.Compute2DCoords(mol)
        mol = Chem.AddHs(mol)
        writer = Chem.SDWriter(sdf_file)
        writer.write(mol)
        writer.close()

        pdbqt_file = self.get_ligand_pdbqt(sdf_file, pdbqt_file)
        return pdbqt_file


    def prepare_protein_pdbqt(self, pdb_file, pqr_file, pdbqt_file):
        
        pqr_file = self.addH(pdb_file, pqr_file)

        pdbqt_file = self.get_receptor_pdbqt(pqr_file, pdbqt_file)
        
        return pdbqt_file
    
    def set_docking_site(self, ligand_path, center=None, size_factor=0, buffer=5.0):
        ligand_rdmol = Chem.SDMolSupplier(ligand_path)[0]
        pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 50, 50, 50
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor + buffer
        self.box_size = [self.size_x, self.size_y, self.size_z]

    def dock(self, save_name=False, mode='dock', exhaustiveness=8):

        receptor_pdbqt = self.prepare_protein_pdbqt(os.path.join(self.tmpdir.name, "receptor.pdb"),
                                                    os.path.join(self.tmpdir.name, "receptor.pqr"),
                                                    os.path.join(self.tmpdir.name, "receptor.pdbqt"))
        
        ligand_pdbqt = self.prepare_peptide_pdbqt(os.path.join(self.tmpdir.name, "ligand.pdb"),
                                                  os.path.join(self.tmpdir.name, "ligand.sdf"), 
                                                  os.path.join(self.tmpdir.name, "ligand.pdbqt"))    
        
        self.vina_task.set_receptor(receptor_pdbqt)
        self.set_docking_site(os.path.join(self.tmpdir.name, "ligand.sdf"))
        self.vina_task.compute_vina_maps(center=self.center, box_size=self.box_size)
        self.vina_task.set_ligand_from_file(ligand_pdbqt)

        if mode == 'score_only': 
            score = self.vina_task.score()[0]
            pose = None 
        elif mode == 'minimize':
            score = self.vina_task.optimize()[0]
            tmp = tempfile.NamedTemporaryFile()
            with open(tmp.name, 'w') as f: 
                self.vina_task.write_pose(tmp.name, overwrite=True)             
            with open(tmp.name, 'r') as f: 
                pose = f.read()
        elif mode == 'dock':
            self.vina_task.dock(exhaustiveness=exhaustiveness, n_poses=1)
            score = self.vina_task.energies(n_poses=1)[0][0]
            pose = self.vina_task.poses(n_poses=1)
            if save_name:
                self.vina_task.write_pose(os.path.join(self.tmpdir.name, "pose.pdbqt"),
                                           overwrite=True)
                
                obConversion = openbabel.OBConversion()
                obConversion.SetInAndOutFormats("pdbqt", "pdb")
                ob_mol = openbabel.OBMol()
                obConversion.ReadFile(ob_mol, os.path.join(self.tmpdir.name, "pose.pdbqt"))

                obConversion.WriteFile(ob_mol, os.path.join(self.save_dir, save_name))
                
        else:
            raise ValueError

        return score, pose

        #return self._dump_complex_pdb(save_all)

if __name__ == "__main__":
    pdb_id = '2qbx'
    gen_bb_file = './PPDbench/{}/peptide.pdb'.format(pdb_id)
    protein_file = './PPDbench/{}/receptor.pdb'.format(pdb_id)

    if not os.path.exists('results/vina/{}'.format(pdb_id)):

        os.mkdirs('results/vina/{}'.format(pdb_id))
    
    dock_method = VinaDock(save_dir='results/vina/{}'.format(pdb_id))
    
    
    dock_method.set_ligand(gen_bb_file)
    dock_method.set_receptor(protein_file)
    dock_method.side_chain_packing('ligand')
    score, _ = dock_method.dock(save_name='ligand_docked.pdb')
    print(score)