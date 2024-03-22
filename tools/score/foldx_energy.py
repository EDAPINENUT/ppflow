import tempfile
import shutil
import os
from tools.relax.rosetta_packing import side_chain_packing
from tools.base import merge_protein_ligand

class FoldXGibbsEnergy():

    def __init__(
        self,
        foldx_path = './bin/FoldX/foldx'
    ):
        super().__init__()
        self.foldx_path = os.path.abspath(foldx_path)
        self.tmpdir = tempfile.TemporaryDirectory()

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
                                      output_file=os.path.join(self.tmpdir.name, '{}_packed.pdb'.format(type)))
        
        if type == 'ligand':
            self._ligand_path = out_path
        elif type == 'receptor':
            self._receptor_path = out_path
        else:
            raise ValueError('No such type')

        return out_path
    
    def merge_protein_ligand(self):
        out_file, self.interface = merge_protein_ligand(self._receptor_path, self._ligand_path, 
                                        out_pdb_file=os.path.join(self.tmpdir.name, 'merge.pdb'))
        return out_file
        
    def cal_interface_energy(self):
        merge_file = self.merge_protein_ligand()
        cmd="cd "+ self.tmpdir.name +"; "
        cmd += self.foldx_path + " --command=Stability" + " --pdb="+ os.path.basename(merge_file)

        os.system(cmd)

        return self.read_output(os.path.join(self.tmpdir.name, 'merge_0_ST.fxout'))

    def read_output(self, file_name):
        with open(file_name, 'r') as f:
            data = f.read().splitlines()
            total_energy = float(data[0].split('\t')[1])
        return total_energy


if __name__ == '__main__':
    gen_bb_file = './PPDbench/2qbx/peptide.pdb'
    protein_file = './PPDbench/2qbx/receptor.pdb'

    score_calculator = FoldXGibbsEnergy()
    score_calculator.set_ligand(gen_bb_file)
    score_calculator.set_receptor(protein_file)
    score_calculator.side_chain_packing('ligand')
    dg = score_calculator.cal_interface_energy()


