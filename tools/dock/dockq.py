import shutil
import tempfile
import os
from tools.relax.rosetta_packing import side_chain_packing
import subprocess

class DockQ(object):

    def __init__(
        self, 
        save_dir,
        dockq_bin='./bin/DockQ/DockQ.py',
    ):
        super().__init__()
        self.dockq_bin = os.path.realpath(dockq_bin)
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

    def eval_dock(self, save_all=True):
        if not (self._has_receptor and self._has_ligand):
            raise ValueError('Missing receptor or ligand.')

        result = subprocess.run(
            [self.createpl_bin, "Hdock.out", "ligand_docked.pdb"], 
            cwd=self.tmpdir.name, check=True
        )
        return self._dump_complex_pdb(save_all)