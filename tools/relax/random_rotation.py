from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta.core import kinematics
import numpy as np
import random
from pyrosetta.rosetta.core.pack.task import TaskFactory
import pyrosetta
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

# 初始化 PyRosetta
init()


def random_rotate_peptide(pdb_file, output_file):
    # 创建一个 Pose 对象
    protein_pose = pose_from_pdb(pdb_file)

    for i in range(len(protein_pose.residues)):
        new_phi_angle = random.uniform(-180, 180)
        new_psi_angle = random.uniform(-180, 180)
        new_omega_angle = random.uniform(-180, 180)

        # 使用 set_phi 和 set_psi 方法来设置新的扭转角度
        protein_pose.set_phi(i+1, new_phi_angle)
        protein_pose.set_psi(i+1, new_psi_angle)
        protein_pose.set_omega(i+1, new_omega_angle)
    
    protein_pose.dump_pdb(output_file)

def random_rotate_peptide_sidechain(pdb_file, output_file):
    # 创建一个 Pose 对象
    protein_pose = pose_from_pdb(pdb_file)
    conformation = protein_pose.conformation()
    for i in range(len(protein_pose.residues)):
        new_chis_angle = [random.uniform(-180, 180),
                          random.uniform(-180, 180),
                          random.uniform(-180, 180),
                          random.uniform(-180, 180)]
        residue = conformation.residue(i+1)
        num_chi = residue.nchi()
        
        for j in range(num_chi):
            protein_pose.set_chi(j+1, i+1, new_chis_angle[j])
    protein_pose.dump_pdb(output_file)


def random_rotate_peptide_sidechain_and_repacking(pdb_file, output_file):
    # 创建一个 Pose 对象
    protein_pose = pose_from_pdb(pdb_file)
    conformation = protein_pose.conformation()
    for i in range(len(protein_pose.residues)):
        new_chis_angle = [random.uniform(-180, 180),
                          random.uniform(-180, 180),
                          random.uniform(-180, 180),
                          random.uniform(-180, 180)]
        residue = conformation.residue(i+1)
        num_chi = residue.nchi()
        
        for j in range(num_chi):
            protein_pose.set_chi(j+1, i+1, new_chis_angle[j])
    task_pack = TaskFactory.create_packer_task(protein_pose)
    task_pack.restrict_to_repacking()
    scorefxn = pyrosetta.get_fa_scorefxn()
    # 创建 PackRotamersMover
    pack_rotamers_mover = PackRotamersMover(scorefxn, task_pack)

    # # 设置任务工厂和蛋白结构
    # pack_rotamers_mover.task_factory(scorefxn, task_pack)

    pack_rotamers_mover.apply(protein_pose)
    
    protein_pose.dump_pdb(output_file)

if __name__ =='__main__':
    pdb_file = './scripts/examples/0001.pdb'
    out_file = './scripts/examples/0001_packed.pdb'
    random_rotate_peptide(pdb_file, out_file)