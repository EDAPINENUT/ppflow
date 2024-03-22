from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta.core.pack.task import TaskFactory
import pyrosetta
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

init()


def get_scorefxn(scorefxn_name:str):
    """
    Gets the scorefxn with appropriate corrections.
    Taken from: https://gist.github.com/matteoferla/b33585f3aeab58b8424581279e032550
    """
    import pyrosetta

    corrections = {
        'beta_july15': False,
        'beta_nov16': False,
        'gen_potential': False,
        'restore_talaris_behavior': False,
    }
    if 'beta_july15' in scorefxn_name or 'beta_nov15' in scorefxn_name:
        # beta_july15 is ref2015
        corrections['beta_july15'] = True
    elif 'beta_nov16' in scorefxn_name:
        corrections['beta_nov16'] = True
    elif 'genpot' in scorefxn_name:
        corrections['gen_potential'] = True
        pyrosetta.rosetta.basic.options.set_boolean_option('corrections:beta_july15', True)
    elif 'talaris' in scorefxn_name:  #2013 and 2014
        corrections['restore_talaris_behavior'] = True
    else:
        pass
    for corr, value in corrections.items():
        pyrosetta.rosetta.basic.options.set_boolean_option(f'corrections:{corr}', value)
    return pyrosetta.create_score_function(scorefxn_name)

def side_chain_packing(pdb_file, output_file, score_fn='ref2015'):
    protein_pose = pose_from_pdb(pdb_file)
    task_pack = TaskFactory.create_packer_task(protein_pose)
    task_pack.restrict_to_repacking()
    scorefxn = get_scorefxn(score_fn)

    pack_rotamers_mover = PackRotamersMover(scorefxn, task_pack)

    pack_rotamers_mover.apply(protein_pose)
    protein_pose.dump_pdb(output_file)
    return output_file

if __name__ =='__main__':
    pdb_file = './scripts/examples/0001.pdb'
    out_file = './scripts/examples/0001_packed.pdb'
    side_chain_packing(pdb_file, output_file=out_file)