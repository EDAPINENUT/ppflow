import argparse
import os 
import numpy as np
import threading
import concurrent.futures
import subprocess

def run_command(command):
    process = subprocess.Popen(command, shell=True)
    process.wait()

def run_commands(commands, block=4):
    with concurrent.futures.ThreadPoolExecutor(max_workers=block) as executor:

        chunks = [commands[i:i+block] for i in range(0, len(commands), block)]

        futures = {executor.submit(run_command, command) for command in chunks[0]}

        concurrent.futures.wait(futures)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_gen_dir', type=str, default='./results/ppflow/codesign_ppflow/')
    parser.add_argument('--benchmark_dir', type=str, default='./PPDbench')
    parser.add_argument('--mode', type=str, choices=['basic_prop', 'bind_aff'], default='basic_prop')
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--block', type=int, default=4)
    
    args = parser.parse_args()

    gen_dir_names = sorted(os.listdir(args.meta_gen_dir))

    cmds = []
    for gen_dir_name in gen_dir_names:

        gen_dir = os.path.join(args.meta_gen_dir, gen_dir_name)
        pdb_id = gen_dir_name.split('_')[1]
        ref_dir = os.path.join(args.benchmark_dir, pdb_id)

        if args.mode == 'basic_prop':
            cmd = 'python eval_struct_seq.py --gen_dir {} --ref_dir {} --save_path {}'.format(gen_dir, ref_dir, gen_dir)
            cmds.append(cmd)
        
        elif args.mode == 'bind_aff':
            cmd = 'python eval_bind.py --gen_dir {} --ref_dir {} --save_path {}'.format(gen_dir, ref_dir, gen_dir)
            cmds.append(cmd)
    
    if args.parallel:
        run_commands(cmds, args.block)
        
    else:
        for cmd in cmds:
            os.system(cmd)

        