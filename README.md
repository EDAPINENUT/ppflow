# PPFlow: Target-Aware Peptide Design with Torsional Flow Matching

## Installation

#### Create the conda environment and activate it.
```
conda create -n ppflow python==3.9
conda activate ppflow
```
#### Install basic packages
```
# install requirements
pip install -r requirements.txt

pip install easydict
pip install biopython
# mmseq
conda install bioconda::mmseqs2

# Alternative: obabel and RDkit
conda install -c openbabel openbabel
conda install conda-forge::rdkit

# Alternative for visualization: py3dmol
conda install conda-forge::py3dmol
```

### Packages for training and generating.

#### Install pytorch 1.13.1 with the cuda version that is compatible with your device. The geomstats package does not support torch>=2.0.1 on GPU. Here we recommend use torch==1.13.1.
```
# torch-geomstats
conda install -c conda-forge geomstats

# torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html  

# OR: stable torch-scatter
pip install ./temp/torch_scatter-2.1.1+pt113cu117-cp39-cp39-linux_x86_64.whl 
```

## Dataset 
We provide the processed dataset of `PPBench2024` through an anonymous google drive: 
Together with processed `PPDBench' through:

Please download it and unzip it, leading to the data file directory as 
```
- data
    - processed
        cluster_result_all_seqs.fasta
        cluster_result_cluster.tsv
        cluster_result_rep_seq.fasta
        parsed_pair.pt
        receptor_sequences.fasta
        split.pt
    - processed_bench
        cluster_result_all_seqs.fasta
        cluster_result_cluster.tsv
        cluster_result_rep_seq.fasta
        parsed_pair.pt
        receptor_sequences.fasta
        split.pt
    pdb_benchmark.pt
    pdb_filtered.pt
```

We cannot provide the raw dataset of `PPDbench` since it is hard to find a anonymous drive for the large dataset. For further test and comparison on reference `pdb` file. Unzip the file of `PPDbench.tar.gz`, leading to the directory as 
```
- dataset
    - PPDbench
        - 1cjr
            peptide.pdb
            recepotor.pdb
        - 1cka
            peptide.pdb
            recepotor.pdb
        ...      
    - ppbench2024
        - 1a0m_A
            peptide.pdb
            recepotor.pdb
```

The raw data of PPBench2024 will be uploaded soon.

## Training and Generating
### Training from scratch
Run the following command for PPFlow training:

```
python train_ppf.py
```

Run the following command for DiffPP training:

```
python train_diffpp.py
```

For RDE finetuning, you should firstly download the pretrained `RDE.pt` model from the [google drive](https://drive.google.com/drive/folders/18bCjncFKDK3eeYf6fyiFfdNgyK40K88g?usp=sharing), then save it as `./pretrained/RDE.pt`, and finally, run the following command for finetuning:

```
python train_rde.py --fine_tune ./pretrained/RDE.pt
```

#### After training, you can choose a epoch for generating the peptides through:

```
python codesign_diffpp.py -ckpt {where-the-trained-ckpt-is}
python codesign_ppflow.py -ckpt {where-the-trained-ckpt-is}
```

### Generating from pretrained checkpoints

Here we give the checkpoints that is pretrained, which is named `ppflow_pretrained.pt` and can be downloaded from the [google drive](https://drive.google.com/drive/folders/18bCjncFKDK3eeYf6fyiFfdNgyK40K88g?usp=sharing). You can directly download it an copy it to `./pretrained/ppflow_pretrained.pt`. Further, run the following to generation:

```
python codesign_diffpp.py -ckpt ./pretrained/ppflow_pretrained.pt
```


## Packages and Scripts for Evaluation

### Packages for docking and other evaluation.

#### For Vina Docking, install the packages through:
```
 conda install conda-forge::vina
 pip install meeko
 pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3.git@aee55d50d5bdcfdbcd80220499df8cde2a8f4b2a
 pip install pdb2pqr
```
`./tools/dock/vinadock.py` gives an example of our python interface for vinadock.

#### For HDock, first libfftw3 is needed for hdock with `apt-get install -y libfftw3-3`. Besides, the HDock softward can be download through: http://huanglab.phys.hust.edu.cn/software/hdocklite/. After downloading it, install or unzip it to the `./bin` directory, leading to the file structure as 
```
- bin
    - hdock
        1CGl_l_b.pdb
        1CGl_r_b.pdb
        createpl
        hdock
```
`./tools/dock/hdock.py`  gives an example of our python interface for hdock.

#### Pyrosetta: For pyrosetta, you should first sign up in https://www.pyrosetta.org/downloads. After the authorizations of liscence, you can install it through
```
 conda config --add channels https://yourauthorizedid:password@conda.rosettacommons.org 
 conda install pyrosetta   
```

Since it is an anonymous github, I have apply a random ID and password for you to test, as 

```
 conda config --add channels https://levinthal:paradox@conda.rosettacommons.org 
 conda install pyrosetta   
```
`./tools/relax/rosetta_packing.py` gives an example of our python interface for rosetta side-chain packing.

#### FoldX: For FoldX, you should register and log in according to https://foldxsuite.crg.eu/foldx4-academic-licence, and download the packages and copy it to `./bin`. Than, unip it will lead directory look like 

```
- bin
    - FoldX
        foldx
```
where foldx is the software. `./tools/score/foldx_energy.py` gives an example of our python interface for foldx stability.

#### ADCP: We provide the available ADFRsuite software in `./bin`. If it does not compatitable with your system, please install it through https://ccsb.scripps.edu/adcp/downloads/. Copy the `ADFRsuite_x86_64Linux_1.0.tar` into `./bin`. Finally, the installed ADCP into `./bin` should look like
```
- bin
    - ADFRsuite_x86_64Linux_1.0
        - Toos
          CCSBpckgs.tar.gz
          ...
      ADFRsuite_Linux-x86_64_1.0_install.run
      uninstall
```
Remember to add it to your env-path as 
```
export PATH={Absolute-path-of-ppfolw}/bin/ADFRsuite_x86_64Linux_1.0/bin:$PATH
```
`./tools/dock/adcpdock.py` gives an example of our python interface for ADCPDocking.

#### TMscore: The available TMscore evaluation software is provided in `./bin`, as 
```
- bin
    - TMscore
        TMscore 
        TMscore.cpp
```

#### PLIP for interaction analysis
If you want to analyze the interaction type of generated protein-peptide, you can use PLIP: https://github.com/pharmai/plip.
First, clone it to `./bin`
```
cd ./bin
git clone https://github.com/pharmai/plip.git
cd plip
python setup.py install
alias plip='python {Absolute-path-of-ppfolw}/bin/plip/plip/plipcmd.py' 
```

`./tools/interaction/interaction_analysis.py` gives an example of our python interface for plip interaction analysis.

### Evaluation scripts
The evaluation scripts are given in `./evaluation` directory, you can run the following for the evalution in the main experiments:

```
# Evaluting the docking energy
python eval_bind.py --gen_dir {where-the-generated-peptide-is} --ref_dir {where-the-protein-pdb-is} --save_path {where-you-want-the-docked-peptide-to-be-saved-in}

# Evaluating the sequence and structure
python eval_bind.py --gen_dir {where-the-generated-peptide-is} --ref_dir {where-the-protein-pdb-is} --save_path {where-you-want-the-docked-peptide-to-be-saved-in}
```

We give an example file pair, so the `gen_dir` can be `./results/ppflow/codesign_ppflow/0008_3tzy_2024_01_19__19_16_21`,  `ref_dir` can be `./PPDbench/3tzy/` and the `save_path` can be `./results/ppflow/codesign_ppflow/0008_3tzy_2024_01_19__19_16_21`.



## Citation
If our paper or the code in the repository is helpful to you, please cite the following:
```
@inproceedings{lin2024ppflow,
	author = {Lin, Haitao and Zhang, Odin and Zhao, Huifeng and Jiang, Dejun and Wu, Lirong and Liu, Zicheng and Huang, Yufei and Li, Stan Z.},
	title = {PPFlow: Target-Aware Peptide Design with Torsional Flow Matching},
	year = {2024},
	booktitle={International Conference on Machine Learning},
	URL = {https://www.biorxiv.org/content/early/2024/03/08/2024.03.07.583831},
	eprint = {https://www.biorxiv.org/content/early/2024/03/08/2024.03.07.583831.full.pdf},
}

```

