from Bio.PDB import PDBParser, PDBIO
parser = PDBParser(QUIET=True)

def merge_protein_ligand(protein_pdb_file, ligand_pdb_file, out_pdb_file):
    structure_protein = parser.get_structure(protein_pdb_file, protein_pdb_file)[0]
    structure_ligand = parser.get_structure(ligand_pdb_file, ligand_pdb_file)[0]
    receptor_chains = set()
    for chain in structure_protein:
        receptor_chains.add(chain.id)

    ligand_chains = set()
    for chain in structure_ligand:
        ligand_chains.add(chain.id)

    for chain in structure_ligand:
        structure_protein.add(chain)
    
    io = PDBIO() 
    io.set_structure(structure_protein) 
    io.save(out_pdb_file)

    receptor_chains = ''.join(list(receptor_chains))
    ligand_chains = ''.join(list(ligand_chains))
    interface = f"{receptor_chains}_{ligand_chains}"
    return out_pdb_file, interface

