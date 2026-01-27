"""Contains all functions fo different genotype to phenotype mappings
All function should take a single sequence as input and output a list
of phenotypes (can be of size one as well)

"""
import numpy as np
from typing import Callable

from rna_folding.base_pairing import BasePairing
from rna_folding.nussinov import BasePairMatrixNussinov
from rna_folding.utils import bp_to_dotbracket, dotbracket_to_genotype, dotbracket_to_genotype_random, is_compatible
from rna_folding.parsing import dict_to_gpmap
import RNA

def test():
    print("D")

def gp_mapper(input: str, output: str, mapping_function: Callable):
    """Takes file with genotypes, maps them to phenotypes and saves them in
    output file

    Args:
        input (str): Path to input file.
        output (str): Path to output file.
        mapping_function (function): A function takes a genotype (str) as 
        single positional argument and returns a list of phenotypes (str).

    Returns:
        None
    
    """
    ph_to_gt = {}

    # Read genotypes and map to phenotypes
    with open(input, "r") as file_in:
        for i, sequence in enumerate(file_in):
            seq = sequence.strip()
            phenotypes_ = mapping_function(seq)
            # add sequence ID to the phenotype that they map to
            for ph in phenotypes_:
                try:
                    ph_to_gt[ph].append(i)
                except KeyError:
                    ph_to_gt[ph] = [i]

    # Write to output file (line example: "{ph} {gt_id} {gt_id} {gt_id}\n"
    dict_to_gpmap(ph_to_gt=ph_to_gt, file=output)
    

def nussinov(genotype: str, 
             base_pairing: BasePairing, 
             min_loop_size: int, 
             suboptimal: int, 
             structures_max: int) -> list:
    """Nussinov genotype-phenotype mapping wrapper

    Args:
        genotype (str): genotype to be mapped
        base_pairing (BasePairing): An BasePairing object defining pairing ules
        min_loop_size (int): minimum size that RNA loops must have
        suboptimal (int): How many base-pairs off from optimum are allowed
        structures_max (int): How many structures to generate at most

    Returns:
        list: List of phenotypes that the genotypes maps to

    """
    P = BasePairMatrixNussinov(n=len(genotype), base_pairing=base_pairing)
    P.fill_matrix(seq=genotype, min_loop_size=min_loop_size)
    strucs = P.traceback_subopt(seq=genotype, d=suboptimal,
                                structures_max=structures_max)
    # print(genotype)
    # for s in strucs:
    #     print(bp_to_dotbracket(s.B, l=len(genotype)))
    #     for pair in s.B:
    #         print(genotype[pair[0]-1], genotype[pair[1]-1])
    phenotypes = [bp_to_dotbracket(s.B, l=len(genotype)) for s in strucs]

    return phenotypes


def nussinov_mfe(genotype: str, 
                 base_pairing: BasePairing, 
                 min_loop_size: int, 
                 suboptimal: int, 
                 structures_max: int,
                 seed: int,
                 base_pair: str = "GC",
                 deterministic: bool = False)-> list:
    """Nussinov + mfe ranking genotype-phenotype mapping wrapper.
    Candidate phenotypes are generated using Nussinov's algorithm which are
    then mapped to a canonical genotype and scored using viennaRNA package

    Args:
        genotype (str): genotype to be mapped
        base_pairing (BasePairing): An BasePairing object defining pairing ules
        min_loop_size (int): minimum size that RNA loops must have
        suboptimal (int): How many base-pairs off from optimum are allowed
        structures_max (int): How many structures to generate at most
        base_pair (str): Which base-pair is used for MFE calc., either "GC" or "AU".
        seed (int): Random seed to use for generation of canonical genotype
        deterministic (bool): If True, always use G to for unpaired sites, else
        randomly pick G or C, if "GC" is given as base-pair

    Returns:
        list: List of phenotypes that the genotypes maps to
        
    """
    phenotypes = nussinov(genotype=genotype, base_pairing=base_pairing, 
                            min_loop_size=min_loop_size, suboptimal=suboptimal, 
                            structures_max=structures_max)
    
    g_fe_map = []

    for ph in phenotypes:
        # turn into a canonical alphabet
        seq_canon = dotbracket_to_genotype(dotbracket=ph,
                                            base_pair=base_pair,
                                            random=deterministic,
                                            seed=seed)
        g_fe_map.append(RNA.eval_structure_simple(seq_canon, ph))
    
    mfe_ph_id = np.argmin(g_fe_map)  # get index of mfe phenotype
    if g_fe_map[mfe_ph_id] >= 0:  # if energy is above 0
        mfe_ph = "."*len(phenotypes[mfe_ph_id])  # phenotype is unf.
    else:
        mfe_ph = phenotypes[mfe_ph_id]  # get mfe phenotype
    
    return [mfe_ph]


def debug_nussinov_mfe(genotype: str,
                 base_pairing: BasePairing,
                 min_loop_size: int,
                 suboptimal: int,
                 structures_max: int,
                 seed: int,
                 base_pair: str = "GC",
                 deterministic: bool = False) -> list:
    """Nussinov + mfe ranking genotype-phenotype mapping wrapper.
    Candidate phenotypes are generated using Nussinov's algorithm which are
    then mapped to a canonical genotype and scored using viennaRNA package

    Args:
        genotype (str): genotype to be mapped
        base_pairing (BasePairing): An BasePairing object defining pairing ules
        min_loop_size (int): minimum size that RNA loops must have
        suboptimal (int): How many base-pairs off from optimum are allowed
        structures_max (int): How many structures to generate at most
        base_pair (str): Which base-pair is used for MFE calc., either "GC" or "AU".
        seed (int): Random seed to use for generation of canonical genotype
        deterministic (bool): If True, always use G to for unpaired sites, else
        randomly pick G or C, if "GC" is given as base-pair

    Returns:
        list: List of phenotypes that the genotypes maps to

    """
    phenotypes = nussinov(genotype=genotype, base_pairing=base_pairing,
                          min_loop_size=min_loop_size, suboptimal=suboptimal,
                          structures_max=structures_max)

    g_fe_map = []

    for ph in phenotypes:
        # turn into a canonical alphabet
        seq_canon = dotbracket_to_genotype_random(dotbracket=ph)
        g_fe_map.append(RNA.eval_structure_simple(seq_canon, ph))
    mfe_ph_id = np.argmin(g_fe_map)  # get index of mfe phenotype
    if g_fe_map[mfe_ph_id] >= 0:  # if energy is above 0
        mfe_ph = "." * len(phenotypes[mfe_ph_id])  # phenotype is unf.
    else:
        mfe_ph = phenotypes[mfe_ph_id]  # get mfe phenotype

    return [mfe_ph], phenotypes, g_fe_map


def viennaRNA_mfe(genotype: str, return_mfe=False) -> list:
    """Predict RNA secondary structure using default ViennaRNA mfe function.
    Args:
        genotype (str): Input genotype

    Returns:
        list: List of phenotype the genotype maps to.
        
    """
    mfe_ph, mfe = RNA.fold(genotype)
    if return_mfe:
        return mfe_ph, mfe
    else:
        return [mfe_ph]

def nussinov_canonical_fe(genotype: str, 
                 base_pairing: BasePairing, 
                 min_loop_size: int, 
                 suboptimal: int, 
                 structures_max: int) -> list:
    """Nussinov + free energy calc genotype-phenotype mapping wrapper.
    Candidate phenotypes are generated using Nussinov's algorithm which are
    then mapped to a canonical genotype and scored using viennaRNA package.
    Only works for the canonical alphabet for now

    Args:
        genotype (str): genotype to be mapped
        base_pairing (BasePairing): An BasePairing object defining pairing ules
        min_loop_size (int): minimum size that RNA loops must have
        suboptimal (int): How many base-pairs off from optimum are allowed
        structures_max (int): How many structures to generate at most

    Returns:
        list: (List of phenotypes where phenotype is a comma-separated string 
                containing dotbracket phenotype and free energy, e.g.:
                ["(((...))),-2.4", "(..).....,8.5", ...] 
                list is sorted by free energy, low to high
        
    """
    phenotypes = nussinov(genotype=genotype, base_pairing=base_pairing, 
                            min_loop_size=min_loop_size, suboptimal=suboptimal, 
                            structures_max=structures_max)

    energies = []
    for ph in phenotypes:
        # turn into a canonical alphabet
        energies.append(RNA.eval_structure_simple(genotype, ph))
    
    # sort both lists based energy values (low to high)
    sorted_gf_map = [p+","+str(np.round(e, 2)) for e, p in sorted(zip(energies, phenotypes), key=lambda pair: pair[0])]

    return sorted_gf_map


def nussinov_with_probabilistic_scoring(genotype: str, base_pairing, scores: dict, rng=None) -> list:
    """Generate suboptimal set with nussinov's algorithm and pick phenotype
    based on probabilistic scores.

    Args:
        genotype (str):     Genotypes string, e.g. "AUGGCA"
        base_pairing (BasePairing): A BasePairing object defining pairing rules
        scores (dict):      Dictionary that maps phenotypes (str) to a score 
                            (float).

    Returns:
        list:               Phenotype string inside a list.

    """
    if not rng:
        rng = np.random.Generator()

    phenotypes = [ph for ph in scores if is_compatible(genotype, ph, base_pairing)]
    scores_subset = np.array([scores[ph] for ph in phenotypes])
    scores_norm = scores_subset/np.sum(scores_subset)  # normalize

    phenotype = rng.choice(phenotypes, p=scores_norm)  # choose based on score

    return [phenotype]



