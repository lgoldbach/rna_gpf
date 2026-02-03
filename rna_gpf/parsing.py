import argparse
import numpy as np
from typing import Type



def many_to_one_map_from_file_to_dict(file: str, source_type: Type = str, target_type: Type = str, delimiter: str = " ", skip_first: bool = False) -> dict:
    """Read in a file that is a many-to-one (source-to-target) mapping of the 
    and turn it into dictionary mapping every source to their target. Order of 
    targets and sources does not matter. Every source can only appear once in
    the file, otherwise it would be a many-to-many map.

    Args:
        file (str):         Path to a file that contains many-to-one mapping.
                            Example:
                            <target1> <source A> <source B> ...
                            <target2> <source D> <source K> ...
                            <target3> <source C> ...
                            ...
                            
        source_type (Type): Desired type of the sources, i.e. dict keys.
                            Default: str. 
        target_type (Type): Desired the type of the target, i.e. dict values.
                            Default: str.
        delimiter (str):    Delimiter used in the file. Default: " ".

    Returns:
        dict:               Dictionary that maps every source of type 
                            source_type to its target of type target_type.
                            Example:
                            {<source X (int)>: target 1 (str), 
                            <source X (int)>: target 1 (str), }
                        
    """
    D = {}
    with open(file, "r") as f:
        for line_ in f:
            line = line_.strip().split(delimiter)
            target = int(line[0])
            if skip_first:
                start_idx = 2
            else:
                start_idx = 1
            for source in line[start_idx:]:
                D[source] = target

    return D


def gpmap_pgdict(gpmap_file: str, genotype_file: str = None) -> dict:
    """Takes a file that stores genotype-phenotype mapping and a list of 
    genotypes and parses it into dictionary {<phe> (str): [genotypes (str)]}. 
    Intended for many-to-many mappings

    Args:
        gpmap_file (str):       Path to file in following format:
                                <phenotype> <genotypeID_x> <genotypeID_y>
                                <phenotype> <genotypeID_y>
                                ...
                                Example for RNA secondary structure:
                                ((..)) 2 1
                                (....) 3
                                ().... 4 1
                                ...

        genotype_file (str):    file path to a file that contains simple list of 
                                genotypes (one per line)

    Returns:
        pgmap (dict):           Dictionary that maps phenotype (str) to list of
                                one or more genotypes (str).

    """
    if genotype_file:
        with open(genotype_file, "r") as g_file:    
            genotype_list = [line.strip() for line in g_file]
         
    pg_map = {}
    with open(gpmap_file, "r") as gp_file:
        for line in gp_file:
            l = line.strip().split(" ")
            pg_map[l[0]] = [genotype_list[int(gt)] for gt in l[1:]]
    
    return pg_map


def gpmap_to_lists(gpmap_file: str) -> tuple:
    """Takes a gp map text file and returns a tuple which contains a list of
    genotypes and a list of phenotypes where the ith genotype of the first list
    maps to the iths phenotype of the second list

    Args:
        gpmap_file (str):       Path to file in following format:
                                <phenotype> <genotypeID_x> <genotypeID_y>
                                <phenotype> <genotypeID_y>
                                ...
                                Example for RNA secondary structure:
                                ((..)) 2 1
                                (....) 3
                                ().... 4 1
                                ...

    Returns:
        tuple (list, list):     One list with genotypes and another phenotypes
                                Both same length

    """
    genotypes = []
    phenotypes = []
    with open(gpmap_file, "r") as gp_file:
        for line in gp_file:
            l = line.split()
            ph = l[0]
            for gt in l[1:]:
                genotypes.append(gt)
                phenotypes.append(ph)
    
    return genotypes, phenotypes


def lists_to_gp_map(genotypes, phenotypes, output_filename) -> None:
    """Take a list of genotypes and phenotypes and saves them in standard
    gp map format.

    Args:
        genotypes (list): list of genotype strings
        phenotypes (list): list of phenotypes strings
        output_filename (str):    Path to file in following format:
                                    <phenotype> <genotypeID_x> <genotypeID_y>
                                    <phenotype> <genotypeID_y>
                                    ...
                                    Example for RNA secondary structure:
                                    ((..)) 2 1
                                    (....) 3
                                    ().... 4 1
                                    ...

    """
    pg_map = {}
    for (gt, ph) in zip(genotypes, phenotypes):
        if ph in pg_map:
            pg_map[ph].append(gt)
        else:
            pg_map[ph] = [gt]

    dict_to_gpmap(pg_map, output_filename)
        

def gpmap_to_dict(gpmap_file: str, genotype_file: str = None) -> dict:
    """Takes a file that stores genotype-phenotype mapping and a list of 
    genotypes and parses it into dictionary 
    {<genotype> (str): [phenotypes (str)]}. Intended for mappings to multiple 
    phenotypes.    

    Args:
        gpmap_file (str): Path to file in following format:
                            <phenotype> <genotypeID_x> <genotypeID_y>
                            <phenotype> <genotypeID_y>
                            ...
                            Example for RNA secondary structure (dot-bracket):
                            ((..)) 2 1
                            (....) 3
                            ().... 4 1
                            ...

        genotype_file (str): file path to a file that contains simple list of 
                                genotypes (one per line)

    Returns:
        gpmap (dict): Dictionary that maps genotype (str) to list of one or 
                        more phenotypes (str).

    """
    # read in genotypes as list
    if genotype_file:
        with open(genotype_file, "r") as g_file:    
            genotype_list = [line.strip() for line in g_file]
         
    gp_map = {}
    with open(gpmap_file, "r") as gp_file:
        for line in gp_file:
            l = line.split()
            db = l[0]
            for gt in l[1:]:
                if genotype_file:
                    gt = genotype_list[int(gt)]  # get genotype using id (i)
                if gt in gp_map:
                    gp_map[gt].append(db)
                else:
                    gp_map[gt] = [db]

    return gp_map


def viennarna_to_gp_map_file(viennarna_output: str) -> dict:
    """Takes an output file and parses it into dictionary 
    {<genotype> (str): [phenotypes (str)]}. Intended for mappings to multiple 
    phenotypes.    

    Args:
        viennarna_output (str): path to file with format:
                                <seq1>
                                <dot-bracket> <free energy>
                                <seq2>
                                <dot-bracket> <free energy>
                                ...
                                e.g:
                                GGGAAACCC
                                (((...))) (-1.20)
                                AAAAAAAAA
                                ......... (0.00)
    Returns:
        gpmap (dict): Dictionary that maps genotype (str) to list of one or 
                        more phenotypes (str).

    """
    gp_map = {}
    with open(viennarna_output, "r") as file:
        for line in file:
            gt = line.strip()
            ph = next(file).split(" ")[0]
            gp_map[gt] = ph

    return gp_map


def dict_to_gpmap(ph_to_gt: dict, file: str) -> None:
    """Take a dict that maps phenotype to list of genotypes and save it
    as a space-separated "c"sv file, where each line looks like this:
    "{ph} {gt_id} {gt_id} {gt_id}"

    Args:
        ph_to_gt (dict): _description_
        file (str): _description_
    """
    # Write to output file (
    with open(file, "w") as file_out:
        for p in ph_to_gt:
            line = p + " " + " ".join(map(str, ph_to_gt[p])) + "\n"
            file_out.write(line)
    file_out.close()

def load_phenotype_and_metric_from_file(file: str, dtype=float, ignore: str = None):
    """Take a file in the common phenotype (col1) metric (col2) data-type 
    I am using and reat it as two array.
    Example file:
    ((...)) 0.8
    (.....) 0.7
    ...

    Args:
        file (str):     Path to the file
        ignore (str):   (optional) Phenotype to ignore.

    Retruns:
        phenotypes, data
    """
    file_data = np.loadtxt(file, dtype=str)
    if file_data.ndim == 1:  # in case there is only one phenotype
        file_data = np.expand_dims(file_data, axis=0)
    phenotypes = file_data[:,0]
    metric = file_data[:,1].astype(dtype)
    
    if ignore:
        phenotypes_ = []
        metric_ = []
        for ph, m in zip(phenotypes, metric):
            if ph != ignore:
                phenotypes_.append(ph)
                metric_.append(m)
        phenotypes = phenotypes_
        metric = metric_

    return phenotypes, metric

def genotype_file_to_numpy(filepath):
    """Take genotype file and return numpy array with each line as one line
    in the numpy array, so a single column, n=#genotyopes row array

    Args:
        filepath (str): Path to genotype file
    
    Returns:
        np.array (dtype='<U4'): Array containing genotypes row by row

    """
    A = np.loadtxt(filepath, dtype=str)

    return A


def read_ruggedness_per_ph_file(filename, n):
    """Read file that contains peak sizes for <n> fitness landscape for each
    phenotype
    instances for each ph. Assumes following file structure:
    <ph1>
    <peak 1 size (int)> <peak 2 size (int)> <peak 3 size (int)> ...  # fl 1
    <peak 1 size (int)> <peak 2 size (int)> <peak 3 size (int)> ...  # fl 2
    <ph2>
    ...

    Args:
        filename (str): File containing peak sizes
        n (int): Number of fitness landscape instances per ph

    Returns:
        r (dict):   Dictionary that maps ph to the peak sizes of each of the
                    fitness landscapes

    """
    r = {}
    with open(filename, "r") as f:
        lines = list(f)
        for i in range(0, len(lines), n+1):
            ph = lines[i].strip().split(" ")[0]  # read ph
            r[ph] = []
            for j in range(i+1, i+1+n):  # loop over next n lines (one for each fl)
                peaks_sizes = lines[j].strip().split(" ")
                r[ph].append([int(k) for k in peaks_sizes])  # append peak sizes as int
    return r

def read_ruggedness_file(filename):
    """Read file that contains peak sizes for <n> fitness landscape
    instances for each ph. Assumes following file structure:
    <peak 1 size (int)> <peak 2 size (int)> <peak 3 size (int)> ...  # fl 1
    <peak 1 size (int)> <peak 2 size (int)> <peak 3 size (int)> ...  # fl 2
    ...

    Args:
        filename (str): File containing peak sizes

    Returns:
        r (dict):   Dictionary that maps ph to the peak sizes of each of the
                    fitness landscapes

    """
    peak_sizes = []
    with open(filename, "r") as f:
        for line in f:
            peaks_sizes_line = line.strip().split(" ")
            peak_sizes.append([int(k) for k in peaks_sizes_line])  # append peak sizes as int
    return peak_sizes

def read_adaptive_walks_w_ph_headers_to_dict(filepath, phenotypes):
    """Read paths from a file that contains phenotyp header in the following 
    format:
    <ph1>
    <gt1> <gt2> <gt3> ...  # path 1
    <gt1> <gt2> <gt3> ...  # path 2
    <ph2>
    ...

    Args:
        filepath (str):     File path for path file
        phenotypes (list):  List of phenotypes (str) which should contain all 
                            the phenotypes potentially found as path headers

    Returns:
        ph_to_paths (dict): Dictionary that maps phenotypes to paths

    """
    ph_to_paths = {}
    with open(filepath, "r") as file:
        for line_ in file:
            line = line_.strip().split()
            # check if we hit a phenotype header
            if line[0] in phenotypes:
                ph = line[0]
                ph_to_paths[ph] = []
            else:
                # all the lines following the header will contain paths
                ph_to_paths[ph].append(line)
    return ph_to_paths
        
def read_navigability_per_fl(file: str) -> dict:
    navig = {}
    with open(file, "r") as file:
        for line_ in file:
            line = line_.strip().split(" ")
            ph = line[0]
            nav = float(line[1])
            if ph in navig:
                navig[ph].append(float(nav))
            else:
                navig[ph] = [float(nav)]
    return navig
    

def read_navigability_per_ph_per_fl_file(file: str) -> dict:
    """Take a text file with navigability values and translate it into a 
    dict that maps each phenotype to a list of navigability values, one for
    each fitness landscape.

    Args:
        file (str): File path to a navigability file of the following format:
                    <ph1 (str)> <navig. fl 1 (float)> <navig. fl 2 (float)> ..
                    <ph2 (str)> <navig. fl 1 (float)> <navig. fl 2 (float)> ..
                    ...

    Returns:
        dict:       Maps every phenotype to a list of navigability values:
                    {<ph1 (str)>: [<navig. fl 1 (float)>, <navig. fl 2 (float)>, ...],
                    <ph1 (str)>: <navig. fl 1 (float)>, <navig. fl 2 (float)> ...],
                    ...}

                    e.g.:
                    {"((...))": [0.8, 0.65, 0.2], "(...)..": [1.0, 0.2, 0.3]}

    """
    navig = {}
    with open(file, "r") as file:
        for line_ in file:
            line = line_.strip().split(" ")
            ph = line[0]
            navig[ph] = [float(n) for n in line[1:]]
    
    return navig


