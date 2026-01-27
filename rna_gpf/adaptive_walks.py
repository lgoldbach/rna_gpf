import numpy as np
from rna_folding.gp_map import GenotypePhenotypeGraph
from typing import Callable, Type
import copy
from rna_folding.utils import remove_nonadaptive_edges
import networkx as nx


def kimura_fixation_from_fitness(f1: float, f2: float, N: int):
    """Formula to compute kimuara's fixation probability

    Args:
        f1 (float): Fitness 1
        f2 (float): Fitness 2
        N (int): Population size

    Returns:
        float: fixation probability (float in [0, 1]).
    
    """
    s = f2-f1  # selection coefficient is the fitness difference
    if s == 0:
        s = -10**-10

    p = (1-np.exp(-2*s))/(1-np.exp(-2*N*s))
    return p


def kimura_fixation(s: float, N: int):
    """Formula to compute kimuara's fixation probability

    Args:
        s (float): Selection coefficient
        N (int): Population size

    Returns:
        float: fixation probability (float in [0, 1]).
    
    """
    if s == 0:
        s = -10**-10

    p = (1-np.exp(-2*s))/(1-np.exp(-2*N*s))
    return p

def kimura_fixation_vectorizable(p1: float, p2: float, N: int):
    """Formula to compute kimuara's fixation probability

    Args:
        p1 (float): Phenotype 1
        p2 (float): Phenotype 2
        N (int): Population size

    Returns:
        float: fixation probability (float in [0, 1]).
    
    """
    s = p2 - p1
    return (1-np.exp(-2*s))/(1-np.exp(-2*N*s))


def adaptive_walk(gpmap: GenotypePhenotypeGraph, 
                  starting_genotype,
                  fitness_function,
                  max_steps,
                  population_size,
                  fixation_function,
                  rng) -> list:
    path = [starting_genotype]
    if fitness_function[gpmap.nodes[path[-1]]["phenotype"]] == 1:
        return path
    
    while len(path) < max_steps:
        candidate = rng.choice(gpmap._neighbors(path[-1]))
        f1 = fitness_function[gpmap.nodes[path[-1]]["phenotype"]]
        f2 = fitness_function[gpmap.nodes[candidate]["phenotype"]]
        if f2 == 1:  # found target phenotype
            path.append(candidate)  # append and break
            break
        s = f2-f1
        p = fixation_function(s, N=population_size)
        # this ignores waiting times for mutations, in a sense that we only
        # track the occurences of mutatons and if the mutation is rejected
        # we instead append the same genotype again. From this sequence we
        # can infer waiting times from a realistic mutation probability
        # with high population size it will be impossible to traverse neutral
        # nets with this approach
        if rng.uniform() < p:
            path.append(candidate)
        else:
            path.append(path[-1])
    return path

def productive_adaptive_walk(gpmap: GenotypePhenotypeGraph, 
                  starting_genotype,
                  fitness_function,
                  max_steps,
                  population_size,
                  fixation_function,
                  rng,
                  max_fit=1) -> list:
    path = [starting_genotype]
    if fitness_function[gpmap.map(path[-1])] == max_fit:
        return path
    
    while len(path) < max_steps:
        probs = []
        f1 = fitness_function[gpmap.map(path[-1])]
        neighbors = gpmap._neighbors(path[-1])
        for neigh in neighbors:
            f2 = fitness_function[gpmap.map(neigh)]
            s = f2-f1
            p = fixation_function(s, N=population_size)
            probs.append(p)
        
        if sum(probs) == 0:  # no way to go
            break
        normed_probs = np.array(probs) / sum(probs)
        candidate = rng.choice(neighbors, p=normed_probs)

        path.append(candidate)
        if fitness_function[gpmap.map(candidate)] == max_fit:  # found target phenotype
            break
    return path


def productive_adaptive_walk_w_T(gpmap: GenotypePhenotypeGraph, 
                  starting_genotype,
                  fitness_function,
                  T,
                  max_steps,
                  rng,
                  max_fit=1) -> list:
    path = [starting_genotype]
    if fitness_function[gpmap.map(path[-1])] == max_fit:
        return path
    
    while len(path) < max_steps:
        probs = []
    
        neighbors = gpmap._neighbors(path[-1])
        for neigh in neighbors:
            probs.append(T[(gpmap.map(path[-1]), gpmap.map(neigh))])
        
        if sum(probs) == 0:  # no way to go
            break
        normed_probs = np.array(probs) / sum(probs)
        candidate = rng.choice(neighbors, p=normed_probs)
        path.append(candidate)
        if fitness_function[gpmap.map(candidate)] == max_fit:  # found target phenotype
            break
    return path


def greedy_adaptive_walk(gpmap: GenotypePhenotypeGraph, 
                  starting_genotype,
                  fitness_function,
                  max_steps,
                  rng) -> list:
    path = [starting_genotype]
    if fitness_function[gpmap.nodes[path[-1]]["phenotype"]] == 1:
        return path
    
    while len(path) < max_steps:
        s_coeffs = []
        f1 = fitness_function[gpmap.nodes[path[-1]]["phenotype"]]
        neighbors = gpmap._neighbors(path[-1])
        for neigh in neighbors:
            f2 = fitness_function[gpmap.nodes[neigh]["phenotype"]]
            s = f2-f1
            s_coeffs.append(s)
        
        s_coeffs = np.array(s_coeffs)
        m = np.max(s_coeffs)  # find max
        if m >= 0:
            maxima = np.where(s_coeffs==m)[0]  # find all maxima
            next_gt = rng.choice(maxima)  # pick one at random
            path.append(neighbors[next_gt])
            if fitness_function[gpmap.nodes[path[-1]]["phenotype"]] == 1:  # found target phenotype
                break
        else:
            break  # no higher or equal fitness found, end path
        
    return path


def greedy_adaptive_walk_no_neutral(gpmap: GenotypePhenotypeGraph, 
                  starting_genotype,
                  fitness_function,
                  max_steps,
                  rng) -> list:
    path = [starting_genotype]
    if fitness_function[gpmap.nodes[path[-1]]["phenotype"]] == 1:
        return path
    
    while len(path) < max_steps:
        s_coeffs = []
        f1 = fitness_function[gpmap.nodes[path[-1]]["phenotype"]]
        neighbors = gpmap._neighbors(path[-1])
        for neigh in neighbors:
            f2 = fitness_function[gpmap.nodes[neigh]["phenotype"]]
            s = f2-f1
            s_coeffs.append(s)
        
        s_coeffs = np.array(s_coeffs)
        m = np.max(s_coeffs)  # find max
        if m > 0:
            maxima = np.where(s_coeffs==m)[0]  # find all maxima
            next_gt = rng.choice(maxima)  # pick one at random
            path.append(neighbors[next_gt])
            if fitness_function[gpmap.nodes[path[-1]]["phenotype"]] == 1:  # found target phenotype
                break
        else:
            break  # no higher fitness found, end path
        
    return path

def nc_uniform_adaptive_walk(nc_graph: nx.DiGraph, 
                             starting_nc,
                             max_steps,
                             rng) -> list:
    path = [starting_nc]

    while len(path) < max_steps:
        print(path, flush=True)
        successors = list(nc_graph.successors(path[-1]))
        # reached a peak if there are no successors in digraph
        if not successors:  
            print(successors, flush=True)
            break
        else:
            print(successors, flush=True)
            # randomly choose a successor (uniform adaptive walk)
            next_nc = rng.choice(successors)  
            print(next_nc, flush=True)
            path.append(next_nc)

    return path

def pairwise_transition_prob_dict(f_map: dict, func: Callable) -> dict:
    """Generate a quick look up array with pairwise transition probabilities

    Args:
        f_map (dict):           dictionary mapping genotype or phenotype to 
                                fitness
        func (Callable):        A transition probability function, e.g. 
                                fixation probability function. The function has
                                to take in two values, fitness 1 and fitness 2.

    Returns:
        dict: dict array where entry (i,j) is the fixation probability of 
        genotype or phenotype i to j (not normalized).

    """
    T = {}

    pair_idx = ((i,j) for i in f_map for j in f_map)

    # Compute transition probability for all pairs
    for p in pair_idx:
        T[(p[0], p[1])] = func(f_map[p[0]], f_map[p[1]])
    return T

def update_T(T: dict, ph, phenotypes, func: Callable, f_map: dict):
    """Update the entries for a single phenotype in an existing transition 
    matrix

    Args:
        T (dict):               Transition matrix 
        ph (immutable):         Phenotype dictionary key.   
        phenotypes (iterable):  List of all phenotype for which to compute new
                                transition probability to or from ph     
        func (Callable):        A transition probability function, e.g. 
                                fixation probability function. The function has
                                to take in two values, fitness 1 and fitness 2.   
        f_map (dict):           dictionary mapping genotype or phenotype to 
                                fitness

    Returns:
        dict: Transiton matrix T with changed entries for ph

    """
    pair_idx_for = ((ph, j) for j in phenotypes)
    pair_idx_back = ((i, ph) for i in phenotypes)
    for p in pair_idx_for:
        T[(p[0], p[1])] = func(f_map[p[0]], f_map[p[1]])
    for p in pair_idx_back:
        T[(p[0], p[1])] = func(f_map[p[0]], f_map[p[1]])
    return T



def pairwise_transition_prob(fitnesses: np.array, func: Callable, loop=False) -> np.array:
    """Generate a quick look up array with pairwise transition probabilities

    Args:
        fitnesses (np.array):   List of fitness values
        func (Callable):        A transition probability function, e.g. 
                                fixation probability function. The function has
                                to take in two values, fitness 1 and fitness 2.
        loop (bool):            Compute probability of remaining in the same 
                                state (Default=False). If false, the diagonal
                                of the transition matrix will be 0.

    Returns:
        np.array: 2D array where the entry i,j is the fixation probability of 
        i to j (not row or column normalized)

    """
    N = len(fitnesses)
    T = np.zeros(shape=(N, N))  # init zeros array

    if loop:  # get all pairs of indices
        pair_idx = ((i,j) for i in range(N) for j in range(N))
    else:  # excluding identical i and j
        pair_idx = ((i,j) for i in range(N) for j in range(N) if i!=j)

    # Compute transition probability for all pairs
    for p in pair_idx:
        T[p[0], p[1]] = func(fitnesses[p[0]], p[1])

    return T

def contains_downhill_steps(path, gp_map, ph_to_f):
    """Check if the genotype path contains downward/fitness descreasing 
    steps

    Args:
        path (list): list of genotypes
        gp_map (GenotypePhenotypeGraph): GenotypePhenotypeGraph instance.
        ph_to_f (dict): Dictionary that maps phenotypes to fitness.

    Returns:
        bool: True if at least one downhill step is found in path

    """
    downhill = False
    ph = gp_map.map(path[0])
    f_prev = ph_to_f[ph]
    for gt in path[1:]:
        f_new = ph_to_f[gp_map.map(gt)]
        if f_prev > f_new:
            downhill = True
            return downhill
        else:
            f_prev = f_new
    return downhill

def load_fl_file_to_dict(path):
    """Take the path to a fitness landscape file and turn it into a dict

    Args:
        path (str): Path to a fitness landscape file

    Returns:
        ph_to_f (dict): A dictionary that maps phenotype to fitness

    """
    ph_to_f = {}
    with open(path, "r") as f:
        for line_ in f:
            line = line_.strip().split(" ")
            ph_to_f[line[0]] = float(line[1])

    return ph_to_f

def genotype_path_to_fitness_path(paths: list, gp_map, ph_to_f, ignore_neutral=True):
    """Map genotype path to fitness paht

    Args:
        paths (list): List of genotype paths
        gp_map (GenotypePhenotypeGraph): GPGraph object
        ph_to_f (dict): Map from phenotypes to fitness
        ignore_neutral (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    fit_paths = []
    for path in paths:
        ph = gp_map.map(path[0])
        fitness = ph_to_f[ph]
        fit_path = [fitness]
        for i, gt in enumerate(path[1:]):
            ph = gp_map.map(gt)
            fitness = ph_to_f[ph]
            if ignore_neutral and fitness != fit_path[-1]:  # only add non-neutral strpds
                fit_path.append(fitness)
            else:  # add all steps
                fit_path.append(fitness)

        fit_paths.append(fit_path)
    return fit_paths

def nc_graph_to_directed_graph(nc_graph, ph_to_f):
    """Take an nc graph and turn it into a directed graph where two nodes
    are connected if they are neighboring NCs and the targed NC has higher
    fitness than the source.

    Args:
        nc_graph:   A networkx graph that has all neutral components and their
                    neighborhood relationships and a "phenotype" attribute for
                    each NC

        ph_to_f:    A phenotype to fitness mapping
    
    Return:
        nx.Graph

    """
    G = nc_graph.to_directed(as_view=False)
    for node in G.nodes:
        G.nodes[node]["fitness"] = ph_to_f[G.nodes[node]["phenotype"]]

    G = remove_nonadaptive_edges(G)
    return G

def read_genotype_paths_from_file(file: str,
                                  delimiter: str = " ",
                                  gt_type: Type = str,
                                  map_to: dict = None) -> list:
    """Read a file that contains paths of genotypes, one path per line, and
    turn it into a list of paths

    Args:
        file (str): Input file with one path per line.
        delimiter (str, optional): Delimiter between consecutive genotypes.
        gt_type (Type): Convert all genotype to a type
        map_to (Callable): Map every genotype to a value, e.g. its phenotype or
                        fitness and then add that value to the path instead
                        of the genotype

    Returns:
        list: List of paths. Example: [[<gt1>, <gt2>], [<gt10>, <gt4>]]. If map is not None:
              [[map[<gt1>], map[<gt2>]], [map[<gt10>], map[<gt4>]]] 
        
    """
    paths = []
    with open(file, "r") as f:
        # this if-else makes the code a bit bloated but creates less comp.
        # overheat because I do not need to check for map for every path or
        # genotype
        if map_to:
            for line in f:
                path_ = line.strip().split(delimiter)  # path to stripped list
                path  = []
                for gt in path_:
                    val = map_to[gt_type(gt)]  # convert to correct type and map
                    path.append(val)  # turn to string for easier writing to file
                paths.append(path)
        else:
            for line in f:
                path_ = line.strip().split(delimiter)
                path = []
                for gt in path_:
                    path.append(gt_type(gt))
                paths.append(path)

    return paths

def write_paths_to_file(paths: list, file: str, delimiter: str = " ") -> None:
    """Write a list of paths (list) to a file, one path per line

    Args:
        paths (list): List of paths, each of which is a list.
        file (str): Output file path.
        delimiter (str, optional): Delitier to use for writing paths to file.
        Defaults to " ".

    Returns:
        None

    """
    with open(file, "w") as f:
        for path in paths:
            # one path per line. Map every element to str first.
            f.write(delimiter.join(map(str, path)) + "\n")

