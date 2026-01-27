import numpy as np
from itertools import product
from scipy.stats.mstats import gmean


def count_gt_per_ph_and_ph_per_gt(gp_map_file, sep=" "):
    """Open a gp_map file and count phenotypes per genotype and vice versa

    Args:
        gp_map_file (str):  gp_map file in format: 
                            <ph1> <gtX> <gtY> ...
                            <ph2> <gtY> <gtX> <gtZ>...
        sep (str):          Column seperator. Defaults to " ".

    Returns:
        gt_per_ph (dict):   Maps genotypes to number of phenotypes they map to
        ph_per_gt (dict):   Maps phenotypes to num. of genot. that map to them

    """
    gt_per_ph = {}
    ph_per_gt = {}

    with open(gp_map_file, "r") as file:
        for line_ in file:  # at start of each loop the garbage collector should delete previous line from memory
            line = line_.strip().split(sep)
            gt_per_ph[line[0]] = len(line)-1
            for gt in line[1:]:
                if gt not in ph_per_gt:
                    ph_per_gt[gt] = 1
                else:
                    ph_per_gt[gt] += 1

    return gt_per_ph, ph_per_gt


def pairwise_consensus_matrix(phenotypes, pg_map, ref_gp_map):
    """Create a pairwise consensus matrix that counts how often any phenotype i
    is ranked above any other phenotype j

    Args:
        phenotypes (list):  List of phenotypes to consider
        ph_map (dict):      Maps phenotypes to list of genotypes that map to 
                            it. One-to-many gp map.
        ref_gp_map (dict):  Reference one-to-one map that maps genotype to the
                            highest ranked phenotype.

    Returns:
        A (np.array):       #(phenotype) x #(phenotypes) matrix where entry 
                            A(i,j) counts how often phenotype i has been the
                            highest ranked phenotype when both i and j appeared
                            in the same suboptimal set.

    """
    A = np.zeros(shape=(len(phenotypes), len(phenotypes)))  # initiate matrix
    # Loop over upper triangle of matrix
    for i in range(len(phenotypes)):
        for j in range(i+1,len(phenotypes)):
            ph_i = phenotypes[i]
            ph_j = phenotypes[j]
            # get genotypes that map to both phenos
            gt_intersect = set(pg_map[ph_i]).intersection(pg_map[ph_j])
            for gt in gt_intersect:  # loop over gt
                # phenotype i is ranked above j
                if ref_gp_map[gt][0] == ph_i:  
                    A[i, j] += 1
                # phenotype j is ranked above i
                elif ref_gp_map[gt][0] == ph_j:
                    A[j, i] += 1
    return A


def infer_bradley_terry_scores(pairwise_rankings, max_iter=10**3, conv_crit=10**-7):
    p = np.ones(pairwise_rankings.shape[0])  # initialize probabilities to 1

    for n in range(max_iter):
        old_p = p
        for i in range(len(p)):  # update each value once
            denom = p + p[i]
            p[i] = np.sum((p * pairwise_rankings[i]) / denom) / np.sum(pairwise_rankings.T[i] / denom)

        p = p/gmean(p)

        if np.sum(np.abs((p - old_p))) < conv_crit:
            print(f"converged after {n} steps")
            return p
        
    raise AssertionError(f"Not converged after {n} steps. Error: {np.sum(np.abs((p - old_p)))}")

def get_peaks(nc_graph, ph_to_f, local_only=False):
    """Count number of peaks in an neutral component graph given a 
    phenotype to fitness mapping"

    Args:
        nc_graph (nx.Graph):    A networkx graph. Nodes have to have a 
                                "phenotype" property.
        ph_to_f (dict):         Mapping from phenotypes to fitness (float). 
    
    Returns:
        (peaks_nc, peaks_f):    (NC ids of peaks (list), 
                                fitness of peaks [list])

    
    """
    peaks_nc = []
    peaks_f = []
    for nc in nc_graph.nodes:
        neighbors = nc_graph.neighbors(nc)
        nc_ph = nc_graph.nodes[nc]["phenotype"]
        nc_f = ph_to_f[nc_ph]
        peak = True
        for ne in neighbors:
            ne_ph = nc_graph.nodes[ne]["phenotype"]
            ne_f = ph_to_f[ne_ph]
            if ne_f >= nc_f:
                peak = False
        if peak:
            peaks_nc.append(nc)
            peaks_f.append(nc_f)

    return peaks_nc, peaks_f
