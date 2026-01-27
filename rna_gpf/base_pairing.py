"""Defines a class for base-pairings. Base-pairings are simple graphs with 
n nodes where n is the number of bases. Enumerating all possible simple
graphs without isomorphism is only possible by brute which is why all the
base-pairing here are hard-coded and taken from:
http://users.cecs.anu.edu.au/~bdm/data/graphs.html

"""

import numpy as np
from rna_folding.utils import canonical_adjacency_matrix


class BasePairing:
    def __init__(self, bases, graph_path, id) -> None:
        """Initialize base-pairing object

        Args:
            n (str): Defines the names of the bases and thus the number of 
                        bases, i.e. number of nodes of the base-pairing graph.
            id (int): Defines which of the possible base-pairing graphs to
                        choose. If -1 then canonical base-pairing is used
            graph_path (str): Path to folder containing the base-pairing 
                        graph's adjacency matrix files.


        """
        self.bases = bases
        self.graph_path = graph_path
        # map each base to its id (position in list) for quick look-up
        self.bases_to_id = dict(zip(self.bases, range(len(self.bases))))
        self.id = id
        self.n = len(bases)
        if self.id == -1:
            self.A = canonical_adjacency_matrix()
            if self.bases != "UCAG":
                raise(ValueError, f"If id is set to -1, canonical base-pairing is assumed and bases should be 'UCAG' not {self.bases}")
        else:
            self.A = self.get_adjacency_matrix(self.n, id)
        
    def get_adjacency_matrix(self, n, id):
        # file name format is graph{n}.adj
        file_path = self.graph_path + "graph" + str(n) + ".adj"
        with open(file_path, "r") as file:
            for line in file:  # loop over lines until we find right graph
                # Graph header format: Graph 3, order 4.
                if line.split(',')[0] == "Graph " + str(id):
                    adjacency_matrix = []
                    # loop over rows of graph adjacency matrix
                    for i in range(n):
                        row_str = next(file).strip()  # get row, e.g. "0101"
                        row_num = [int(num_str) for num_str in row_str]  # to int
                        adjacency_matrix.append(row_num)  # add row to adj. ma.
                    break

        A = np.array(adjacency_matrix)

        return A

    def pairs(self, A, B):
        """Returns True if A and B can pair according to the adjacency matrix 
        and False otherwise

        Args:
            A (str): The first base
            B (str): The second base
        
        Returns:
            (bool): True if bases can pair, False otherwise 
        
        """
        a = self.bases_to_id[A]
        b = self.bases_to_id[B]
        return self.A[a, b]

