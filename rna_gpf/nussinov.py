"""
Implementation of Nussinov's algorithm for RNA structure prediction
Author: Leander Goldbach
GitHub: lgolbach

"""
import numpy as np
from rna_folding.secondary_structure import SecondaryStructure
from rna_folding.base_pairing import BasePairing


class BasePairMatrixNussinov:
    """A matrix that hold the maximum number of possible base-pairs for a RNA sequence of length l, where position
    i+1, j+1 of the matrix holds the maximum number of base pairs for segment [i,j] of the RNA sequence.

    """
    def __init__(self, n: int, base_pairing: BasePairing):
        """Initialize a (L+1)x(L+1) matrix with zeros on diagonal and the diagonal below.

        Args:
            l (int): Sequence length.
            base_pairing (BasePairing): Instance of a BasePairing object.

        """
        self.base_pairing = base_pairing
        self._P = np.zeros((n+1, n+1), dtype=int)
        self._min_loop_size = None
        self._n = n

    @property
    def P(self):
        return self._P

    @property
    def min_loop_size(self):
        return self._min_loop_size

    @min_loop_size.setter
    def min_loop_size(self, value):
        raise AttributeError("Cannot change attribute min_loop_size directly. Can only be set via fill_matrix method."
                             "If matrix was already filled in, create new matrix with different min_loop_size")

    def fill_matrix(self, seq: str, min_loop_size: int = 1):
        """Main step of Nussinov's algorithm, i.e. filling the P matrix to find maximum base-pairing for all seqments
        subject only to minimum loop size constraint.

        Args:
            seq (string): The RNA sequence comprised of the letters A, U, G or C.
            min_loop_size (int): Minimum loop length. Default = 1

        Returns:
            None

        """
        self._min_loop_size = min_loop_size

        for k in range(1, self._n):  # loop over segment sizes
            for i in range(1, self._n - k + 1):  # loop over starting index of segment
                j = i + k
                j_unpaired = self._P[i, j-1]
                l_j_paired = [self._P[i, l-1] + self._P[l+1, j-1] + 1 for l in range(i, j-self._min_loop_size)
                              if self.base_pairing.pairs(seq[l-1], seq[j-1])]

                if l_j_paired:
                    self._P[i, j] = max(j_unpaired, *l_j_paired)
                else:
                    self._P[i, j] = j_unpaired


    def traceback(self, seq: str):
        """Find a base-pairs of an optimal secondary structure, i.e. a structure with maximum number of base-pairs

        Args:
            seq (string): The RNA sequence

        Returns:
            B (list): List of base pairs in tuple format

        Remarks:
            Only returns one structure. There may be more structures with the same number of base-pairs.

        """
        s = SecondaryStructure(sigma=[(1, self._n)], B=[])  # initiate first suboptimal structure

        while s.sigma:
            i, j = s.sigma.pop()
            if i >= j:  # ignore segments too small for a base-pair (produced when two neighboring sites pair or first)
                continue
            if self._P[i, j] == self._P[i, j-1]:
                s.sigma.append((i, j-1))
            else:
                for l in range(i, j):
                    if self.base_pairing.pairs(seq[l-1], seq[j-1]):
                        if self._P[i, j] == self._P[i, l-1] + self._P[l+1, j-1] + 1:
                            s.B.append((l, j))
                            s.sigma.extend([(i, l-1), (l+1, j-1)])
                            break
        return s

    def traceback_subopt(self, seq: str, d: int = 0, structures_max = np.inf):
        """Find all suboptimal structures within a certain number of base-pairs from the maximum according to the
        Wuchty1999 algorithm.

        Args:
            seq (string): The RNA sequence comprised of the letters A, U, G or C.
            d (int): allowed difference in number of base-pairs between optimal and suboptimal structures. Default d = 0
                         generates all possible optimal structures.

        Returns:
            B (list): List of base pairs in tuple format

        """
        s_init = SecondaryStructure(sigma=[(1, self._n)], B=[])  # initiate first suboptimal structure
        R = [s_init]  # initiate stack of suboptimal structures
        final_structures = []  # where we collect suboptimal structures
        p_max = self._P[1, -1]  # maximum possible number of base-pairs

        if p_max == 0:
            s_init.pop()
            final_structures.append(s_init)
            return final_structures

        while R:
            added_to_R = False  # track whether something has been put on R stack since popping s
            s = R.pop()
            if s.is_folded():
                final_structures.append(s)
                if len(final_structures) == structures_max:
                    break
                continue

            while not s.is_folded():
                i, j = s.pop()
                if j-i > self._min_loop_size:
                    s_ = SecondaryStructure(sigma=[(i, j-1), *s.sigma], B=s.B)
                    if s_.maximum_bp(self._P) >= p_max - d:
                        R.append(s_)
                        added_to_R = True
                    for l in range(i, j):
                        if self.base_pairing.pairs(seq[l - 1], seq[j - 1]) and j-l > self._min_loop_size:
                            s_ = SecondaryStructure(sigma=[(i, l-1), (l+1, j-1), *s.sigma], B=[*s.B, (l, j)])
                            if s_.maximum_bp(self._P) >= p_max - d:
                                R.append(s_)
                                added_to_R = True
            if not added_to_R:  # nothing has been put on stack since popping s
                R.append(s)  # continue with s next iteration (no infinite loop because each iteration we pop from s.sigma)
        return final_structures

