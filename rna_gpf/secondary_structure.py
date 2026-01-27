import numpy as np


class SecondaryStructure:
    """Implements data structure for backtracking an RNA secondary structure. Holds identified base-pairs and sequence
    segments that have yet to be explored by backtracking.

    """
    def __init__(self, sigma: list, B: list):
        """Initialize a stack of segments (sigma) and a list of base-pairs

        Args:
            sigma (list): Stack of segments (tuples of size 2)
            B (list): Stack of base-pairs (tuples of size 2)
        """
        self._sigma = []
        self.__add_segments(sigma)
        self._B = []
        self.__add_base_pairs(B)

    def maximum_bp(self, P: np.ndarray) -> int:
        """Determine the maximum possible number of base pairs this structure can have

        Args:
            P (np.ndarray): Matrix containing the maximum number of base-pairs possible for each segment [i, j] in P_i,j

        Returns:
            max_bps (int): Maximum number of base-pairs possible for this structure.

        """
        determined_bps = len(self._B)
        potential_bps = sum([P[seg] for seg in self._sigma])
        max_bps = determined_bps + potential_bps
        return max_bps

    @property
    def B(self):
        return self._B

    def __add_base_pairs(self, bp: list):
        self._B.extend(bp)

    @property
    def sigma(self):
        return self._sigma

    def __add_segments(self, segments: list):
        for s in segments:
            if s[0] < s[1]:  # ignore segments too small for a base-pair (produced when two neighboring sites pair or first)
                self._sigma.append(s)

    def pop(self):
        """Return segment last added to the stack

        Returns:
            (tuple): Interval (i, j), with i, j in [0, L] (L=seq. length) and i<j, which defines a sequence segment.

        """
        if not self.is_folded():
            return self._sigma.pop()

    def is_folded(self):
        if self._sigma:
            return False
        else:
            return True

    def update(self, segments: list, base_pairs: list):
        """Updates segment stack and base pairs simultaneously to avoid unsynched access to either of the two

        Args:
            segments (list): list of tuples defining segments to be added to the stack
            base_pairs (list): list of tuples defining new base pairs to be added

        Returns:
            None

        """
        self.__add_segments(segments)
        self.__add_base_pairs(base_pairs)
