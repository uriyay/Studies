"""
A more generic implementation of Cellular Automata with unlimited dimensions number
"""
import itertools
import numpy as np

class CellularAutomata:
    def __init__(self, d, n, states, init_state):
        self.d = d
        self.n = n
        self.init_state = init_state
        self._mat = self._init_matrix()
        self._gen_id = 0

    def iterate_matrix(self):
        return itertools.product(*[range(n) for _ in range(self.d)])

    def _init_matrix(self):
        matrix = np.zeros([self.n for _ in range(self.d)], dtype=int)
        matrix.fill(self.init_state)

    def get_neighborhood(self, pos):
        raise NotImplementedError()

    def transition_rule(self, cur_state, states):
        raise NotImplementedError()

    def next(self):
        new_mat = self._init_matrix()
        for pos in self.iterate_matrix():
            new_mat.__setitem__(pos,
                    self.transition_rule(
                        self._mat.__getitem__(pos),
                        self.get_neighborhood(pos)
                    )
            )
        self._gen_id += 1
        self._mat = new_mat

    def set_state(self, pos, state):
        self._mat.__setitem__(pos, state)
