import numpy as np


class Buffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.terminals = []

    def clear(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.terminals = []

    def __len__(self):
        return len(self.states)


class Discretizer:
    def __init__(
        self,
        min_points,
        max_points,
        buckets,
        dimensions=None,
        ):

        self.min_points = np.array(min_points)
        self.max_points = np.array(max_points)
        self.buckets = np.array(buckets)

        self.range = self.max_points - self.min_points
        self.spacing = self.range / self.buckets

        self.dimensions = dimensions
        if dimensions:
            self.n_states = np.round(self.buckets).astype(int)
            self.row_n_states = [self.n_states[dim] for dim in self.dimensions[0]]
            self.col_n_states = [self.n_states[dim] for dim in self.dimensions[1]]

            self.N = np.prod(self.row_n_states)
            self.M = np.prod(self.col_n_states)

            self.row_offset = [int(np.prod(self.row_n_states[i + 1:])) for i in range(len(self.row_n_states))]
            self.col_offset = [int(np.prod(self.col_n_states[i + 1:])) for i in range(len(self.col_n_states))]

    def get_index(self, state):
        state = np.clip(state, a_min=self.min_points, a_max=self.max_points)
        scaling = (state - self.min_points) / self.range
        idx = np.round(scaling * (self.buckets - 1)).astype(int)

        if not self.dimensions:
            return idx

        row_idx = idx[:, self.dimensions[0]]
        row = np.sum(row_idx*self.row_offset, axis=1)

        col = None
        if self.dimensions[1]:
            col_idx = idx[:, self.dimensions[1]]
            col = np.sum(col_idx*self.col_offset, axis=1)

        return [row, col]
