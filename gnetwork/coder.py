import numpy as np
import itertools as it
from gnetwork.orthogonal_patterns import random_hadamard

class Coder(dict):
    def __init__(self, activator, ortho=None):
        self.activator = activator

        if ortho:
             had = random_hadamard(activator.N, activator.N)
             self.ortho_patterns = [
                 had[:,i].reshape((activator.N,1))
                 for i in range(activator.N)]
        else:
            self.ortho_patterns = None

    def encode(self, token, pattern=None):
        if token not in self:
            if pattern is None:
                if self.ortho_patterns:
                    pattern = self.ortho_patterns[len(self)]
                else:
                    pattern = self.activator.make_pattern()
            self[token] = pattern
            return pattern
        else:
            return self[token]

    def get_sign_distances(self, pattern):
        if pattern is not None and len(self):
            return tuple(
                (k, np.sum(np.sign(pattern) != np.sign(v)))
                for k,v in self.items())
        else:
            return ((None, 0.),)

    def get_distances(self, pattern):
        if pattern is not None and len(self):
            return tuple(
                (k, np.sum(np.abs(pattern - v)))
                for k,v in self.items())
        else:
            return ((None, 0.),)

    def get_similarities(self, pattern):
        if pattern is not None and len(self):
            N = self.activator.N
            if self.activator.label == "heaviside":
                return tuple(
                    (k, np.sum(np.equal(pattern, v)) / N)
                    for k,v in self.items())
            else:
                return tuple(
                    (k, np.sum(np.multiply(pattern, v)) / N)
                    for k,v in self.items())
        else:
            return ((None, 0.),)

    def decode(self, pattern):
        sims = self.get_similarities(pattern)
        return max(sims, key = lambda x: x[1])

        #dists = self.get_distances(pattern)
        #sym,dist = min(dists, key = lambda x: x[1])
        #return sym,1 - (dist / self.activator.N)


if __name__ == "__main__":
    
    N = 8
    PAD = 0.9
    
    from activator import *
    act = tanh_activator(PAD, N)

    c = Coder(act)
    v = c.encode("TEST")
    print(v.T)
    print(c.decode(v))
    print(c.encode("TEST").T)
    print(c.decode(act.make_pattern()))
