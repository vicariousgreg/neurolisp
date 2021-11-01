import numpy as np

class Activator:
    def __init__(self, N, lam, rho, label):
        self.N = N
        self.lam = lam
        self.rho = rho
        self.label = label

    def make_pattern(self):
        return self.saturate(np.random.uniform(self.lam-1, self.lam, (self.N,1)))

    def invert(self, p):
        return self.f(-self.g(p))

class tanh_activator(Activator):
    def __init__(self, pad, N, lam=.5):
        super().__init__(N, lam, (1. - pad) ** 2, "tanh")
        self.pad = pad

    def f(self, p):
        return np.tanh(p)

    def g(self, p):
        return np.arctanh(np.clip(p, self.pad - 1., 1 - self.pad))

    def saturate(self, p):
        return np.sign(p) * (1. - self.pad)

class heaviside_activator(Activator):
    def __init__(self, N, lam=.5):
        super().__init__(N, lam, 1., "heaviside")

    def f(self, p):
        return (p > 0).astype(np.float32)

    def g(self, p):
        return (-1.)**(p <= 0)

    def saturate(self, p):
        return (p > 0).astype(np.float32)

class sign_activator(Activator):
    def __init__(self, N, lam=.5):
        super().__init__(N, lam, 1., "sign")

    def f(self, p):
        return np.sign(p)

    def g(self, p):
        return (p != 0) * (-1.)**(p < 0)

    def saturate(self, p):
        return np.sign(p)
