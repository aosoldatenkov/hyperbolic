import numpy as np
import math
import itertools as it

def integers(signed=True, length=-1):
    n = length
    if signed:
        for i in it.count(0, 1):
            yield (i >> 1) - (i & 1) * (i | 1)
            n -= 1
            if n == 0:
                break
    else:
        for i in it.count(0, 1):
            yield i
            n -= 1
            if n == 0:
                break

def int_seq(dim, signed=None, length=-1):
    if signed == None:
        signed = [1 for i in range(dim)]
    v = np.zeros(dim, 'int64')
    head = dim - 1
    width = 0
    n = length
    while n != 0:
        yield [(v[i] >> signed[i]) - (v[i] & signed[i]) * (v[i] | 1)
               for i in range(dim)]
        if head == dim - 1:
            width += 1
            v[head] = 0
            v[0] = width
            head = 0
        else:
            v[head + 1] += 1
            w = v[head]
            v[head] = 0
            v[0] = w - 1 if w > 0 else 0
            head = 0 if v[0] > 0 else head + 1
        n -= 1
        
def sproject(v):
    return [x / (1 - v[0]) for x in v[1:]]

def sunproject(dim, w):
    d = np.inner(w, w)
    v = [(d - 1) / (d + 1)]
    v.append(2 * x / (d + 1) for x in w)
    return v


class LatticeEnum:

    def __init__(self, dim):
        self.dimension = dim
        self.masksize = 2 ** self.dimension
        self.reset()

    def reset(self):
        self.v = np.zeros(self.dimension, 'int32')
        self.head = self.dimension - 1
        self.width = 0
        self.bitmask = 0

    def __iter__(self):
        self.reset()
        return self

    def iterate(self):
        if self.bitmask < self.masksize:
            b = self.bitmask % self.masksize
            b = b ^ (b + 1)
            for i in range(self.dimension):
                if b % 2 > 0:
                    self.v[i] = -1 - self.v[i]
                b = b // 2
            self.bitmask += 1
            if self.bitmask < self.masksize:
                return
            self.bitmask = 0
        if self.head == self.dimension - 1:
            self.width += 1
            self.v[self.head] = 0
            self.v[0] = self.width
            self.head = 0
        else:
            self.v[self.head + 1] += 1
            w = self.v[self.head]
            self.v[self.head] = 0
            self.v[0] = w - 1 if w > 0 else 0
            self.head = 0 if self.v[0] > 0 else self.head + 1

    def __next__(self):
        oldv = self.v.copy()
        self.iterate()
        return oldv


class HLattice:

    def __init__(self, dimension, prod_matrix):
        self.dim = dimension
        self.A = prod_matrix.copy()
        eval, evec = np.linalg.eigh(self.A)
        eval = np.sqrt(np.fabs(np.flip(eval)))
        evalm = np.diag(eval)
        evalm = np.linalg.inv(evalm.reshape((dimension, dimension)))
        self.evec = np.matmul(np.fliplr(evec), evalm)
        self.basis = np.linalg.inv(self.evec)

    def dimension(self):
        return self.dim

    def get_basis(self):
        return self.basis

    def prod(self, u, v):
        return np.dot(u, np.matmul(self.A, v))

    def square(self, u):
        return self.prod(u, u)

    def hcoord(self, u):
        return np.matmul(self.basis, u)

    def lcoord(self, u):
        return np.matmul(self.evec, u)
