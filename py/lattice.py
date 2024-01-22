import numpy as np
import math
import itertools as it
from circle import Circle, CircleArrangement


def hscalar(a, b):
    return 2 * a[0] * b[0] - np.inner(a, b)


def hsquare(a):
    return hscalar(a, a)


def sproject(v):
    return [x / (1 - v[0]) for x in v[1:]]


def sunproject(dim, w):
    d = np.inner(w, w)
    v = [(d - 1) / (d + 1)]
    v.append(2 * x / (d + 1) for x in w)
    return v


def print_progress(percent, e="\r"):
    N = int(percent)
    print("Progress: " + "=" * N + f" {percent:.2f}%", end=e)


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


def int_seq(dim, signs=None, nonzero=False, length=-1):
    if signs == None:
        signs = [0 for i in range(dim)]
    v = np.zeros(dim, 'int')
    if nonzero:
        head = 0
        width = 1
        v[0] = 1
    else:
        head = dim - 1
        width = 0
    n = length
    while n != 0:
        out = [(v[i] >> 1) - (v[i] & 1) * (v[i] | 1) for i in range(dim)]
        if all(out[j] * signs[j] >= 0 for j in range(dim)):
            yield out
            n -= 1
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


class HLattice:

    def __init__(self, dimension, prod_matrix):
        self.dim = dimension
        self.A = prod_matrix.copy()
        eval, evec = np.linalg.eigh(self.A)
        self.eval = np.flip(eval)
        self.evec = np.fliplr(evec)
        eval = np.sqrt(np.fabs(self.eval))
        evalm = np.diag(eval)
        evalm = np.linalg.inv(evalm.reshape((dimension, dimension)))
        self.hbasis = np.matmul(self.evec, evalm)
        self.lbasis = np.linalg.inv(self.hbasis)

    def dimension(self):
        return self.dim

    def get_eigh(self):
        return self.eval, self.evec

    def prod(self, u, v):
        return np.dot(u, np.matmul(self.A, v))

    def square(self, u):
        return self.prod(u, u)

    def hcoord(self, u):
        return np.matmul(self.lbasis, u)

    def lcoord(self, u):
        return np.matmul(self.hbasis, u)

    def trace_ray(self, hdir, steps, q=[-2], max_vectors=0):
        qmin = min(a for a in q)
        qmax = max(a for a in q)
        if qmax >= 0:
            return
        offset = [np.array(x) for x in it.product([-1, 0, 1], repeat=self.dim)]
        scale = 0.5
        eval, evec = self.get_eigh()
        lmax = math.sqrt(abs(eval[3]))
    
        hor_h = np.zeros(self.dim)
        hor_h[1:] = hdir[1:]
        hor_l = self.lcoord(hor_h)
        hor_lnorm = math.sqrt(np.inner(hor_l, hor_l))
        hor_h = scale * hor_h / hor_lnorm
        hor_l = scale * hor_l / hor_lnorm
                          
        vert_h = np.zeros(self.dim)
        vert_h[0] = 1
        vert_l = self.lcoord(vert_h)
        vert_lnorm = math.sqrt(np.inner(vert_l, vert_l))
        vert_h = scale * vert_h / vert_lnorm
        vert_l = scale * vert_l / vert_lnorm
        vert_hnorm = scale / vert_lnorm

        a = math.sqrt(abs(qmax / hsquare(hor_h)))
        hor_pos_h = hor_h * a
        hor_pos_l = hor_l * a

        for i in range(steps):
            pos_h = hor_pos_h
            pos_l = hor_pos_l
            a = np.inner(pos_h, pos_h)
            hnorm = math.sqrt(a)

            hmax = (hnorm + lmax) ** 2
            hmin = max(0, hnorm - lmax) ** 2
            vmax = math.ceil(math.sqrt((qmax + hmax) / vert_hnorm)) + 1
            vmin = math.floor(math.sqrt(max(0, qmin + hmin) / vert_hnorm)) - 1
            vmin = max(0, vmin)        

            t = vmin
            pos = pos_l + t * vert_l
            while t <= vmax:
                ipos = np.array([round(a) for a in pos])
                for x in offset:
                    new_vec = ipos + x
                    new_vec_q = self.square(new_vec)
                    if new_vec_q in q:
                        yield new_vec, new_vec_q
                pos += vert_l
                t += 1

            hor_pos_h += hor_h
            hor_pos_l += hor_l


    def scan(self, ca, N, depth, q=[-2], qdisc=[-2], dlimit=0):
        if self.dimension() != 4:
            return
        total = 4 * N * N
        n_vectors = 0
        print(f"Scan {N}, {depth}:")
        for i in range(4 * N):
            for j in range(N):
                print_progress(100 * (N * i + j) / total)
                v = [0,
                     -math.sin(j * math.pi / (2 * N)),
                     math.cos(j * math.pi / (2 * N)) * math.cos(i * math.pi / (2 * N)),
                     math.cos(j * math.pi / (2 * N)) * math.sin(i * math.pi / (2 * N))]
                w = sproject(v[1:])
                if not ca.pt_is_visible(w[0], w[1]):
                    continue
                n_circles = dlimit
                for int_v, qv in self.trace_ray(np.array(v), depth, q, dlimit):
                    n_vectors += 1
                    w = self.hcoord(int_v)
                    if w[1] - w[0] != 0:
                        x = w[2] / (w[1] - w[0])
                        y = w[3] / (w[1] - w[0])
                        r = math.sqrt(w[1] ** 2 + w[2] ** 2 + w[3] ** 2 - w[0] ** 2) / abs(w[1] - w[0])
                        if qv in qdisc:
                            new_circ = Circle(x, y, r, disc=1, square=int(qv))
                        else:
                            new_circ = Circle(x, y, r, square=int(qv))
                        if ca.add_circle(new_circ):
                            n_circles -= 1
                            if n_circles == 0:
                                break
        print_progress(100, e="\n")
        print(f"{n_vectors} new vectors found")


    def scan_plus(self, ca, limit, q=[-2], qdisc=[-2], max_circles=-1):
        if self.dim != 4:
            return
        eval, evec = self.get_eigh()
        scale = 0.5
        evec = evec * scale
        lpos = eval[0] * scale * scale
        lmax = math.sqrt(abs(eval[3]))
        #evec[:, 1] = -evec[:, 1]
        qmin = min(a for a in q)
        qmax = max(a for a in q)
        nvec = 0
        for u in int_seq(3, signs=[0, 0, 0], length=limit):
            nvec += 1
            print(f"{ca.get_number()} circles found", end="\r")
            h = np.matmul(evec[:, 1:], u)
            if u[0] == u[1] == u[2] == 0:
                continue
            hnorm = math.sqrt(abs(self.square(h)))
            if hnorm + lmax < 0:
                continue
            hmax = (hnorm + lmax) ** 2
            hmin = max(0, hnorm - lmax) ** 2
            tmax = math.ceil(math.sqrt((qmax + hmax) / lpos)) + 1
            tmin = math.floor(math.sqrt(max(0, qmin + hmin) / lpos)) - 1
            tmin = max(0, tmin)
            for t in range(tmin, tmax + 1):
                v = h + t * evec[:, 0]
                vint = np.array([round(a) for a in v])
                qv = self.square(vint)
                if qv in q:
                    w = self.hcoord(vint)
                    if w[0] <= Circle.epsilon:
                        continue
                    if abs(w[1] - w[0]) >= Circle.epsilon:
                        x = w[2] / (w[1] - w[0])
                        y = w[3] / (w[1] - w[0])
                        r = math.sqrt(w[1] ** 2 + w[2] ** 2 + w[3] ** 2 - w[0] ** 2) / abs(w[1] - w[0])
                        new_circ = Circle(x, y, r, v=[int(x) for x in vint], square=int(qv))
                        if qv in qdisc:
                            if w[1] < w[0]:
                                new_circ.set_attr(disc=1)
                            else:
                                new_circ.set_attr(hole=1)
                    else:
                        new_circ = Circle(w[2], w[3], w[0], line=1, v=[int(x) for x in vint], square=int(qv))
                        if qv in qdisc:
                            new_circ.set_attr(disc=1)
                    ca.add_circle(new_circ)
                    if max_circles == ca.get_number():
                        break
        print(f"{ca.get_number()} circles found")
            

    def list_vectors(self, q: list, limit: int = -1) -> np.array:
        eval, evec = self.get_eigh()
        scale = 1 / (self.dim ** 0.5) - Circle.epsilon
        evec = evec * scale
        lpos = eval[0] * scale * scale
        lmax = math.sqrt(abs(eval[self.dim - 1]))
        qmin = min(a for a in q)
        qmax = max(a for a in q)
        for u in int_seq(self.dim - 1, nonzero=True, length=limit):
            h = np.matmul(evec[:, 1:], u)
            hnorm = math.sqrt(abs(self.square(h)))
            if hnorm + lmax < 0:
                continue
            hmax = (hnorm + lmax) ** 2
            hmin = max(0, hnorm - lmax) ** 2
            tmax = math.ceil(math.sqrt((qmax + hmax) / lpos)) + 1
            tmin = math.floor(math.sqrt(max(0, qmin + hmin) / lpos)) - 1
            tmin = max(0, tmin)
            for t in range(tmin, tmax + 1):
                v = h + t * evec[:, 0]
                vint = np.array([round(a) for a in v])
                qv = self.square(vint)
                if qv in q:
                    yield vint
        
