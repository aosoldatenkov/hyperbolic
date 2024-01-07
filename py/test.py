import numpy as np
import math
import lattice as lt
import itertools as it
from circle import Circle, CircleArrangement

def hscalar(a, b):
    return 2 * a[0] * b[0] - np.inner(a, b)

def hsquare(a):
    return hscalar(a, a)

int_vectors = set()

def trace_ray(lat, hdir, steps, q=[-2], max_vectors=0):
    global int_vectors
    dim = lat.dimension()
    qmin = min(a for a in q)
    qmax = max(a for a in q)
    if qmax >= 0:
        return
    offset = [np.array(x) for x in it.product([-1, 0, 1], repeat=dim)]

    hor_h = np.zeros(dim)
    hor_h[1:] = hdir[1:]
    hor_l = lat.lcoord(hor_h)
    hor_lnorm = math.sqrt(np.inner(hor_l, hor_l))
    hor_h = hor_h / hor_lnorm
    hor_l = hor_l / hor_lnorm
                          
    vert_h = np.zeros(dim)
    vert_h[0] = 1
    vert_l = lat.lcoord(vert_h)
    vert_lnorm = math.sqrt(np.inner(vert_l, vert_l))
    vert_h = vert_h / vert_lnorm
    vert_l = vert_l / vert_lnorm
    vert_hnorm = 1 / vert_lnorm

    a = math.sqrt(abs(qmax / hsquare(hor_h)))
    hor_pos_h = hor_h * a
    hor_pos_l = hor_l * a

#    n_vectors = max_vectors
    for i in range(steps):
        pos_h = hor_pos_h
        pos_l = hor_pos_l
        a = np.inner(pos_h, pos_h)
        vmin = math.sqrt(max(0, qmin + a)) / vert_hnorm
        vmax = math.sqrt(max(0, qmax + a)) / vert_hnorm

        t = vmin
        pos = pos_l + t * vert_l
        while t <= vmax:
            ipos = np.array([round(a) for a in pos])
            for x in offset:
                new_vec = ipos + x
#                new_vec_tuple = tuple(new_vec)
#                if new_vec_tuple in int_vectors:
#                    continue
#                int_vectors = int_vectors | {new_vec_tuple}
                new_vec_q = lat.square(new_vec)
                if new_vec_q in q:
                    yield new_vec, new_vec_q
#                    n_vectors -= 1
#                    if n_vectors == 0:
#                        return
            pos += vert_l
            t += 1

        hor_pos_h += hor_h
        hor_pos_l += hor_l


def print_progress(percent, e="\r"):
    N = int(percent)
    print("Progress: " + "=" * N + f" {percent:.2f}%", end=e)


def scan(lat, ca, N, depth, q=[-2], qdisc=[-2], dlimit=0):
    global int_vectors
    if lat.dimension() != 4:
        return
    total = 4 * N * N
    n_vectors = 0
    print(f"Scan {N}, {depth}:")

    vectors_str = ""
    for i in range(4 * N):
        for j in range(N):
            print_progress(100 * (N * i + j) / total)
            v = [0,
                 -math.sin(j * math.pi / (2 * N)),
                 math.cos(j * math.pi / (2 * N)) * math.cos(i * math.pi / (2 * N)),
                 math.cos(j * math.pi / (2 * N)) * math.sin(i * math.pi / (2 * N))]
            w = lt.sproject(v[1:])
            if not ca.pt_is_visible(w[0], w[1]):
                continue
            n_circles = dlimit
            for int_v, qv in trace_ray(lat, np.array(v), depth, q, dlimit):
                n_vectors += 1
                w = lat.hcoord(int_v)
                if w[1] - w[0] != 0:
                    x = w[2] / (w[1] - w[0])
                    y = w[3] / (w[1] - w[0])
                    r = math.sqrt(w[1] ** 2 + w[2] ** 2 + w[3] ** 2 - w[0] ** 2) / abs(w[1] - w[0])
                    if qv in qdisc:
                        new_circ = Circle(x, y, r, disc=1, square=int(qv))
                    else:
                        new_circ = Circle(x, y, r, square=int(qv))
                    if ca.add_circle(new_circ):
                        #vectors_str += str(int_v) + f", q={qv}, r={r}\n"
                        n_circles -= 1
                        if n_circles == 0:
                            break

    print_progress(100, e="\n")
    print(f"{n_vectors} new vectors found")
    #print(vectors_str)

def scan_lattice1():
    dimension = 4
    p = 5
    A = np.array([[-2, 0, 0, 0],
                  [0, 2 * p, 0, 0],
                  [0, 0, -2 * p, 0],
                  [0, 0, 0, -2 * p]])
    lat = lt.HLattice(dimension, A)
    ca = CircleArrangement()
    int_vectors = set()
    scan(lat, ca, 10, 10, q=[-2, -4, -6, -8])
    #scan(lat, ca, 20, 100, q=[-2])
    #scan(lat, ca, 50, 100, q=[-2], dlimit=3)
    #scan(lat, ca, 150, 150, q=[-2], dlimit=3)
    #ca.dump("out.json")
    #ca.load("out.json")
    ca.tikz_out("tikzcode.tex")

def scan_lattice2():
    dimension = 4
    p = 3
    A = np.array([[-2, 0, 0, 0],
                  [0, -2 * p, 0, 0],
                  [0, 0, -2 * p * p, 0],
                  [0, 0, 0, 2 * p * p * p]])
    lat = lt.HLattice(dimension, A)
    ca = CircleArrangement()
    int_vectors = set()
    scan(lat, ca, 20, 10, q=[-2, -4])
    scan(lat, ca, 50, 100, q=[-2])
    scan(lat, ca, 150, 150, q=[-2], dlimit=1)
    #scan(lat, ca, 150, 150, q=[-2], dlimit=3)
    #ca.dump("out.json")
    #ca.load("out.json")
    ca.tikz_out("tikzcode.tex")

def scan_lattice3():
    dimension = 4
    A = np.array([[2 * 3, 0, 0, 0],
                  [0, -2, 0, 0],
                  [0, 0, -2 * 7 * 9, 0],
                  [0, 0, 0, -2 * 7 * 9]])
    lat = lt.HLattice(dimension, A)
    ca = CircleArrangement()
    int_vectors = set()
    scan(lat, ca, 20, 10, q=[-2, -4])
    scan(lat, ca, 50, 100, q=[-2])
    scan(lat, ca, 140, 200, q=[-2], dlimit=1)
    #scan(lat, ca, 150, 150, q=[-2], dlimit=3)
    ca.dump("out.json")
    #ca.load("out.json")
    ca.tikz_out("tikzcode.tex")

def scan_lattice4():
    dimension = 4
    p = 5
    A = np.array([[-2, p, 0, 0],
                  [p, 0, 0, 0],
                  [0, 0, -2 * p, p],
                  [0, 0, p, -2 * p]])
    lat = lt.HLattice(dimension, A)
    ca = CircleArrangement()
    int_vectors = set()
    #scan(lat, ca, 20, 10, q=[-2])
    #scan(lat, ca, 50, 100, q=[-2])
    #scan(lat, ca, 140, 200, q=[-2], dlimit=1)
    ca.load("out.json")
    scan(lat, ca, 150, 1000, q=[-2], dlimit=1)
    ca.dump("out.json")
    #ca.load("out.json")
    ca.tikz_out("tikzcode.tex")


def main():
    scan_lattice4()
    
if __name__ == '__main__':
    main()
