import numpy as np
import math
import lattice as lt
import itertools as it
from circle import Circle, CircleArrangement

def scan_lattice1():
    dimension = 4
    p = 5
    A = np.array([[-2, 0, 0, 0],
                  [0, 2 * p, 0, 0],
                  [0, 0, -2 * p, 0],
                  [0, 0, 0, -2 * p]])
    lat = lt.HLattice(dimension, A)
    ca = CircleArrangement()
    lat.scan_plus(ca, 100000, q=[-2], qdisc=[-2])
    #scan(lat, ca, 20, 100, q=[-2])
    #scan(lat, ca, 50, 100, q=[-2], dlimit=3)
    #scan(lat, ca, 150, 150, q=[-2], dlimit=3)
    ca.dump("out.json")
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
    lat.scan_plus(ca, -1, q=[-2], qdisc=[-2])
    #scan(lat, ca, 20, 10, q=[-2, -4])
    #scan(lat, ca, 50, 100, q=[-2])
    #scan(lat, ca, 150, 150, q=[-2], dlimit=1)
    #scan(lat, ca, 150, 150, q=[-2], dlimit=3)
    #ca.dump("out.json")
    #ca.load("out.json")
    #ca.tikz_out("tikzcode.tex")

def scan_lattice3():
    dimension = 4
    A = np.array([[2 * 3, 0, 0, 0],
                  [0, -2, 0, 0],
                  [0, 0, -2 * 7 * 9, 0],
                  [0, 0, 0, -2 * 7 * 9]])
    lat = lt.HLattice(dimension, A)
    ca = CircleArrangement()
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
    #ca.load("out.json")
    lat.scan_plus(ca, 1000000, q=[-2])
    #scan(lat, ca, 20, 10, q=[-2])
    #scan(lat, ca, 50, 100, q=[-2])
    #scan(lat, ca, 140, 200, q=[-2], dlimit=1)
    #ca.load("out.json")
    #lat.scan(ca, 150, 1000, q=[-2], dlimit=1)
    ca.dump("out.json")
    #ca.load("out.json")
    ca.tikz_out("tikzcode.tex")

def scan_lattice5():
    dimension = 4
    p = 5
    A = np.array([[-1, 2, 0, 0],
                  [2, -1, 1, 0],
                  [0, 1, -2, 1],
                  [0, 0, 1, -2]])
    lat = lt.HLattice(dimension, A)
    ca = CircleArrangement()
    ca.load("out.json")
    lat.scan_plus(ca, 500000, q=[-1], qdisc=[-1])
    #lat.scan(ca, 20, 10, q=[-1], qdisc=[-1])
    #lat.scan(ca, 50, 100, q=[-1], qdisc=[-1])
    #scan(lat, ca, 140, 200, q=[-1], qdisc=[-1], dlimit=1)
    #ca.load("out.json")
    #ca.load("backup.json")
    #lat.scan(ca, 150, 200, q=[-1], qdisc=[-1])
    ca.dump("out.json")
    #ca.load("backup.json")
    ca.tikz_out("tikzcode.tex", use_colors=True)

def scan_lattice6():
    dimension = 4
    p = 5
    A = np.array([[-1, 2, 0, 0],
                  [2, 0, 0, 0],
                  [0, 0, -2, 0],
                  [0, 0, 0, -2]])
    lat = lt.HLattice(dimension, A)
    ca = CircleArrangement()
    ca.load("out.json")
    lat.scan_plus(ca, 100000, q=[-1], qdisc=[-1])
    #lat.scan(ca, 20, 10, q=[-1], qdisc=[-1])
    #lat.scan(ca, 50, 100, q=[-1], qdisc=[-1])
    #scan(lat, ca, 140, 200, q=[-1], qdisc=[-1], dlimit=1)
    #ca.load("out.json")
    #ca.load("backup.json")
    #lat.scan(ca, 150, 200, q=[-1], qdisc=[-1])
    ca.dump("out.json")
    #ca.load("backup.json")
    ca.tikz_out("tikzcode.tex", use_colors=True)

def scan_latticeA():
    dimension = 4
    A = np.array([[-1, 1, 1, 1],
                  [1, -1, 1, 1],
                  [1, 1, -1, 1],
                  [1, 1, 1, -1]])
    lat = lt.HLattice(dimension, A)
    #print(lat.lbasis @ lat.evec[:, 0])
    #print(lat.eval)
    eval = np.sqrt(np.fabs(lat.eval))
    evec = np.array([[1, 0, 1,  1],
                     [1, 0, 1, -1],
                     [1, 1, -1,  0],
                     [1, -1,-1,  0]])
    evec2 = evec.T @ evec
    evec = evec @ np.sqrt(np.linalg.inv(evec2))
    lat.evec = evec
    evalm = np.diag(eval)
    evalm = np.linalg.inv(evalm)
    lat.hbasis = np.matmul(evec, evalm)
    lat.lbasis = np.linalg.inv(lat.hbasis)
    ca = CircleArrangement()
    lat.scan_plus(ca, 1000000, q=[-1], qdisc=[-1])
    #lat.scan(ca, 20, 10, q=[-1], qdisc=[-1])
    #lat.scan(ca, 50, 50, q=[-1], qdisc=[-1])
    #scan(lat, ca, 140, 200, q=[-1], qdisc=[-1], dlimit=1)
    #ca.load("out.json")
    #ca.load("backup.json")
    #lat.scan(ca, 150, 200, q=[-1], qdisc=[-1])
    ca.dump("out.json")
    #ca.load("backup.json")
    ca.tikz_out("tikzcode.tex")

def scan_latticeB():
    dimension = 4
    A = np.array([[0, 2, 2, 1],
                  [2, 0, 2, 0],
                  [2, 2, 0, 0],
                  [1, 0, 0, -2]])
    lat = lt.HLattice(dimension, A)
    ca = CircleArrangement()
    lat.scan_plus(ca, 1000000, q=[-2], qdisc=[-2])
    #lat.scan(ca, 20, 10, q=[-1], qdisc=[-1])
    #lat.scan(ca, 50, 50, q=[-1], qdisc=[-1])
    #scan(lat, ca, 140, 200, q=[-1], qdisc=[-1], dlimit=1)
    #ca.load("out.json")
    #ca.load("backup.json")
    #lat.scan(ca, 150, 200, q=[-1], qdisc=[-1])
    #ca.dump("out.json")
    #ca.load("backup.json")
    ca.tikz_out("tikzcode.tex")

def scan_latticeC():
    dimension = 4
    A = np.array([[2, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, -2, 0],
                  [0, 0, 0, -2]])
    lat = lt.HLattice(dimension, A)
    ca = CircleArrangement()
    #lat.scan_plus(ca, 1000000, q=[-1], qdisc=[-1])
    #ca.dump("out.json")
    ca.load("out.json")
    ca.tikz_out("tikzcode.tex")



def main():
    scan_lattice5()
    #ca = CircleArrangement()
    #ca.load("backup.json")
    #ca.load("out.json")
    #ca.tikz_out("tikzcode.tex", max_circles=18, use_colors=True)
    #ca.dump("out.json")
    
    
if __name__ == '__main__':
    main()
