import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import lattice as lt
from circle import Circle, CircleArrangement
import time


def test_iterable():
    n_points = 5000
    sig = [0, 0]
    ptx, pty = [0 for i in range(n_points)], [0 for i in range(n_points)]
    lat = (x for x in lt.int_seq(2, signs=sig, nonzero=True))
    for i in range(1000):
        v = next(lat)
        ptx.append(v[0])
        pty.append(v[1])
    plotsize = 100
    fig, ax = plt.subplots()
    scat = ax.scatter(ptx, pty, s=5)
    ax.grid(False)
    fig.tight_layout()
    pt = [(0, 0) for i in range(n_points)]

    def update(frame_number):
        global lat, pt
        index = frame_number % n_points
        if index == 0:
            lat = (x for x in lt.int_seq(2, signs=sig))
            pt = [(0, 0) for i in range(n_points)]
        v = next(lat)
        pt[index] = (v[0], v[1])
        scat.set_offsets(pt)

    animation = FuncAnimation(fig, update, interval=20, save_count=n_points)
    plt.show()


def iterable_speed():
    n_max = 100000
    n = 0
    last_time = time.time() 
    for x in lt.int_seq(3):
        n += 1
        cur_time = time.time()
        if n >= n_max and cur_time - last_time >= 1:
            print(f"{n / (cur_time - last_time):5f} pts/sec", end="\r")
            last_time = cur_time
            n = 0


def test_lattice_listing():
    dimension = 4
    p = 5
    A = np.array([[-2, 0, 0, 0],
                  [0, 2 * p, 0, 0],
                  [0, 0, -2 * p, 0],
                  [0, 0, 0, -2 * p]])
    lat = lt.HLattice(dimension, A)
    n_max = 100
    n = 0
    last_time = time.time() 
    for x in lat.list_vectors(q=[-2]):
        n += 1
        cur_time = time.time()
        if n >= n_max and cur_time - last_time >= 1:
            print(f"{n / (cur_time - last_time):5f} vectors/sec", end="\r")
            last_time = cur_time
            n = 0


def test_circle():
    c1 = Circle(0, 0, 1)
    c2 = Circle(1, 0, 1)
    c3 = Circle(0.5, 0, 0.5)
    c4 = Circle(2, 1, 10)
    c5 = Circle(0, 0.5, 1)

    assert c1.contains_pt(0.1, 0.7)
    assert not c1.contains_circ(c2)
    assert c1.contains_circ(c3)
    assert c2.intersects_circ(c1)
    assert c2.intersects_circ(c3)
    assert not c3.contains_circ(c4)
    assert c4.contains_circ(c3)
    assert not c4.intersects_circ(c5)
    assert not c5.contains_pt(0.0001, -0.5)
    assert c5.intersects_circ(c1)
    
    print("Circle test passed")


if __name__ == '__main__':
    #iterable_speed()
    #test_circle()
    #test_iterable()
    test_lattice_listing()
    #ca = CircleArrangement()
    #ca.load_xyr("circles.txt")
    #ca.load("out.json")
    #ca.dump("circles.json")
