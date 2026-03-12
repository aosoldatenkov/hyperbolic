import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Set, Iterator

def int_seq(dim: int, signs: Optional[List[int]] = None, nonzero: bool = False, length: int = -1) -> Iterator[List[int]]:
    """Lists all integer sequences of length dim, with given restrictions on signs"""
    if signs is None:
        signs = [0] * dim
    
    v = [0] * dim
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
        
        # Verify sign constraints
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


@dataclass
class BinBasis:
    e: Tuple[int, int]
    f: Tuple[int, int]
    a: int
    b: int
    h: int

    def flip(self) -> 'BinBasis':
        return BinBasis(self.e, (-self.f[0], -self.f[1]), self.a, self.b, -self.h)

    def swap(self) -> 'BinBasis':
        return BinBasis(self.f, self.e, self.b, self.a, self.h)

    def negate(self) -> 'BinBasis':
        return BinBasis(self.e, self.f, -self.a, -self.b, -self.h)

    def copy(self) -> 'BinBasis':
        return BinBasis(self.e, self.f, self.a, self.b, self.h)

    def sum(self) -> Tuple[int, int]:
        return (self.e[0] + self.f[0], self.e[1] + self.f[1])


class BinLattice:
    def __init__(self, a: int, b: int, h: int):
        self.a = a
        self.b = b
        self.h = h
        self.disc = a * b - h * h  # Determinant of the form
        
        self.zero: List[Tuple[int, int]] = []   # List of all primitive vectors representing zero, if any such vectors exist
        self.river: List[BinBasis] = []         # List of bases with vectors of opposite signs, forming the 'river' of Conway's topograph
        self.pos: List[BinBasis] = []           # List of bases that generate the positive direction of the topograph
        self.neg: List[BinBasis] = []           # List of bases that generate the negative direction of the topograph
        self.shift: Optional[Tuple[int, int, int, int]] = None # An automorphism of the lattice that gives the translation along the 'river', if it exists
        self.can: Optional[BinBasis] = None     # Canonical basis of the lattice
        
        self._initialize_lattice()

    def _initialize_lattice(self) -> None:
        """Routes initialization based on the discriminant and signature."""
        if self.a == 0 and self.b == 0 and self.h == 0:
            self.signature = (0, 0)
        elif self.disc == 0:
            self._init_degenerate()
        elif self.disc > 0:
            self._init_definite()
        else:
            self._init_indefinite()

    def _init_degenerate(self) -> None:
        self.can = self.descend(BinBasis((1, 0), (0, 1), self.a, self.b, self.h))
        self.zero = [self.can.e, (-self.can.e[0], -self.can.e[1])]
        e, f = self.can.f, self.can.sum()
        
        if self.a > 0 or self.b > 0:
            self.signature = (1, 0)
            self.pos = [BinBasis(e, f, self.sqr(e), self.sqr(f), self.prod(e, f))]
        else:
            self.signature = (0, 1)
            self.neg = [BinBasis(e, f, self.sqr(e), self.sqr(f), self.prod(e, f))]

    def _init_definite(self) -> None:
        self.can = self.descend(BinBasis((1, 0), (0, 1), self.a, self.b, self.h))
        s = self.can.sum()
        
        if self.a > 0:
            self.signature = (2, 0)
            target_list = self.pos
        else:
            self.signature = (0, 2)
            target_list = self.neg
            
        target_list.extend([
            self.can.flip(),
            BinBasis(self.can.e, s, self.can.a, self.sqr(s), self.prod(self.can.e, s)),
            BinBasis(self.can.f, s, self.can.b, self.sqr(s), self.prod(self.can.f, s))
        ])

    def _init_indefinite(self) -> None:
        self.signature = (1, 1)
        bas = self.descend(BinBasis((1, 0), (0, 1), self.a, self.b, self.h))
        
        if bas.h < 0:
            bas = bas.flip()
            
        if bas.a == 0:
            r = bas.b % (2 * bas.h)
            t = (r - bas.b) // (2 * bas.h)
            if r == 0:
                bas.f = (bas.f[0] + t * bas.e[0], bas.f[1] + t * bas.e[1])
            else:
                e = (bas.f[0] + t * bas.e[0], bas.f[1] + t * bas.e[1])
                f = (bas.f[0] + (t - 1) * bas.e[0], bas.f[1] + (t - 1) * bas.e[1])
                bas.e, bas.f = e, f
            bas.a, bas.b, bas.h = self.sqr(bas.e), self.sqr(bas.f), self.prod(bas.e, bas.f)
            
        if bas.a == 0 and bas.b == 0:
            self.can = bas if bas.h > 0 else bas.flip()
            self.zero = [self.can.e, (-self.can.e[0], -self.can.e[1]), 
                         self.can.f, (-self.can.f[0], -self.can.f[1])]
            self.pos = [self.can.copy()]
            self.neg = [self.can.flip()]
        else:
            self._process_river(bas)

    def _process_river(self, bas: BinBasis) -> None:
        river = self.flow(bas)
        bas1 = river[-1]
        
        if bas1.a + bas1.b + 2 * bas1.h == 0:
            river = self.flow(bas1.flip())
            bas2 = river[-1]
            self.can = bas1.flip() if self.comp_bases(bas1, bas2) < 0 else bas2.flip()
            self.river = self.flow(self.can)
            s1, s2 = bas1.sum(), bas2.sum()
            self.zero = [s1, (-s1[0], -s1[1]), s2, (-s2[0], -s2[1])]
            
            b1 = BinBasis(s1, bas1.e, 0, bas1.a, self.prod(s1, bas1.e))
            b2 = BinBasis(s2, bas2.e, 0, bas2.a, self.prod(s2, bas2.e))
            self.pos = [b1.copy() if b1.h > 0 else b1.flip(), b2.copy() if b2.h > 0 else b2.flip()]
            
            b1_neg = BinBasis(s1, bas1.f, 0, bas1.b, self.prod(s1, bas1.f))
            b2_neg = BinBasis(s2, bas2.f, 0, bas2.b, self.prod(s2, bas2.f))
            self.neg = [b1_neg.copy() if b1_neg.h < 0 else b1_neg.flip(), 
                        b2_neg.copy() if b2_neg.h < 0 else b2_neg.flip()]
        else:
            nmin = 0
            for i in range(len(river)):
                nmin = i if self.comp_bases(river[i], river[nmin]) < 0 else nmin
            self.can = river[nmin] if river[nmin].h > 0 else river[nmin].flip()
            self.river = self.flow(self.can)
            river.append(self.flow(river[-1])[1])
            
            e0, e1 = river[0].e
            f0, f1 = river[0].f
            E0, E1 = river[-1].e
            F0, F1 = river[-1].f
            det = e0 * f1 - e1 * f0
            self.shift = (det * (E0 * f1 - F0 * e1), det * (-E0 * f0 + F0 * e0), 
                          det * (E1 * f1 - F1 * e1), det * (-E1 * f0 + F1 * e0))

        # Populate pos and neg from the river
        for r1, r2 in zip(river[:-1], river[1:]):
            if r1.e != r2.e:
                b = BinBasis(r1.e, r2.e, r1.a, r2.a, self.prod(r1.e, r2.e))
                self.pos.append(b.copy() if b.h > 0 else b.flip())
            if r1.f != r2.f:
                b = BinBasis(r1.f, r2.f, r1.b, r2.b, self.prod(r1.f, r2.f))
                self.neg.append(b.copy() if b.h < 0 else b.flip())

    @staticmethod
    def _normalize(u: Tuple[int, int]) -> Tuple[int, int]:
        """Ensures vectors are canonically oriented."""
        if u[0] > 0 or (u[0] == 0 and u[1] > 0):
            return u
        return (-u[0], -u[1])

    def comp_bases(self, x: BinBasis, y: BinBasis) -> int:
        for attr in ['h', 'a', 'b']:
            diff = abs(getattr(x, attr)) - abs(getattr(y, attr))
            if diff != 0:
                return diff
        return 0

    def is_isomorphic(self, other: 'BinLattice') -> bool:
        if self.signature != other.signature:
            return False
        return (self.can.a, self.can.b, self.can.h) == (other.can.a, other.can.b, other.can.h)

    def list_positive(self, bound: int) -> Dict[int, Set[Tuple[int, int]]]:
        p = list(self.pos)
        vals = defaultdict(set)
        
        for bas in p:
            if abs(bound) >= bas.a > 0:
                vals[bas.a].add(self._normalize(bas.e))
            if abs(bound) >= bas.b > 0:
                vals[bas.b].add(self._normalize(bas.f))
                
        while p:
            bas = p.pop()
            s = bas.sum()
            c = self.sqr(s)
            if 0 < c <= abs(bound):
                vals[c].add(self._normalize(s))
                p.append(BinBasis(bas.e, s, bas.a, c, self.prod(bas.e, s)))
                p.append(BinBasis(bas.f, s, bas.b, c, self.prod(bas.f, s)))
        return dict(vals)

    def list_negative(self, bound: int) -> Dict[int, Set[Tuple[int, int]]]:
        p = list(self.neg)
        vals = defaultdict(set)
        
        for bas in p:
            if -abs(bound) <= bas.a < 0:
                vals[bas.a].add(self._normalize(bas.e))
            if -abs(bound) <= bas.b < 0:
                vals[bas.b].add(self._normalize(bas.f))
                
        while p:
            bas = p.pop()
            s = bas.sum()
            c = self.sqr(s)
            if 0 > c >= -abs(bound):
                vals[c].add(self._normalize(s))
                p.append(BinBasis(bas.e, s, bas.a, c, self.prod(bas.e, s)))
                p.append(BinBasis(bas.f, s, bas.b, c, self.prod(bas.f, s)))
        return dict(vals)

    def is_root(self, r: Tuple[int, int]) -> bool:
        norm_sq = self.sqr(r)
        if norm_sq == 0:
            return False
        val1 = 2 * (self.a * r[0] + self.h * r[1])
        val2 = 2 * (self.h * r[0] + self.b * r[1])
        return (math.gcd(val1, val2) % norm_sq) == 0
    
    def list_roots(self) -> Dict[int, Set[Tuple[int, int]]]:
        bound = 2 * abs(self.disc)
        p = self.list_positive(bound)
        n = self.list_negative(bound)
        roots = defaultdict(set)
        
        for pool in (p, n):
            for v, vectors in pool.items():
                r = {x for x in vectors if self.is_root(x)}
                if r:
                    roots[v] |= r
        return dict(roots)

    def info(self) -> str:
        if self.signature == (0, 0):
            return 'Zero lattice'
            
        parity = 'Even' if self.a % 2 == 0 and self.b % 2 == 0 else 'Odd'
        lines = [
            f"{parity} lattice of signature {self.signature} and discriminant {self.disc}"
        ]
        
        if self.zero:
            lines.append("Represents zero at " + ', '.join(f"{x}" for x in self.zero))
        else:
            lines.append("Does not represent zero")
            
        lines.append(f"Canonical basis: {self.can.e}, {self.can.f}")
        lines.append(f"Canonical form: {self.can.a}, {self.can.b}, {self.can.h}")
        
        if self.river:
            lines.append(f"River length: {len(self.river)}")
            
        return '\n'.join(lines) + '\n'

    def min_positive(self) -> Optional[int]:
        v = {self.sqr(x.e) for x in self.pos} | {self.sqr(x.f) for x in self.pos}
        positive_vals = v - {0}
        return min(positive_vals) if positive_vals else None

    def max_negative(self) -> Optional[int]:
        v = {self.sqr(x.e) for x in self.neg} | {self.sqr(x.f) for x in self.neg}
        negative_vals = v - {0}
        return max(negative_vals) if negative_vals else None

    def prod(self, u: Tuple[int, int], v: Tuple[int, int]) -> int:
        return self.a * u[0] * v[0] + self.b * u[1] * v[1] + self.h * (u[0] * v[1] + v[0] * u[1])

    def sqr(self, u: Tuple[int, int]) -> int:
        return self.prod(u, u)

    def descend(self, bas: BinBasis) -> BinBasis:
        if bas.a * bas.b == 0:
            return bas.copy() if bas.a == 0 else bas.swap()
        elif bas.a * bas.b < 0:
            return bas.copy() if bas.a > 0 else bas.swap()
            
        b0 = bas.negate() if bas.a < 0 else bas.copy()
        bn = bas.flip() if b0.h > 0 else b0.copy()
        
        while bn.a * bn.b > 0:
            c = bn.a + bn.b + 2 * bn.h
            gn = bn.sum()
            
            if bn.a + bn.h < 0:
                bn.b, bn.f, bn.h = c, gn, bn.a + bn.h
            elif bn.b + bn.h < 0:
                bn.a, bn.e, bn.h = c, gn, bn.b + bn.h
            else:
                triple = sorted([(bn.a, bn.e), (bn.b, bn.f), (c, (-gn[0], -gn[1]))], key=lambda x: x[0])
                e, f = triple[0][1], triple[1][1]
                return BinBasis(e, f, self.sqr(e), self.sqr(f), self.prod(e, f))
                
        if bn.a * bn.b == 0:
            return bn if bn.a == 0 else bn.swap()
        return bn if bn.a > 0 else bn.swap()

    def flow(self, bas: BinBasis) -> List[BinBasis]:
        if bas.a * bas.b >= 0:
            return []
            
        b0 = bas if bas.a > 0 else bas.swap()
        bn = b0.copy()
        river = []
        
        while bn.a * bn.b < 0:
            river.append(bn.copy())
            c = bn.a + bn.b + 2 * bn.h
            g = bn.sum()
            
            if c > 0:
                bn.a, bn.e, bn.h = c, g, bn.h + bn.b
            else:
                bn.b, bn.f, bn.h = c, g, bn.h + bn.a
                
            if (bn.a, bn.b, bn.h) == (b0.a, b0.b, b0.h):
                break
                
        return river