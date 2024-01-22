import sys
import json
import math


class Circle:

    epsilon: float = sys.float_info.epsilon * 1000

    def __init__(self, x: float = 0, y: float = 0, r: float = 1, **attr) -> None:
        self.x, self.y, self.r = x, y, r
        self.attr = attr

    def radius(self) -> float:
        return self.r

    def center(self) -> tuple[float]:
        return self.x, self.y

    def get_attr(self) -> dict:
        return self.attr
    
    def set_attr(self, **attr) -> None:
        self.attr.update(attr)

    def contains_pt(self, x: float, y: float) -> bool:
        if (self.x - x) ** 2 + (self.y - y) ** 2 - self.r ** 2 <= self.epsilon:
            return True
        return False

    def contains_circ(self, c) -> bool:
        x, y = c.center()
        r = c.radius()
        if r - self.r > self.epsilon:
            return False
        if (self.x - x) ** 2 + (self.y - y) ** 2 - (self.r - r) ** 2 <= self.epsilon:
            return True
        return False

    def intersects_circ(self, c) -> bool:
        x, y = c.center()
        r = c.radius()
        sq_dist = (self.x - x) ** 2 + (self.y - y) ** 2
        if all([(self.r - r) ** 2 - sq_dist <= self.epsilon,
               sq_dist - (self.r + r) ** 2 <= self.epsilon]):
            return True
        return False

    def is_outside(self, c) -> bool:
        if self.contains_circ(c):
            return True
        x, y = c.center()
        r = c.radius()
        sq_dist = (self.x - x) ** 2 + (self.y - y) ** 2
        if sq_dist - (self.r + r) ** 2 >= self.epsilon:
            return True
        return False

    def equiv(self, c) -> bool:
        x, y = c.center()
        r = c.radius()
        if all([abs(x - self.x) < self.epsilon,
                abs(y - self.y) < self.epsilon,
                abs(r - self.r) < self.epsilon]):
            return True
        return False

    def dump_data(self) -> dict:
        data = {"x": self.x,
                "y": self.y,
                "r": self.r,
                "attr": self.attr}
        return data

    def load_data(self, d: dict) -> None:
        self.x = d["x"]
        self.y = d["y"]
        self.r = d["r"]
        self.attr = d["attr"]


class CircleArrangement:

    def reset(self) -> None:
        self.circles = []
        self.discs = []
        self.holes = []
        self.lines = []
        self.frame = None
    
    def __init__(self, backup_file="backup.json", backup_interval=200) -> None:
        self.b_file = backup_file
        self.b_int = backup_interval
        self.b_last = 0
        self.reset()

    def get_size(self) -> int:
        return len(self.discs) + len(self.holes) + len(self.circles) + len(self.lines)
        
    def dump(self, fname: str) -> None:
        data = [c.dump_data() for c in self.holes]
        data.extend(c.dump_data() for c in self.discs)
        data.extend(c.dump_data() for c in self.circles)
        data.extend(c.dump_data() for c in self.lines)
        with open(fname, "w") as file_out:
            json.dump(data, file_out, indent=1)
        
    def load(self, fname: str, preserve_existing: bool = False) -> None:
        print(f"Loading data from {fname}...", end="")
        if not preserve_existing:
            self.reset()
        with open(fname) as file_in:
            data = json.load(file_in)
        for c in data:
            circ = Circle()
            circ.load_data(c)
            self.add_circle(circ, backup=False)
        print(f"done; {len(self.discs) + len(self.circles)} circles loaded")

    def load_xyr(self, fname: str, preserve_existing: bool = False) -> None:
        if not preserve_existing:
            self.reset()
        print(f"Loading data from {fname}...", end="")
        with open(fname) as file_in:
            for c in file_in:
                xyr = c.split()
                self.discs.append(Circle(float(xyr[0]),
                                         float(xyr[1]),
                                         float(xyr[2]),
                                         disc=1))
        print(f"done; {len(self.discs)} circles loaded")

    def add_circle(self, c, backup: bool = True) -> bool:
        x, y = c.center()
        r = c.radius()
        if "line" in c.get_attr():
            self.lines.append(c)
            return True
        elif "hole" in c.get_attr():
            if self.frame != None and c.contains_circ(self.frame):
                return False
            insert_at = 0
            for i in range(len(self.holes)):
                if self.holes[i].radius() >= r + Circle.epsilon:
                    insert_at = i
                    break
                if c.contains_circ(self.holes[i]):
                    return False
            else:
                insert_at = len(self.holes)
            if self.holes == []:
                self.holes.append(c)
            else:
                self.holes.insert(insert_at, c)
                self.holes[insert_at + 1:] = [a for a in self.holes[insert_at + 1:] if not a.contains_circ(c)]
            self.discs = [a for a in self.discs if not a.is_outside(c)]
            self.circles = [a for a in self.circles if not a.is_outside(c)]
            if self.frame == None or self.frame.radius() > r:
                self.frame = Circle(x, y, r)
        elif "disc" in c.get_attr():
            for a in self.holes:
                if c.is_outside(a):
                    return False
            insert_at = 0
            for i in range(len(self.discs)):
                if self.discs[i].radius() <= r - Circle.epsilon:
                    insert_at = i
                    break
                if self.discs[i].contains_circ(c):
                    return False
            else:
                insert_at = len(self.discs)
            if self.discs == []:
                self.discs.append(c)
            else:
                self.discs.insert(insert_at, c)
                self.discs[insert_at + 1:] = [a for a in self.discs[insert_at + 1:] if not c.contains_circ(a)]
            self.circles = [a for a in self.circles if not c.contains_circ(a)]
        else:
            for a in self.holes:
                if c.is_outside(a):
                    return False
            for i in range(len(self.circles)):
                if self.circles[i].equiv(c):
                    return False
            self.circles.append(c)
        if backup and self.get_size() >= self.b_last + self.b_int:
            self.dump(self.b_file)
            self.b_last = self.get_size()
        return True
    
    def pt_is_visible(self, x: float, y: float) -> bool:
        return all(not c.contains_pt(x, y) for c in self.discs)

    def circ_is_visible(self, c) -> bool:
        return all(not a.contains_circ(c) for a in self.discs)
    
    def get_statistics(self) -> dict:
        return {"holes": len(self.holes),
                "discs": len(self.discs),
                "circles": len(self.circles),
                "lines": len(self.lines)}

    def print_statistics(self) -> None:
        print("\n".join(f"Number of {x}: {self.get_statistics()[x]}" for x in self.get_statistics()))
    
    def tikz_out(self, fname: str, width: int = 150, max_circles: int = -1, use_colors: bool = False) -> None:
        lines = []
        self.print_statistics()
        self.frame = None
        if self.frame == None:
            o_x = o_y = 0
            o_r = 1
        else:
            o_x, o_y = self.frame.center()
            o_r = self.frame.radius()
        #name_opacity = "\\TikzOpacity"
        #var_opacity = 0.2
        #name_lwidth = "\\TikzLineWidth"
        #var_lwidth = "0.1pt"
        #lines.append(f"\\newcommand{chr(123)}{name_opacity}{chr(125)}{chr(123)}{var_opacity}{chr(125)}\n")
        #lines.append(f"\\newcommand{chr(123)}{name_lwidth}{chr(125)}{chr(123)}{var_lwidth}{chr(125)}\n\n")
        lines.append("\\definecolor{bright}{rgb}{1.0,0.6,0.1}\n")
        lines.append("\\definecolor{dark}{rgb}{0.7,0.3,0.1}\n")
        lines.append("\\begin{tikzpicture}\n")
        for c in self.holes:
            x, y = c.center()
            x -= o_x
            y -= o_y
            if abs(x) < Circle.epsilon:
                x = 0
            if abs(y) < Circle.epsilon:
                y = 0
            x = x * width / o_r
            y = y * width / o_r
            r = c.radius() * width / o_r
            lines.append(f"  \\clip ({x}pt,{y}pt) circle ({r}pt);\n")
        lines.append(f"  \\fill[fill=black] (0,0) circle ({width}pt);\n")
        for c in self.discs:
            x, y = c.center()
            x -= o_x
            y -= o_y
            if abs(x) < Circle.epsilon:
                x = 0
            if abs(y) < Circle.epsilon:
                y = 0
            x = x * width / o_r
            y = y * width / o_r
            r = c.radius() * width / o_r
            color = min(100, int(o_r / c.radius()))
            if use_colors:
                lines.append(f"  \\fill[fill=dark!{color}!bright] ({x}pt, {y}pt) circle ({r}pt);\n")
            else:
                lines.append(f"  \\fill[fill=white] ({x}pt, {y}pt) circle ({r}pt);\n")
        ncircles = max_circles
        for c in self.circles:
            if ncircles == 0:
                break
            ncircles -= 1
            x, y = c.center()
            x -= o_x
            y -= o_y
            if abs(x) < Circle.epsilon:
                x = 0
            if abs(y) < Circle.epsilon:
                y = 0
            x = x * width / o_r
            y = y * width / o_r
            r = c.radius() * width / o_r
            lines.append(f"  \\draw[line width=0.6,draw=red] ({x}pt, {y}pt) circle ({r}pt);\n")
        #lines.append(f"  \\draw[line width=1,draw=red] ({-o_x * width / o_r}pt, {-o_y * width / o_r}pt) circle ({width / o_r}pt);\n")
        lines.append("\\end{tikzpicture}")
        with open(fname, "w", encoding="ASCII") as file_out:
            file_out.writelines(lines)
