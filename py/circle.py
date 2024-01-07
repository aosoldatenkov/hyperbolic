import sys
import json


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
    
    def set_atta(self, **attr) -> None:
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

    def __init__(self) -> None:
        self.circles = []
        self.discs = []

    def dump(self, fname: str) -> None:
        data = [c.dump_data() for c in self.discs]
        data.extend(c.dump_data() for c in self.circles)
        print(f"Writing data to {fname}...", end="")
        with open(fname, "w") as file_out:
            json.dump(data, file_out, indent=1)
        print("done")

    def load(self, fname: str) -> None:
        print(f"Loading data from {fname}...", end="")
        with open(fname) as file_in:
            data = json.load(file_in)
        for c in data:
            if "disc" in c["attr"]:
                self.discs.append(Circle())
                self.discs[-1].load_data(c)
            else:
                self.circles.append(Circle())
                self.circles[-1].load_data(c)
        print(f"done; {len(self.discs) + len(self.circles)} circles loaded")

    def load_xyr(self, fname: str) -> None:
        self.circles = []
        self.discs = []
        print(f"Loading data from {fname}...", end="")
        with open(fname) as file_in:
            for c in file_in:
                xyr = c.split()
                self.discs.append(Circle(float(xyr[0]),
                                         float(xyr[1]),
                                         float(xyr[2]),
                                         disc=1))
        print(f"done; {len(self.discs)} circles loaded")

    def add_circle(self, c) -> bool:
        x, y = c.center()
        r = c.radius()
        if x ** 2 + y ** 2 > 1 or r > 1 - Circle.epsilon:
            return False
        for i in range(len(self.discs)):
            if self.discs[i].radius() <= r - Circle.epsilon:
                continue
            if self.discs[i].contains_circ(c):
                return False
        for i in range(len(self.circles)):
            if self.circles[i].equiv(c):
                return False
        if "disc" in c.get_attr():
            if self.discs == []:
                self.discs.append(c)
            else:
                self.discs.insert(i, c)
            self.circles = [a for a in self.circles if not c.contains_circ(a)]
        else:
            self.circles.append(c)
        return True
    
    def pt_is_visible(self, x: float, y: float) -> bool:
        return all(not c.contains_pt(x, y) for c in self.discs)

    def circ_is_visible(self, c) -> bool:
        return all(not a.contains_circ(c) for a in self.discs)
    
    def tikz_out(self, fname: str, width: int = 150) -> None:
        lines = []
        #name_opacity = "\\TikzOpacity"
        #var_opacity = 0.2
        #name_lwidth = "\\TikzLineWidth"
        #var_lwidth = "0.1pt"
        #lines.append(f"\\newcommand{chr(123)}{name_opacity}{chr(125)}{chr(123)}{var_opacity}{chr(125)}\n")
        #lines.append(f"\\newcommand{chr(123)}{name_lwidth}{chr(125)}{chr(123)}{var_lwidth}{chr(125)}\n\n")
        lines.append("\\begin{tikzpicture}\n")
        lines.append("  \\draw[clip] (0,0) circle (150pt);\n")
        lines.append("  \\filldraw[fill=black] (0,0) circle (150pt);\n")
        print(f"Number of discs: {len(self.discs)}")
        print(f"Number of circles: {len(self.circles)}")
        for c in self.discs:
            x, y = c.center()
            x *= width
            y *= width
            r = c.radius() * width
            lines.append(f"  \\fill[fill=white] ({x}pt, {y}pt) circle ({r}pt);\n")
        for c in self.circles:
            x, y = c.center()
            x *= width
            y *= width
            r = c.radius() * width
            lines.append(f"  \\draw[line width=0.6,draw=red] ({x}pt, {y}pt) circle ({r}pt);\n")
        lines.append("\\end{tikzpicture}")
        with open(fname, "w", encoding="ASCII") as file_out:
            file_out.writelines(lines)
