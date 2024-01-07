import numpy as np
import math
import lattice as lt

name_opacity = "\\TikzOpacity"
var_opacity = 0.2
name_lwidth = "\\TikzLineWidth"
var_lwidth = "0.1pt"

lines = []
lines.append(f"\\newcommand{chr(123)}{name_opacity}{chr(125)}{chr(123)}{var_opacity}{chr(125)}\n")
lines.append(f"\\newcommand{chr(123)}{name_lwidth}{chr(125)}{chr(123)}{var_lwidth}{chr(125)}\n\n")

lines.append("\\begin{tikzpicture}\n")
lines.append("  \\draw[clip] (0,0) circle (150pt);\n")

limits = 150
width = 150

def hyperbolic_pict():
  dimension = 3
  A = np.array([[3, 0, 0],
                [0, -2, 1],
                [0, 1, -2]])
  lat = lt.HLattice(dimension, A)
  lenum = lt.LatticeEnum(dimension)
  count = 0
  for i in range(2 ** 20):
    v = next(lenum)
    w = lat.hcoord(v)
    square = lat.prod(v, v)
    if w[0] > 0 and -6 < square < 0:
#    x = width * w[1] / (1 + w[0])
#    y = width * w[2] / (1 + w[0])
#    r = width / (1 + w[0])
      x = width * w[1] / w[0]
      y = width * w[2] / w[0]
      r = width * math.sqrt((w[1] / w[0]) ** 2 + (w[2] / w[0]) ** 2 - 1)
      lines.append(f"  \\filldraw[line width={name_lwidth},fill=black,fill opacity={name_opacity}] ({x:.5f}pt, {y:.5f}pt) circle ({r:.5f}pt);\n")
      count += 1
      if count > 200:
        break
  return count

def boundary_pict():
  dimension = 4
  #  A = np.array([[14, 0, 0, 0],
  #                [0, -2, 0, 0],
  #                [0, 0, -14, 7],
  #                [0, 0, 7, -14]])
  p = 5
  A = np.array([[2 * p, 0, 0, 0],
                [0, -2, 0, 0],
                [0, 0, -2 * p, p],
                [0, 0, p, -2 * p]])
  lat = lt.HLattice(dimension, A)
  lenum = lt.LatticeEnum(dimension)
  count = 0
  for i in range(2 ** 30):
    v = next(lenum)
    w = lat.hcoord(v)
    square = lat.prod(v, v)
    if w[1] - w[0] != 0 and -3 < square < 0:
      x = width * w[2] / (w[1] - w[0])
      y = width * w[3] / (w[1] - w[0])
      r = width * math.sqrt(w[1] ** 2 + w[2] ** 2 + w[3] ** 2 - w[0] ** 2) / (w[1] - w[0])
      lines.append(f"  \\draw[line width={name_lwidth}] ({x:.5f}pt, {y:.5f}pt) circle ({r:.5f}pt);\n")
      count += 1
      if count > 1000:
        break
      if count % 100 == 0:
        print(count)
  return count

count = boundary_pict()

lines.append("\\end{tikzpicture}")

with open("tikzcode.tex", "w", encoding="ASCII") as file_out:
    file_out.writelines(lines)

print(count)
