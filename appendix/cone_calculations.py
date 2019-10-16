#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2016 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of fip-walking
# <https://github.com/stephane-caron/fip-walking>.
#
# fip-walking is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# fip-walking is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# fip-walking. If not, see <http://www.gnu.org/licenses/>.

"""
Symbolic calculations used to derive the formula of the FIP contact-stability
condition (Appendix A).
"""

import sympy

X = sympy.Symbol('X', real=True, positive=True)
Y = sympy.Symbol('Y', real=True, positive=True)
mu = sympy.Symbol('mu', real=True, positive=True, latex='\\mu')
ld = sympy.Symbol('lamda', real=True, positive=True, latex='\\lambda')

x_Z = sympy.Symbol('x_Z', real=True)
y_Z = sympy.Symbol('y_Z', real=True)
x_G = sympy.Symbol('x_G', real=True)
y_G = sympy.Symbol('y_G', real=True)
z_G = sympy.Symbol('z_G', real=True)

F = sympy.Matrix([
    # fx fy             fz taux tauy tauz
    [-1,  0,           -mu,   0,   0,   0],
    [+1,  0,           -mu,   0,   0,   0],
    [0,  -1,           -mu,   0,   0,   0],
    [0,  +1,           -mu,   0,   0,   0],
    [0,   0,            -Y,  -1,   0,   0],
    [0,   0,            -Y,  +1,   0,   0],
    [0,   0,            -X,   0,  -1,   0],
    [0,   0,            -X,   0,  +1,   0],
    [-Y, -X, -(X + Y) * mu, +mu, +mu,  -1],
    [-Y, +X, -(X + Y) * mu, +mu, -mu,  -1],
    [+Y, -X, -(X + Y) * mu, -mu, +mu,  -1],
    [+Y, +X, -(X + Y) * mu, -mu, -mu,  -1],
    [+Y, +X, -(X + Y) * mu, +mu, +mu,  +1],
    [+Y, -X, -(X + Y) * mu, +mu, -mu,  +1],
    [-Y, +X, -(X + Y) * mu, -mu, +mu,  +1],
    [-Y, -X, -(X + Y) * mu, -mu, -mu,  +1]])

# f = lambda * (p_G - p_Z)
# tau_O = tau_Z + OZ x f
f_x = ld * (x_G - x_Z)
f_y = ld * (y_G - y_Z)
f_z = ld * z_G
tau_Ox = y_Z * f_z
tau_Oy = -x_Z * f_z
tau_Oz = x_Z * f_y - y_Z * f_x

w_O = sympy.Matrix([f_x, f_y, f_z, tau_Ox, tau_Oy, tau_Oz])


def print_v1():
    print """
    \\section{Rewriting step 1}
    \\begin{eqnarray*}
        |x_G - x_Z| & \leq & \mu z_G \\\\
        |y_G - y_Z| & \leq & \mu z_G \\\\
        |x_Z| & \leq & X \\\\
        |y_Z| & \leq & Y \\\\
        +X (y_Z - y_G) + Y (x_Z - x_G) + x_G y_Z - x_Z y_G & \leq & \mu z_G \left(X + Y + x_Z - y_Z\\right) \\\\
        -X (y_Z - y_G) + Y (x_Z - x_G) + x_G y_Z - x_Z y_G & \leq & \mu z_G \left(X + Y - x_Z - y_Z\\right) \\\\
        +X (y_Z - y_G) - Y (x_Z - x_G) + x_G y_Z - x_Z y_G & \leq & \mu z_G \left(X + Y + x_Z + y_Z\\right) \\\\
        -X (y_Z - y_G) - Y (x_Z - x_G) + x_G y_Z - x_Z y_G & \leq & \mu z_G \left(X + Y - x_Z + y_Z\\right)  \\\\
        -X (y_Z - y_G) - Y (x_Z - x_G) - x_G y_Z + x_Z y_G & \leq & \mu z_G \left(X + Y + x_Z - y_Z\\right)  \\\\
        +X (y_Z - y_G) - Y (x_Z - x_G) - x_G y_Z + x_Z y_G & \leq & \mu z_G \left(X + Y - x_Z - y_Z\\right)  \\\\
        -X (y_Z - y_G) + Y (x_Z - x_G) - x_G y_Z + x_Z y_G & \leq & \mu z_G \left(X + Y + x_Z + y_Z\\right) \\\\
        +X (y_Z - y_G) + Y (x_Z - x_G) - x_G y_Z + x_Z y_G & \leq & \mu z_G \left(X + Y - x_Z + y_Z\\right)
    \end{eqnarray*}""",


def print_v2():
    print """
    \\section{Rewriting step 2}
    \\begin{eqnarray*}
        |x_G - x_Z| & \leq & \mu z_G \\\\
        |y_G - y_Z| & \leq & \mu z_G \\\\
        |x_Z| & \leq & X \\\\
        |y_Z| & \leq & Y \\\\
        - \mu z_G x_Z - x_Z y_G - \mu z_G y_Z + x_G y_Z & \leq & X (\mu z_G - (y_Z - y_G)) + Y (\mu z_G + (x_Z - x_G)) \\\\
        - \mu z_G x_Z - x_Z y_G + \mu z_G y_Z + x_G y_Z & \leq & X (\mu z_G - (y_Z - y_G)) + Y (\mu z_G - (x_Z - x_G)) \\\\
        - \mu z_G x_Z + x_Z y_G - \mu z_G y_Z - x_G y_Z & \leq & X (\mu z_G + (y_Z - y_G)) + Y (\mu z_G - (x_Z - x_G)) \\\\
        - \mu z_G x_Z + x_Z y_G + \mu z_G y_Z - x_G y_Z & \leq & X (\mu z_G + (y_Z - y_G)) + Y (\mu z_G + (x_Z - x_G)) \\\\
        + \mu z_G x_Z + x_Z y_G + \mu z_G y_Z - x_G y_Z & \leq & X (\mu z_G - (y_Z - y_G)) + Y (\mu z_G + (x_Z - x_G)) \\\\
        + \mu z_G x_Z - x_Z y_G + \mu z_G y_Z + x_G y_Z & \leq & X (\mu z_G + (y_Z - y_G)) + Y (\mu z_G - (x_Z - x_G)) \\\\
        + \mu z_G x_Z - x_Z y_G - \mu z_G y_Z + x_G y_Z & \leq & X (\mu z_G + (y_Z - y_G)) + Y (\mu z_G + (x_Z - x_G)) \\\\
        + \mu z_G x_Z + x_Z y_G - \mu z_G y_Z - x_G y_Z & \leq & X (\mu z_G - (y_Z - y_G)) + Y (\mu z_G - (x_Z - x_G))
    \end{eqnarray*}""",


def print_v3():
    print """
    \\section{Rewriting step 3}
    \\begin{eqnarray*}
        |x_G - x_Z| & \leq & \mu z_G \\\\
        |y_G - y_Z| & \leq & \mu z_G \\\\
        |x_Z| & \leq & X \\\\
        |y_Z| & \leq & Y \\\\
        -x_Z (\mu z_G - (y_Z - y_G)) - y_Z (\mu z_G + (x_Z - x_G)) & \leq & X (\mu z_G - (y_Z - y_G)) + Y (\mu z_G + (x_Z - x_G)) \\\\
        -x_Z (\mu z_G - (y_Z - y_G)) + y_Z (\mu z_G - (x_Z - x_G)) & \leq & X (\mu z_G - (y_Z - y_G)) + Y (\mu z_G - (x_Z - x_G)) \\\\
        +x_Z (\mu z_G - (y_Z - y_G)) + y_Z (\mu z_G + (x_Z - x_G)) & \leq & X (\mu z_G - (y_Z - y_G)) + Y (\mu z_G + (x_Z - x_G)) \\\\
        +x_Z (\mu z_G - (y_Z - y_G)) - y_Z (\mu z_G - (x_Z - x_G)) & \leq & X (\mu z_G - (y_Z - y_G)) + Y (\mu z_G - (x_Z - x_G)) \\\\
        -x_Z (\mu z_G + (y_Z - y_G)) - y_Z (\mu z_G - (x_Z - x_G)) & \leq & X (\mu z_G + (y_Z - y_G)) + Y (\mu z_G - (x_Z - x_G)) \\\\
        -x_Z (\mu z_G + (y_Z - y_G)) + y_Z (\mu z_G + (x_Z - x_G)) & \leq & X (\mu z_G + (y_Z - y_G)) + Y (\mu z_G + (x_Z - x_G)) \\\\
        +x_Z (\mu z_G + (y_Z - y_G)) + y_Z (\mu z_G - (x_Z - x_G)) & \leq & X (\mu z_G + (y_Z - y_G)) + Y (\mu z_G - (x_Z - x_G)) \\\\
        +x_Z (\mu z_G + (y_Z - y_G)) - y_Z (\mu z_G + (x_Z - x_G)) & \leq & X (\mu z_G + (y_Z - y_G)) + Y (\mu z_G + (x_Z - x_G))
    \end{eqnarray*}""",


def print_final():
    print """
    \\section{Rewriting step 4 (final)}
    \\begin{eqnarray}
        |x_G - x_Z| & \leq & \mu z_G \\\\
        |y_G - y_Z| & \leq & \mu z_G \\\\
        |x_Z| & \leq & X \\\\
        |y_Z| & \leq & Y
    \end{eqnarray}
    \\begin{eqnarray*}
        0 & \leq & (X + x_Z) (\mu z_G - (y_Z - y_G)) + (Y + y_Z) (\mu z_G + (x_Z x_G)) \\\\
        0 & \leq & (X + x_Z) (\mu z_G + (y_Z - y_G)) + (Y + y_Z) (\mu z_G - (x_Z x_G)) \\\\
        0 & \leq & (X - x_Z) (\mu z_G - (y_Z - y_G)) + (Y + y_Z) (\mu z_G - (x_Z x_G)) \\\\
        0 & \leq & (X - x_Z) (\mu z_G + (y_Z - y_G)) + (Y + y_Z) (\mu z_G + (x_Z x_G)) \\\\
        0 & \leq & (X - x_Z) (\mu z_G + (y_Z - y_G)) + (Y - y_Z) (\mu z_G - (x_Z x_G)) \\\\
        0 & \leq & (X + x_Z) (\mu z_G - (y_Z - y_G)) + (Y - y_Z) (\mu z_G - (x_Z x_G)) \\\\
        0 & \leq & (X + x_Z) (\mu z_G + (y_Z - y_G)) + (Y - y_Z) (\mu z_G + (x_Z x_G)) \\\\
        0 & \leq & (X - x_Z) (\mu z_G - (y_Z - y_G)) + (Y - y_Z) (\mu z_G + (x_Z x_G))
    \end{eqnarray*}""",


if __name__ == "__main__":
    E = F.dot(w_O)
    print """\\documentclass[10pt]{article}
\\usepackage{color}
\\usepackage{enumerate}
\\usepackage{algorithm}
\\usepackage{algorithmic}
\\usepackage{multicol}
\\usepackage{amssymb, amsmath}
\\usepackage{stmaryrd}
\\usepackage{graphicx}
\\usepackage{setspace}
\\usepackage{mathrsfs}
\\usepackage{bm}
\\begin{document}
    \section{Initial SymPy output}"""
    print '    \\begin{eqnarray*}\n       ',
    print ' & \\leq & 0 \\\\\n        '.join([
        sympy.latex((expr / ld).factor(mu, z_G)) for expr in E]),
    print ' & \\leq & 0'
    print '    \\end{eqnarray*}',
    print_v1()
    print_v2()
    print_v3()
    print_final()
    print ""
    print "\\end{document}"
