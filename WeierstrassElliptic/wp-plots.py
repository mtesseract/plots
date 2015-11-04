# Copyright (C) 2014, 2015 Moritz Schulte <mtesseract@silverratio.net>

from   mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import cmath
import itertools
import math
from   mpl_toolkits.mplot3d import proj3d
import matplotlib.animation as animation

# Draw a point, given its (x, y, z) coordinates.
def addPoint(x1, x2, x3, color):
    ax.scatter([x1], [x2], zs=[x3], c=color, s=[100], depthshade=False)

# Draw a complex number, using stereographic projection.
def addNumberColored(z, color):
    if cmath.isinf (z):
        addPoint(0, 0, 1, color)
    else:
        (x1, x2, x3) = stePro (z.real, z.imag)
        addPoint(x1, x2, x3, color)

# Stereographic Projection: Given x and y, compute the coordinates (X,
# Y, Z) on the unit sphere embedded in $\mathbb{R}^3$ .
def stePro(x,y):
    x1 = 2*x / (1 + x**2 + y**2)
    x2 = 2*y / (1 + x**2 + y**2)
    x3 = (x**2 + y**2 - 1) / (1 + x**2 + y**2)
    return (x1, x2, x3)

# This draws the complex projective space $\mathbb{P}_1$ as a sphere.
def drawRiemannSphere():
    # Begin with polar coordinates...
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    # ... and convert them to cartesian coordinates.
    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b', alpha=0.5, linewidth=0, cmap='cool')
    addNumberColored(float("inf"), 'g')
    addNumberColored(0, 'y')

################################################
# Implementation of the Weierstrass P function #
################################################

# These are the lattice points which we use for approximating the Weierstrass elliptic functions.
# First in cartesian coordinates...
latticepointsXY = list(filter(lambda x: x != (0,0), itertools.product(range(-10,11),range(-10,11))))
# ... then in the complex plane.
latticepoints = list(map (lambda x: x[0] + x[1]*1j, latticepointsXY))

# $w = \wp(z)$

def wp(z):
    if z == 0 or z in latticepoints:
        return float("inf") + float("inf") * 1j
    else:
        w = 1/(z**2) + sum(map (lambda a: 1/(z-a)**2 - 1/a**2, latticepoints))
        return w

# Compute the stereographic projection of $\wp(x + t i)$.
def wpPlot(t, x):
    z = x + t*1j
    w = wp (z)
    return stePro (w.real, w.imag)

#################################################
# Implementation of the Weierstrass P' function #
#################################################

# $w = \wp'(z)$
def wpP(z):
    if z == 0 or z in latticepoints:
        return float("inf") + float("inf") * 1j
    else:
        w = -2/(z**3) - 2 * sum(map (lambda a: 1/(z-a)**3, latticepoints))
        return w

# Compute the stereographic projection of $\wp'(x + t i)$.
def wpPPlot(t, x):
    z = x + t*1j
    w = wpP (z)
    return stePro (w.real, w.imag)

# [[x0, y0, z0], [x1, y1, y1], ...] => ([x0, x1, ...], [y0, y1, ...], [z0, z1, ...])
def extractComponents(tuples):
    xs = []
    ys = []
    zs = []
    while True:
        if tuples == []:
            return (xs, ys, zs)
        else:
            t = tuples[0]
            xs.append (t[0])
            ys.append (t[1])
            zs.append (t[2])
            tuples = tuples[1:]

def takeUntil(l,x):
    lRes = []
    while l != []:
        lRes.append(l[0])
        if l[0] == x:
            break
        else:
            l = l[1:]
    return lRes

# We use this for bookkeeping.
lastpaths = []

# Initialize drawing canvas.
fig = plt.figure(frameon=True, figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)

# Plot a single "slice" of the Weierstrass P function. I.e., on the
# standard square torus an interval of the form
# $I_{y(t)} = {(x, y(t)) | 0 \leq x \leq 1}$.
def plotSlice(t, f):
    global lastpaths
    xGrainsN = 100
    xGrainsNHalf = round (xGrainsN / 2)
    xGrains = np.linspace (0, 1, xGrainsN)
    xGrains1 = xGrains[0:xGrainsNHalf]
    xGrains2 = xGrains[xGrainsNHalf:xGrainsN]
    (xs, ys, zs) = extractComponents (list (map  (lambda x: f (t, x), xGrains1)))
    path1, = ax.plot (xs, ys, zs=zs, c='g')
    lastpaths.append (path1)
    (xs, ys, zs) = extractComponents (list (map  (lambda x: f (t, x), xGrains2)))
    path2, = ax.plot (xs, ys, zs=zs, c='k')
    lastpaths.append (path2)

def plotCircle(tEnd, y, f):
    global lastpaths
    tValues = takeUntil (ts, tEnd)
    tValuesTail = tValues[-2:]
    (xs, ys, zs) = extractComponents (list (map  (lambda t: f (y, t), tValues)))
    path, = ax.plot (xs, ys, zs=zs, c='g')
    lastpaths.append (path)
    (xs, ys, zs) = extractComponents (list (map  (lambda t: f (y, t), tValuesTail)))
    path, = ax.plot (xs, ys, zs=zs, c='k')
    lastpaths.append (path)

# Remove last drawn path.
def clearLastPath():
    global lastpaths
    if lastpaths != False:
        for p in lastpaths:
            ax.lines.remove(p)
        lastpaths = []

# Main drawing function, varying in time.
def plotWpSlices(t):
    clearLastPath ()
    plotSlice (t, wpPlot)

# Main drawing function, varying in time.
def plotWpPSlices(t):
    clearLastPath ()
    plotSlice (t, wpPPlot)

def plotWpCircle0(t):
    clearLastPath ()
    plotCircle (t, 0, wpPlot)

def plotWpCircle1(t):
    clearLastPath ()
    plotCircle (t, 0.5, wpPlot)

# Construct the list of t's for which we do the plot:
def makeTs(a, b):
    ts = list(np.linspace (a, b, 100))
    # Pause for a bit:
    for i in range(5):
        ts.insert(0, a)
        ts.append(b)
    return ts

# Create the Animation object
ax.axis("off")
drawRiemannSphere()

def init_fig():
    print ("computing...")

# Enable the desired plot:

# Plot global behaviour of $\wp$:
ts = makeTs(0, 0.5)
line_ani = animation.FuncAnimation(fig, plotWpSlices, frames=list(ts), interval=100,
                                   init_func=init_fig, blit=False)
# Save as video, if desired.
# line_ani.save('wp-global.mp4')

# Plot global behaviour of $\wp'$:
# ts = makeTs(0, 0.5)
# line_ani = animation.FuncAnimation(fig, plotWpPSlices, frames=list(makeTs(0, 0.5)), interval=300,
#                                    init_func=init_fig,  blit=False)
# Save as video, if desired.
# line_ani.save('wpP-global.mp4')

# Plot the behaviour of $\wp$ along the fixed point circle at $y=0$:
# ts = makeTs(0, 1)
# line_ani = animation.FuncAnimation(fig, plotWpCircle0, frames=list(ts), interval=100, blit=False,
#                                    init_func=init_fig)
# Save as video, if desired.
# line_ani.save('wp-circle0.mp4')

# Plot the behaviour of $\wp$ along the fixed point circle at $y=0.5$:
# ts = makeTs(0, 1)
# line_ani = animation.FuncAnimation(fig, plotWpCircle1, frames=list(ts), interval=100, blit=False,
#                                    init_func=init_fig)
# Save as video, if desired.
# line_ani.save('wp-circle1.mp4')

plt.show()
