import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from lattice import PrimitiveCell
# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_parallelepiped(ax, a, origin=np.array([0, 0, 0])):
    '''
        @param a  a [3,3] vector of the form [a0 a1 a2]^T, i.e. a0 = a[0] starting from xyz
    '''
    a = np.array(a).astype(np.float64)
    points = np.zeros((8, 3))

    points[1] = a[0]
    points[2] = a[1]
    points[3] = a[2]
    points[4] = a[0]+a[1]
    points[5] = a[0]+a[2]
    points[6] = a[1]+a[2]
    points[7] = a[0]+a[1]+a[2]

    points += origin

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]


    faces = Poly3DCollection(edges, linewidths=0.2, edgecolors='k')
    faces.set_facecolor((0, 0, 1, 0.1))

    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0)


def plot_cell(ax, p: PrimitiveCell):
    plot_parallelepiped(ax, np.array(p.lattice_vectors.T).astype(np.float64))

    # plot atomic sites
    for a in p.atoms:
        p.lattice_vectors
        ax.plot(*a.xyz, 'o', **a.plot_args)

    # plot bonds
    for b in p.bonds:
        x0 = p.atoms[b['from_idx']].xyz
        parts = [[x0]]
        denom = 100
        for i in range(denom):
            x = p.wrap_coordinate(x0 + (i+1)*b['bond_delta']/denom)

            if (x-parts[-1][-1]).norm() < 2*b['bond_delta'].norm()/denom:
                parts[-1].append(x)
            else:
                # suggests a jump has occurred; new part
                parts.append([x])

        for link in parts:
            XX = np.array(link)[:, :, 0]
            ax.plot(*XX.T, **b['fmt'])

    set_axes_equal(ax)
