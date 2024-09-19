"""Script to run and compare the place algorithms."""
from voxcell.voxel_data import VoxelData
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
from brainbuilder.cell_positions import _create_cell_positions_uniform
from brainbuilder.cell_positions import _create_cell_positions_poisson_disc
from brainbuilder.cell_positions import _create_cell_positions_accept_reject

if __name__ == "__main__":
    raw = 2e3 * np.ones((50, 2, 40))
    voxel_dimensions = [25, 25, 25]
    density = VoxelData(raw, voxel_dimensions)
    print(density)

    cells_uniform = _create_cell_positions_uniform(density, density_factor=1.0)
    cells_poisson = _create_cell_positions_poisson_disc(density, density_factor=1.0)
    cells_ar = _create_cell_positions_accept_reject(density, density_factor=1.0, min_distance=100)

    d_u = distance_matrix(cells_uniform, cells_uniform)
    d_u = np.tril(d_u).flatten()
    d_u = d_u[d_u > 0]

    d_ar = distance_matrix(cells_ar, cells_ar)
    d_ar = np.tril(d_ar).flatten()
    d_ar = d_ar[d_ar > 0]

    plt.figure()
    plt.axvline(100, c="r")
    plt.hist(d_u, 100, histtype="step", label="uniform")
    plt.hist(d_ar, 100, histtype="step", label="accept reject")
    plt.xlabel("inter cell distance")
    plt.ylabel("number of pairs of cells")
    plt.savefig("inter_cell_distance.pdf")

    plt.figure()
    plt.scatter(cells_uniform[:, 0], cells_uniform[:, 2], s=1, c="C0", label="uniform")
    plt.scatter(cells_poisson[:, 0], cells_poisson[:, 2], s=1, c="blue", label="poisson")
    plt.scatter(cells_ar[:, 0], cells_ar[:, 2], s=1, c="C1", label="accept reject")

    plt.legend()
    plt.axis([0, 50 * 25, 0, 40 * 25])
    plt.savefig("cells.pdf")
