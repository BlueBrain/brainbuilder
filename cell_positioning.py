'''task that positions cell bodies from a density distribution'''
from brainbuilder.utils import genbrain as gb
import numpy as np
from scipy.stats import itemfreq  # pylint: disable=E0611

import logging
L = logging.getLogger(__name__)


def _unique_with_counts(array):
    '''return two arrays: the unique values of array and the number of times they appear
    this is equivalent to np.unique(array, return_counts=True)
    However, this is a numpy 1.9 function so we need a custom implementation to run on numpy 1.8
    '''
    if array.shape != (0,):
        unique, counts = tuple(itemfreq(array).transpose())
    else:
        unique, counts = (np.array([]), np.array([]))

    return unique.astype(array.dtype), counts.astype(np.int)


def assign_cell_counts(cell_body_density_raw, total_cell_count):
    '''create a matrix of the same dimensions of the cell_body_density where each value
    is an integer with the number of cells expected to ocurr in that volume'''

    all_voxels = np.arange(cell_body_density_raw.size)

    probs = cell_body_density_raw.flatten().astype(np.float64)
    probs /= np.sum(probs)

    chosen_voxels = np.random.choice(all_voxels, size=total_cell_count, p=probs)

    unique, counts = _unique_with_counts(chosen_voxels)

    chosen_indexes = np.unravel_index(unique, cell_body_density_raw.shape)

    assigned = np.zeros_like(cell_body_density_raw)

    assigned[chosen_indexes] = counts

    assert np.sum(assigned) == total_cell_count

    return assigned.astype(np.int)


def cell_counts_to_cell_voxel_indices(cell_counts_per_voxel):
    '''take a matrix with an element per voxel that represents the number of cells in that space
    and return a matrix with a row for each cell where the values represent its corresponding
    voxel's X Y Z indices'''

    idx = np.nonzero(cell_counts_per_voxel)
    repeats = cell_counts_per_voxel[idx]
    locations = np.repeat(np.array(idx).transpose(), repeats, 0)

    return locations


def cell_positioning(density_raw, voxel_dimensions, total_cell_count):
    '''
    Accepts:
        density: voxel data from Allen Brain Institute.
            Called "atlasVolume" in their website.
            Each voxel represents a value that once normalised, can be treated as a probability of
            cells appearing in this voxel.
        voxel_dimensions: tuple with the size of the voxels in microns in each axis
        total_cell_count: an int

    Returns:
        positions: list of positions for soma centers (x, y, z).
    '''

    cell_counts_per_voxel = assign_cell_counts(density_raw, total_cell_count)

    assert np.sum(cell_counts_per_voxel) == total_cell_count, \
        '%s != %s' % (np.sum(cell_counts_per_voxel), total_cell_count)

    cell_voxel_indices = cell_counts_to_cell_voxel_indices(cell_counts_per_voxel)

    positions = gb.cell_voxel_indices_to_positions(cell_voxel_indices, voxel_dimensions)

    return positions