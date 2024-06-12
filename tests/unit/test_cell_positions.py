# SPDX-License-Identifier: Apache-2.0
import numpy as np
import numpy.testing as npt
import scipy.spatial.distance as distance
from pytest import raises
from voxcell import VoxelData

import brainbuilder.cell_positions as test_module


def test_create_cell_positions_1():
    density = VoxelData(1000 * np.ones((3, 3, 3)), voxel_dimensions=(100, 100, 100))
    result = test_module.create_cell_positions(density)
    assert np.shape(result) == (27, 3)
    assert np.all((result >= 0) & (result <= 3 * 100))


def test_create_cell_positions_2():
    density = VoxelData(1000 * np.ones((3, 3, 3)), voxel_dimensions=(100, 100, 100))
    result = test_module.create_cell_positions(density, density_factor=0.2)
    assert np.shape(result) == (5, 3)


def test_create_cell_positions_reproducible():
    density = VoxelData(250 * np.ones((3, 3, 3)), voxel_dimensions=(10, 10, 10))
    density.raw[0:5, 0:5, 0:5] = 100
    density.raw[0, 0, 0] = 1500
    result_1 = test_module.create_cell_positions(density, density_factor=200, seed=0)
    result_2 = test_module.create_cell_positions(density, density_factor=200, seed=0)
    npt.assert_array_equal(result_1, result_2)


def test_create_equidistributed_cell_positions_1():
    density = VoxelData(1000 * np.ones((3, 3, 3)), voxel_dimensions=(100, 100, 100))
    max_expected_nb_points = 27

    result = test_module.create_cell_positions(density, method="poisson_disc")

    assert result.shape[0] <= max_expected_nb_points
    assert np.all((result >= 0) & (result <= 3 * 100))

    min_distance = 0.84 * 100
    min_distance_between_pts = np.min(distance.pdist(result).flatten())
    assert min_distance <= min_distance_between_pts


def test_create_equidistributed_cell_positions_2():
    density = VoxelData(1000 * np.ones((3, 3, 3)), voxel_dimensions=(100, 100, 100))
    max_expected_nb_points = 5

    result = test_module.create_cell_positions(density, density_factor=0.2, method="poisson_disc")

    assert result.shape[0] <= max_expected_nb_points
    assert np.all((result >= 0) & (result <= 3 * 100))

    min_distance = 0.84 * 100
    min_distance_between_pts = np.min(distance.pdist(result).flatten())
    assert min_distance <= min_distance_between_pts


def test_create_cell_positions_black_white():
    density = VoxelData(np.zeros((3, 3, 3)), voxel_dimensions=(100, 100, 100))
    density.raw[1, 1, 1] = 27000
    max_expected_nb_points = 27

    result = test_module.create_cell_positions(density, method="poisson_disc")

    assert result.shape[0] <= max_expected_nb_points
    assert np.all((result >= 100) & (result <= 2 * 100))

    min_distance = 0.84 * 100.0 / np.power(max_expected_nb_points, 1.0 / 3.0)
    min_distance_between_pts = np.min(distance.pdist(result).flatten())
    assert min_distance <= min_distance_between_pts


def test_create_cell_positions_black_grey():
    density = VoxelData(1000 * np.ones((3, 3, 3)), voxel_dimensions=(100, 100, 100))
    density.raw[1, 1, 1] = 4000
    max_expected_nb_points = 30

    result = test_module.create_cell_positions(density, method="poisson_disc")

    assert result.shape[0] <= max_expected_nb_points
    assert np.all((result >= 0) & (result <= 3 * 100))
    assert not np.all((result >= 100) & (result <= 2 * 100))

    # max expected nb points in middle voxel: 4
    min_distance = 0.84 * 100.0 / np.power(4.0, 1.0 / 3.0)
    min_distance_between_pts = np.min(distance.pdist(result).flatten())
    assert min_distance <= min_distance_between_pts


def test_create_cell_positions_negative_density():
    density = VoxelData(np.ones((3, 3, 3)), voxel_dimensions=(100, 100, 100))
    density.raw[1, 1, 1] = -1.0
    with raises(Exception):
        test_module.create_cell_positions(density)


def test_create_cell_positions__zero_counts__basic():
    """Test that algorithm outputs an (0, 3) array when counts are zero."""
    density = VoxelData(np.zeros((3, 3, 3)), voxel_dimensions=(10, 10, 10))
    result = test_module.create_cell_positions(density)
    assert result.shape == (0, 3) and result.dtype == np.float32


def test_create_cell_positions__zero_counts__poisson_disc():
    """Test that algorithm outputs an (0, 3) array when counts are zero."""
    density = VoxelData(np.zeros((3, 3, 3)), voxel_dimensions=(10, 10, 10))
    result = test_module.create_cell_positions(density, method="poisson_disc")
    assert result.shape == (0, 3) and result.dtype == np.float32


def test_get_bbox_indices_nonzero_entries():
    data = np.array(
        [
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    bbox = np.array([[1, 1, 2], [3, 2, 3]])

    result = test_module.get_bbox_indices_nonzero_entries(data)

    assert np.array_equal(result, bbox)


def test_get_bbox_nonzero_entries():
    data = np.array(
        [
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    bbox = np.array([[0.0, 0.0, 0.0], [50.0, 15.0, -4.0]])
    bbox_nonzero = np.array([[10.0, 5.0, -2.0], [40.0, 15.0, -4.0]])
    voxel_dimensions = np.array([10.0, 5.0, -1.0])

    result = test_module.get_bbox_nonzero_entries(data, bbox, voxel_dimensions)

    assert np.array_equal(result, bbox_nonzero)
