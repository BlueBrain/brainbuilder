# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import scipy.spatial.distance as distance

import brainbuilder.poisson_disc_sampling as test_module
from brainbuilder.exceptions import BrainBuilderError


@pytest.fixture
def setup_func():
    np.random.seed(42)


def test_grid_empty_cell():
    domain = np.array([[0, 0, 0], [100, 200, 500]])
    grid = test_module.Grid(domain, 100)
    # mark some cells as non-empty
    grid.grid[0, 0, 0] = 0
    grid.grid[0, 1, 3] = 0
    grid.grid[0, 2, 4] = 0

    empty_cell = grid.get_random_empty_grid_cell()

    assert grid.grid[empty_cell] == -1


def test_grid_no_empty_cell():
    domain = np.array([[0, 0, 0], [100, 200, 500]])
    grid = test_module.Grid(domain, 100)
    # no empty cells
    grid.grid = np.ones(grid.grid.shape)

    with pytest.raises(BrainBuilderError):
        grid.get_random_empty_grid_cell()


def test_generate_random_point_in_empty_cell():
    domain = np.array([[0, 0, 0], [100, 200, 500]])
    grid = test_module.Grid(domain, 100)
    # mark some cells as non-empty
    grid.grid[0, 0, 0] = 0
    grid.grid[0, 1, 3] = 0
    grid.grid[0, 2, 4] = 0

    point = grid.generate_random_point_in_empty_grid_cell()

    grid_point = grid.get_grid_coords(point)
    assert grid.grid[grid_point] == -1


def test_generate_random_point_in_empty_cell2():
    domain = np.array([[0, 0, 0], [100, 200, 500]])
    grid = test_module.Grid(domain, 100)
    # no empty cells
    grid.grid = np.ones(grid.grid.shape)

    with pytest.raises(BrainBuilderError):
        grid.generate_random_point_in_empty_grid_cell()


def test_generate_points(setup_func):
    domain = np.array([[0, 0, 0], [100, 200, 500]])
    nb_points = 20
    min_distance = 5
    seed = np.array([0, 0, 0])

    def min_distance_func(point=None):
        return min_distance

    points = test_module.generate_points(domain, nb_points, min_distance_func, seed)

    assert len(points) == nb_points
    assert np.all(np.equal(seed, points[0]))
    min_distance_between_pts = np.min(distance.pdist(points).flatten())
    assert min_distance <= min_distance_between_pts
    for point in points:
        assert np.all(point >= domain[0, :]) and np.all(point <= domain[1, :])


def test_generate_points_too_many(setup_func):
    domain = np.array([[0, 0, 0], [10, 20, 5]])
    nb_points = 1000
    min_distance = 1
    seed = np.array([0, 0, 0])

    def min_distance_func(point=None):
        return min_distance

    points = test_module.generate_points(domain, nb_points, min_distance_func, seed)

    assert len(points) < nb_points
    min_distance_between_pts = np.min(distance.pdist(points).flatten())
    assert min_distance <= min_distance_between_pts
    for point in points:
        assert np.all(point >= domain[0, :]) and np.all(point <= domain[1, :])


def test_generate_points_random_seed(setup_func):
    domain = np.array([[0, 0, 0], [100, 200, 50]])
    nb_points = 20
    min_distance = 5

    def min_distance_func(point=None):
        return min_distance

    points = test_module.generate_points(domain, nb_points, min_distance_func)

    assert len(points) == nb_points
    min_distance_between_pts = np.min(distance.pdist(points).flatten())
    assert min_distance <= min_distance_between_pts
    for point in points:
        assert np.all(point >= domain[0, :]) and np.all(point <= domain[1, :])


def test_generate_points_random_seed_neg_domain(setup_func):
    domain = np.array([[-50, -100, -25], [50, 100, 25]])
    nb_points = 20
    min_distance = 5

    def min_distance_func(point=None):
        return min_distance

    points = test_module.generate_points(domain, nb_points, min_distance_func)

    assert len(points) == nb_points
    min_distance_between_pts = np.min(distance.pdist(points).flatten())
    assert min_distance <= min_distance_between_pts
    for point in points:
        assert np.all(point >= domain[0, :]) and np.all(point <= domain[1, :])


def test_generate_points_random_seed_neg_domain_2(setup_func):
    domain = np.array([[-50, -100, -25], [-150, 100, 25]])
    nb_points = 20
    min_distance = 5

    def min_distance_func(point=None):
        return min_distance

    points = test_module.generate_points(domain, nb_points, min_distance_func)

    assert len(points) == nb_points
    min_distance_between_pts = np.min(distance.pdist(points).flatten())
    assert min_distance <= min_distance_between_pts
    for point in points:
        assert (point[0] <= domain[0, 0]) and (point[0] >= domain[1, 0])
        assert np.all(point[1:] >= domain[0, 1:]) and np.all(point[1:] <= domain[1, 1:])
