# SPDX-License-Identifier: Apache-2.0
import numpy as np
import voxcell

from brainbuilder.app import cells as test_module
from brainbuilder.cell_positions import _get_cell_count


def test_load_density__dangerously_low_densities(tmp_path):
    """Test for very low densities where the float precision affects the total count."""

    shape = (10, 10, 10)
    voxel_dimensions = np.array([25, 25, 25])
    filepath = tmp_path / "test_load_density__dangerously_low_densities.nrrd"
    raw = np.full(shape, dtype=np.float64, fill_value=2.04160691e02)

    raw[1, :, 1] = 8.36723867e10

    density = voxcell.VoxelData(raw=raw, voxel_dimensions=voxel_dimensions)
    density.save_nrrd(filepath)

    loaded_density = density.with_data(
        test_module._load_density(str(filepath), mask=np.ones(shape, int), atlas=None)
    )

    _, count = _get_cell_count(loaded_density, 1.0)

    assert count == 13073813


def test_load_density__near_zero_values_are_ignored(tmp_path):
    """Test that values smaller than 1e-7 are ignored."""
    shape = (3, 3, 3)
    voxel_dimensions = np.array([25, 25, 25])

    filepath = tmp_path / "test_load_density__near_zero_values_are_ignored.nrrd"
    raw = np.zeros(shape, dtype=np.float64)

    raw[:, 0, :] = -0.001
    raw[:, 1, :] = 1e-8
    raw[:, 2, :] = 10.0

    density = voxcell.VoxelData(raw=raw, voxel_dimensions=voxel_dimensions)
    density.save_nrrd(filepath)

    mask = np.ones_like(shape, dtype=bool)
    result = test_module._load_density(str(filepath), mask, atlas=None)

    # Non close to zero negative and positive values remain
    assert np.count_nonzero(result < 0.0) == 9
    assert np.count_nonzero(result > 0.0) == 9

    # Close to zero values are zeroed
    assert np.count_nonzero(result == 1e-8) == 0

    # Sanity check for the remaining entries
    assert np.count_nonzero(result) == 18
