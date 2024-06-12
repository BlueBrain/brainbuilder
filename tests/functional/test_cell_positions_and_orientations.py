# SPDX-License-Identifier: Apache-2.0
"""test positions_and_orientations"""

from unittest.mock import patch

import h5py
import numpy as np
import numpy.testing as npt
from click.testing import CliRunner
from voxcell import CellCollection, VoxelData  # type: ignore

import brainbuilder.app.cells as tested
from brainbuilder.utils import dump_yaml


def get_result(runner):
    return runner.invoke(
        tested.positions_and_orientations,
        [
            "--annotation-path",
            "annotation.nrrd",
            "--orientation-path",
            "orientation.nrrd",
            "--config-path",
            "config.yaml",
            "--output-path",
            "positions_and_orientations.h5",
        ],
    )


def create_density_configuration():
    config = {
        "inputDensityVolumePath": {
            "inhibitory neuron": "inhibitory_neuron_density.nrrd",
            "excitatory neuron": "excitatory_neuron_density.nrrd",
            "oligodendrocyte": "oligodendrocyte_density.nrrd",
            "astrocyte": "astrocyte_density.nrrd",
            "microglia": "microglia_density.nrrd",
        }
    }

    return config


def create_input():
    input_ = {
        "annotation": np.array(
            [
                [[512, 512, 1143]],
                [[512, 512, 1143]],
                [[477, 56, 485]],
            ],
            dtype=np.uint32,
        ),
        "orientation": np.array(
            [
                [
                    [
                        [1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                    ]
                ],
                [
                    [
                        [0.0, 1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0],
                        [1, 0, 0, 1],
                        [1, 0, 0, 0],
                    ]
                ],
                [
                    [
                        [0, 0, 0, 1],
                        [0.0, 1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)],
                        [1, 0, 0, 0],
                    ]
                ],
            ]
        ),
        "inhibitory neuron": np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0]],
            ]
        ),
        "excitatory neuron": np.array(
            [
                [[0.0, 1.0, 0.0]],
                [[9.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0]],
            ]
        ),
        "astrocyte": np.array(
            [
                [[0.0, 1.0, 5.0]],
                [[1.0, 4.0, 5.0]],
                [[0.0, 1.0, 0.0]],
            ]
        ),
        "microglia": np.array(
            [
                [[0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0]],
            ]
        ),
        "oligodendrocyte": np.array(
            [
                [[1.0, 1.0, 0.0]],
                [[2.0, 1.0, 4.0]],
                [[0.0, 0.0, 0.0]],
            ]
        ),
    }

    return input_


def test_positions_and_orientations_valid_input():
    voxel_dimensions = [25] * 3  # a voxel of size 25um x 25um x 25um
    input_ = create_input()
    config = create_density_configuration()
    runner = CliRunner()
    with runner.isolated_filesystem():
        dump_yaml("config.yaml", config)
        for cell_type, path in config["inputDensityVolumePath"].items():
            # the input densities are expressed in number of cells per voxel
            VoxelData(
                input_[cell_type] * (1e9 / 25**3), voxel_dimensions=voxel_dimensions
            ).save_nrrd(path)
        for input_voxel_data in ["annotation", "orientation"]:
            VoxelData(input_[input_voxel_data], voxel_dimensions=voxel_dimensions).save_nrrd(
                input_voxel_data + ".nrrd"
            )
        result = get_result(runner)
        assert result.exit_code == 0

        # Check sanity through voxcell.CellCollection interface
        cell_collection = CellCollection.load_sonata("positions_and_orientations.h5")
        npt.assert_array_equal(cell_collection.positions.shape, (43, 3))
        # CellCollection orientations are 3 x 3 orthogonal matrices
        npt.assert_array_equal(cell_collection.orientations.shape, (43, 3, 3))
        properties = cell_collection.properties
        assert properties["region_id"].dtype == np.uint32
        npt.assert_array_equal(
            properties["cell_type"].dtype.categories,
            [
                "astrocyte",
                "excitatory neuron",
                "inhibitory neuron",
                "microglia",
                "oligodendrocyte",
            ],
        )

        # Check directly through the h5py interface
        cell_collection = h5py.File("positions_and_orientations.h5", "r")
        npt.assert_array_almost_equal(
            cell_collection.get("/nodes/atlas_cells/0/orientation_x")[()].shape, (43,)
        )
        npt.assert_array_equal(cell_collection.get("/nodes/atlas_cells/0/y")[()].shape, (43,))
        assert cell_collection.get("/nodes/atlas_cells/0/orientation_y")[()].dtype == np.float32
        assert cell_collection.get("/nodes/atlas_cells/0/z")[()].dtype == np.float32
        assert cell_collection.get("/nodes/atlas_cells/0/region_id")[()].dtype == np.uint32
        assert cell_collection.get("/nodes/atlas_cells/0/cell_type")[()].dtype == np.uint32
        # In order to retrieve cell types, we need to use an intermediate dataset,
        # namely /nodes/atlas_cells/0/@library/cell_type which map uint32 index to strings.
        npt.assert_array_equal(
            cell_collection.get("/nodes/atlas_cells/0/cell_type")[()].shape, (43,)
        )
        npt.assert_array_equal(
            cell_collection.get("/nodes/atlas_cells/0/@library/cell_type"),
            np.array(
                [
                    b"astrocyte",
                    b"excitatory neuron",
                    b"inhibitory neuron",
                    b"microglia",
                    b"oligodendrocyte",
                ],
                dtype=object,
            ),
        )


def test_positions_and_orientations_invalid_input():
    config = create_density_configuration()
    input_ = create_input()
    runner = CliRunner()
    with runner.isolated_filesystem():
        dump_yaml("config.yaml", config)
        for cell_type, path in config["inputDensityVolumePath"].items():
            # the input densities are expressed in number of cells per voxel
            VoxelData(input_[cell_type] * (1e9 / 25**3), voxel_dimensions=[25] * 3).save_nrrd(path)
        for input_voxel_data in ["annotation", "orientation"]:
            # Intentional mismatch of voxel dimensions: 10um != 25um
            VoxelData(input_[input_voxel_data], voxel_dimensions=[10] * 3).save_nrrd(
                input_voxel_data + ".nrrd"
            )
        result = get_result(runner)
        assert result.exit_code == 1
        assert "different voxel dimensions" in str(result.exception)


@patch("brainbuilder.app.cells.L.warning")
def test_positions_and_orientations_negative_density(L_warning_mock):
    config = create_density_configuration()
    input_ = create_input()
    input_["microglia"][0, 0, 1] = -1.0  # negative density value on purpose
    runner = CliRunner()
    with runner.isolated_filesystem():
        dump_yaml("config.yaml", config)
        for cell_type, path in config["inputDensityVolumePath"].items():
            # the input densities are expressed in number of cells per voxel
            VoxelData(input_[cell_type] * (1e9 / 25**3), voxel_dimensions=[25] * 3).save_nrrd(path)
        for input_voxel_data in ["annotation", "orientation"]:
            VoxelData(input_[input_voxel_data], voxel_dimensions=[25] * 3).save_nrrd(
                input_voxel_data + ".nrrd"
            )
        result = get_result(runner)
        assert result.exit_code == 0
        assert "Negative density values" in L_warning_mock.call_args[0][0]
