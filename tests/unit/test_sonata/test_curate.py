# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path

import h5py
import morphio
import numpy as np
import pytest

from brainbuilder.utils import bbp
from brainbuilder.utils.sonata import curate

TEST_DATA_PATH = Path(__file__).parent.parent / "data"
DATA_PATH = TEST_DATA_PATH / "sonata" / "curate"
NODES_FILE = DATA_PATH / "nodes.h5"
EDGES_FILE = DATA_PATH / "edges.h5"


def test_get_population_names(tmp_path):
    assert ["not-default"] == curate.get_population_names(NODES_FILE)
    assert ["not-default"] == curate.get_population_names(EDGES_FILE)

    nodes_copy_file = shutil.copy2(NODES_FILE, tmp_path)
    with h5py.File(nodes_copy_file, "r+") as h5f:
        del h5f["/nodes"]
    with pytest.raises(AssertionError):
        curate.get_population_names(nodes_copy_file)

    edges_copy_file = shutil.copy2(EDGES_FILE, tmp_path)
    with h5py.File(edges_copy_file, "r+") as h5f:
        del h5f["/edges"]
    with pytest.raises(AssertionError):
        curate.get_population_names(edges_copy_file)


def test_get_population_name(tmp_path):
    assert "not-default" == curate.get_population_name(NODES_FILE)
    assert "not-default" == curate.get_population_name(EDGES_FILE)

    with pytest.raises(ValueError):
        curate.get_population_name(EDGES_FILE, "unknown")

    nodes_copy_file = shutil.copy2(NODES_FILE, tmp_path)
    with h5py.File(nodes_copy_file, "r+") as h5f:
        h5f["nodes"].create_group("2nd_population")
    with pytest.raises(AssertionError):
        curate.get_population_name(nodes_copy_file)


def test_rename_node_population(tmp_path):
    nodes_copy_file = shutil.copy2(NODES_FILE, tmp_path)
    curate.rename_node_population(nodes_copy_file, "newname")
    assert ["newname"] == curate.get_population_names(nodes_copy_file)

    nodes_copy_file = shutil.copy2(NODES_FILE, tmp_path)
    curate.rename_node_population(nodes_copy_file, "newname", "not-default")
    assert ["newname"] == curate.get_population_names(nodes_copy_file)


def test_rename_edge_population(tmp_path):
    edges_copy_file = shutil.copy2(EDGES_FILE, tmp_path)
    curate.rename_edge_population(edges_copy_file, "newname")
    assert ["newname"] == curate.get_population_names(edges_copy_file)

    edges_copy_file = shutil.copy2(EDGES_FILE, tmp_path)
    curate.rename_edge_population(edges_copy_file, "newname", "not-default")
    assert ["newname"] == curate.get_population_names(edges_copy_file)


def test_add_edge_type_id(tmp_path):
    edges_copy_file = shutil.copy2(EDGES_FILE, tmp_path)
    with h5py.File(edges_copy_file, "r+") as h5f:
        del h5f["edges/not-default/edge_type_id"]
    curate.add_edge_type_id(edges_copy_file, "not-default")
    with h5py.File(edges_copy_file, "r") as h5f:
        edge_type_id = np.asarray(h5f["edges/not-default/edge_type_id"])
        assert (edge_type_id == -1).all()


def test_set_nodes_attribute(tmp_path):
    nodes_copy_file = shutil.copy2(NODES_FILE, tmp_path)
    with h5py.File(nodes_copy_file, "r+") as h5f:
        del h5f["nodes/not-default/0/model_type"]
    curate.set_group_attribute(
        nodes_copy_file, "nodes", "not-default", "0", "model_type", "biophysical"
    )
    with h5py.File(nodes_copy_file, "r") as h5f:
        assert [b"biophysical"] == h5f["nodes/not-default/0/@library/model_type"][:].tolist()
        model_type = np.asarray(h5f["nodes/not-default/0/model_type"])
        assert model_type.dtype == int
        assert (model_type == 0).all()


def test_set_edges_attribute(tmp_path):
    edges_copy_file = shutil.copy2(EDGES_FILE, tmp_path)
    curate.set_group_attribute(
        edges_copy_file, "edges", "not-default", "0", "syn_weight", 2.2, True
    )
    with h5py.File(edges_copy_file, "r") as h5f:
        syn_weight = np.asarray(h5f["edges/not-default/0/syn_weight"])
        assert syn_weight.dtype == float
        assert (syn_weight == 2.2).all()


def test_rewire_edge_population(tmp_path):
    nodes_copy_file = shutil.copy2(NODES_FILE, tmp_path)
    edges_copy_file = shutil.copy2(EDGES_FILE, tmp_path)

    curate.rename_node_population(nodes_copy_file, "newname")
    curate.rewire_edge_population(edges_copy_file, nodes_copy_file, nodes_copy_file, "chemical")
    expected_name = "newname__newname__chemical"
    assert [expected_name] == curate.get_population_names(edges_copy_file)
    with h5py.File(edges_copy_file, "r") as h5f:
        expected_name = "/edges/" + expected_name
        assert "newname" == h5f[expected_name]["source_node_id"].attrs["node_population"]
        assert "newname" == h5f[expected_name]["target_node_id"].attrs["node_population"]


def test_get_source_nodes_range():
    edge_population_name = curate.get_population_name(EDGES_FILE)
    start, end = curate.get_source_nodes_range(EDGES_FILE, edge_population_name)
    assert start == 0
    assert end == 2

    projection_file = DATA_PATH / "projection.h5"
    edge_population_name = curate.get_population_name(projection_file)
    start, end = curate.get_source_nodes_range(projection_file, edge_population_name)
    assert start == 10
    assert end == 11


def test_create_projection_source_nodes(tmp_path):
    projection_file = DATA_PATH / "projection.h5"
    source_nodes_file = curate.create_projection_source_nodes(
        projection_file, tmp_path, "projections", fix_offset=False
    )
    assert source_nodes_file.stem == "nodes_projections"
    assert [
        "projections",
    ] == curate.get_population_names(source_nodes_file)
    with h5py.File(source_nodes_file, "r") as h5f:
        assert [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] == h5f["/nodes/projections/0/model_type"][
            :
        ].tolist()

    source_nodes_file = curate.create_projection_source_nodes(
        projection_file, tmp_path, "projections", fix_offset=True
    )
    assert source_nodes_file.stem == "nodes_projections"
    assert [
        "projections",
    ] == curate.get_population_names(source_nodes_file)
    with h5py.File(source_nodes_file, "r") as h5f:
        assert [b"virtual", b"virtual"] == h5f["/nodes/projections/0/model_type"][:].tolist()


def test_correct_source_nodes_offset(tmp_path):
    shutil.copy2(DATA_PATH / "projection.h5", tmp_path)
    edges_file = tmp_path / "projection.h5"
    curate.correct_source_nodes_offset(edges_file, edge_population_name="not-default", offset=10)
    with h5py.File(edges_file, "r") as h5f:
        assert h5f["/edges/not-default/0/syn_weight"].shape == (4,)
        assert [0, 0, 1, 1] == h5f["/edges/not-default/source_node_id"][:].tolist()


def test_merge_h5_files(tmp_path):
    nodes_copy_file = shutil.copy2(NODES_FILE, tmp_path)
    curate.rename_node_population(nodes_copy_file, "newname")

    merged_file = tmp_path / "merged_nodes.h5"
    curate.merge_h5_files([NODES_FILE, nodes_copy_file], "nodes", merged_file)
    assert ["newname", "not-default"] == curate.get_population_names(merged_file)
    with h5py.File(merged_file, "r") as h5f:
        assert "/nodes/not-default/0" in h5f
        assert "/nodes/newname/0" in h5f


def test__has_unifurcations():
    morph = morphio.Morphology(DATA_PATH / "wrong-order-with-unifurcations.h5")
    assert curate._has_unifurcations(morph)


def test__has_sonata_ordering():
    morph = morphio.Morphology(DATA_PATH / "wrong-order-with-unifurcations.h5")
    assert not curate._has_sonata_ordering(morph)


def test_check_morphology_invariants():
    morph_names = bbp.load_extneurondb(str(DATA_PATH / "neurondbExt.dat")).morphology.to_list()
    incorrect_ordering, have_unifurcations = curate.check_morphology_invariants(
        DATA_PATH, morph_names
    )
    assert incorrect_ordering == {"wrong-order-with-unifurcations"}
    assert have_unifurcations == {"wrong-order-with-unifurcations"}


def test__update_dtype(tmp_path):
    with h5py.File(tmp_path / "test__update_dtype.h5", "w") as h5:
        dset = h5.create_dataset("ints", (100,), dtype="i8")
        dset.attrs["foo"] = "bar"
        dset.attrs["bar"] = "baz"
        name, dtype = curate._update_dtype(h5, "ints", np.float32)
        assert name == "/ints"
        assert dtype == np.float32
        assert h5["ints"].dtype == np.float32
        assert {"foo": "bar", "bar": "baz"} == dict(h5["ints"].attrs)


def test_update_node_dtypes(tmp_path):
    nodes_copy_file = shutil.copy2(NODES_FILE, tmp_path)
    with h5py.File(nodes_copy_file, "r+") as h5:
        data = h5["/nodes/not-default/0/dynamics_params/holding_current"].astype(int)
        del h5["/nodes/not-default/0/dynamics_params/holding_current"]
        h5.create_dataset("/nodes/not-default/0/dynamics_params/holding_current", data=data)
    converted = curate.update_node_dtypes(nodes_copy_file, "not-default", "biophysical")
    assert converted["/nodes/not-default/0/x"] == np.float32  # was np.float64
    assert converted["/nodes/not-default/0/rotation_angle_xaxis"] == np.float32  # was np.float64
    assert converted["/nodes/not-default/0/dynamics_params/holding_current"] == np.float32


def test_update_edge_dtypes(tmp_path):
    edges_copy_file = shutil.copy2(EDGES_FILE, tmp_path)
    converted = curate.update_edge_dtypes(edges_copy_file, "not-default", "chemical", virtual=False)
    assert converted["/edges/not-default/0/efferent_surface_z"] == np.float32
    assert converted["/edges/not-default/edge_type_id"] == np.int64
