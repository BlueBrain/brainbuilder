# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import utils
from numpy.testing import assert_array_equal

from brainbuilder.utils import load_json
from brainbuilder.utils.sonata import split_population

DATA_PATH = (Path(__file__).parent / "../data/sonata/split_population/").resolve()


def test__get_population_name():
    assert "src__dst__chemical" == split_population._get_population_name(src="src", dst="dst")
    assert "src" == split_population._get_population_name(src="src", dst="src")


def test__get_unique_population():
    nodes = DATA_PATH / "split_subcircuit" / "networks" / "nodes" / "nodes.h5"
    with h5py.File(nodes, "r") as h5:
        with pytest.raises(ValueError):
            split_population._get_unique_population(h5["nodes"])

    nodes = DATA_PATH / "nodes.h5"
    with h5py.File(nodes, "r") as h5:
        assert split_population._get_unique_population(h5["nodes"]) == "default"


def test__get_unique_group(tmp_path):
    nodes = DATA_PATH / "nodes.h5"
    with h5py.File(nodes, "r") as h5:
        parent = h5["nodes/default"]
        assert split_population._get_unique_group(parent)

    with h5py.File(tmp_path / "nodes.h5", "w") as h5:
        parent = h5.create_group("/edges/")
        parent.create_group("/pop_name/0")
        parent.create_group("/pop_name/1")
        with pytest.raises(ValueError):
            split_population._get_unique_group(parent)


def test__write_nodes(tmp_path):
    split_nodes = {
        "A": pd.DataFrame({"fake_prop": range(10)}, index=np.arange(10)),
        "B": pd.DataFrame({"fake_prop": range(5)}, index=np.arange(10, 15)),
    }
    split_population._write_nodes(tmp_path, split_nodes)
    assert (tmp_path / "nodes_A.h5").exists()
    assert (tmp_path / "nodes_B.h5").exists()

    with h5py.File(tmp_path / "nodes_A.h5", "r") as h5:
        assert_array_equal(h5["/nodes/A/0/fake_prop"], np.arange(10))
        assert_array_equal(h5["/nodes/A/node_type_id"], np.full(10, -1))
    with h5py.File(tmp_path / "nodes_B.h5", "r") as h5:
        assert_array_equal(h5["/nodes/B/0/fake_prop"], np.arange(5))
        assert_array_equal(h5["/nodes/B/node_type_id"], np.full(5, -1))


def test__get_node_id_mapping():
    split_nodes = {
        "A": pd.DataFrame(index=np.arange(0, 10)),
        "B": pd.DataFrame(index=np.arange(10, 15)),
    }
    ret = split_population._get_node_id_mapping(split_nodes)
    assert len(ret) == 2
    assert ret["A"].new_id.to_list() == list(range(10))
    assert ret["B"].new_id.to_list() == list(range(5))


def test__split_population_by_attribute():
    # nodes.h5 contains 3 nodes with mtypes "L2_X", "L6_Y", "L6_Y"
    nodes_path = DATA_PATH / "nodes.h5"
    ret = split_population._split_population_by_attribute(nodes_path, "mtype")
    assert len(ret) == 2
    assert isinstance(ret["L2_X"], pd.DataFrame)

    assert len(ret["L2_X"]) == 1
    assert ret["L2_X"].mtype.unique()[0] == "L2_X"
    assert_array_equal(ret["L2_X"].index, [0])

    assert len(ret["L6_Y"]) == 2
    assert ret["L6_Y"].mtype.unique()[0] == "L6_Y"
    assert_array_equal(ret["L6_Y"].index, [1, 2])


def test__write_circuit_config(tmp_path):
    split_nodes = {
        "A": pd.DataFrame(index=np.arange(0, 10)),
        "B": pd.DataFrame(index=np.arange(10, 15)),
    }
    split_population._write_circuit_config(tmp_path, split_nodes)
    ret = load_json(tmp_path / "circuit_config.json")
    assert "manifest" in ret
    assert "networks" in ret
    assert "nodes" in ret["networks"]
    assert "edges" in ret["networks"]
    assert len(ret["networks"]["edges"]) == 0  # no edge files

    open(tmp_path / "edges_A.h5", "w").close()
    open(tmp_path / "edges_B.h5", "w").close()
    open(tmp_path / "edges_A__B__chemical.h5", "w").close()
    split_population._write_circuit_config(tmp_path, split_nodes)
    ret = load_json(tmp_path / "circuit_config.json")
    assert len(ret["networks"]["edges"]) == 3


@pytest.mark.parametrize(
    "id_mapping, h5_read_chunk_size, expected_dir",
    [
        (
            {
                # edges: A -> B (2), B -> A, B -> B
                "A": pd.DataFrame({"new_id": np.arange(4)}, index=[5, 4, 3, 0]),
                "B": pd.DataFrame({"new_id": np.arange(2)}, index=[1, 2]),
            },
            10,
            DATA_PATH / "01",
        ),
        (
            {
                # edges: A -> A (4)
                "A": pd.DataFrame({"new_id": np.arange(4)}, index=[3, 2, 1, 0]),
                "B": pd.DataFrame({"new_id": np.arange(2)}, index=[5, 4]),
            },
            10,
            DATA_PATH / "02",
        ),
        (
            {
                # edges: B -> B (4), reduced chunk size
                "A": pd.DataFrame({"new_id": np.arange(3)}, index=[5, 4, 3]),
                "B": pd.DataFrame({"new_id": np.arange(3)}, index=[2, 1, 0]),
            },
            3,
            DATA_PATH / "03",
        ),
        (
            {
                # edges: A -> A, A -> B (3)
                "A": pd.DataFrame({"new_id": np.arange(4)}, index=[2, 0, 4, 5]),
                "B": pd.DataFrame({"new_id": np.arange(2)}, index=[1, 3]),
            },
            10,
            DATA_PATH / "04",
        ),
        (
            {
                # edges: B -> B, B -> A (3)
                "A": pd.DataFrame({"new_id": np.arange(4)}, index=[1, 3, 4, 5]),
                "B": pd.DataFrame({"new_id": np.arange(2)}, index=[2, 0]),
            },
            10,
            DATA_PATH / "05",
        ),
    ],
)
def test__write_edges(tmp_path, id_mapping, h5_read_chunk_size, expected_dir):
    # edges.h5 contains the following edges:
    # '/edges/default/source_node_id': [2, 0, 0, 2]
    # '/edges/default/target_node_id': [0, 1, 1, 1]
    edges_path = DATA_PATH / "edges.h5"
    # iterate over different id_mappings to split the edges in different ways
    split_population._write_edges(
        tmp_path,
        edges_path,
        id_mapping,
        expect_to_use_all_edges=True,
        h5_read_chunk_size=h5_read_chunk_size,
    )
    utils.assert_h5_dirs_equal(tmp_path, expected_dir, pattern="edges_*.h5")


def test_split_population(tmp_path):
    attribute = "mtype"
    nodes_path = DATA_PATH / "nodes.h5"
    edges_path = DATA_PATH / "edges.h5"
    expected_dir = DATA_PATH / "00"

    split_population.split_population(tmp_path, attribute, nodes_path, edges_path)
    utils.assert_h5_dirs_equal(tmp_path, expected_dir)
    utils.assert_json_files_equal(
        tmp_path / "circuit_config.json", expected_dir / "circuit_config.json"
    )


def test__split_population_by_node_set():
    nodes_path = DATA_PATH / "nodes.h5"
    node_set_name = "L2_X"
    node_set_path = DATA_PATH / "node_sets.json"

    ret = split_population._split_population_by_node_set(nodes_path, node_set_name, node_set_path)

    assert len(ret) == 1
    assert isinstance(ret["L2_X"], pd.DataFrame)

    assert len(ret["L2_X"]) == 1
    assert ret["L2_X"].mtype.unique()[0] == "L2_X"
    assert_array_equal(ret["L2_X"].index, [0])


def test_simple_split_subcircuit(tmp_path):
    nodes_path = DATA_PATH / "nodes.h5"
    edges_path = DATA_PATH / "edges.h5"
    node_set_name = "L6_Y"
    node_set_path = DATA_PATH / "node_sets.json"

    split_population.simple_split_subcircuit(
        tmp_path, node_set_name, node_set_path, nodes_path, edges_path
    )

    assert (tmp_path / "nodes_L6_Y.h5").exists()
    with h5py.File(tmp_path / "nodes_L6_Y.h5", "r") as h5:
        population = h5["nodes/L6_Y/"]
        assert list(population["node_type_id"]) == [-1, -1]
        assert len(population["0/layer"]) == 2

    assert (tmp_path / "edges_L6_Y.h5").exists()
    with h5py.File(tmp_path / "edges_L6_Y.h5", "r") as h5:
        group = h5["edges/L6_Y/"]
        assert list(group["source_node_id"]) == [1]
        assert list(group["target_node_id"]) == [0]


def test__gather_layout_from_networks():
    res = split_population._gather_layout_from_networks({"nodes": [], "edges": []})
    assert res == ({}, {})

    nodes, edges = split_population._gather_layout_from_networks(
        {
            "nodes": [
                {
                    "nodes_file": "a/b/a.h5",
                    "populations": {"a": {"type": "biophysical"}},
                },
                {
                    "nodes_file": "a/b/bc.h5",
                    "populations": {"b": {"type": "biophysical"}, "c": {"type": "biophysical"}},
                },
                {
                    "nodes_file": "a/b/a.h5",
                    "populations": {"A": {"type": "biophysical"}},
                },
            ],
            "edges": [
                {
                    "edges_file": "a/b/a.h5",
                    "populations": {"a_a": {"type": "biophysical"}},
                },
                {
                    "edges_file": "a/b/bc.h5",
                    "populations": {"b_c": {"type": "biophysical"}, "c_b": {"type": "biophysical"}},
                },
                {
                    "edges_file": "a/a/bc.h5",
                    "populations": {"a_c": {"type": "biophysical"}, "a_b": {"type": "biophysical"}},
                },
                {
                    "edges_file": "a/b/a.h5",
                    "populations": {"A_a": {"type": "biophysical"}},
                },
            ],
        }
    )
    assert nodes == {
        "A": "A/a.h5",
        "a": "a/a.h5",
        "b": "b/bc.h5",
        "c": "b/bc.h5",
    }
    assert edges == {
        "A_a": "A_a/a.h5",
        "a_a": "a_a/a.h5",
        "a_b": "a/bc.h5",
        "a_c": "a/bc.h5",
        "b_c": "b/bc.h5",
        "c_b": "b/bc.h5",
    }

    nodes, edges = split_population._gather_layout_from_networks(
        {
            "nodes": [
                {
                    "nodes_file": "a/b/bc.h5",
                    "populations": {"b": {"type": "biophysical"}, "c": {"type": "biophysical"}},
                },
                {
                    "nodes_file": "a/b/bc.h5",
                    "populations": {"B": {"type": "biophysical"}, "C": {"type": "biophysical"}},
                },
            ],
            "edges": [],
        }
    )
    assert nodes == {"B": "b/bc.h5", "C": "b/bc.h5", "b": "b/bc.h5", "c": "b/bc.h5"}

    nodes, edges = split_population._gather_layout_from_networks(
        {
            "nodes": [
                {
                    "nodes_file": "a/b/a.h5",
                    "populations": {"a": {"type": "biophysical"}},
                },
                {
                    "nodes_file": "a/b/bc.h5",
                    "populations": {"b": {"type": "biophysical"}},
                },
                {
                    "nodes_file": "a/b/bc.h5",
                    "populations": {"c": {"type": "biophysical"}},
                },
                {
                    "nodes_file": "a/b/a.h5",
                    "populations": {"A": {"type": "biophysical"}},
                },
            ],
            "edges": [],
        }
    )
    assert nodes == {
        "A": "A/a.h5",
        "a": "a/a.h5",
        "b": "b/bc.h5",
        "c": "c/bc.h5",
    }


def test__update_node_sets():
    ret = split_population._update_node_sets(node_sets={}, id_mapping={})
    assert ret == {}

    node_sets = {
        "CopiedNoNodeIds": ["All"],
        "MissingPopluationNotCopied": {"node_id": [15, 280, 397, 509, 555, 624, 651, 789]},
        "HasPopulationCopied": {
            "population": "A",
            "node_id": [
                # exist in the mapping
                3,
                4,
                5,
                # not in the mapping
                1003,
                1004,
                1005,
            ],
            "mtype": "foo",
        },
    }
    id_mapping = {
        "A": pd.DataFrame({"new_id": np.arange(4)}, index=[0, 5, 4, 3]),
    }
    ret = split_population._update_node_sets(node_sets, id_mapping)

    expected = {
        "CopiedNoNodeIds": ["All"],
        "HasPopulationCopied": {
            "node_id": [1, 2, 3],
            "population": "A",
            "mtype": "foo",
        },
    }
    assert ret == expected


def test_get_subcircuit_external_ids(monkeypatch):
    all_sgids = np.array([10, 10, 11, 11, 12, 12, 10, 10, 11, 11, 12, 12, 10, 10, 11, 11, 12, 12])
    all_tgids = np.array([10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12])

    def get_ids(wanted_src_ids, wanted_dst_ids):
        monkeypatch.setenv("H5_READ_CHUNKSIZE", "3")
        return split_population._get_subcircuit_external_ids(
            all_sgids, all_tgids, wanted_src_ids, wanted_dst_ids
        )

    wanted_src_ids = [10, 12]
    wanted_dst_ids = [10]
    expected = pd.DataFrame({"new_id": np.array([0, 1], np.uint)}, index=[10, 12])
    pd.testing.assert_frame_equal(expected, get_ids(wanted_src_ids, wanted_dst_ids))

    wanted_src_ids = [10]
    wanted_dst_ids = [10, 12, 11]
    expected = pd.DataFrame({"new_id": np.array([0], np.uint)}, index=[10])
    pd.testing.assert_frame_equal(expected, get_ids(wanted_src_ids, wanted_dst_ids))

    wanted_src_ids = [10, 12, 11]
    wanted_dst_ids = [10, 12]
    expected = pd.DataFrame({"new_id": np.array([0, 1, 2], np.uint)}, index=[10, 11, 12])
    pd.testing.assert_frame_equal(expected, get_ids(wanted_src_ids, wanted_dst_ids))


def _find_populations_by_path(networks, key, name):
    populations = {
        k: v
        for population in networks[key]
        for k, v in population["populations"].items()
        if population[f"{key}_file"] == name
    }
    return populations


def _check_biophysical_nodes(path, has_virtual, has_external):
    mapping = load_json(path / "id_mapping.json")
    assert mapping["A"] == {"new_id": [0, 1, 2], "old_id": [0, 2, 4]}
    assert mapping["B"] == {"new_id": [0, 1, 2, 3], "old_id": [0, 2, 4, 5]}
    assert mapping["C"] == {"new_id": [0, 1, 2, 3], "old_id": [0, 2, 4, 5]}

    with h5py.File(path / "nodes" / "nodes.h5", "r") as h5:
        nodes = h5["nodes"]
        for src in ("A", "B", "C"):
            assert src in nodes
            assert len(nodes[src]["0/@library/mtype"]) == 1
            assert np.all(nodes[src]["0/@library/mtype"][0] == b"a")
            assert np.all(nodes[src]["0/mtype"][:] == 0)

        assert len(nodes["A/node_type_id"]) == 3
        assert len(nodes["B/node_type_id"]) == 4
        assert len(nodes["C/node_type_id"]) == 4

    with h5py.File(path / "edges" / "edges.h5", "r") as h5:
        edges = h5["edges"]

        assert "A__B" in edges
        assert list(edges["A__B"]["source_node_id"]) == [0, 0, 0]
        assert list(edges["A__B"]["target_node_id"]) == [0, 0, 1]  # 2nd is duplicate edge

        assert "B__A" not in edges

        assert "A__C" in edges
        assert list(edges["A__C"]["source_node_id"]) == [2]
        assert list(edges["A__C"]["target_node_id"]) == [2]

        assert "B__C" in edges
        assert list(edges["B__C"]["source_node_id"]) == [1]
        assert list(edges["B__C"]["target_node_id"]) == [1]

        assert "C__A" in edges
        assert list(edges["C__A"]["source_node_id"]) == [2]
        assert list(edges["C__A"]["target_node_id"]) == [2]

        config = load_json(path / "circuit_config.json")

        assert "manifest" in config
        assert config["manifest"]["$BASE_DIR"] == "./"
        assert "networks" in config
        assert "nodes" in config["networks"]
        node_pops = _find_populations_by_path(
            config["networks"], "nodes", "$BASE_DIR/nodes/nodes.h5"
        )
        assert node_pops == {
            "A": {"type": "biophysical"},
            "B": {"type": "biophysical"},
            "C": {"type": "biophysical"},
        }
        assert "edges" in config["networks"]
        edge_pops = _find_populations_by_path(
            config["networks"], "edges", "$BASE_DIR/edges/edges.h5"
        )
        assert edge_pops == {
            "A__B": {"type": "chemical"},
            "A__C": {"type": "chemical"},
            "B__A": {"type": "chemical"},
            "B__C": {"type": "chemical"},
            "C__A": {"type": "chemical"},
            "C__B": {"type": "chemical"},
        }

        virtual_node_count = sum(
            population["type"] == "virtual"
            for node in config["networks"]["nodes"]
            for population in node["populations"].values()
        )
        if has_virtual:
            assert virtual_node_count > 0
        else:
            assert virtual_node_count == 0
            assert len(node_pops) == 3
            assert len(edge_pops) == 6

        node_sets = load_json(path / "node_sets.json")
        assert node_sets == {
            "mtype_a": {"mtype": "a"},
            "someA": {"node_id": [0, 1], "population": "A"},
            "allB": {"node_id": [0, 1, 2, 3], "population": "B"},
            "noC": {"node_id": [], "population": "C"},
        }

        expected_mapping = {
            "A": {"old_id": [0, 2, 4], "new_id": [0, 1, 2]},
            "B": {"old_id": [0, 2, 4, 5], "new_id": [0, 1, 2, 3]},
            "C": {"old_id": [0, 2, 4, 5], "new_id": [0, 1, 2, 3]},
        }

        if has_virtual:
            expected_mapping["V1"] = {"old_id": [0, 2, 3], "new_id": [0, 1, 2]}
            expected_mapping["V2"] = {"old_id": [0], "new_id": [0]}

        if has_external:
            expected_mapping["external_A__B"] = {"old_id": [5], "new_id": [0]}
            expected_mapping["external_A__C"] = {"old_id": [5], "new_id": [0]}

        mapping = load_json(path / "id_mapping.json")
        assert mapping == expected_mapping


def test_split_subcircuit_with_no_externals(tmp_path):
    node_set_name = "mtype_a"
    circuit_config_path = str(DATA_PATH / "split_subcircuit" / "circuit_config.json")

    split_population.split_subcircuit(
        tmp_path, node_set_name, circuit_config_path, do_virtual=False, create_external=False
    )

    _check_biophysical_nodes(path=tmp_path, has_virtual=False, has_external=False)


def test_split_subcircuit_with_externals(tmp_path):
    node_set_name = "mtype_a"
    circuit_config_path = str(DATA_PATH / "split_subcircuit" / "circuit_config.json")

    split_population.split_subcircuit(
        tmp_path, node_set_name, circuit_config_path, do_virtual=False, create_external=True
    )

    _check_biophysical_nodes(path=tmp_path, has_virtual=False, has_external=True)

    mapping = load_json(tmp_path / "id_mapping.json")
    assert mapping["external_A__B"] == {"new_id": [0], "old_id": [5]}
    assert mapping["external_A__C"] == {"new_id": [0], "old_id": [5]}
    assert "external_B" not in mapping
    assert "external_C" not in mapping

    with h5py.File(tmp_path / "nodes_external_A__B.h5", "r") as h5:
        assert len(h5["nodes/external_A__B/0/model_type"]) == 1

    with h5py.File(tmp_path / "nodes_external_A__C.h5", "r") as h5:
        assert len(h5["nodes/external_A__C/0/model_type"]) == 1

    with h5py.File(tmp_path / "external_A__B.h5", "r") as h5:
        assert h5["edges/external_A__B/source_node_id"].attrs["node_population"] == "A"
        assert h5["edges/external_A__B/target_node_id"].attrs["node_population"] == "B"
        assert len(h5["edges/external_A__B/0/delay"]) == 1

    networks = load_json(tmp_path / "circuit_config.json")["networks"]
    assert len(networks["nodes"]) == 1
    assert len(networks["edges"]) == 1

    with h5py.File(tmp_path / "external_A__C.h5", "r") as h5:
        assert len(h5["edges/external_A__C/0/delay"]) == 1
        assert h5["edges/external_A__C/0/delay"][0] == 0.5


def test_split_subcircuit_with_virtual(tmp_path):
    node_set_name = "mtype_a"
    circuit_config_path = str(DATA_PATH / "split_subcircuit" / "circuit_config.json")
    split_population.split_subcircuit(
        tmp_path, node_set_name, circuit_config_path, do_virtual=True, create_external=False
    )

    _check_biophysical_nodes(path=tmp_path, has_virtual=True, has_external=False)

    with h5py.File(tmp_path / "V1" / "nodes.h5", "r") as h5:
        assert len(h5["nodes/V1/0/model_type"]) == 3

    with h5py.File(tmp_path / "V2" / "nodes.h5", "r") as h5:
        assert len(h5["nodes/V2/0/model_type"]) == 1

    with h5py.File(tmp_path / "edges" / "virtual_edges_V1.h5", "r") as h5:
        assert len(h5["edges/V1__A/0/delay"]) == 2
        assert list(h5["edges/V1__A/source_node_id"]) == [0, 2]
        assert list(h5["edges/V1__A/target_node_id"]) == [0, 0]

        assert len(h5["edges/V1__B/0/delay"]) == 1
        assert list(h5["edges/V1__B/source_node_id"]) == [1]
        assert list(h5["edges/V1__B/target_node_id"]) == [0]

    with h5py.File(tmp_path / "V2__C" / "virtual_edges_V2.h5", "r") as h5:
        assert len(h5["edges/V2__C/0/delay"]) == 1

        assert list(h5["edges/V2__C/source_node_id"]) == [0]
        assert list(h5["edges/V2__C/target_node_id"]) == [1]

    networks = load_json(tmp_path / "circuit_config.json")["networks"]

    # nodes
    for pop in (1, 2):
        virtual_pop = _find_populations_by_path(networks, "nodes", f"$BASE_DIR/V{pop}/nodes.h5")
        assert len(virtual_pop) == 1
        assert virtual_pop[f"V{pop}"] == {"type": "virtual"}

    # edges
    virtual_pop = _find_populations_by_path(
        networks, "edges", "$BASE_DIR/edges/virtual_edges_V1.h5"
    )
    assert virtual_pop == {"V1__A": {"type": "chemical"}, "V1__B": {"type": "chemical"}}

    virtual_pop = _find_populations_by_path(
        networks, "edges", "$BASE_DIR/V2__C/virtual_edges_V2.h5"
    )
    assert virtual_pop == {"V2__C": {"type": "chemical"}}
