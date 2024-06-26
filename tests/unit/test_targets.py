# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
import pandas as pd
import voxcell
from voxcell.nexus.voxelbrain import Atlas

import brainbuilder.targets as tested

TEST_DATA_PATH = Path(__file__).parent.parent / "unit/data"


def test__add_occupied_hierarchy():
    region_map_df = pd.DataFrame(
        columns=["id", "acronym", "parent_id"],
        data=[
            [997, "root", -1],
            [8, "grey", 997],
            [688, "CTX", 8],
            [695, "CTXpl", 688],
            [315, "Isocortex", 695],
            [184, "FRP", 315],
            [68, "FRP1", 184],
            [667, "FRP2/3", 184],
        ],
    ).set_index("id")

    occupied_regions = {"FRP1", "FRP2/3", "CTXpl"}
    result = {"CTXpl": {"region": "CTXpl"}}
    tested._add_occupied_hierarchy(region_map_df, occupied_regions, result)
    expected = {
        "FRP": ["FRP1", "FRP2/3"],
        "Isocortex": ["FRP"],
        "CTXpl": ["CTXpl-only", "Isocortex"],
        "CTXpl-only": {
            "region": "CTXpl",
        },
        "CTX": ["CTXpl"],
        "grey": ["CTX"],
        "root": ["grey"],
    }

    assert result == expected


class MockAtlas(Atlas):
    def __init__(self, hierarchy, data):
        super().__init__()
        self._memcache[("region_map",)] = voxcell.RegionMap.from_dict(hierarchy)
        self._memcache.update(data)

    def fetch_data(self, data_type):
        """Fetch `data_type` NRRD."""

    def fetch_hierarchy(self):
        """Fetch brain region hierarchy JSON."""


def test_create_node_sets():
    population = "not-default"
    cells_path = TEST_DATA_PATH / "target_nodes.h5"
    cells = voxcell.CellCollection.load(cells_path)

    full_hierarchy = False
    targets = None
    atlas = None
    allow_empty = True
    res = tested.create_node_sets(cells, full_hierarchy, atlas, targets, allow_empty, population)
    expected = {
        "All": {"population": "not-default"},
        "Excitatory": {"synapse_class": "EXC"},
        "Inhibitory": {"synapse_class": "INH"},
        "L2_X": {"mtype": "L2_X"},
        "L6_Y": {"mtype": "L6_Y"},
        "a": {
            "region": "a",
        },
        "b": {
            "region": "b",
        },
        "c": {
            "region": "c",
        },
    }
    assert res == expected

    hierarchy = {
        "id": 0,
        "name": "a",
        "acronym": "a",
        "children": [
            {"id": 1, "name": "b", "acronym": "b", "children": []},
            {"id": 2, "name": "c", "acronym": "c", "children": []},
        ],
    }

    full_hierarchy = True
    allow_empty = True
    targets = TEST_DATA_PATH / "targets.yaml"
    mask = voxcell.ROIMask(raw=np.ones((3, 3, 3), dtype=bool), voxel_dimensions=(1000, 1000, 1000))
    atlas = MockAtlas(hierarchy, {("data", "{S1HL-cylinder}", voxcell.ROIMask): mask})
    res = tested.create_node_sets(cells, full_hierarchy, atlas, targets, allow_empty, population)

    expected = {
        "All": {"population": "not-default"},
        "Excitatory": {"synapse_class": "EXC"},
        "Inhibitory": {"synapse_class": "INH"},
        "L2_X": {"mtype": "L2_X"},
        "L6_Y": {"mtype": "L6_Y"},
        "a-only": {
            "region": "a",
        },
        "b": {
            "region": "b",
        },
        "c": {
            "region": "c",
        },
        "a": ["a-only", "b", "c"],
        "L3_MC": {
            "mtype": "L23_MC",
            "region": {
                "$regex": ".*3",
            },
        },
        "L5_CUSTOM": {
            "layer": "5",
        },
        "L5_TPC": {
            "mtype": [
                "L5_TPC:A",
            ],
        },
        "cylinder": {
            "node_id": [
                0,
                1,
                2,
            ],
            "population": "not-default",
        },
    }
    assert res == expected
