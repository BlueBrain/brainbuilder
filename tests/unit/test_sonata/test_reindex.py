# SPDX-License-Identifier: Apache-2.0
import os
import shutil
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from brainbuilder.utils.sonata import reindex
from brainbuilder.utils.sonata.reindex import FIRST_POINT_ID, PARENT_GROUP_ID

TEST_DATA_PATH = Path(__file__).parent.parent / "data"
DATA_PATH = TEST_DATA_PATH / "sonata" / "reindex"
STRUCTURE_SOMA_SINGLE_CHILD = np.array(
    [
        [0, 1, -1],
        [3, 2, 0],
        [5, 2, 0],
    ]
)
# single/only children marked with *
STRUCTURE_SOMA_SINGLE_CHILDREN = np.array(
    [
        [0, 1, -1],
        [3, 2, 0],
        [5, 2, 1],  # *
        [7, 2, 2],  # *
        [9, 2, 3],
        [11, 2, 3],
    ]
)

STRUCTURE_MID_SINGLE_CHILDREN = np.array(
    [
        [0, 1, -1],  # 0
        [3, 2, 0],
        [5, 2, 1],
        [7, 2, 2],
        [9, 2, 2],
        [11, 2, 1],  # 5
        [13, 2, 5],  # *
        [15, 2, 6],  # *
        [17, 2, 7],  # *
        [19, 2, 8],  # 9
        [21, 2, 9],  # *
        [23, 2, 8],
        [25, 2, 11],  # *
    ]
)

STRUCTURE_LEAF_SINGLE_CHILDREN = np.array(
    [
        [0, 1, -1],
        [3, 2, 0],
        [5, 2, 1],  # *
        [7, 2, 2],
        [9, 2, 2],
        [11, 2, 2],  # 5
        [13, 2, 5],  # *
        [15, 2, 6],  # *
        [17, 2, 7],  # *
    ]
)

# this is the same as DATA_PATH/morphs/two_child_unmerged.h5
# copied here so it can be annotated w/ single parents
STRUCTURE_TWO_CHILD_UNMERGED = np.array(
    [
        [0, 1, -1],  # 0
        [1, 3, 0],  # 1 *
        [5, 3, 1],  # 2 *
        [8, 3, 2],  # 3
        [10, 3, 3],  # 4 *
        [12, 3, 4],  # 5
        [14, 3, 3],  # 6 *
        [16, 3, 6],  # 7 *
        [18, 3, 7],
    ]
)  # 8 *

# this is the same as DATA_PATH/morphs/three_child_unmerged.h5
STRUCTURE_THREE_CHILD_UNMERGED = np.array(
    [
        [0, 1, -1],  # 0
        [1, 3, 0],  # 1
        [3, 3, 1],  # 2 *
        [5, 3, 2],  # 3
        [7, 3, 3],  # 4 *
        [9, 3, 2],  # 5
        [11, 3, 5],  # 6 *
        [13, 3, 6],  # 7
        [15, 3, 6],  # 8
        [17, 3, 8],  # 9 *
        [19, 3, 2],  # 10
        [21, 3, 10],  # 11 *
    ]
)


def test__get_only_children():
    ret = reindex._get_only_children(STRUCTURE_SOMA_SINGLE_CHILD[:, PARENT_GROUP_ID])
    assert not len(ret)  # first section is not collapsed

    ret = reindex._get_only_children(STRUCTURE_SOMA_SINGLE_CHILDREN[:, PARENT_GROUP_ID])
    assert_allclose(
        ret,
        [
            2,
            3,
        ],
    )

    ret = reindex._get_only_children(STRUCTURE_MID_SINGLE_CHILDREN[:, PARENT_GROUP_ID])
    assert_allclose(ret, [6, 7, 8, 10, 12])

    ret = reindex._get_only_children(STRUCTURE_LEAF_SINGLE_CHILDREN[:, PARENT_GROUP_ID])
    assert_allclose(ret, [2, 6, 7, 8])

    ret = reindex._get_only_children(STRUCTURE_TWO_CHILD_UNMERGED[:, PARENT_GROUP_ID])
    assert_allclose(ret, [2, 3, 5, 7, 8])

    ret = reindex._get_only_children(STRUCTURE_THREE_CHILD_UNMERGED[:, PARENT_GROUP_ID])
    assert_allclose(ret, [2, 4, 6, 9, 11])


def test__only_child_removal():
    new_parents, new_segment_offset = reindex._only_child_removal(
        STRUCTURE_SOMA_SINGLE_CHILDREN[:, PARENT_GROUP_ID],
        STRUCTURE_SOMA_SINGLE_CHILDREN[:, FIRST_POINT_ID],
    )
    assert new_parents == [2, 3]
    assert new_segment_offset == {2: 1, 3: 2}

    new_parents, new_segment_offset = reindex._only_child_removal(
        STRUCTURE_MID_SINGLE_CHILDREN[:, PARENT_GROUP_ID],
        STRUCTURE_MID_SINGLE_CHILDREN[:, FIRST_POINT_ID],
    )
    assert new_parents == [6, 7, 8, 10, 12]
    assert new_segment_offset == {6: 1, 7: 2, 8: 3, 10: 1, 12: 1}

    new_parents, new_segment_offset = reindex._only_child_removal(
        STRUCTURE_LEAF_SINGLE_CHILDREN[:, PARENT_GROUP_ID],
        STRUCTURE_LEAF_SINGLE_CHILDREN[:, FIRST_POINT_ID],
    )
    assert new_parents == [2, 6, 7, 8]
    assert new_segment_offset == {2: 1, 6: 1, 7: 2, 8: 3}

    new_parents, new_segment_offset = reindex._only_child_removal(
        STRUCTURE_TWO_CHILD_UNMERGED[:, PARENT_GROUP_ID],
        STRUCTURE_TWO_CHILD_UNMERGED[:, FIRST_POINT_ID],
    )
    assert new_parents == [2, 3, 5, 7, 8]
    assert new_segment_offset == {2: (5 - 1 - 1), 3: 3 + 2, 5: 1, 7: 1, 8: 1 + 1}

    new_parents, new_segment_offset = reindex._only_child_removal(
        STRUCTURE_THREE_CHILD_UNMERGED[:, PARENT_GROUP_ID],
        STRUCTURE_THREE_CHILD_UNMERGED[:, FIRST_POINT_ID],
    )
    assert new_parents == [2, 4, 6, 9, 11]
    assert new_segment_offset == {2: 1, 4: 1, 6: 1, 9: 1, 11: 1}


def test__update_structure_and_points():
    def test(structure, new_parents):
        points = np.vstack([np.arange(structure[-1, 0] + 5)] * 3).T
        new_structure, new_points = reindex._update_structure_and_points(
            structure, points, new_parents
        )
        assert len(new_parents) == len(structure) - len(new_structure)
        assert len(new_parents) == len(points) - len(new_points)

    test(STRUCTURE_SOMA_SINGLE_CHILDREN, [2, 3])
    test(STRUCTURE_MID_SINGLE_CHILDREN, [6, 7, 8, 10, 12])
    test(STRUCTURE_LEAF_SINGLE_CHILDREN, [2, 6, 7, 8])


def test__update_section_ids():
    section_id = np.array([], dtype=int)
    segment_id = np.array([], dtype=int)
    new_section_id, new_segment_id = reindex._update_section_and_segment_ids(
        section_id, segment_id, {"new_parents": {}, "new_segment_offset": {}}
    )
    assert len(section_id) == 0
    assert len(segment_id) == 0

    section_id = np.array([0, 1, 2, 3, 4], dtype=int)
    segment_id = np.array([9, 10, 11, 12, 13], dtype=int)
    updates = {
        "new_parents": [2, 3],
        "new_segment_offset": {2: 2, 3: 4},
    }
    new_section_id, new_segment_id = reindex._update_section_and_segment_ids(
        section_id, segment_id, updates
    )
    assert list(new_section_id) == [0, 1, 1, 1, 2]
    assert list(new_segment_id) == [9, 10, 11 + 2, 12 + 4, 13]


def test__apply_to_edges():
    names = (
        "afferent_section_id",
        "afferent_segment_id",
        "efferent_section_id",
        "efferent_segment_id",
    )
    edges = {
        "0": {name: np.array([], dtype=int) for name in names},
        "target_node_id": np.array([], dtype=int),
        "source_node_id": np.array([], dtype=int),
    }
    updates = {"new_parents": {}, "new_segment_offset": {}}
    edges_orig = deepcopy(edges)

    # run w/ no changes ids
    reindex._apply_to_edges(np.array([], int), updates, edges)

    for name in names:
        assert_allclose(edges_orig["0"][name], edges["0"][name])

    # run w/ missing ids, nothing will match, so no changes
    reindex._apply_to_edges(np.array([1], int), updates, edges)

    for name in names:
        assert_allclose(edges_orig["0"][name], edges["0"][name])
    assert_allclose(edges_orig["target_node_id"], edges["target_node_id"])
    assert_allclose(edges_orig["source_node_id"], edges["source_node_id"])

    edges = {
        "0": {
            "afferent_section_id": np.array([0, 0, 0, 1, 1, 2], dtype=int),
            "afferent_segment_id": np.array([0, 1, 2, 0, 1, 3], dtype=int),
            "efferent_section_id": np.array([10, 10, 10, 11, 11, 12], dtype=int),
            "efferent_segment_id": np.array([0, 1, 2, 0, 1, 3], dtype=int),
        },
        "target_node_id": np.array([0, 0, 1, 1, 2, 2], dtype=int),
        "source_node_id": np.array([10, 10, 11, 11, 12, 12], dtype=int),
    }
    edges_orig = deepcopy(edges)

    updates = {"new_parents": {}, "new_segment_offset": {}}
    reindex._apply_to_edges(np.array([1], int), updates, edges)

    # no changes; updates empty
    for name in names:
        assert_allclose(edges_orig["0"][name], edges["0"][name])
    assert_allclose(edges_orig["target_node_id"], edges["target_node_id"])
    assert_allclose(edges_orig["source_node_id"], edges["source_node_id"])

    updates = {"new_parents": {1: []}, "new_segment_offset": {1: 10}}
    ids = np.array([1], int)
    reindex._apply_to_edges(ids, updates, edges)

    # only afferent changes, since ids == 1
    assert_allclose(edges["0"]["afferent_section_id"], [0, 0, 0, 0, 1, 2])
    assert_allclose(edges["0"]["afferent_segment_id"], [0, 1, 2, 10, 1, 3])

    assert_allclose(edges["0"]["efferent_section_id"], edges_orig["0"]["efferent_section_id"])
    assert_allclose(edges["0"]["efferent_segment_id"], edges_orig["0"]["efferent_segment_id"])

    assert_allclose(edges_orig["target_node_id"], edges["target_node_id"])
    assert_allclose(edges_orig["source_node_id"], edges["source_node_id"])


def test_reindex(tmp_path):
    temp_dir = os.path.join(tmp_path, "reindex")
    morphs_path = str(DATA_PATH / "morphs")
    h5_updates = reindex.generate_h5_updates(morphs_path)
    assert len(h5_updates) == 2

    # {'two_child_unmerged.h5':
    # {'new_parents': [2, 3, 5, 7, 8], 'new_segment_offset': {2: 3, 3: 5, 5: 1, 7: 1, 8: 2}},
    # 'three_child_unmerged.h5':
    # {'new_parents': [2, 4, 6, 9, 11], 'new_segment_offset': {2: 1, 4: 1, 6: 1, 9: 1, 11: 1}}}

    new_morph_path = os.path.join(temp_dir, "morphs")
    reindex.write_new_h5_morphs(h5_updates, morphs_path, new_morph_path)
    for path in (
        "three_child_merged.h5",
        "three_child_unmerged.h5",
        "two_child_merged.h5",
        "two_child_unmerged.h5",
    ):
        assert os.path.exists(os.path.join(new_morph_path, path))

    # TODO: h5diff three_child_merged.h5 three_child_unmerged.h5

    edge = os.path.join(temp_dir, "edges.h5")
    shutil.copy(str(DATA_PATH / "edges.h5"), edge)
    morphologies = pd.Series(
        ["two_child_unmerged", "two_child_unmerged", "three_child_unmerged"],
        # id:         0                   1                     2
        name="morphology",
    )
    population = "default"

    reindex.apply_edge_updates(morphologies, edge, h5_updates, population)

    """
    Looking at results of new_parents:

        position id  afferent_section_id
                      old   new
        --------------------------------
        [ 0 ]     0    8     3 = 8 - (1 + 4) (# of parents deleted + 1)
        [ 2 ]     0    5     2 = 5 - (1 + 2)  # of parents deleted element count in new_parents
        [ 4 ]     2    11    6 = 11 - (1 + 4)   before old section
        [ 5 ]     2    6     3 = 6 - (1 + 2)
        [ 6 ]     2    7     4 = 7 - (1 + 2)
        [ 8 ]     0    3     1 = 3 - (1 + 1)
        [ 9 ]     0    7     3 = 7 - (1 + 3)
        [ 10 ]    0    5     2 = 5 - (1 + 2)

    Looking at results of new_segment_offset:
        position  id   sec   afferent_segment_id
                       old   old   new
        ----------------------------------
        [ 0 ]      0     8   0     2
        [ 2 ]      0     5   0     1
        [ 4 ]      2     11  0     1
        [ 5 ]      2     6   0     1
        [ 6 ]      2     7   0     0 # not merged with parent, shifted b/c a prev. section deleted
        [ 8 ]      0     3   0     5
        [ 9 ]      0     7   0     1
        [ 10 ]     0     5   0     1
    """
    with h5py.File(edge, "r") as h5:
        expected = [3, 2, 6, 3, 4, 1, 3, 2]
        assert (
            list(h5["edges/default/0/afferent_section_id"][[0, 2, 4, 5, 6, 8, 9, 10]]) == expected
        )
        expected = [2, 1, 1, 1, 0, 5, 1, 1]
        assert (
            list(h5["edges/default/0/afferent_segment_id"][[0, 2, 4, 5, 6, 8, 9, 10]]) == expected
        )

    reindex.write_sonata_pos(new_morph_path, morphologies, population, [edge])

    """
    position  id afferent_section_pos afferent_section_pos
    --------------------------------------------------------------
    [ 0 ]      0     0.5             0.785714
    [ 1 ]      0     0.5             0.291667
    [ 2 ]      0     0.5             0.875
    [ 3 ]      0     0.285714        0.166667
    [ 4 ]      2     0.333333        0.666667
    [ 5 ]      2     0.666667        0.833333
    [ 10 ]     0     0               0.75
    [ 11 ]     0     1               0.583333
    """

    with h5py.File(edge, "r") as h5:
        assert_allclose(
            h5["edges/default/0/afferent_section_pos"][[0, 1, 2, 3, 4, 5, 10, 11]],
            [0.785714, 0.291667, 0.875, 0.166667, 0.666667, 0.833333, 0.75, 0.583333],
            rtol=1e-5,
        )
