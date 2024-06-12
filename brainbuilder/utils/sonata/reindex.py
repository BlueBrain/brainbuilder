# SPDX-License-Identifier: Apache-2.0
r"""
Single parents; collapse to single segment::

    Leaf node

    ...---O---O---| => ...---------|

Which is a general case of::

    Internal node

                    /-----|                   /-----|
    ...---O---O---O<         => ...---------O<
                    \-----|                   \-----|

For h5 morph:
  - deleted sections result in *all* the parent ids being decreased, for
    all the sections in the file, not just those that are children, grand-children, etc
  - extra point is removed, so all the start offsets need to be decreased, in
    the same fasion

For synapses:
 - section_id: same change as for morph: since a section is deleted, every
   section_id with an id >= than the one deleted needs be decreased by one
 - segment ID: collapsed sections result in the segment_id increasing by
   the count of the parent's section count
"""

import glob
import logging
import os
import shutil

import h5py
import morphio
import numpy as np
import pandas as pd

L = logging.getLogger(__name__)

FIRST_POINT_ID, PARENT_GROUP_ID = 0, 2
XYZ = slice(0, 3)


def _get_only_children(parents):
    """get children that have no siblings"""
    ids_, counts = np.unique(parents, return_counts=True)
    single_child_parents = ids_[(counts == 1) & (ids_ != -1)]
    ret = np.nonzero(np.isin(parents, single_child_parents))[0]
    # make sure we don't collapse the soma
    ret = ret[parents[ret] != 0]
    return ret


def _only_child_removal(parents, first_points):
    """find `new_parents` and `new_segment_offset` so only children can be removed

    Args:
        parents(np.array):  h5 morphology structure of parents
        first_points(np.array):  h5 morphology structure of first points

    Returns:
        new_parents(list): deleted_child_id
        new_segment_offset(dict): deleted_child_id -> segment_id increase needed
    """
    only_children = _get_only_children(parents)

    assert np.all(
        (only_children - 1) == parents[only_children]
    ), "Some only_children sections do not follow their parent"

    only_children = [int(c) for c in only_children]

    new_segment_offset = {}

    # handle the case where multiple single_children were deleted; need to
    # update their new segment w/ the final parent, and save the total offset
    # Note: the offset is the number of segments (ie # of points in the
    # sections - 1) comes before the current section, in all the parents
    def get_offset(child_id):
        """give `child_id`, return the parent and the number of sections in its parent"""
        parent_id = parents[child_id]
        offset = first_points[child_id] - first_points[parent_id] - 1
        return parent_id, offset

    for child_id in only_children:
        parent_id, offset = get_offset(child_id)
        while parent_id in only_children:
            parent_id, o = get_offset(parent_id)
            offset += o
        new_segment_offset[child_id] = int(offset)

    return list(only_children), new_segment_offset


def generate_h5_updates(h5_morph_path):
    """Generate dict of updates required to fix h5 morphologies

    For each file, create the necesary data to:
        1) remove the single children from the h5 morphology file
        2) update the sonata edges file with correct section_id/segment_id

    Args:
        h5_morph_path(str): path to location of h5 morphology files

    Returns:
        dict of updates with the following structure::

            {
                <morph_name>: {
                    'new_parents': list(<deleted_child_id>)
                    'new_segment_offset': {
                        <deleted_child_id>: <segment_id increase needed>
                    }
                }
            }
    """
    ret = {}
    for file_ in glob.glob(os.path.join(h5_morph_path, "*.h5")):
        L.debug("generate_h5_updates for %s", file_)
        with h5py.File(file_, "r") as h5:
            structure = h5["structure"][:]
            new_parents, new_segment_offset = _only_child_removal(
                structure[:, PARENT_GROUP_ID], structure[:, FIRST_POINT_ID]
            )
            if new_parents:
                ret[os.path.basename(file_)] = {
                    "new_parents": new_parents,
                    "new_segment_offset": new_segment_offset,
                }

    return ret


def _update_structure_and_points(structure, points, new_parents):
    """Update the h5v1 points based on `new_parents` so that there are no more unifurcations

    Args:
        structure(nd.array): h5v1 morph structure
        points(nd.array): h5v1 morph points
        new_parents(list): [old_id, ...]

    Returns:
        tuple(new_structure(nd.array), new_points(nd.array))
    """
    new_structure = structure.copy()

    deleted_structure = []

    for old_id in sorted(new_parents, reverse=True):
        mask = old_id <= new_structure[:, PARENT_GROUP_ID]
        new_structure[mask, PARENT_GROUP_ID] -= 1

        # we will be deleting a point, so everything from this point on is offset by 1
        new_structure[old_id:, FIRST_POINT_ID] -= 1

        deleted_structure.append(old_id)

    deleted_points = structure[deleted_structure, FIRST_POINT_ID]

    new_structure = np.delete(new_structure, deleted_structure, axis=0)
    new_points = np.delete(points.astype(np.float32), deleted_points, axis=0)

    return new_structure, new_points


def write_new_h5_morphs(h5_updates, h5_morph_path, output):
    """Given `h5_updates`, fix unifurcations in h5 files in `h5_morph_path`

    Args:
        h5_updates(dict): result of `sonate_reindex.generate_h5_updates`
        h5_morph_path(str): path to original morphologies
        output(str): path to write new morphologies
    """
    shutil.copytree(h5_morph_path, output)

    for file_, updates in h5_updates.items():
        with h5py.File(os.path.join(output, os.path.basename(file_)), "r+") as h5:
            structure = h5["structure"][:]
            del h5["structure"]
            points = h5["points"][:]
            del h5["points"]

            new_structure, new_points = _update_structure_and_points(
                structure, points, updates["new_parents"]
            )
            h5["structure"], h5["points"] = new_structure, new_points


def _update_section_and_segment_ids(section_id, segment_id, updates):
    """fix the section_id, segment_id based on the updates

    Args:
        section_id(np.array-like):
        segment_id(np.array-like)
        updates(dict): {'new_parents': list(deleted_child_id),
                        'new_segment_offset':
                            deleted_child_id -> segment_id increase needed,
                        }
    Returns:
        tuple(new_section_id, new_segment_id)

    Note: section_id and segment_id are modified in place
    """
    # need to do this before section_ids, since new_segments_offsets is keyed on old ids
    for old_id, offset in updates["new_segment_offset"].items():
        mask = section_id == old_id
        L.debug("updating[%d]: %d instances", old_id, np.count_nonzero(mask))
        segment_id[mask] += offset

    for old_id in sorted(updates["new_parents"], reverse=True):
        section_id[section_id >= old_id] -= 1

    return section_id, segment_id


def _apply_to_edges(node_ids, updates, edges):
    """update a circuit's edges to reflect the new morphology organization

    Args:
        node_ids(nd.array): which node ids to change
        updates(dict): {'new_parents': list(deleted_child_id)
                        'new_segment_offset':
                            deleted_child_id -> segment_id increase needed,
                        }
        edges(h5_group): sonata h5py object of the population's group

    Note: the updates are done in place on 'edges' for efficiency reasons
    """
    assert "0" in edges, "Missing default group of 0"
    group = edges["0"]

    for direction in (
        "afferent",
        "efferent",
    ):
        section_name, segment_name = direction + "_section_id", direction + "_segment_id"
        if section_name in group and segment_name in group:
            index = "target_node_id" if direction == "afferent" else "source_node_id"
            mask = np.isin(edges[index][:], node_ids)
            new_ids = _update_section_and_segment_ids(
                group[section_name][mask], group[segment_name][mask], updates
            )
            group[section_name][mask] = new_ids[0]
            group[segment_name][mask] = new_ids[1]
        else:
            L.warning("'%s' or '%s' missing", section_name, segment_name)


def apply_edge_updates(morphologies, edge_path, h5_updates, population):
    """update a sonata edge file reflect the new morphology organization

    morphologies
    edge_path(str): path to sonata edge file
    h5_updates(dict): result of `sonate_reindex.generate_h5_updates`
    """
    with h5py.File(edge_path, "r+") as h5:
        for morphology, node_ids in morphologies.groupby(morphologies):
            if morphology + ".h5" not in h5_updates:
                continue

            node_ids = node_ids.index.to_numpy()
            L.debug("apply_edge_updates for morph: %s[%d]: %s", morphology, len(node_ids), node_ids)
            _apply_to_edges(node_ids, h5_updates[morphology + ".h5"], h5["edges"][population])


def _morph_section_lengths(morph_path, morph_name):
    """load h5 morphology, and calculate section_lengths and cumulative_length"""
    L.debug("Loading: %s", morph_name)
    m = morphio.Morphology(os.path.join(morph_path, morph_name + ".h5"), morphio.Option.nrn_order)
    dfs = [
        pd.DataFrame.from_dict(
            {
                "segment_id": [
                    0,
                ],
                "section_id": 0,
                "cumulative_length": 0.0,
                "section_length": 1.0,
            }
        ),
    ]
    for section in m.iter():
        df = pd.DataFrame()
        df["segment_id"] = np.arange(len(section.points) - 1)
        df["section_id"] = section.id + 1  # morphio doesn't consider the soma a section
        lengths = np.cumsum(np.linalg.norm(section.points[1:] - section.points[:-1], axis=1))
        df["cumulative_length"] = np.hstack(([0], lengths[:-1]))
        df["section_length"] = lengths[-1]

        dfs.append(df)

    return pd.concat(dfs, sort=False, ignore_index=True).set_index(["section_id", "segment_id"])


def _get_synapse_mask_for_ids(h5_indices, ids, total_synapse_count):
    """find the mask for the synapses belonging to ids"""
    mask = np.full(total_synapse_count, False)
    for id_ in ids:
        start, end = h5_indices["node_id_to_ranges"][id_]
        for s, e in h5_indices["range_to_edge_id"][start:end]:
            mask[s:e] = True

    return mask


def _calculate_section_position(section_lengths, section_ids, segment_ids, offsets):
    """find the _section_pos for all synapses"""
    df = pd.DataFrame.from_dict(
        {
            "section_id": section_ids,
            "segment_id": segment_ids,
            "offset": offsets,
        }
    )

    df = df.join(section_lengths, on=["section_id", "segment_id"])

    assert len(df[df.isnull().any(axis=1)]) == 0, "All section/segments should exist in morphology"

    section_pos = (df["offset"] + df["cumulative_length"]) / df["section_length"]
    section_pos[np.isnan(section_pos)] = 0.0

    max_pos = np.max(section_pos.to_numpy())
    if not 0.0 <= max_pos <= 1.00001:
        L.warning("pos %s should be between [0, 1]", max_pos)

    section_pos = np.clip(section_pos, 0.0, 1.0)

    return section_pos


def _update_pos(h5_population, section_lengths, ids):
    """for `ids` in population, update the *_section_pos values"""
    assert "0" in h5_population, "Missing default group of 0"

    h5_group = h5_population["0"]

    def _data(name, direction, mask):
        path = f"{direction}_{name}"
        if path not in h5_group:
            return None
        return h5_group[path][mask]

    for direction in ("afferent", "efferent"):
        index = "target_to_source" if direction == "afferent" else "source_to_target"
        mask = _get_synapse_mask_for_ids(
            h5_population["indices/" + index],
            ids.index.to_list(),
            total_synapse_count=len(h5_group["syn_type_id"]),
        )

        section_id = _data("section_id", direction, mask)
        segment_id = _data("segment_id", direction, mask)
        offset = _data("segment_offset", direction, mask)

        if section_id is None or segment_id is None or offset is None:
            L.info("Cannot update %s positions in %s", direction, h5_group.file.filename)
            continue

        h5_group[f"{direction}_section_pos"][mask] = _calculate_section_position(
            section_lengths, section_id, segment_id, offset
        )


def write_sonata_pos(morph_path, morphologies, population, edges):
    """Update synapses in edges with new `_section_pos` SONATA attribute

    Args:
        morph_path(str): path to h5 morphologogies
        morphologies(pd.DataFrame()): morph names with index of NodeID
        population(str): population name in edge files
        edges(list(str)): paths to edge files to be updated
    """

    def create_section_pos(h5_group, name):
        if not (
            name + "_section_id" in h5_group
            and name + "_segment_id" in h5_group
            and name + "_segment_offset" in h5_group
        ):
            return

        name += "_section_pos"
        new_name = name + "_orig"

        # create backup, if the original exists
        if new_name in h5_group:
            del h5_group[new_name]
        if name in h5_group:
            h5_group.move(name, new_name)

        size = len(h5_group["syn_type_id"])
        h5_group[name] = np.zeros(size, dtype=np.float32)

    for edge in edges:
        with h5py.File(edge, "r+") as h5:
            h5_group = h5["edges"][population]["0"]

            create_section_pos(h5_group, "afferent")
            create_section_pos(h5_group, "efferent")

    for edge in edges:
        with h5py.File(edge, "r+") as h5:
            for morphology, ids in morphologies.groupby(morphologies):
                section_lengths = _morph_section_lengths(morph_path, morphology)
                if ids.empty:
                    L.debug("No nodes for %s, skipping", morphology)
                else:
                    _update_pos(h5["edges"][population], section_lengths, ids)
