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
import multiprocessing
import os
import shutil
import sys
from functools import partial

import click
import h5py
import morph_tool.transform
import morphio
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

L = logging.getLogger(__name__)

FIRST_POINT_ID, PARENT_GROUP_ID = 0, 2
XYZ = slice(0, 3)
INDEX_DIRECTION = {
    "afferent": "target_to_source",
    "efferent": "source_to_target",
}
INDEX_ID = {
    "afferent": "target_node_id",
    "efferent": "source_node_id",
}
FLOAT_LIMIT = sys.float_info.epsilon * 64  # from touchdetector


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
            mask = np.isin(edges[INDEX_ID[direction]][:], node_ids)
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


def _get_synapse_ids(h5_indices, id_):
    """find the mask for the synapses belonging to ids"""
    syn_ids = []

    start, end = h5_indices["node_id_to_ranges"][id_]

    for s, e in h5_indices["range_to_edge_id"][start:end]:
        syn_ids.append(np.arange(s, e, dtype=np.uint64))

    return np.sort(np.concatenate(syn_ids)) if syn_ids else None


def _calculate_section_position(morph, section_ids, segment_ids, segment_offsets):
    """Computes the section position

    Args:
        morph(morphio.Morphology): A morphio mmutable morphology.
        section_ids(np.array): Mx1 array containing the section ids of the synapses
        segment_ids(np.array): Mx1 array containing the segment ids of the synapses
        segment_offsets(np.array): Mx1 array containing the segment offsets of the synapses

    Returns:
        np.array: Mx1 array containing the respective section positions
    """
    section_pos = []

    for section_id, segment_id, segment_offset in zip(section_ids, segment_ids, segment_offsets):
        len_to_segment = 0
        total_len = 1

        # If an actual section, not a soma
        if section_id > 0:
            # morphio doesn't consider soma a section, so section_id is shifted
            section = morph.section(section_id - 1)
            total_len = sum(np.linalg.norm(np.diff(section.points, axis=0), axis=1))

            if segment_id > 0:
                len_to_segment = sum(
                    np.linalg.norm(np.diff(section.points[: segment_id + 1], axis=0), axis=1)
                )
        section_pos.append((len_to_segment + segment_offset) / total_len)

    max_pos = np.max(section_pos)
    if not 0.0 <= max_pos <= 1.00001:
        L.warning("pos %s should be between [0, 1]", max_pos)

    return np.clip(section_pos, 0, 1)


def _get_section_pos_data(gid, edge_path, direction, population):
    """Gets the necessary data from edges file to compute the section position."""
    with h5py.File(edge_path) as h5:
        pop = h5["edges"][population]
        edge_idx = _get_synapse_ids(pop[f"indices/{INDEX_DIRECTION[direction]}"], gid)

        if edge_idx is None:
            return None

        index = INDEX_ID[direction]
        assert len(set(pop[index][edge_idx])) == 1, f"{index} contains erroneous entries"

        return (
            np.asarray(pop[f"0/{direction}_section_id"][edge_idx]),
            np.asarray(pop[f"0/{direction}_segment_id"][edge_idx]),
            np.asarray(pop[f"0/{direction}_segment_offset"][edge_idx]),
            edge_idx,
        )


def compute_section_positions_worker(id_morph, edge_path, direction, population):
    """Worker function computing the section pos values.

    Args:
        id_morph(list,tuple): node ID and full path to morphology
        edge_path(str): path to the edges.h5 file containing the edges' data
        direction(str): afferent/efferent
        population(str): edge population name

    Returns:
        np.array: the section pos values with their respective indices in the edge file
        for given node id.
        None: if the node has no connections in the edge file
    """
    gid, morph_path = id_morph
    data = _get_section_pos_data(gid, edge_path, direction, population)

    if data is None:
        return None

    morph = morphio.Morphology(morph_path, morphio.Option.nrn_order)

    edge_idx = data[-1]
    positions = _calculate_section_position(morph, *data[:-1])

    return positions, edge_idx


def backup_and_create_dataset(h5_group, field, data, dtype):
    """Helper function to create a dataset and backup the existing one."""
    field_bu = field + "_orig"

    if field_bu in h5_group:
        del h5_group[field_bu]
    if field in h5_group:
        h5_group.move(field, field_bu)
        h5_path = h5_group[field_bu].name
        filepath = h5_group.file.filename
        msg = (
            f"Original field has been backed up to '{h5_path}'.\n"
            "To remove the backed up field (in python):\n"
            f" >>> h5 = h5py.File('{filepath}', 'r+')\n"
            f" >>> del h5['{h5_path}']\n"
            "\n"
            "To regain space after removing the field (on commandline):\n"
            f" $ h5repack -i {filepath} \\\n"
            f"            -o {filepath}.repacked\n"
            f" $ mv {filepath}.repacked \\\n"
            f"      {filepath}"
        )
        L.info(click.style(msg, fg="green"))

    h5_group.create_dataset(field, data=data, dtype=dtype)


def write_sonata_pos(morphologies, population, direction, edge_path):
    """Computes section positions in parallel and writes them to the edges file.

    Args:
        morphologies(pd.DataFrame): full morph paths with index of NodeID
        population(str): edge_population name
        direction(str): `afferent` or `efferent`
        edge_path(str): path to the edges.h5 file
    """
    func = partial(
        compute_section_positions_worker,
        edge_path=edge_path,
        direction=direction,
        population=population,
    )

    with multiprocessing.Pool() as pool:
        res = pool.map(func, tqdm(morphologies.reset_index().values))

    positions, edge_ids = np.concatenate([r for r in res if r is not None], axis=1)

    # sort values by edge ids
    positions = positions[np.argsort(edge_ids)]

    with h5py.File(edge_path, "r+") as h5:
        pop0 = h5[f"edges/{population}/0"]
        field = direction + "_section_pos"
        backup_and_create_dataset(pop0, field, positions, np.float32)


def _get_data_for_resolving_section_type(gid, edge_path, direction, population):
    """Gets the necessary data from edges file to resolve section type."""
    with h5py.File(edge_path) as h5:
        pop = h5["edges"][population]
        edge_idx = _get_synapse_ids(pop[f"indices/{INDEX_DIRECTION[direction]}"], gid)

        if edge_idx is None:
            return None

        index = INDEX_ID[direction]
        assert len(set(pop[index][edge_idx])) == 1, f"{index} contains erroneous entries"

        return (
            np.asarray(pop[f"0/{direction}_section_id"][edge_idx]),
            edge_idx,
        )


def _get_section_type_worker(id_morph, edge_path, direction, population):
    gid, morph_path = id_morph
    data = _get_data_for_resolving_section_type(gid, edge_path, direction, population)

    if data is None:
        return None

    morph = morphio.Morphology(morph_path, morphio.Option.nrn_order)

    section_ids, edge_idx = data
    types = np.fromiter(
        (
            morphio.SectionType.soma if id_ == 0 else morph.section(id_ - 1).type
            for id_ in section_ids
        ),
        dtype=np.uint32,
    )

    return types, edge_idx


def write_section_types(morphologies, population, direction, edge_path):
    """Update synapses in edges with new `_section_pos` SONATA attribute

    Args:
        morphologies(pd.DataFrame): full morph paths with index of NodeID
        population(str): population name in edge files
        direction(str): 'afferent' / 'efferent'
        edge_path(str): path to edge file to be updated
    """
    func = partial(
        _get_section_type_worker,
        edge_path=edge_path,
        direction=direction,
        population=population,
    )

    with multiprocessing.Pool() as pool:
        res = pool.map(func, tqdm(morphologies.reset_index().values))

    types, edge_ids = np.concatenate([r for r in res if r is not None], axis=1)

    # sort values by edge ids
    types = types[np.argsort(edge_ids)]

    with h5py.File(edge_path, "r+") as h5:
        pop0 = h5[f"edges/{population}/0"]
        field = direction + "_section_type"
        backup_and_create_dataset(pop0, field, types, np.uint32)


def _compute_center_point(morph, section_ids, segment_ids, segment_offsets):
    """Compute center point along the segment."""
    segment_start, segment_end = np.zeros((2, len(section_ids), 3))

    for i, (section_id, segment_id) in enumerate(zip(section_ids, segment_ids)):
        if section_id == 0:
            segment_start[i] = morph.soma.center
            segment_end[i] = segment_start[i] + 1
        else:
            section = morph.section(section_id - 1)  # off-by-one gotcha

            if segment_id == len(section.points) - 1:
                # In rare cases, synapse at the end of last segment has a non-existing
                # Segment ID (1 too much) and an offset of zero.
                if segment_offsets[i] != 0:
                    raise RuntimeError(f"Unexpected offset {segment_offsets[i]}")

                # Imaginary segment starting at the end of last segment
                segment_start[i] = section.points[segment_id]
                segment_end[i] = section.points[segment_id] + 1
            else:
                segment_start[i] = section.points[segment_id]
                segment_end[i] = section.points[segment_id + 1]

    along = segment_end - segment_start
    center_point = segment_start + (segment_offsets * along.T / np.linalg.norm(along, axis=1)).T

    return center_point


def _get_data_for_computing_center_points(gid, edge_path, direction, population):
    """Gets the necessary data from edges file to compute center point."""
    with h5py.File(edge_path) as h5:
        pop = h5["edges"][population]
        edge_idx = _get_synapse_ids(pop[f"indices/{INDEX_DIRECTION[direction]}"], gid)

        if edge_idx is None:
            return None

        index = INDEX_ID[direction]
        assert len(set(pop[index][edge_idx])) == 1, f"{index} contains erroneous entries"

        return (
            np.asarray(pop[f"0/{direction}_section_id"][edge_idx]),
            np.asarray(pop[f"0/{direction}_segment_id"][edge_idx]),
            np.asarray(pop[f"0/{direction}_segment_offset"][edge_idx]),
            edge_idx,
            pop[index].attrs["node_population"],
        )


def _transform_morphology(morph, node_path, node_population, gid):
    """Apply transformation to morphology."""
    with h5py.File(node_path, "r") as h5:
        pop0 = h5[f"nodes/{node_population}/0"]
        x = pop0["x"][gid]
        y = pop0["y"][gid]
        z = pop0["z"][gid]
        qw = pop0["orientation_w"][gid]
        qx = pop0["orientation_x"][gid]
        qy = pop0["orientation_y"][gid]
        qz = pop0["orientation_z"][gid]

    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = np.array([x, y, z])
    morph_tool.transform.transform(morph, T)

    return morph.as_immutable()


def _compute_center_point_worker(id_morph, edge_path, node_path, direction, population):
    """Worker function computing the synapse center points."""
    gid, morph_path = id_morph
    data = _get_data_for_computing_center_points(gid, edge_path, direction, population)

    if data is None:
        return None

    section_id, segment_id, segment_offset, edge_idx, node_population = data

    morph = morphio.mut.Morphology(morph_path, morphio.Option.nrn_order)
    morph = _transform_morphology(morph, node_path, node_population, gid)

    center_points = _compute_center_point(morph, section_id, segment_id, segment_offset)

    return np.hstack((center_points, edge_idx[:, np.newaxis]))


def write_center_points(morphologies, population, direction, edge_path, node_path):
    """Write center x, y, z points for given edge population

    Args:
        morphologies(pd.DataFrame): full morph paths with index of NodeID
        population(str): edge_population name
        direction(str): `afferent` or `efferent`
        edge_path(str): path to the edges.h5 file
        node_path(str): path to the *fferent nodes.h5 file
    """
    func = partial(
        _compute_center_point_worker,
        edge_path=edge_path,
        node_path=node_path,
        direction=direction,
        population=population,
    )

    with multiprocessing.Pool() as pool:
        res = pool.map(func, tqdm(morphologies.reset_index().values))

    res = np.concatenate([r for r in res if r is not None])
    positions, edge_ids = res[:, :-1], res[:, -1]

    # sort values by edge ids
    positions = positions[np.argsort(edge_ids)].T

    with h5py.File(edge_path, "r+") as h5:
        pop0 = h5[f"edges/{population}/0"]
        fields = [direction + "_center_" + a for a in list("xyz")]
        for field, position in zip(fields, positions):
            backup_and_create_dataset(pop0, field, position, np.float32)


def _get_data_for_computing_surface_points(gid, edge_path, direction, population):
    """Gets the necessary data from edges file to compute surface point."""
    with h5py.File(edge_path) as h5:
        pop = h5["edges"][population]
        edge_idx = _get_synapse_ids(pop[f"indices/{INDEX_DIRECTION[direction]}"], gid)

        if edge_idx is None:
            return None

        index = INDEX_ID[direction]
        assert len(set(pop[index][edge_idx])) == 1, f"{index} contains erroneous entries"

        opposite = "efferent" if direction == "afferent" else "afferent"

        return (
            np.asarray(h5[f"edges/{population}/0/{direction}_section_id"][edge_idx]),
            np.asarray(h5[f"edges/{population}/0/{direction}_segment_id"][edge_idx]),
            np.asarray(h5[f"edges/{population}/0/{direction}_segment_offset"][edge_idx]),
            np.vstack(
                (
                    np.asarray(h5[f"edges/{population}/0/{direction}_center_x"][edge_idx]),
                    np.asarray(h5[f"edges/{population}/0/{direction}_center_y"][edge_idx]),
                    np.asarray(h5[f"edges/{population}/0/{direction}_center_z"][edge_idx]),
                )
            ).T,
            np.vstack(
                (
                    np.asarray(h5[f"edges/{population}/0/{opposite}_center_x"][edge_idx]),
                    np.asarray(h5[f"edges/{population}/0/{opposite}_center_y"][edge_idx]),
                    np.asarray(h5[f"edges/{population}/0/{opposite}_center_z"][edge_idx]),
                )
            ).T,
            edge_idx,
            pop[index].attrs["node_population"],
        )


def _get_segment_info_for_surface_points(morph, section_ids, segment_ids):
    """Get necessary segment information to compute surface point."""
    radius, delta = np.full((2, len(section_ids)), np.nan)
    segment = np.full((len(section_ids), 3), np.nan)

    for i, (section_id, segment_id) in enumerate(zip(section_ids, segment_ids)):
        if section_id == 0:
            if morph.soma.type != morphio.SomaType.SOMA_SIMPLE_CONTOUR:
                L.warning("Morph soma type is: %s", morph.soma.type)
            radius[i] = morph.soma.max_distance
        else:
            section = morph.section(section_id - 1)  # off-by-one gotcha

            # if touch in the end of the last segment (rare case)
            if segment_id + 1 == len(section.points):
                # In rare cases, synapse at the end of last segment has a non-existing
                # Segment ID (1 too much) and an offset of zero. Assume the segment direction
                # is the same, radius is that of the end of the segment and delta = 0
                segment[i] = np.diff(section.points[segment_id - 1 : segment_id + 1], axis=0)
                radius[i] = section.diameters[segment_id]
                delta[i] = 0

            else:
                segment_slice = slice(segment_id, segment_id + 2)
                segment[i] = np.diff(section.points[segment_slice], axis=0)
                radius[i] = section.diameters[segment_id]
                delta[i] = np.diff(section.diameters[segment_slice])

    return (
        segment[np.all(np.isfinite(segment), axis=1)],
        delta[np.isfinite(delta)],
        radius,
    )


def _compute_surface_point_segment(center_point, segment, connection, radius, delta_radius, offset):
    """Compute surface points for synapses in segments."""
    radial = np.cross(segment, np.cross(connection, segment, axis=1), axis=1)
    radial_norm = np.linalg.norm(radial, axis=1)

    if any(under_limit := radial_norm <= FLOAT_LIMIT):
        L.warning("Radial norm too small")
        radial_norm[under_limit] = FLOAT_LIMIT

    center_radius = radius + delta_radius * offset / np.linalg.norm(segment, axis=1)

    return center_point + center_radius[:, np.newaxis] * radial / radial_norm[:, np.newaxis]


def _compute_surface_point(
    morph, section_id, segment_id, segment_offset, center_point, opposite_center
):
    """Compute the surface point for the synapses"""
    segment, delta_radius, radius = _get_segment_info_for_surface_points(
        morph, section_id, segment_id
    )

    surface_points = np.full_like(center_point, np.nan)

    connection = opposite_center - center_point
    len_connection = np.linalg.norm(connection, axis=1)
    zeros = len_connection < FLOAT_LIMIT

    if any(zeros):
        # surface too close?
        L.warning("Surface too close, assuming equal to center")
        surface_points[zeros] = center_point[zeros]

    # for soma
    soma_mask = np.logical_and(np.invert(zeros), section_id == 0)
    surface_points[soma_mask] = (
        center_point[soma_mask]
        + connection[soma_mask]
        / len_connection[soma_mask, np.newaxis]
        * radius[soma_mask, np.newaxis]
    )

    # for segments
    segment_mask = np.logical_and(np.invert(soma_mask), np.invert(zeros))
    surface_points[segment_mask] = _compute_surface_point_segment(
        center_point[segment_mask],
        segment,
        connection[segment_mask],
        radius[segment_mask],
        delta_radius,
        segment_offset[segment_mask],
    )

    return surface_points


def _get_surface_point_worker(id_morph, edge_path, direction, population, node_path):
    """Worker function computing the synapse surface points."""
    # pylint: disable=too-many-locals
    gid, morph_path = id_morph
    data = _get_data_for_computing_surface_points(gid, edge_path, direction, population)

    if data is None:
        return None

    (
        section_id,
        segment_id,
        segment_offset,
        center_point,
        opposite_center,
        edge_idx,
        node_population,
    ) = data

    morph = morphio.mut.Morphology(morph_path, morphio.Option.nrn_order)
    morph = _transform_morphology(morph, node_path, node_population, gid)

    surface_points = _compute_surface_point(
        morph, section_id, segment_id, segment_offset, center_point, opposite_center
    )
    return np.hstack((surface_points, edge_idx[:, np.newaxis]))


def write_surface_points(morphologies, population, direction, edge_path, node_path):
    """Write surface x, y, z points for given edge population

    Args:
        morphologies(pd.DataFrame): full morph paths with index of NodeID
        population(str): edge_population name
        direction(str): `afferent` or `efferent`
        edge_path(str): path to the edges.h5 file
        node_path(str): path to the *fferent nodes.h5 file
    """
    func = partial(
        _get_surface_point_worker,
        edge_path=edge_path,
        direction=direction,
        population=population,
        node_path=node_path,
    )

    with multiprocessing.Pool() as pool:
        res = pool.map(func, tqdm(morphologies.reset_index().values))

    res = np.concatenate([r for r in res if r is not None])
    positions, edge_ids = res[:, :-1], res[:, -1]

    # sort values by edge ids
    positions = positions[np.argsort(edge_ids)].T

    with h5py.File(edge_path, "r+") as h5:
        pop0 = h5[f"edges/{population}/0"]
        fields = [direction + "_surface_" + a for a in list("xyz")]
        for field, position in zip(fields, positions):
            backup_and_create_dataset(pop0, field, position, np.float32)
