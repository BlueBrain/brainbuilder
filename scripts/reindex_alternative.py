# SPDX-License-Identifier: Apache-2.0
'''alternative reindex implementation'''
import glob
import os
from collections import OrderedDict

import h5py
import numpy as np
from morphio import AnnotationType
from morphio.mut import Morphology
from neurom.morphmath import path_distance, path_fraction_id_offset


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'reindex')
MORPHS_PATH = os.path.join(DATA_PATH, 'morphs')


def _get_filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def _cut_points_until(points, cut_point):
    """Returns the points before a ``cut_point`` in ``points``.

    Args:
        points: points
        cut_point: point where to cut ``points``

    Returns:
        np.array: points before ``cut_point`` inclusive with ``cut_point`` itself.
    """
    cut_start_idx = np.where((points == cut_point).all(axis=1))[0]
    # assure that only one such point exists
    assert len(cut_start_idx) == 1
    return points[:cut_start_idx[0] + 1, :]


def _get_merged_sections(merged_morphology):
    """Gets merged single child sections.

    Args:
        merged_morphology: morphology with merged single children

    Returns:
        dict: single child section id => its original unmerged points
    """
    return {annotation.section_id: annotation.points for annotation in merged_morphology.annotations
            if annotation.type == AnnotationType.single_child}


def _map_unmerged_to_merged_sections(merged_morphology, merged_sections):
    """Maps ids of original unmerged sections to ids of new merged sections.

    Args:
        merged_morphology: morphology with merged single children
        merged_sections: dict[single child section id, single child section points]

    Returns:
        dict: unmerged section id => merged section id
    """
    unmerged_section_num = len(merged_sections) + len(merged_morphology.sections)
    unmerged_to_merged = OrderedDict({i: i for i in range(unmerged_section_num)})
    for unmerged_sec_id in merged_sections.keys():
        for i in range(unmerged_sec_id, len(unmerged_to_merged)):
            unmerged_to_merged[i] -= 1
    return unmerged_to_merged


def _get_absorbing_sections(merged_morphology, merged_sections, unmerged_to_merged):
    """Gets original unmerged sections that absorbed their single child sections.

    Args:
        merged_morphology: morphology with merged single children
        merged_sections: dict[single child section id, single child section points]
        unmerged_to_merged: dict[unmerged section id, merged section id]

    Returns:
        dict: unmerged absorbing section id => its original unmerged points
    """
    absorb_sections = {}
    for unmerged_sec_id in merged_sections:
        absorb_sec_id = unmerged_sec_id - 1
        if absorb_sec_id not in merged_sections:
            merged_sec_id = unmerged_to_merged[absorb_sec_id]
            unmerged_sec_points = merged_sections[unmerged_sec_id]
            merged_sec_points = merged_morphology.sections[merged_sec_id].points
            absorb_sec_points = _cut_points_until(merged_sec_points, unmerged_sec_points[0])
            absorb_sections[absorb_sec_id] = absorb_sec_points
    return absorb_sections


def _map_merged_section_pos(unmerged_sec_points, unmerged_pos, merged_sec_points):
    """Maps unmerged section pos to its new merged section pos

    Args:
        unmerged_sec_points: section points of unmerged section
        unmerged_pos: unmerged section pos
        merged_sec_points: new merged section points

    Returns:
        float: merged section pos
    """
    unmerged_sec_len = path_distance(unmerged_sec_points)
    unmerged_pos_len = unmerged_pos * unmerged_sec_len
    merged_sec_len = path_distance(merged_sec_points)
    before_points = _cut_points_until(merged_sec_points, unmerged_sec_points[0])
    before_len = path_distance(before_points)
    merged_pos = (before_len + unmerged_pos_len) / merged_sec_len
    return merged_pos


def update_morphologies(asc_morphs, nodes, _, output, edges):  # pylint: disable=too-many-locals, too-many-statements
    """Update h5 morphologies"""
    # pylint: disable=import-outside-toplevel
    from voxcell import CellCollection

    morph_filenames = CellCollection.load(nodes).as_dataframe()['morphology']
    morph_filenames.index = morph_filenames.index - 1
    all_morph_files = glob.glob(os.path.join(asc_morphs, '*.asc'))
    morph_files = [morph_file for morph_file in all_morph_files
                   if _get_filename(morph_file) in morph_filenames.unique()]

    morph_morphologies = {}
    morph_merged_sections = {}
    morph_unmerged_to_merged = {}
    for morph_file in morph_files:
        merged_morph = Morphology(morph_file)
        merged_morph.write(os.path.join(output, _get_filename(morph_file) + '.swc'))
        merged_sections = _get_merged_sections(merged_morph)
        if len(merged_sections) > 0:
            unmerged_to_merged = _map_unmerged_to_merged_sections(merged_morph, merged_sections)
            absorbing_sections = _get_absorbing_sections(
                merged_morph, merged_sections, unmerged_to_merged)
            merged_sections.update(absorbing_sections)

            morph_filename = _get_filename(morph_file)
            morph_morphologies[morph_filename] = merged_morph
            morph_merged_sections[morph_filename] = merged_sections
            morph_unmerged_to_merged[morph_filename] = unmerged_to_merged

    morph_filenames = morph_filenames.loc[morph_filenames.isin(list(morph_morphologies.keys()))]
    print('starting edges')

    for edges_file in edges:
        with h5py.File(edges_file, 'r+') as h5:
            for morph_filename, node_ids in morph_filenames.groupby(morph_filenames):
                if morph_filename not in morph_morphologies:
                    print('NOT presented morph', morph_filename)
                    continue
                merged_morph = morph_morphologies[morph_filename]
                pop = h5['edges/default']
                grp = pop['0']
                merged_sections = morph_merged_sections[morph_filename]
                unmerged_to_merged = morph_unmerged_to_merged[morph_filename]

                # edges of node_ids with changed sections
                afferent_edge_idxs = np.logical_and(
                    np.isin(pop['target_node_id'], node_ids.index),
                    np.isin(grp['afferent_section_id'][:] - 1, list(unmerged_to_merged.keys())))
                for edge_idx in np.where(afferent_edge_idxs)[0]:
                    unmerged_sec_id = grp['afferent_section_id'][edge_idx] - 1
                    merged_sec_id = unmerged_to_merged[unmerged_sec_id]
                    if unmerged_sec_id in merged_sections.keys():
                        unmerged_pos = grp['afferent_section_pos'][edge_idx]
                        unmerged_sec_points = merged_sections[unmerged_sec_id]
                        merged_sec_points = merged_morph.sections[merged_sec_id].points
                        merged_pos = _map_merged_section_pos(
                            unmerged_sec_points, unmerged_pos, merged_sec_points)
                        grp['afferent_section_pos'][edge_idx] = merged_pos
                        seg_id, offset = path_fraction_id_offset(merged_sec_points, merged_pos)
                        grp['afferent_segment_id'][edge_idx] = seg_id
                        grp['afferent_segment_offset'][edge_idx] = offset
                    grp['afferent_section_id'][edge_idx] = merged_sec_id + 1

                # edges of node_ids with changed sections
                efferent_edge_idxs = np.logical_and(
                    np.isin(pop['source_node_id'], node_ids.index),
                    np.isin(grp['efferent_section_id'][:] - 1, list(unmerged_to_merged.keys())))
                for edge_idx in np.where(efferent_edge_idxs)[0]:
                    unmerged_sec_id = grp['efferent_section_id'][edge_idx] - 1
                    merged_sec_id = unmerged_to_merged[unmerged_sec_id]
                    if unmerged_sec_id in merged_sections.keys():
                        unmerged_pos = grp['efferent_section_pos'][edge_idx]
                        unmerged_sec_points = merged_sections[unmerged_sec_id]
                        merged_sec_points = merged_morph.sections[merged_sec_id].points
                        merged_pos = _map_merged_section_pos(
                            unmerged_sec_points, unmerged_pos, merged_sec_points)
                        grp['efferent_section_pos'][edge_idx] = merged_pos
                        seg_id, offset = path_fraction_id_offset(merged_sec_points, merged_pos)
                        grp['efferent_segment_id'][edge_idx] = seg_id
                        grp['efferent_segment_offset'][edge_idx] = offset
                    grp['efferent_section_id'][edge_idx] = merged_sec_id + 1
