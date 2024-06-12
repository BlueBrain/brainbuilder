# SPDX-License-Identifier: Apache-2.0
"""
Temporary SONATA converters.

https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md
"""

import logging
import os
import re

import h5py
import numpy as np
import pandas as pd
from voxcell import CellCollection

from brainbuilder.exceptions import BrainBuilderError
from brainbuilder.utils import dump_json

L = logging.getLogger("brainbuilder")


def _add_me_info(cells, mecombo_info):
    assert not mecombo_info.duplicated(
        "combo_name"
    ).any(), "Duplicate me-combos, strange ModelMangement run?"

    mecombo_info = mecombo_info.set_index("combo_name")

    if "me_combo" in cells.properties:
        me_combos = cells.properties["me_combo"]
        missing_me_combos = me_combos[mecombo_info.index.get_indexer(me_combos) == -1]
        if len(missing_me_combos) != 0:
            raise BrainBuilderError(
                f"The me_combo :{missing_me_combos.tolist()} are missing from "
                f"the e-model release"
            )

        mecombo_params = mecombo_info.loc[me_combos]
        for prop, column in mecombo_params.items():
            values = column.values
            if prop == "emodel":
                values = [("hoc:" + v) for v in values]
                cells.properties["model_template"] = values
            else:
                cells.properties[cells.SONATA_DYNAMIC_PROPERTY + prop] = values


def provide_me_info(
    cells_path,
    out_cells_path,
    model_type="biophysical",
    mecombo_info_path=None,
    population=None,
):
    """Provides cells collection with ME info."""
    cells = CellCollection.load(cells_path)
    if population is not None:
        cells.population_name = population

    def usecols(name):
        """Pick the needed columns."""
        return name not in ("morph_name", "layer", "subregion", "fullmtype", "etype")

    if mecombo_info_path is not None:
        mecombo_info = pd.read_csv(mecombo_info_path, sep=r"\s+", usecols=usecols)
        _add_me_info(cells, mecombo_info)

    cells.properties["model_type"] = model_type
    cells.save(out_cells_path)


def _write_edge_group(group, out):
    # TODO: pick only those used, remap to those mentioned in "spec"
    # conductance -> syn_weight
    # ...
    MAPPING = {
        "morpho_section_id_post": "afferent_section_id",
        "morpho_segment_id_post": "afferent_segment_id",
        "morpho_offset_segment_post": "afferent_segment_offset",
        "morpho_section_fraction_post": "afferent_section_pos",
        "morpho_section_type_post": "afferent_section_type",
        "morpho_section_id_pre": "efferent_section_id",
        "morpho_segment_id_pre": "efferent_segment_id",
        "morpho_offset_segment_pre": "efferent_segment_offset",
        "morpho_section_fraction_pre": "efferent_section_pos",
        "morpho_section_type_pre": "efferent_section_type",
        "morpho_spine_length": "spine_length",
        "morpho_type_id_pre": "efferent_morphology_id",
        "position_center_post_x": "afferent_center_x",
        "position_center_post_y": "afferent_center_y",
        "position_center_post_z": "afferent_center_z",
        "position_center_pre_x": "efferent_center_x",
        "position_center_pre_y": "efferent_center_y",
        "position_center_pre_z": "efferent_center_z",
        "position_contour_post_x": "afferent_surface_x",
        "position_contour_post_y": "afferent_surface_y",
        "position_contour_post_z": "afferent_surface_z",
        "position_contour_pre_x": "efferent_surface_x",
        "position_contour_pre_y": "efferent_surface_y",
        "position_contour_pre_z": "efferent_surface_z",
    }
    for prop in group:
        if prop in ("connected_neurons_post", "connected_neurons_pre"):
            continue
        name = MAPPING.get(prop, prop)
        L.info("'%s' -> '%s'...", prop, name)
        group.copy(prop, out, name=name)


def _write_edge_index(index, out):
    index.copy("neuron_id_to_range", out, name="node_id_to_ranges")
    index.copy("range_to_synapse_id", out, name="range_to_edge_id")


def _write_edge_population(population, source, target, out):
    properties, indices = population["properties"], population["indexes"]
    count = len(properties["connected_neurons_pre"])

    L.info("Writing population-level datasets...")

    L.info("'edge_type_id'...")
    out.create_dataset("edge_type_id", shape=(count,), dtype=np.int8, fillvalue=-1)

    L.info("'source_node_id'...")
    properties.copy("connected_neurons_pre", out, name="source_node_id")
    out["source_node_id"].attrs["node_population"] = str(source)

    L.info("'target_node_id'...")
    properties.copy("connected_neurons_post", out, name="target_node_id")
    out["target_node_id"].attrs["node_population"] = str(target)

    L.info("Writing group-level datasets...")
    _write_edge_group(properties, out.create_group("0"))

    L.info("Writing indices...")

    L.info("'source_to_target'...")
    _write_edge_index(
        indices["connected_neurons_pre"], out.create_group("indices/source_to_target")
    )

    L.info("'target_to_source'...")
    _write_edge_index(
        indices["connected_neurons_post"], out.create_group("indices/target_to_source")
    )


def write_edges_from_syn2(syn2_path, population, source, target, out_h5_path):
    """Export SYN2 to SONATA edge collection."""
    with h5py.File(syn2_path, "r") as syn2:
        with h5py.File(out_h5_path, "w") as h5f:
            assert len(syn2["/synapses"]) == 1
            syn2_population = next(iter(syn2["/synapses"].values()))
            _write_edge_population(
                syn2_population, source, target, h5f.create_group(f"/edges/{population}")
            )


def validate_node_set(node_set, cells):
    """Validate a node_set file"""

    def get_ids(cells, node_set, target):
        """Get ids for a target."""
        if isinstance(target, list):
            return np.hstack([get_ids(cells, node_set, node_set[t]) for t in target])
        if "node_id" in target:
            return np.array(target["node_id"]) + 1
        return cells.ids(target)

    for name, target in node_set.items():
        L.info("Validating %s...", name)
        target_ids = cells.ids(name)
        node_set_ids = get_ids(cells, node_set, target)
        if (
            np.setdiff1d(target_ids, node_set_ids).size > 0
            or np.setdiff1d(node_set_ids, target_ids).size > 0
        ):
            raise BrainBuilderError(f"Target {name} differs in target file and node set file")


def _parse_targets(target_files):
    """Return a dict of all targets.

    This function basically repeats the implementation of targets for bluepy<2.4.1. As of 2.4.1
    the implementation is optimized to reduce memory. That's why the unoptimized implementation
    is kept here for conversion purposes only.
    """

    def _parse_target_file(filepath):
        """Parse .target file, return generator of `Target`s."""
        with open(filepath, "r", encoding="utf-8") as f:
            contents = f.read()

        target_regex = re.compile(
            r"""
            Target               # Start token
            \s+                  # 1 or more whitespace
            (?P<type>\S+)        # type
            \s+                  # 1 or more whitespace
            (?P<name>\S+)        # name
            \s+                  # 1 or more whitespace
            (?P<start>\{)        # start brace
            (?P<contents>[^}]*)  # not the end brace (contents is needed by brainbuilder)
            (?P<end>})           # end brace
            """,
            re.VERBOSE | re.MULTILINE | re.DOTALL,
        )
        for m in target_regex.finditer(contents):
            yield m.group("name"), m.group("contents").strip().split()

    targets = {}
    for file in target_files:
        for target_name, target_content in _parse_target_file(file):
            if target_name in targets:
                raise BrainBuilderError(f"{target_name} is duplicated, please check target files")
            targets[target_name] = target_content

    return targets


def write_node_set_from_targets(target_files, output_file, cells_path):
    """Write SONATA node_set from `target_files`

    This function allows to directly convert a set of target files created from a mvd3 file
    into the corresponding node_set.json file. This is useful if user does not have the yaml target
    rules anymore.

    The 'brainbuilder targets node-sets' should be preferred if possible.
    """
    from bluepy import Circuit  # pylint: disable=import-error,import-outside-toplevel

    cells = Circuit({"cells": cells_path, "targets": target_files}).cells
    if not os.path.basename(output_file) == "node_sets.json":
        basename = os.path.basename(output_file)
        L.warning(
            'basename "%s" is not "node_sets.json" change your config file accordingly.', basename
        )

    properties = set.intersection(cells.available_properties, ["etype", "mtype", "region"])
    # Create a mapping from unique property values (possible target names) to property names
    # e.g. {'dNAC': {'etype': 'dNAC'}} (i.e, for all cells in target 'dNAC': etype==dNAC)
    mapping = {
        target_name: {property_name: target_name}
        for property_name in properties
        for target_name in cells.get(properties=property_name).unique()
    }
    mapping.update({"Excitatory": {"synapse_class": "EXC"}, "Inhibitory": {"synapse_class": "INH"}})

    targets = _parse_targets(target_files)
    re_layer = re.compile(r"^layer(\d)$", re.IGNORECASE)
    re_node_id = re.compile(r"^a\d+$")

    def target_to_node_set_entry(target_name, target_content):
        L.info("Converting %s...", target_name)
        if target_name in mapping:
            return mapping[target_name]
        if re_layer.match(target_name):
            return {"layer": str(re_layer.match(target_name).group(1))}
        if re_node_id.match(target_content[0]):
            # targets are built from a mvd3 file so indexing starts from 1 compare to 0 in SONATA
            return {"node_id": (cells.ids(target_name) - 1).tolist()}

        return list(set(target_content))

    output_dict = {
        name: target_to_node_set_entry(name, contents) for name, contents in targets.items()
    }

    validate_node_set(output_dict, cells)

    dump_json(output_file, output_dict)
