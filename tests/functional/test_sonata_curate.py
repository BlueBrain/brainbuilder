# SPDX-License-Identifier: Apache-2.0
"""Application of `curate` functionality to existing Sonata circuits."""

import re
import shutil
from pathlib import Path

import pytest
from bluepysnap.circuit_validation import validate

from brainbuilder.utils.sonata import curate
from brainbuilder.utils.sonata.write_config import write_network_config

CIRCUIT_PATH = Path("/gpfs/bbp.cscs.ch/project/proj42/circuits/CA1.O0/20191017/")


@pytest.mark.skipif(not CIRCUIT_PATH.exists(), reason="Do not have access to proj42")
def test_hippocampus(tmp_path):
    """Example of curating a Hippocampus circuit"""

    proj_edges_file = CIRCUIT_PATH / "projections" / "v3.2k" / "O0_ca1_20191017_sorted.sonata"
    edges_file = CIRCUIT_PATH / "sonata" / "networks" / "edges" / "functional" / "All" / "edges.h5"
    nodes_file = CIRCUIT_PATH / "sonata" / "networks" / "nodes" / "All" / "nodes.h5"

    original_dir = tmp_path / "original"
    original_dir.mkdir()

    # `copyfile` so that the permissions aren't preserved
    shutil.copyfile(edges_file, original_dir / edges_file.name)
    edges_file = original_dir / edges_file.name

    shutil.copyfile(nodes_file, original_dir / nodes_file.name)
    nodes_file = original_dir / nodes_file.name

    shutil.copyfile(proj_edges_file, original_dir / proj_edges_file.name)
    proj_edges_file = original_dir / proj_edges_file.name

    target_nodes_name = "hippocampus_neurons"
    source_nodes_name = "hippocampus_projections"
    syn_type = "chemical"

    curate.rename_node_population(nodes_file, target_nodes_name)
    curate.set_group_attribute(
        nodes_file, "nodes", target_nodes_name, "0", "model_type", "biophysical", True
    )
    curate.rewire_edge_population(edges_file, nodes_file, nodes_file, syn_type)
    curate.add_edge_type_id(edges_file, curate.get_population_name(edges_file))

    proj_source_nodes_file = curate.create_projection_source_nodes(
        proj_edges_file, original_dir, source_nodes_name, fix_offset=True
    )
    start, _ = curate.get_source_nodes_range(proj_edges_file, edge_population_name="default")
    curate.correct_source_nodes_offset(
        proj_edges_file, edge_population_name="default", offset=start
    )

    curate.rewire_edge_population(proj_edges_file, proj_source_nodes_file, nodes_file, syn_type)

    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()

    curate.merge_h5_files([nodes_file, proj_source_nodes_file], "nodes", curated_dir / "nodes.h5")
    curate.merge_h5_files([edges_file, proj_edges_file], "edges", curated_dir / "edges.h5")

    # update dtypes
    curate.update_node_dtypes(
        curated_dir / "nodes.h5", population_name=target_nodes_name, population_type="biophysical"
    )
    curate.update_node_dtypes(
        curated_dir / "nodes.h5", population_name=source_nodes_name, population_type="biophysical"
    )

    curate.update_edge_dtypes(
        curated_dir / "edges.h5",
        population_name="hippocampus_neurons__hippocampus_neurons__chemical",
        population_type=syn_type,
        virtual=False,
    )
    curate.update_edge_dtypes(
        curated_dir / "edges.h5",
        population_name="hippocampus_projections__hippocampus_neurons__chemical",
        population_type=syn_type,
        virtual=True,
    )

    sonata_config_file = curated_dir / "circuit_config.json"
    curated_dir = curated_dir.resolve()
    write_network_config(
        base_dir="/",
        morph_dir="/gpfs/bbp.cscs.ch/project/proj12/NSE/brainbuilder/CA1.O0-20191017/entities/morphologies",
        emodel_dir="/gpfs/bbp.cscs.ch/project/proj12/NSE/brainbuilder/CA1.O0-20191017/entities/hoc",
        nodes_dir=curated_dir,
        nodes=[
            {
                "nodes_file": "nodes.h5",
                "populations": {
                    target_nodes_name: {"type": "biophysical"},
                    "hippocampus_projections": {"type": "virtual"},
                },
            }
        ],
        node_sets="",
        edges_dir=curated_dir,
        edges_suffix="",
        edges=[
            {
                "edges_file": "edges.h5",
                "populations": {
                    "hippocampus_neurons__hippocampus_neurons__chemical": {"type": "chemical"},
                    "hippocampus_projections__hippocampus_neurons__chemical": {"type": "chemical"},
                },
            }
        ],
        output_path=sonata_config_file,
    )

    # the hippocampus circuit used in the tests is missing the following
    # 'mandatory' elements, but we're not going to add them
    allowed_missing_hippocampus_attributes = {
        "afferent_section_id",
        "afferent_section_pos",
        "afferent_section_type",
        "afferent_segment_id",
        "afferent_segment_offset",
        "afferent_surface_x",
        "afferent_surface_y",
        "afferent_surface_z",
        "efferent_center_x",
        "efferent_center_y",
        "efferent_center_z",
        "efferent_section_id",
        "efferent_section_pos",
        "efferent_section_type",
        "efferent_segment_id",
        "efferent_segment_offset",
        "efferent_surface_x",
        "efferent_surface_y",
        "efferent_surface_z",
        "spine_length",
    }

    errors = []
    for err in validate(str(sonata_config_file), skip_slow=False, only_errors=True):
        filtered_line = []
        lines = err.message.splitlines()
        for line in lines:
            m = re.match("^.*: '([^']*)' is a required property$", line)
            if not m or m.group(1) not in allowed_missing_hippocampus_attributes:
                filtered_line.append(line)

        if len(filtered_line) == 1 and filtered_line[0] == lines[0]:
            # first line of error message is the h5 path, so if that's all that's left
            # we don't *actually* have an error
            continue

        errors.append("\n".join(filtered_line))

    assert errors == []
