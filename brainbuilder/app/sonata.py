# SPDX-License-Identifier: Apache-2.0
"""Tools for working with SONATA."""

# pylint: disable=import-outside-toplevel
import glob
import os
import sys
from pathlib import Path

import click

from brainbuilder.app._utils import REQUIRED_PATH, REQUIRED_PATH_DIR
from brainbuilder.utils import dump_json, load_json


def _get_morphs_with_full_path(nodes_path, morph_path):
    from voxcell import CellCollection

    # load morphs, extend them to have the full path to the h5 file
    morph_path += "/" if not morph_path.endswith("/") else ""
    morphologies = CellCollection.load(nodes_path).as_dataframe()["morphology"]
    morphologies = morph_path + morphologies.astype(str) + ".h5"
    morphologies.index -= 1

    return morphologies


@click.group()
def app():
    """Tools for working with SONATA"""


@app.command()
@click.argument("mvd3")
@click.option("-o", "--output", help="Path to output SONATA nodes", required=True)
@click.option("--model-type", help="Type of neurons", required=True)
@click.option("--mecombo-info", help="Path to TSV file with ME-combo table", default=None)
@click.option("--population", help="Population name", default="default", show_default=True)
def from_mvd3(mvd3, output, model_type, mecombo_info, population):
    """Convert MVD3 to SONATA nodes"""
    from brainbuilder.utils.sonata import convert

    assert mvd3.endswith(".mvd3"), mvd3 + ' must end with ".mvd3" suffix'
    convert.provide_me_info(mvd3, output, model_type, mecombo_info, population)


@app.command()
@click.argument("cells-path")
@click.option("-o", "--output", help="Path to output SONATA nodes", required=True)
@click.option("--model-type", help="Type of neurons", required=True)
@click.option("--mecombo-info", help="Path to TSV file with ME-combo table", default=None)
def provide_me_info(cells_path, output, model_type, mecombo_info):
    """Provide SONATA nodes with MorphoElectrical info"""
    from brainbuilder.utils.sonata import convert

    convert.provide_me_info(cells_path, output, model_type, mecombo_info)


@app.command()
@click.argument("syn2")
@click.option("--population", help="Population name", default="default", show_default=True)
@click.option("--source", help="Source node population name", default="default", show_default=True)
@click.option("--target", help="Target node population name", default="default", show_default=True)
@click.option("-o", "--output", help="Path to output HDF5", required=True)
def from_syn2(syn2, population, source, target, output):
    """Convert SYN2 to SONATA edges"""
    from brainbuilder.utils.sonata import convert

    convert.write_edges_from_syn2(
        syn2_path=syn2,
        population=population,
        source=source,
        target=target,
        out_h5_path=output,
    )


@app.command()
@click.option("--base-dir", help="Path to base directory", required=True)
@click.option("--morph-dir", help="Morphologies directory (relative to BASE_DIR)", required=True)
@click.option(
    "--emodel-dir", help="Cell electrical models directory (relative to BASE_DIR)", required=True
)
@click.option("--nodes-dir", help="Node files directory (relative to BASE_DIR)", required=True)
@click.option(
    "--nodes",
    help=(
        "Nodes file(s) (';'-separated). For each file, the populations must be specified "
        "in the format 'path:pop1,pop2...'"
    ),
    required=True,
)
@click.option("--node-sets", help="Node sets file (JSON)", required=True)
@click.option("--edges-dir", help="Edge files directory (relative to BASE_DIR)", required=True)
@click.option("--edges-suffix", help="Edge file suffix", default="")
@click.option(
    "--edges",
    help=(
        "Edges file(s) (';'-separated). For each file, the populations must be specified "
        "in the format 'path:pop1,pop2...'"
    ),
    required=True,
)
@click.option("-o", "--output", help="Path to output file (JSON)", required=True)
def network_config(
    base_dir,
    morph_dir,
    emodel_dir,
    nodes_dir,
    nodes,
    node_sets,
    edges_dir,
    edges_suffix,
    edges,
    output,
):
    """Write SONATA network config"""
    # pylint: disable=too-many-arguments
    from brainbuilder.utils.sonata import write_config

    write_config.write_network_config(
        base_dir=base_dir,
        morph_dir=morph_dir,
        emodel_dir=emodel_dir,
        nodes_dir=nodes_dir,
        nodes=nodes.split(";"),
        node_sets=node_sets,
        edges_dir=edges_dir,
        edges_suffix=edges_suffix,
        edges=edges.split(";"),
        output_path=output,
    )


@app.command()
@click.option(
    "--input-dir",
    type=REQUIRED_PATH_DIR,
    help="Path to the input directory containing the targets files",
)
@click.option("--target-file", multiple=True, type=REQUIRED_PATH, help="Path to target file")
@click.option("-c", "--cells-path", help="Path to cells file", required=True)
@click.option("-o", "--output", help="Path to output the .json file", required=True)
def node_set_from_targets(input_dir, target_file, cells_path, output):
    """Convert target files into a single node_set.json like file.

    Please check 'brainbuilder targets node-sets' also.
    """
    from brainbuilder.utils.sonata import convert

    target_files = set(target_file)

    if input_dir is not None:
        target_files = target_files | set(glob.glob(f"{input_dir}/*.target"))

    if target_files:
        convert.write_node_set_from_targets(target_files, output, cells_path)
    else:
        click.secho("No target files specified; output won't exist", fg="red")


@app.command()
@click.option(
    "--h5-morphs", required=True, type=REQUIRED_PATH_DIR, help="Path to h5 morphology directory"
)
@click.option("--morphdb", required=True, type=REQUIRED_PATH, help="Path to extNeuronDB.dat")
def check_morphologies(h5_morphs, morphdb):
    """Make sure the h5 morphology files pass the required SONATA invariants"""
    from brainbuilder.utils import bbp
    from brainbuilder.utils.sonata import curate

    morphs = bbp.load_extneurondb(morphdb).morphology.to_list()
    with click.progressbar(morphs) as morphs:
        incorrect_ordering, have_unifurcations = curate.check_morphology_invariants(
            Path(h5_morphs), morphs
        )

    if incorrect_ordering:
        click.secho(f"The following morphs have incorrect ordering: {incorrect_ordering}", fg="red")

    if have_unifurcations:
        click.secho(f"The following morphs have unifurcations: {have_unifurcations}", fg="red")

    if not incorrect_ordering and not have_unifurcations:
        click.secho("morphologies appear correct", fg="green")

    sys.exit(int(len(have_unifurcations) > 0 or len(incorrect_ordering) > 0))


@app.command()
@click.option(
    "--h5-morphs", required=True, type=REQUIRED_PATH_DIR, help="Path to h5 morphology directory"
)
@click.option(
    "-o", "--output", required=True, help="Path to output directory for HDF5 morphologies"
)
def update_morphologies(h5_morphs, output):
    """Update h5 morphologies"""
    from brainbuilder.utils.sonata import reindex

    assert not os.path.exists(output), "output directory must not already exist"

    h5_updates = reindex.generate_h5_updates(h5_morphs)

    reindex.write_new_h5_morphs(h5_updates, h5_morphs, output)

    h5_updates_path = Path(output) / "h5_updates.json"
    dump_json(h5_updates_path, h5_updates)

    click.echo(f"h5_updates output to {h5_updates_path}")


@app.command()
@click.option(
    "--h5-updates",
    required=True,
    type=REQUIRED_PATH,
    help="h5_updates.json produced by update_morphologies",
)
@click.option("--nodes", required=True, type=REQUIRED_PATH, help="Node file")
@click.option("--population", default="default", show_default=True, help="Population name")
@click.argument("edges", nargs=-1, required=True)
def update_edge_population(h5_updates, nodes, population, edges):
    """Given h5_updates from removing single children, update synapses"""
    from voxcell import CellCollection

    from brainbuilder.utils.sonata import reindex

    h5_updates = load_json(h5_updates)

    for v in h5_updates.values():
        v["new_parents"] = [int(k) for k in v["new_parents"]]
        v["new_segment_offset"] = {int(k): vv for k, vv in v["new_segment_offset"].items()}

    morphologies = CellCollection.load(nodes).as_dataframe()["morphology"]
    morphologies.index = morphologies.index - 1
    for edge in edges:
        reindex.apply_edge_updates(morphologies, edge, h5_updates, population)


@app.command()
@click.option(
    "--morph-path", required=True, type=REQUIRED_PATH_DIR, help="path to h5 morphology files"
)
@click.option("--population", default="default", show_default=True, help="Population name")
@click.option("--nodes", required=True, type=REQUIRED_PATH, help="Node file")
@click.option("--direction", type=click.Choice(["afferent", "efferent"]), required=True)
@click.argument("edge-file", required=True)
def update_edge_pos(morph_path, population, nodes, direction, edge_file):
    """Using: section_id, segment_id and offset, create the sonata *_section_pos"""
    from brainbuilder.utils.sonata import reindex

    morphologies = _get_morphs_with_full_path(nodes, morph_path)
    reindex.write_sonata_pos(morphologies, population, direction, edge_file)


@app.command()
@click.option(
    "--morph-path", required=True, type=REQUIRED_PATH_DIR, help="path to h5 morphology files"
)
@click.option("--population", default="default", show_default=True, help="Population name")
@click.option("--nodes", required=True, type=REQUIRED_PATH, help="Node file")
@click.option("--direction", type=click.Choice(["afferent", "efferent"]), required=True)
@click.argument("edge-file", required=True, type=REQUIRED_PATH)
def update_edge_section_types(morph_path, population, nodes, direction, edge_file):
    """Update edge afferent/efferent section types using section_id."""
    from brainbuilder.utils.sonata import reindex

    morphologies = _get_morphs_with_full_path(nodes, morph_path)
    reindex.write_section_types(morphologies, population, direction, edge_file)


@app.command()
@click.option(
    "--morph-path", required=True, type=REQUIRED_PATH_DIR, help="path to h5 morphology files"
)
@click.option("--population", default="default", show_default=True, help="Population name")
@click.option("--nodes", required=True, type=REQUIRED_PATH, help="Node file")
@click.option("--direction", type=click.Choice(["afferent", "efferent"]), required=True)
@click.argument("edge-file", required=True, type=REQUIRED_PATH)
def update_edge_center_points(morph_path, population, nodes, direction, edge_file):
    """Update edge afferent/efferent center points."""
    from brainbuilder.utils.sonata import reindex

    morphologies = _get_morphs_with_full_path(nodes, morph_path)
    reindex.write_center_points(morphologies, population, direction, edge_file, nodes)


@app.command()
@click.option("--population", default="default", show_default=True, help="Population name")
@click.argument("edge-file", required=True, type=REQUIRED_PATH)
def update_projection_efferent_section_type(population, edge_file):
    """Write projections' efferent section types as axons."""
    import h5py
    import numpy as np
    from morphio import SectionType

    from brainbuilder.utils.sonata import reindex

    with h5py.File(edge_file, "r+") as h5:
        pop0 = h5["edges"][population]["0"]
        datalen = len(pop0[list(pop0)[0]])
        types = np.full(datalen, int(SectionType.axon))
        reindex.backup_and_create_dataset(pop0, "efferent_section_type", types, np.uint32)


@app.command()
@click.option("--attribute", required=True, help="Name of attribute to split on")
@click.option("--nodes", required=True, type=REQUIRED_PATH, help="Input node file")
@click.option("--edges", required=True, type=REQUIRED_PATH, help="Input edge file")
@click.option("-o", "--output", required=True, type=REQUIRED_PATH_DIR, help="Output directory")
def split_population(attribute, nodes, edges, output):
    """Split a single Node and Edges file into multiple nodes and edges based on attribute"""
    from brainbuilder.utils.sonata import split_population as module

    module.split_population(output, attribute, nodes, edges)


@app.command()
@click.option("--nodeset", required=True, help="Name of nodeset")
@click.option(
    "--nodeset-path", required=True, type=REQUIRED_PATH, help="path to node_sets.json file"
)
@click.option("--nodes", required=True, type=REQUIRED_PATH, help="Input node file")
@click.option("--edges", required=True, type=REQUIRED_PATH, help="Input edge file")
@click.option("-o", "--output", required=True, type=REQUIRED_PATH_DIR, help="Output directory")
def simple_split_subcircuit(nodeset, nodeset_path, nodes, edges, output):
    """Split a subcircuit out from a SONATA circuit based on node_set"""
    from brainbuilder.utils.sonata import split_population as module

    module.simple_split_subcircuit(output, nodeset, nodeset_path, nodes, edges)


@app.command()
@click.option("--nodeset", required=True, help="Name of nodeset")
@click.option("--circuit", required=True, type=REQUIRED_PATH, help="path to circuit_config.json")
@click.option(
    "--include-virtual/--no-include-virtual",
    default=False,
    help="Extract virtual nodes that project into the extracted subcircuit",
)
@click.option(
    "--create-external/--no-create-external",
    default=False,
    help="Extract external connections; ones that are non-virtual, but sourced from"
    "outside the extracted subcircuit - they become virtual nodes",
)
@click.option("-o", "--output", required=True, type=REQUIRED_PATH_DIR, help="Output directory")
def split_subcircuit(nodeset, circuit, include_virtual, create_external, output):
    """Split a subcircuit out from a SONATA circuit based on node_set"""
    from brainbuilder.utils.sonata import split_population as module

    module.split_subcircuit(
        output,
        node_set_name=nodeset,
        circuit_config_path=circuit,
        do_virtual=include_virtual,
        create_external=create_external,
    )

    click.echo(
        click.style(
            f"Please remember to update {output}/circuit_config.json "
            "to reflect the location of all necessary files",
            fg="green",
        )
    )


@app.command()
@click.option("-o", "--output", required=True, type=REQUIRED_PATH_DIR, help="Output directory")
@click.option("--circuit", required=True, type=REQUIRED_PATH, help="path to circuit_config.json")
@click.option(
    "--population-name", required=True, help="Name of population to clip out morphologies"
)
def clip_morphologies(output, circuit, population_name):
    """Clip (ie: copy) morphologies actually used by a population into the `output` directory"""
    from brainbuilder.utils.sonata import clip

    clip.morphologies(output, circuit, population_name)
