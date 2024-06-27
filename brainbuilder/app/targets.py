# SPDX-License-Identifier: Apache-2.0
"""Target generation."""

# pylint: disable=import-outside-toplevel
import logging

import click
import voxcell

import brainbuilder
import brainbuilder.targets
from brainbuilder.exceptions import BrainBuilderError
from brainbuilder.utils import bbp, dump_json

L = logging.getLogger("brainbuilder")


@click.group()
def app():
    """Tools for working with .target files"""


def _synapse_class_name(synclass):
    return {
        "EXC": "Excitatory",
        "INH": "Inhibitory",
    }[synclass]


def _layer_name(layer):
    return f"Layer{layer}"


def write_default_targets(cells, output):
    """Write default property-based targets."""
    bbp.write_target(output, "Mosaic", include_targets=["All"])
    bbp.write_target(output, "All", include_targets=sorted(cells["mtype"].unique()))
    bbp.write_property_targets(output, cells, "synapse_class", mapping=_synapse_class_name)
    bbp.write_property_targets(output, cells, "mtype")
    bbp.write_property_targets(output, cells, "etype")
    bbp.write_property_targets(output, cells, "region")


def write_query_targets(query_based, circuit, output, allow_empty=False):
    """Write targets based on BluePy-like queries."""
    for name, query in query_based.items():
        gids = circuit.cells.ids(query)
        if len(gids) < 1:
            msg = f"Empty target: {name} {query}"
            if allow_empty:
                L.warning(msg)
            else:
                raise BrainBuilderError(msg)
        bbp.write_target(output, name, gids=gids)


@app.command()
@click.argument("cells-path")
@click.option("--atlas", help="Atlas URL / path", default=None, show_default=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("-t", "--targets", help="Path to target definition YAML file", default=None)
@click.option("--allow-empty", is_flag=True, help="Allow empty targets", show_default=True)
@click.option("-o", "--output", help="Path to output .target file", required=True)
def from_input(cells_path, atlas, atlas_cache, targets, allow_empty, output):
    """Generate .target file from MVD3 or SONATA (and target definition YAML)"""
    # pylint: disable=too-many-locals
    from bluepy import Circuit  # pylint: disable=import-error

    circuit = Circuit({"cells": cells_path})
    cells = circuit.cells.get()
    with open(output, "w", encoding="utf-8") as f:
        write_default_targets(cells, f)
        if targets is None:
            if "layer" in cells:
                bbp.write_property_targets(f, cells, "layer", mapping=_layer_name)
        else:
            query_based, atlas_based = brainbuilder.targets.load_targets(targets)
            if query_based is not None:
                write_query_targets(query_based, circuit, f, allow_empty=allow_empty)
            if atlas_based is not None:
                atlas = brainbuilder.targets.load_atlas(atlas, atlas_cache)
                xyz = cells[["x", "y", "z"]].to_numpy()
                for name, dset in atlas_based.items():
                    mask = atlas.load_data(dset, cls=voxcell.ROIMask).lookup(xyz)
                    bbp.write_target(f, name, cells.index[mask])


@app.command(name="node-sets")
@click.argument("cells-path")
@click.option(
    "--full-hierarchy",
    is_flag=True,
    help="Include, from leaf to root, all the region names as node_sets",
)
@click.option("--atlas", help="Atlas URL / path", default=None, show_default=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("-t", "--targets", help="Path to target definition YAML file", default=None)
@click.option("--allow-empty", is_flag=True, help="Allow empty targets", show_default=True)
@click.option("--population", help="Population name", default="default", show_default=True)
@click.option("-o", "--output", help="Path to output JSON file", required=True)
def node_sets_cli(
    cells_path, full_hierarchy, atlas, atlas_cache, targets, allow_empty, population, output
):
    """Generate JSON node sets from MVD3 or SONATA (and target definition YAML)"""
    node_sets(
        cells_path, full_hierarchy, atlas, atlas_cache, targets, allow_empty, population, output
    )


def node_sets(
    cells_path, full_hierarchy, atlas, atlas_cache, targets, allow_empty, population, output
):
    """Generate JSON node sets from MVD3 or SONATA (and target definition YAML)"""

    cells = voxcell.CellCollection.load(cells_path)

    atlas = brainbuilder.targets.load_atlas(atlas, atlas_cache)

    result = brainbuilder.targets.create_node_sets(
        cells, full_hierarchy, atlas, targets, allow_empty, population
    )

    dump_json(output, result)
