# SPDX-License-Identifier: Apache-2.0
"""Target generation."""

# pylint: disable=import-outside-toplevel
import collections
import logging

import click
from voxcell import ROIMask
from voxcell.nexus.voxelbrain import Atlas

from brainbuilder.exceptions import BrainBuilderError
from brainbuilder.utils import bbp, dump_json, load_yaml

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


def _load_atlas(atlas_path, atlas_cache_path):
    """Try and load the atlas."""
    if atlas_path is None:
        raise BrainBuilderError("Atlas not provided")
    atlas = Atlas.open(atlas_path, cache_dir=atlas_cache_path)
    return atlas


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


def _enforce_layer_to_str(data):
    for key, value in data.items():
        if isinstance(value, dict):
            _enforce_layer_to_str(value)
        elif key == "layer":
            data[key] = str(data[key])


def _load_targets(filepath):
    """
    Load target definition YAML, e.g.:

    >
      targets:
        # BluePy-like queries a.k.a. "smart targets"
        query_based:
            mc2_Column: {'region': '@^mc2'}
            Layer1: {'region': '@1$'}

        # 0/1 masks registered in the atlas
        atlas_based:
            cylinder: '{S1HL-cylinder}'
    """
    content = load_yaml(filepath)["targets"]
    _enforce_layer_to_str(content)

    return (
        content.get("query_based"),
        content.get("atlas_based"),
    )


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
    from bluepy import Circuit

    circuit = Circuit({"cells": cells_path})
    cells = circuit.cells.get()
    with open(output, "w", encoding="utf-8") as f:
        write_default_targets(cells, f)
        if targets is None:
            if "layer" in cells:
                bbp.write_property_targets(f, cells, "layer", mapping=_layer_name)
        else:
            query_based, atlas_based = _load_targets(targets)
            if query_based is not None:
                write_query_targets(query_based, circuit, f, allow_empty=allow_empty)
            if atlas_based is not None:
                atlas = _load_atlas(atlas, atlas_cache)
                xyz = cells[["x", "y", "z"]].to_numpy()
                for name, dset in atlas_based.items():
                    mask = atlas.load_data(dset, cls=ROIMask).lookup(xyz)
                    bbp.write_target(f, name, cells.index[mask])


def _add_occupied_hierarchy(region_map_df, occupied_regions, result):
    """Create node_sets for `occupied_regions`

    For regions that have children AND contents, we have a '$region-only' nodeset

    Note that result is passed with already populated regions such that
    an '$result-only' can be created when there are conflicts
    """
    occupied_regions = set(occupied_regions)

    region2parent_id = region_map_df.set_index("acronym")["parent_id"].to_dict()
    id2region = region_map_df["acronym"].to_dict()

    to_add = collections.defaultdict(set)
    for region in occupied_regions:
        parent_region_id = region2parent_id[region]
        while parent_region_id != -1:
            parent_region = id2region[parent_region_id]
            to_add[parent_region].add(region)
            region, parent_region_id = parent_region, region2parent_id[parent_region]

    for region, subregions in to_add.items():
        if region in result:
            result[f"{region}-only"] = result[region]
            subregions.add(f"{region}-only")
        result[region] = sorted(subregions)


@app.command()
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
def node_sets(
    cells_path, full_hierarchy, atlas, atlas_cache, targets, allow_empty, population, output
):
    """Generate JSON node sets from MVD3 or SONATA (and target definition YAML)"""
    # pylint: disable=too-many-locals
    from bluepy import Circuit

    result = {}

    def _add_node_sets(to_add):
        for name, query in sorted(to_add.items()):
            if name in result:
                raise BrainBuilderError(f"Duplicate node set: '{name}'")

            if not allow_empty and cells.count(query) == 0:
                raise BrainBuilderError(f"Empty target: {name} {query}")

            result[name] = query

    cells = Circuit({"cells": cells_path}).cells

    result["All"] = {"population": population}
    _add_node_sets(
        {
            "Excitatory": {"synapse_class": "EXC"},
            "Inhibitory": {"synapse_class": "INH"},
        }
    )
    _add_node_sets(
        {
            val: {prop: val}
            for prop in ["mtype", "etype"]
            for val in cells.get(properties=prop).unique()
        }
    )

    occupied_regions = {val: {"region": val} for val in cells.get(properties="region").unique()}
    _add_node_sets(occupied_regions)

    if full_hierarchy:
        region_map_df = _load_atlas(atlas, atlas_cache).load_region_map().as_dataframe()
        _add_occupied_hierarchy(region_map_df, occupied_regions, result)

    if targets is not None:
        query_based, atlas_based = _load_targets(targets)

        if query_based is not None:
            _add_node_sets(query_based)

        if atlas_based is not None:
            atlas = _load_atlas(atlas, atlas_cache)
            xyz = cells.get(properties=list("xyz"))
            for name, dset in atlas_based.items():
                mask = atlas.load_data(dset, cls=ROIMask).lookup(xyz.to_numpy())
                ids = xyz.index[mask] - 1
                assert name not in result
                result[name] = {"population": population, "node_id": ids.tolist()}

    dump_json(output, result)
