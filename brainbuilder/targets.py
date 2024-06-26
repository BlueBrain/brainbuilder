"""Handling of targets and nodesets"""
import collections

import bluepysnap
import voxcell
from voxcell.nexus.voxelbrain import Atlas

from brainbuilder.exceptions import BrainBuilderError
from brainbuilder.utils import load_yaml


def load_atlas(atlas_path, atlas_cache_path):
    """Try and load the atlas."""
    if atlas_path is None:
        raise BrainBuilderError("Atlas not provided")
    atlas = Atlas.open(atlas_path, cache_dir=atlas_cache_path)
    return atlas


def _enforce_layer_to_str(data):
    for key, value in data.items():
        if isinstance(value, dict):
            _enforce_layer_to_str(value)
        elif key == "layer":
            data[key] = str(data[key])


def load_targets(filepath):
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


def create_node_sets(cells, full_hierarchy, atlas, targets, allow_empty, population):
    """Create and return a node sets dictionary

    Args:
        cells(voxcell.CellCollection): cells to be added to targets
        full_hierarchy(bool): Include, from leaf to root, all the region names as node_sets
        atlas(voxcell.nexus.voxelbrain.Atlas): atlas
        targets(str): Path to target definition YAML file
        allow_empty(bool): Allow empty targets
        population(str): name of the population
    """
    result = {}

    cells = cells.as_dataframe()

    def _add_node_sets(to_add):
        for name, query in sorted(to_add.items()):
            if name in result:
                raise BrainBuilderError(f"Duplicate node set: '{name}'")

            ids = bluepysnap.query.resolve_ids(cells, population, query)
            if not allow_empty and ids.any():
                raise BrainBuilderError(f"Empty target: {name} {query}")

            result[name] = query

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
            for prop in ["mtype", "etype"] if prop in cells.columns
            for val in cells[prop].unique()
        }
    )

    occupied_regions = {}
    if "region" in cells.columns:
        occupied_regions = {val: {"region": val} for val in cells['region'].unique()}
        _add_node_sets(occupied_regions)

    if full_hierarchy:
        region_map_df = atlas.load_region_map().as_dataframe()
        _add_occupied_hierarchy(region_map_df, occupied_regions, result)

    if targets is not None:
        query_based, atlas_based = load_targets(targets)

        if query_based is not None:
            _add_node_sets(query_based)

        if atlas_based is not None:
            xyz = cells[list("xyz")]
            for name, dset in atlas_based.items():
                mask = atlas.load_data(dset, cls=voxcell.ROIMask).lookup(xyz.to_numpy())
                ids = xyz.index[mask] - 1  # CellCollection is 1 based, SONATA is 0 based
                assert name not in result
                result[name] = {"population": population, "node_id": ids.tolist()}

    return result
