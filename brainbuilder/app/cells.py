# SPDX-License-Identifier: Apache-2.0
"""
CellCollection building.

A collection of commands for creating CellCollection and augmenting its properties, in particular:

----

# `brainbuilder cells place`

Based on YAML cell composition recipe, create MVD3 with:
 - cell positions
 - required cell properties: 'mtype', 'etype'
 - additional cell properties prescribed by the recipe and / or atlas

----

# `brainbuilder cells assign_emodels`

Based on `extNeuronDB.dat` file, add 'me_combo' to existing MVD3.
MVD3 is expected to have the following properties already assigned:
 - 'layer', 'mtype', 'etype'


# `brainbuilder cells positions_and_orientations`

Based on:
 - volumetric density nrrd files (a.k.a density fields)
 - the orientation nrrd file.
 - the annotation nrrd file.

 The output sonata file also contains the region identifier of each cell.
 This is why the annotation nrrd file is part of the input.

 The output sonata file is to be consumed by:
 - the BBP web Cell Atlas (https://bbpcode.epfl.ch/code/#/admin/projects/nexus/cell-atlas)
 - the point neuron whole brain workflow
    (https://bbpcode.epfl.ch/code/#/admin/projects/bbpnr/genBrain)
"""

import logging
import numbers
from collections.abc import Mapping

import click
import numpy as np
import pandas as pd
from voxcell import (
    CellCollection,
    ROIMask,
    VoxelData,
    values_to_hemisphere,
    values_to_region_attribute,
)
from voxcell.nexus.voxelbrain import Atlas

from brainbuilder import BrainBuilderError
from brainbuilder.app._utils import REQUIRED_PATH
from brainbuilder.cell_positions import create_cell_positions
from brainbuilder.utils import bbp, deprecate, load_yaml
from brainbuilder.utils.bbp import load_cell_composition

L = logging.getLogger("brainbuilder")


@click.group()
def app():
    """Building CellCollection"""


def load_mtype_taxonomy(filepath):
    """
    Load mtype taxonomy from TSV file.

    TODO: link to spec
    """
    # TODO: validate
    return pd.read_csv(filepath, sep=r"\s+", index_col="mtype")


def load_mini_frequencies(filepath):
    """
    Load mini frequencies from a TSV file.
    """
    return pd.read_csv(filepath, sep=r"\s+", index_col="layer", dtype={"layer": str})


def _load_density(value, mask, atlas):
    """Load density as 3D numpy array.

    Args:
        value: one of
            - float value (constant density per `mask`)
            - path to NRRD file (load from file, filter by `mask`)
            - dataset in `atlas` (load from atlas, filter by `mask`)
        mask: 0/1 3D mask
        atlas: Atlas to use for loading atlas datasets

    `value` of form '{name}' is recognized as atlas dataset 'name'.

    Returns:
        3D float32 numpy array of same shape as `mask`.
    """
    if isinstance(value, numbers.Number):
        result = np.zeros_like(mask, dtype=np.float64)
        result[mask] = float(value)
    elif value.startswith("{"):
        assert value.endswith("}")
        dataset = value[1:-1]
        L.info("Loading 3D density profile from '%s' atlas dataset...", dataset)
        result = atlas.load_data(dataset, cls=VoxelData).raw.astype(np.float64)
    elif value.endswith(".nrrd"):
        L.info("Loading 3D density profile from '%s'...", value)
        result = VoxelData.load_nrrd(value).raw.astype(np.float64)
    else:
        raise BrainBuilderError(f"Unexpected density value: '{value}'")

    # Mask away density values outside region mask (NaNs are fine there)
    result[~mask] = 0

    if np.any(np.isnan(result)):
        raise BrainBuilderError("NaN density values within region mask")

    # Densities smaller than 1e-7 per mm3 correspond to less than 1 cell for the whole brain.
    # For example, mouse brain volume is ~600 mm3 and human brain ~1260000 mm3.
    # Allowing extremely small numbers introduces noise into the placement and should be ideally
    # addressed at the density generation stage. However, given that this is not always the case,
    # the near zero values will be zeroed to ensure the correct behavior of the algorithm.
    mask = np.isclose(result, 0.0, atol=1e-7, rtol=0.0) & (result != 0.0)
    if mask.any():
        L.warning("{mask.sum()} Near zero values smaller than 1e-7 found and wil be zeroed.")
        result[mask] = 0.0

    return result


def _create_cell_group(conf, atlas, root_mask, density_factor, soma_placement):
    region_mask = atlas.get_region_mask(conf["region"], with_descendants=True, memcache=True)
    if root_mask is not None:
        region_mask.raw &= root_mask.raw
    if not np.any(region_mask.raw):
        raise BrainBuilderError(f"Empty region mask for region: '{conf['region']}'")

    density = region_mask.with_data(_load_density(conf["density"], region_mask.raw, atlas))

    pos = create_cell_positions(density, density_factor=density_factor, method=soma_placement)
    result = pd.DataFrame(pos, columns=["x", "y", "z"])

    for prop, value in conf["traits"].items():
        if isinstance(value, Mapping):
            values, probs = zip(*value.items())
            if not np.allclose(np.sum(probs), 1.0):
                L.warning("Weights don't sum up to 1.0 for %s; renormalizing them", str(value))
                probs = probs / np.sum(probs)
            result[prop] = np.random.choice(values, size=len(pos), p=probs)
        else:
            result[prop] = value

    L.info("%s... [%d cells]", conf["traits"], len(result))
    return result


def _assign_subregions(cells, brain_regions, region_map):
    cell_coordinates = cells[["x", "y", "z"]].to_numpy()
    subregion_index = brain_regions.lookup(cell_coordinates)
    subregion_index_to_acronym = region_map.as_dataframe()["acronym"]
    _assign_property(cells, "subregion", subregion_index_to_acronym[subregion_index].to_numpy())


def _assign_property(cells, prop, values):
    if prop in cells:
        raise BrainBuilderError(f"Duplicate property: '{prop}'")
    cells[prop] = values


def _assign_mtype_traits(cells, mtype_taxonomy):
    traits = mtype_taxonomy.loc[cells["mtype"]]
    _assign_property(cells, "morph_class", traits["mClass"].values)
    _assign_property(cells, "synapse_class", traits["sClass"].values)


def _assign_mini_frequencies(cells, mini_frequencies):
    """
    Add the mini_frequency column to `cells`.
    """
    if "layer" in cells:
        idx = cells.layer.to_numpy()
    else:
        # fallback to subregion; this requires that the mini_frequencies file uses subregions,
        # and not layer names
        idx = cells.subregion.to_numpy()

    mfreqs_cells = mini_frequencies.loc[idx]

    _assign_property(cells, "exc_mini_frequency", mfreqs_cells.exc_mini_frequency.to_numpy())
    _assign_property(cells, "inh_mini_frequency", mfreqs_cells.inh_mini_frequency.to_numpy())


def _assign_atlas_property(cells, prop, atlas, dset):
    xyz = cells[["x", "y", "z"]].values
    if dset == "FAST-HEMISPHERE":
        # TODO: remove as soon as "slow" way of assigning hemisphere
        # (with a volumetric dataset) is available
        deprecate.warn("`FAST-HEMISPHERE` is deprecated, use a volumetric dataset")
        values = np.where(xyz[:, 2] < 5700, "left", "right")
    elif prop == "hemisphere":
        values = values_to_hemisphere(atlas.load_data(dset).lookup(xyz))
    elif dset.startswith("~"):
        dset = dset[1:]
        values = values_to_region_attribute(
            atlas.load_data(dset).lookup(xyz),
            region_map=atlas.load_region_map(),
            attr="acronym",
        )
    else:
        values = atlas.load_data(dset).lookup(xyz)

    _assign_property(cells, prop, values)


def _place(
    input_path,
    composition_path,
    mtype_taxonomy_path,
    atlas_url,
    mini_frequencies_path=None,
    atlas_cache=None,
    region=None,
    mask_dset=None,
    soma_placement="basic",
    density_factor=1.0,
    atlas_properties=None,
    sort_by=None,
    append_hemisphere=False,
):
    # pylint: disable=too-many-arguments, too-many-locals
    atlas = Atlas.open(atlas_url, cache_dir=atlas_cache)

    recipe = load_cell_composition(composition_path)
    mtype_taxonomy = load_mtype_taxonomy(mtype_taxonomy_path)

    # Cache frequently used atlas data
    atlas.load_data("brain_regions", memcache=True)
    atlas.load_region_map(memcache=True)

    if mask_dset is None:
        root_mask = None
    else:
        root_mask = atlas.load_data(mask_dset, cls=ROIMask)

    if region is not None:
        region_mask = atlas.get_region_mask(region, with_descendants=True)
        if root_mask is None:
            root_mask = region_mask
        else:
            root_mask.raw &= region_mask.raw

    L.info("Creating cell groups...")
    groups = [
        _create_cell_group(conf, atlas, root_mask, density_factor, soma_placement)
        for conf in recipe["neurons"]
    ]

    L.info("Merging into single CellCollection...")
    result = pd.concat(groups)

    L.info("Total cell count: %d", len(result))

    L.info("Assigning 'subregion'")
    _assign_subregions(
        result,
        atlas.load_data("brain_regions"),
        atlas.load_region_map(),
    )

    L.info("Assigning 'morph_class' / 'synapse_class'...")
    _assign_mtype_traits(result, mtype_taxonomy)

    if mini_frequencies_path is not None:
        mini_frequencies = load_mini_frequencies(mini_frequencies_path)
        L.info("Assigning mini-frequencies")
        _assign_mini_frequencies(result, mini_frequencies)

    for prop, dset in atlas_properties or []:
        L.info("Assigning '%s'...", prop)
        _assign_atlas_property(result, prop, atlas, dset)

    if append_hemisphere:
        result["region"] = result["region"] + "@" + result["hemisphere"]

    if sort_by:
        L.info("Sorting CellCollection...")
        result.sort_values(sort_by, inplace=True)

    L.info("Done!")

    result.index = 1 + np.arange(len(result))
    if input_path is None:
        return CellCollection.from_dataframe(result)
    input_cells = CellCollection.load(input_path)
    out_cells = CellCollection.from_dataframe(pd.concat([input_cells.as_dataframe(), result]))
    out_cells.population_name = input_cells.population_name
    return out_cells


@app.command(short_help="Initialize cell collection")
@click.option("--population-name", help="Name of population to create", required=True)
@click.option(
    "-o",
    "--output",
    required=True,
    help="Path to output. Use .mvd3 file extension for MVD3, otherwise SONATA is used",
)
def init(population_name, output):
    """Initialize a cell collection (ie: SONATA node file) with the specified population name"""
    cells = CellCollection(population_name)
    cells.save(output)


@app.command(short_help="Generate cell positions and me-types", name="place")
@click.option("--composition", help="Path to ME-type composition YAML", required=True)
@click.option("--mtype-taxonomy", help="Path to mtype taxonomy TSV", required=True)
@click.option("--atlas", help="Atlas URL / path", required=True)
@click.option(
    "--mini-frequencies", help="Path to the mini frequencies TSV", default=None, show_default=True
)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("--region", help="Region name filter", default=None, show_default=True)
@click.option("--mask", help="Dataset with volumetric mask filter", default=None, show_default=True)
@click.option("--density-factor", help="Density factor", type=float, default=1.0, show_default=True)
@click.option("--soma-placement", help="Soma placement method", default="basic", show_default=True)
@click.option(
    "--atlas-property", type=(str, str), multiple=True, help="Property based on atlas dataset"
)
@click.option("--sort-by", help="Sort by properties (comma-separated)", default=None)
@click.option(
    "--append-hemisphere", is_flag=True, help="Append hemisphere to region name", default=False
)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option(
    "-o",
    "--output",
    help="Path to output MVD3 or SONATA. Use .mvd3 file extension for"
    " MVD3, otherwise SONATA is used",
    required=True,
)
@click.option(
    "--input",
    "input_path",
    default=None,
    help="Existing cells which are extended with" "the new positioned cells",
)
def place_cli(
    composition,
    mtype_taxonomy,
    atlas,
    mini_frequencies,
    atlas_cache,
    region,
    mask,
    density_factor,
    soma_placement,
    atlas_property,
    sort_by,
    append_hemisphere,
    seed,
    output,
    input_path,
):
    """Places new cells into an existing cells or creates new cells if no existing were provided."""
    # pylint: disable=too-many-arguments, too-many-locals
    place(
        composition,
        mtype_taxonomy,
        atlas,
        mini_frequencies,
        atlas_cache,
        region,
        mask,
        density_factor,
        soma_placement,
        atlas_property,
        sort_by,
        append_hemisphere,
        seed,
        output,
        input_path,
    )


def place(
    composition,
    mtype_taxonomy,
    atlas,
    mini_frequencies,
    atlas_cache,
    region,
    mask,
    density_factor,
    soma_placement,
    atlas_property,
    sort_by,
    append_hemisphere,
    seed,
    output,
    input_path,
):
    """Places new cells into an existing cells or creates new cells if no existing were provided."""
    # pylint: disable=too-many-arguments, too-many-locals
    np.random.seed(seed)

    if sort_by is not None:
        sort_by = sort_by.split(",")

    cells = _place(
        input_path,
        composition,
        mtype_taxonomy,
        atlas,
        mini_frequencies_path=mini_frequencies,
        atlas_cache=atlas_cache,
        region=region,
        mask_dset=mask,
        density_factor=density_factor,
        soma_placement=soma_placement,
        atlas_properties=atlas_property,
        sort_by=sort_by,
        append_hemisphere=append_hemisphere,
    )

    L.info("Export to %s", output)
    cells.save(output)


@app.command()
@click.argument("cells-path")
@click.option("--morphdb", help="Path to extNeuronDB.dat", required=True)
@click.option("--seed", type=int, help="Pseudo-random generator seed", default=0)
@click.option(
    "-o",
    "--output",
    help="Path to output MVD3 or SONATA. Use .mvd3 file extension for"
    " MVD3, otherwise SONATA is used",
    required=True,
)
def assign_emodels(cells_path, morphdb, seed, output):
    """Assign 'me_combo' property"""
    np.random.seed(seed)

    cells = CellCollection.load(cells_path)
    morphdb = bbp.load_extneurondb(morphdb)
    result = bbp.assign_emodels(cells, morphdb)

    result.save(output)


@app.command(
    short_help="Generate cell positions and save them together with orientations,"
    " region annotations and cell types"
)
@click.option(
    "--annotation-path",
    type=REQUIRED_PATH,
    required=True,
    help="Path to the whole mouse brain annotation file (nrrd).",
)
@click.option(
    "--orientation-path",
    type=REQUIRED_PATH,
    required=True,
    help="Path to the whole mouse brain orientation file (nrrd). "
    "Quaternion field whose underlying array is of type float and of shape (W, H, D, 4) where "
    "quaternions under the format [w, x, y, z]. The value w is the real part of the quaternion "
    "while xi + yj + zk is the imaginary part. The direction vector of a cell is assumed to be "
    "given by q.rotate([0, 1, 0]) where q is the quaternion assigned to the cell voxel.",
)
@click.option(
    "--config-path",
    type=REQUIRED_PATH,
    required=True,
    help=(
        "Path to the densities configuration file (yaml)."
        " This file indicates which cell densities are used to generate cell positions."
        " It contains the paths to the density volumes (a.k.a as density fields)."
        "\n"
        " Configuration example:"
        " inputDensityVolumePath:"
        '   inhibitory neuron: "inhibitory_neuron_density.nrrd"'
        '   excitatory neuron: "excitatory_neuron_density.nrrd"'
        '   oligodendrocyte: "oligodendrocyte_density.nrrd"'
        '   astrocyte: "astrocyte_density.nrrd"'
        '   microglia: "microglia_density.nrrd"'
        "\n"
        "A density file holds an array of type double and of shape (W, H, D) "
        "with non-negative entries. A density value is a number of cells per voxel."
    ),
)
@click.option(
    "--output-path",
    type=click.Path(file_okay=False, dir_okay=False, resolve_path=True),
    required=True,
    help="Path where to write the cell positions and orientations (a single sonata .h5 file).",
)
def positions_and_orientations(annotation_path, orientation_path, config_path, output_path):
    """Generate 3D cell positions and store the corresponding cell orientations.\n

    See https://bbpteam.epfl.ch/project/issues/browse/BBPP82-499 for the full context.

    The output is a single sonata file to be consumed by:\n
    - the BBP web Cell Atlas (https://bbpcode.epfl.ch/code/#/admin/projects/nexus/cell-atlas)\n
    - the point neuron whole brain workflow\n

    Cell positions of the form (x, y, z) are generated by means of an acceptance-rejection method
    based on prescribed cell densities.
    The algorithm is described in the section "Computing Cell Positions" of
    "A Cell Atlas for the Mouse Brain" by C. Eroe et al. (2018),
    https://www.frontiersin.org/articles/10.3389/fninf.2018.00084/full.
    See np.random.choice and its p parameter for its implementation:
    https://docs.scipy.org/doc//numpy-1.10.4/reference/generated/numpy.random.choice.html\n


    The inputs are:
    - volumetric density nrrd files (a.k.a density fields) of the following cell types:
        * inhibitory neurons\n
        * excitatory neurons\n
        * astrocytes\n
        * microglia\n
        * oligodendrocytes\n
        These datasets are non-negative arrays of type double and of shape (W, H, D). Density is
        expressed a number of cells per voxel. These files are specified by means of yaml
        configuration, see `config_path` doc.
    - the orientation nrrd file. This file holds a float32 array of shape (W, H, D, 4) with
        quaternions of the form [w, x, y, z] on the last axis.\n
    - the annotation nrrd file. This file holds uint32 array of shape (W, H, D) where voxels are
    labeled with the AIBS Structure IDs, a.k.a region identifiers.\n

    The output sonata file also contains the region identifier of each cell.
    This is why the annotation nrrd file is part of the input.\n

    The output sonata file is to be consumed by:
    - the BBP web Cell Atlas (https://bbpcode.epfl.ch/code/#/admin/projects/nexus/cell-atlas)\n
    - the point neuron whole brain workflow
    (https://bbpcode.epfl.ch/code/#/admin/projects/bbpnr/genBrain)\n

    The output sonata file contains as many cell collections as input density files.
    The format of each individual collection is the sonata format of voxcell.CellCollection.
    Each cell collection is identified by a population name and stores the 3D cell positions
    and orientations of a given cell type together with the cell region identifiers
    (AIBS Structure IDs).\n

    Positions are stored by means of 3 datasets named 'x', 'y' and 'z'. Each is a float32 array
    of shape (N,) where N is the number of cells of a given cell type in the mouse brain.\n

    Orientations are stored by means of 4 datasets named 'orientation_x', 'orientation_y',
    'orientation_z' and 'orientation_w'. Each dataset is a float32 array of shape (N,).
    Input quaternions are assumed to be encoded by 4D vectors of the form [w, x, y, z].
    The cell direction vector associated to a quaternion q is by convention q.rotate([0, 1, 0]).\n

    Region identifiers are stored by means of a dataset named 'region_id'. This is a uint32 array
    of shape (N, ).\n

    The output 3D coordinates are expressed within the 3D orthonormal frame associated with the
    annotated volume. Hence coordinates take the voxel dimensions and the offset of the annotated
    volume into account. Coordinates expressed in um. The voxel dimensions and the offsets of the
    input density files specified in `config_path` are assumed to coincide with the voxel
    dimensions and the offset of the annotated volume.\n

    Output layout as depicted by h5ls -r `output_path`:\n
        /                        Group\n
        /nodes                   Group\n
        /nodes/atlas_cells       Group\n
        /nodes/atlas_cells/0     Group\n
        /nodes/atlas_cells/0/@library Group\n
        /nodes/atlas_cells/0/@library/cell_type Dataset\n
        /nodes/atlas_cells/0/cell_type Dataset\n
        /nodes/atlas_cells/0/orientation_w Dataset\n
        /nodes/atlas_cells/0/orientation_x Dataset\n
        /nodes/atlas_cells/0/orientation_y Dataset\n
        /nodes/atlas_cells/0/orientation_z Dataset\n
        /nodes/atlas_cells/0/region_id Dataset\n
        /nodes/atlas_cells/0/x   Dataset\n
        /nodes/atlas_cells/0/y   Dataset\n
        /nodes/atlas_cells/0/z   Dataset\n
        /nodes/atlas_cells/node_type_id Dataset\n

        Note: The node_type_ids are all set to -1 by voxcell.CellCollection.save_sonata)\n

    How to read the output file:\n
        # The recommanded way: use voxcell.CellCollection support for libsonata\n
        from voxcell import CellCollection\n
        cell_collection = CellCollection.load_sonata('positions_and_orientations.h5')\n
        positions = cell_collection.positions # float32 array of shape (N, 3)\n
        # CellCollection orientations are 3 x 3 orthogonal matrices\n
        orientations = cell_collection.orientations # float32 of shape (N, 3, 3)\n
        properties = cell_collection.properties # pandas.DataFrame\n
        region_ids = properties['region_id'] # uint32 array of shape (N,)\n
        cell_types = properties['cell_type']  # str array of shape (N,)\n

        import h5py\n
        cell_collections = h5py.File('positions_and_orientations.h5', 'r')
        orientation_x = cell_collections.get('/nodes/atlas_cells/0/orientation_x')[()]\n
        position_y = cell_collections.get('/nodes/atlas_cells/0/y')[()]\n
        region_ids = cell_collections.get('/nodes/atlas_cells/0/region_id')[()]\n
        # uint32 array of shape (N,)\n
        cell_types = cell_collections.get('/nodes/atlas_cells/0/cell_type')[()]\n
        # An extra dataset is needed to map utin32 indices to cell type strings,\n
        # that is, the following literal string array of shape (5,) \n
        cell_type_literals = cell_collection.get('/nodes/atlas_cells/0/@library/cell_type')
    """
    # pylint: disable=too-many-arguments, too-many-locals
    L.info("Loading density configuration file %s ...", config_path)
    config = load_yaml(config_path)
    L.info("Loading annotation file %s ...", annotation_path)
    annotation = VoxelData.load_nrrd(annotation_path)
    L.info("Loading orientation file %s ...", orientation_path)
    orientation = VoxelData.load_nrrd(orientation_path)

    assert np.allclose(
        annotation.offset, orientation.offset
    ), "The annotation and orientation files have different offsets."
    assert np.allclose(
        annotation.voxel_dimensions, orientation.voxel_dimensions
    ), "The annotation and orientation files have different voxel dimensions."

    # The columns to be populated
    positions = []
    orientations = []
    region_ids = []
    cell_types = []

    for cell_type, density_path in config["inputDensityVolumePath"].items():
        L.info("Loading density file %s ...", density_path)
        density_voxel_data = VoxelData.load_nrrd(density_path)
        if not np.allclose(density_voxel_data.offset, annotation.offset):
            raise BrainBuilderError(
                f"The input density file {density_path} and the input annotation file "
                f"{annotation_path} have different offsets: "
                f"{density_voxel_data.offset} != {annotation.offset}"
            )
        if not np.allclose(density_voxel_data.voxel_dimensions, annotation.voxel_dimensions):
            raise BrainBuilderError(
                f"The input density file {density_path} and the input annotation file "
                f"{annotation_path} have different voxel dimensions: "
                f"{density_voxel_data.voxel_dimensions} != {annotation.voxel_dimensions}"
            )

        # Microglia cell density can take negative values, see
        # https://bbpteam.epfl.ch/project/issues/browse/NSETM-1260.
        # As a temporary fix, negative values are zeroed. Hence -S extra cells
        # are created where S is the sum of negative values.
        # TODO: implement a long term solution in atlas-building-tools
        negative_mask = density_voxel_data.raw < 0.0
        if np.any(negative_mask):
            L.warning(
                "Negative density values in %s summing up to %f. Zeroing negative values.",
                density_path,
                np.sum(density_voxel_data.raw[negative_mask]),
            )
            density_voxel_data.raw[negative_mask] = 0.0

        L.info('Creating cell positions for the cell type "%s" ...', cell_type)
        positions_ = create_cell_positions(density_voxel_data, seed=0)
        positions.append(positions_)

        L.info('Retrieving voxel indices for the cell type "%s\n" ...', cell_type)
        voxel_indices = annotation.positions_to_indices(positions_)
        voxel_indices = tuple(voxel_indices.T)
        orientations.append(orientation.raw[voxel_indices])
        region_ids.append(annotation.raw[voxel_indices])
        cell_types += [cell_type] * len(voxel_indices[0])

    L.info(
        "Creation of the cell collection dataframe."
        " \n Setting positions, orientations, region ids and cell types ..."
    )
    datasets = np.hstack(
        [
            np.asarray(np.concatenate(positions), dtype=np.float32),
            np.asarray(np.concatenate(orientations), dtype=np.float32),
        ]
    )
    df = pd.DataFrame(
        datasets,
        columns=[
            "x",
            "y",
            "z",
            # We assume quaternions to be under the form [w, x, y, z]
            "orientation_w",
            "orientation_x",
            "orientation_y",
            "orientation_z",
        ],
    )
    df["region_id"] = np.concatenate(region_ids)
    df["cell_type"] = cell_types
    df.index = 1 + np.arange(len(df))  # CellCollection has a 1-based index
    L.info("Building a voxcell.CellCollection ...")
    cells = CellCollection.from_dataframe(df)
    cells.population_name = "atlas_cells"

    L.info("Saving %s to sonata format ...", output_path)
    cells.save_sonata(output_path)
