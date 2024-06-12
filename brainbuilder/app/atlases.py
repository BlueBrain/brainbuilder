# SPDX-License-Identifier: Apache-2.0
"""Tool for artificial atlas building."""

import itertools
import logging
import os
from collections import OrderedDict
from pathlib import Path

import click
import numpy as np
from voxcell import VoxelData, math_utils

from brainbuilder.masks import regular_convex_polygon_mask_from_side
from brainbuilder.utils import dump_json

L = logging.getLogger("brainbuilder")


def _align_thickness(thickness, voxel_side):
    """Align layer boundaries along voxel grid."""
    result = []
    y0 = 0
    for y1 in np.cumsum(thickness):
        dy = voxel_side * max(1, np.round(y1) // voxel_side - y0 // voxel_side)
        y0 = y0 + dy
        result.append(dy)
    return result


def _compact(mask):
    """Trim zero values on mask borders."""
    aabb = math_utils.minimum_aabb(mask)
    return math_utils.clip(mask, aabb)


def _build_2D_mosaic(width, hex_side, voxel_side):
    """Build 2D matrix representing O<K> mosaic."""
    hexagon = _compact(regular_convex_polygon_mask_from_side(hex_side, 6, voxel_side))
    w, h = hexagon.shape

    hex_center = {
        0: [
            [0, 0],
        ],
        1: [
            [1, 1],
            [0, 2],
            [1, 3],
            [2, 2],
            [0, 4],
            [1, 5],
            [2, 4],
        ],
    }[width]

    shift = np.array(hex_center) * (3 * w // 4, h // 2)

    shape = np.max(shift, axis=0) + (w, h)
    mosaic = np.full(shape, -1, dtype=np.int16)
    for column_label, (dx, dz) in enumerate(shift):
        mosaic[dx : dx + w, dz : dz + h][hexagon] = column_label

    offset = -0.5 * np.array([w, h]) * voxel_side
    return mosaic, offset


def _build_column_brain_regions(width, hex_side, layers, voxel_side):
    """Build 'brain_regions' VoxelData."""
    # pylint: disable=too-many-locals
    mosaic_2D, offset_2D = _build_2D_mosaic(width, hex_side, voxel_side)
    mosaic_3d_layers = []

    columns = np.unique(mosaic_2D[mosaic_2D >= 0])

    region_ids = OrderedDict(((column_label, None), k) for k, column_label in enumerate(columns, 1))

    for name, thickness in layers.items():
        pattern = np.zeros_like(mosaic_2D, dtype=np.uint16)
        for column_label in columns:
            region_id = max(region_ids.values()) + 1
            region_ids[(column_label, name)] = region_id
            pattern[mosaic_2D == column_label] = region_id
        mosaic_3d_layers.append(np.repeat([pattern], thickness // voxel_side, axis=0))

    mosaic_3D = np.swapaxes(np.vstack(mosaic_3d_layers), 0, 1)
    offset_3D = np.array([offset_2D[0], 0, offset_2D[1]])

    brain_regions = VoxelData(mosaic_3D, 3 * (voxel_side,), offset_3D).compact()

    # Add zero-voxel margin for better rendering
    margin = 1
    brain_regions = VoxelData(
        np.pad(brain_regions.raw, margin, "constant", constant_values=0),
        brain_regions.voxel_dimensions,
        brain_regions.offset - margin * brain_regions.voxel_dimensions,
    )

    return brain_regions, region_ids


def _build_hyperrectangle_brain_regions(x, z, layers, voxel_side):
    """Build hyperrectangle 'brain_regions' VoxelData."""
    tot_y = sum(layers.values())
    raw = np.zeros(
        (int(x // voxel_side), int(tot_y // voxel_side), int(z // voxel_side)), dtype=np.int32
    )

    region_id = OrderedDict()
    cumulative = 0
    for id_, (name, thickness) in enumerate(layers.items(), 1):
        region_id[name] = id_
        y_min = int(cumulative // voxel_side)
        y_max = int(cumulative + thickness // voxel_side)
        raw[:, y_min:y_max, :] = id_
        cumulative += thickness

    brain_regions = VoxelData(raw, np.repeat(voxel_side, 3), [0, 0, 0])

    # Add zero-voxel margin for better rendering
    margin = 1
    brain_regions = VoxelData(
        np.pad(brain_regions.raw, margin, "constant", constant_values=0),
        brain_regions.voxel_dimensions,
        brain_regions.offset - margin * brain_regions.voxel_dimensions,
    )
    return brain_regions, region_id


def _initialize_raw(brain_regions, dtype, value, add_dim=0):
    """Initialize a np array that will contain atlas data."""
    if add_dim > 0:
        shape = brain_regions.shape + (add_dim,)
    else:
        shape = brain_regions.shape
    layer_array = np.full(shape, value, dtype=dtype)
    return layer_array


def _build_orientation(brain_regions):
    """Build 'orientation' VoxelData."""
    raw = _initialize_raw(brain_regions, np.int8, 0, add_dim=4)
    raw[:, :, :, 0] = 127
    return brain_regions.with_data(raw)


def _build_layer_profile(brain_regions, boundaries):
    """Build '[PH]<layer>' VoxelData."""
    raw = _initialize_raw(brain_regions, np.float32, np.nan, add_dim=2)
    for j in range(brain_regions.raw.shape[1]):
        raw[:, j, :] = list(boundaries)
    return brain_regions.with_data(raw)


def _build_y(brain_regions):
    """Build '[PH]y' VoxelData."""
    raw = _initialize_raw(brain_regions, np.float32, np.nan)
    voxel_side = brain_regions.voxel_dimensions[1]
    for j in range(brain_regions.raw.shape[1]):
        raw[:, j, :] = brain_regions.offset[1] + voxel_side * (0.5 + j)
    return brain_regions.with_data(raw)


def _add_layers_atlases(datasets, layers, brain_regions):
    """Update the layer dataset with layer profiles"""

    def _pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    thickness_cumsum = [0.0] + list(np.cumsum(list(layers.values())))
    boundaries = np.array(list(_pairwise(thickness_cumsum)))
    for (name, _), bounds in zip(layers.items(), boundaries):
        datasets.update({"[PH]" + name: _build_layer_profile(brain_regions, bounds)})


def _dump_atlases(brain_regions, layers, output_dir):
    """Dump mandatory circuit building atlases."""
    datasets = {
        "brain_regions": brain_regions,
        "orientation": _build_orientation(brain_regions),
        "[PH]y": _build_y(brain_regions),
    }
    _add_layers_atlases(datasets, layers, brain_regions)
    for name, data in datasets.items():
        L.info("Write '%s.nrrd'...", name)
        data.save_nrrd(os.path.join(output_dir, name + ".nrrd"))


def _column_hierarchy(column_label, layers, region_ids):
    """Build 'hierarchy' dict for single hypercolumn."""
    return OrderedDict(
        [
            ("id", region_ids[(column_label, None)]),
            ("acronym", f"mc{column_label}_Column"),
            ("name", f"hypercolumn {column_label}"),
            (
                "children",
                [
                    OrderedDict(
                        [
                            ("id", region_ids[(column_label, layer)]),
                            ("acronym", f"mc{column_label};{layer}"),
                            ("name", f"hypercolumn {column_label}, {layer}"),
                        ]
                    )
                    for layer in layers
                ],
            ),
        ]
    )


def _mosaic_hierarchy(width, layers, region_ids):
    """Build 'hierarchy' dict for 'mosaic' atlas."""
    columns = sorted(set(col for col, _ in region_ids))
    return OrderedDict(
        [
            ("id", 65535),
            ("acronym", f"O{width}"),
            ("name", f"O{width} mosaic"),
            ("children", [_column_hierarchy(c, layers, region_ids) for c in columns]),
        ]
    )


def _hyperrectangle_hierarchy(region_ids):
    """Build 'hierarchy' dict for 'Hyperrectangle' atlas."""

    def _layer_hierarchy(layer_name, layer_id):
        """Build 'hierarchy' dict for hyperrectangle layer"""
        return OrderedDict(
            [
                ("id", layer_id),
                ("acronym", layer_name),
                ("name", f"Hyperrectangle layer{layer_id}, {layer_name}"),
            ]
        )

    return OrderedDict(
        [
            ("id", 65535),
            ("acronym", "H"),
            ("name", "Hyperrectangle"),
            (
                "children",
                [_layer_hierarchy(name, region_id) for name, region_id in region_ids.items()],
            ),
        ]
    )


def _normalize_hierarchy(hierarchy):
    """Sort keys in hierarchy dict."""
    result = OrderedDict((key, hierarchy[key]) for key in ["id", "acronym", "name"])
    if "children" in hierarchy:
        result["children"] = [_normalize_hierarchy(c) for c in hierarchy["children"]]
    return result


@click.group()
@click.option(
    "-n",
    "--layer-names",
    help="Layer's names as they appear going from 'bottom' to 'top'",
    required=True,
)
@click.option("-t", "--thickness", help="Layer thickness (um)", required=True)
@click.option("-d", "--voxel-side", type=float, help="Voxel side (um)", required=True)
@click.option("-o", "--output-dir", help="Path to output folder", required=True)
@click.pass_context
def app(ctx, layer_names, thickness, voxel_side, output_dir):
    """Building Synthetic Atlases."""

    ctx.ensure_object(dict)
    output_dir = os.path.abspath(output_dir)

    logging.basicConfig(level=logging.WARN)
    L.setLevel(logging.INFO)

    names = layer_names.split(",")
    raw_thickness = list(map(float, thickness.split(",")))
    assert len(names) == len(raw_thickness)

    aligned_thickness = _align_thickness(raw_thickness, voxel_side)
    L.info("Layer thickness aligned to voxel grid: %s", ",".join(map(str, aligned_thickness)))
    L.info("Total thickness before alignment: %s", sum(raw_thickness))
    L.info("Total thickness after alignment: %s", sum(aligned_thickness))

    layers = OrderedDict(zip(names, aligned_thickness))

    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ctx.obj["voxel_side"] = voxel_side
    ctx.obj["output_dir"] = output_dir
    ctx.obj["layers"] = layers


@app.command()
@click.option("-w", "--width", type=int, help="Mosaic width (0 for single column)", default=0)
@click.option("-a", "--hex-side", type=float, help="Hexagon side (um)", required=True)
@click.pass_context
def column(ctx, width, hex_side):
    """Build synthetic hexagonal column atlas."""

    voxel_side = ctx.obj["voxel_side"]
    output_dir = ctx.obj["output_dir"]
    layers = ctx.obj["layers"]

    assert width in (0, 1)

    brain_regions, region_ids = _build_column_brain_regions(width, hex_side, layers, voxel_side)

    _dump_atlases(brain_regions, layers, output_dir)

    hierarchy = _mosaic_hierarchy(width, list(layers.keys()), region_ids)
    L.info("Write 'hierarchy.json'...")
    dump_json(Path(output_dir) / "hierarchy.json", _normalize_hierarchy(hierarchy))

    L.info("Done!")


@app.command()
@click.option(
    "-x", "--x-length", type=float, help="Atlas length in the x direction (um)", required=True
)
@click.option(
    "-z", "--z-length", type=float, help="Atlas length in the z direction (um)", required=True
)
@click.pass_context
def hyperrectangle(ctx, x_length, z_length):
    """Build synthetic hyperrectangle atlas."""

    voxel_side = ctx.obj["voxel_side"]
    output_dir = ctx.obj["output_dir"]
    layers = ctx.obj["layers"]

    brain_regions, region_ids = _build_hyperrectangle_brain_regions(
        x_length, z_length, layers, voxel_side
    )

    _dump_atlases(brain_regions, layers, output_dir)

    hierarchy = _hyperrectangle_hierarchy(region_ids)
    L.info("Write 'hierarchy.json'...")
    dump_json(Path(output_dir) / "hierarchy.json", _normalize_hierarchy(hierarchy))

    L.info("Done!")
