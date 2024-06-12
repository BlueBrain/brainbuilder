# SPDX-License-Identifier: Apache-2.0
""" Tools for working with MVD3 """

import shutil
from builtins import input  # pylint: disable=redefined-builtin

import click
import numpy as np
import pandas as pd
from voxcell import CellCollection, VoxelData

from brainbuilder.utils import bbp


@click.group()
def app():
    """Tools for working with MVD3"""


@app.command()
@click.argument("mvd3")
@click.option("--recipe", help="Path to builder recipe XML", required=True)
@click.option("-o", "--output", help="Path to output MVD3", required=True)
def reorder_mtypes(mvd3, recipe, output):
    """Align /library/mtypes with builder recipe"""
    tmp_path = output + "~"
    shutil.copy(mvd3, tmp_path)
    bbp.reorder_mtypes(tmp_path, recipe)
    shutil.move(tmp_path, output)


@app.command()
@click.argument("mvd3")
@click.option("-p", "--prop", help="Property name to use", required=True)
@click.option("-d", "--voxel-data", help="Path NRRD with to volumetric data", required=True)
@click.option("-o", "--output", help="Path to output MVD3", required=True)
def add_property(mvd3, prop, voxel_data, output):
    """Add property to MVD3 based on volumetric data"""
    cells = CellCollection.load_mvd3(mvd3)
    if prop in cells.properties:
        choice = input(
            f"There is already '{prop}' property in the provided MVD3. Overwrite (y/n)? "
        )
        if choice.lower() not in ("y", "yes"):
            return
    voxel_data = VoxelData.load_nrrd(voxel_data)
    cells.properties[prop] = voxel_data.lookup(cells.positions)
    cells.save_mvd3(output)


@app.command()
@click.argument("mvd3", nargs=-1)
@click.option("-o", "--output", help="Path to output MVD3", required=True)
def merge(mvd3, output):
    """Merge multiple MVD3 files"""
    chunks = [CellCollection.load_mvd3(filepath).as_dataframe() for filepath in mvd3]
    merged = pd.concat(chunks, ignore_index=True)
    merged.index = 1 + np.arange(len(merged))
    CellCollection.from_dataframe(merged).save_mvd3(output)
