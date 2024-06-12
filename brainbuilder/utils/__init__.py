# SPDX-License-Identifier: Apache-2.0
""" libraries of common functionality for circuit building """

import json

import yaml


def create_appendable_dataset(h5_root, name, dtype, chunksize=1000):
    """create an h5 appendable dataset at `h5_root` w/ `name`"""
    h5_root.create_dataset(
        name,
        dtype=dtype,
        chunks=(chunksize,),
        shape=(0,),
        maxshape=(None,),
    )


def append_to_dataset(dset, values):
    """append `values` to `dset`, which should be an appendable dataset"""
    dset.resize(dset.shape[0] + len(values), axis=0)
    dset[-len(values) :] = values


def load_json(filepath):
    """Load from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(filepath):
    """Load from YAML file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_json(filepath, data, indent=2):
    """Dump to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def dump_yaml(filepath, data):
    """Dump to YAML file."""
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
