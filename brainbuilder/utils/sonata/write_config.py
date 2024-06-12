# SPDX-License-Identifier: Apache-2.0
"""SONATA configuration functions."""

from collections import OrderedDict
from functools import partial
from pathlib import Path

import jsonschema

from brainbuilder.utils import dump_json

# minimal nodes schema to validate data before building the sonata configuration
nodes_schema = {
    "type": "array",
    "minitems": 1,
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["nodes_file", "populations"],
        "properties": {
            "nodes_file": {"type": "string"},
            "populations": {
                "type": "object",
                "minProperties": 1,
                "patternProperties": {"^.+$": {"type": "object"}},
                "additionalProperties": False,
            },
        },
    },
}
# minimal edges schema to validate data before building the sonata configuration
edges_schema = {
    "type": "array",
    "minitems": 1,
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["edges_file", "populations"],
        "properties": {
            "edges_file": {"type": "string"},
            "populations": {
                "type": "object",
                "minProperties": 1,
                "patternProperties": {"^.+$": {"type": "object"}},
                "additionalProperties": False,
            },
        },
    },
}


def _resolve_dir(base_dir, dir_):
    return base_dir if dir_ is None else str(Path(base_dir, dir_))


def _resolve_components(components):
    resolved = {}
    for k, v in components.items():
        if k == "alternate_morphologies":
            resolved[k] = _resolve_components(v)
        elif k == "type":
            resolved[k] = v
        else:
            # all the other properties are absolute paths or paths relative to $COMPONENTS_DIR
            resolved[k] = _resolve_dir("$COMPONENTS_DIR", v)
    return resolved


def _make_network(iterable, file_key, base_dir):
    return [
        {
            file_key: _resolve_dir(base_dir, item[file_key]),
            "populations": {
                name: _resolve_components(pops) for name, pops in item["populations"].items()
            },
        }
        for item in iterable
    ]


_make_nodes_network = partial(_make_network, file_key="nodes_file", base_dir="$NETWORK_NODES_DIR")
_make_edges_network = partial(_make_network, file_key="edges_file", base_dir="$NETWORK_EDGES_DIR")


def make_network_config_bbp(
    base_dir,
    nodes,
    edges,
    node_sets,
    components=None,
    nodes_dir=None,
    edges_dir=None,
    components_dir=None,
):
    """Write SONATA_ config of version 2 that is extended for BBP usage.

    Args:
        base_dir(str|Path): an absolute path to a directory of the circuit that the config is
            written for. It is represented in the config as `$BASE_DIR`.
        nodes(list): list where an item must be a dict with keys 'nodes_file' and 'populations'.
            'nodes_file' must contain a path to a nodes file.
            'populations' must be a dict as described in SONATA_.
            Relative paths to nodes files are prepended with `$NETWORK_NODES_DIR` prefix.
        edges(list): list where an item must be a dict with keys 'edges_file' and 'populations'.
            'edges_file' must contain a path to an edges file.
            'populations' must be a dict as described in SONATA_.
            Relative paths to edges files are prepended with `$NETWORK_EDGES_DIR` prefix.
        node_sets(str|Path): a path to a nodesets file. If relative then it is prepended with
            `BASE_DIR` prefix in the config.
        components(dict): a 'components' field dict as described in SONATA_. Relative paths
            here are prepended with `$COMPONENTS_DIR`.
        nodes_dir(str|Path): a path to a directory that contains nodes files by default, it is
            represented in the config as `$NETWORK_NODES_DIR`. If not set then it equals to
            `$BASE_DIR`.
        edges_dir(str|Path): a path to a directory that contains edges files by default, it is
            represented in the config as `$NETWORK_EDGES_DIR`. If not set then it equals to
            `$BASE_DIR`.
        components_dir(str|Path): a path to a directory that contains components by default, it is
            represented in the config as `$COMPONENTS_DIR`. If not set then it equals to
            `$BASE_DIR`.

    .. _SONATA: https://sonata-extension.readthedocs.io/en/latest/sonata_config.html
    """
    jsonschema.validate(instance=nodes, schema=nodes_schema)
    jsonschema.validate(instance=edges, schema=edges_schema)

    config = OrderedDict()
    config["version"] = 2

    config["manifest"] = {
        "$BASE_DIR": base_dir,
        "$COMPONENTS_DIR": _resolve_dir("$BASE_DIR", components_dir),
        "$NETWORK_NODES_DIR": _resolve_dir("$BASE_DIR", nodes_dir),
        "$NETWORK_EDGES_DIR": _resolve_dir("$BASE_DIR", edges_dir),
    }

    if components is not None:
        config["components"] = _resolve_components(components)

    config["node_sets_file"] = _resolve_dir("$BASE_DIR", node_sets)
    config["networks"] = {
        "nodes": _make_nodes_network(nodes),
        "edges": _make_edges_network(edges),
    }
    return config


def write_network_config(
    base_dir,
    morph_dir,
    emodel_dir,
    nodes_dir,
    nodes,
    node_sets,
    edges_dir,
    edges_suffix,
    edges,
    output_path,
):
    """Write SONATA network config to ``output_path``.

    If a relative path is used for any filepath argument then it will be prepended with a
    corresponding SONATA path. For example ``morph_dir`` will be prepended with ``base_dir``.
    If an absolute path is used then it will be used as is.

    Args:
        base_dir (str|Path): $BASE_DIR of the written config
        morph_dir (str|Path): 'morphologies_dir' of the written config
        emodel_dir (str|Path): 'biophysical_neuron_models_dir' of the written config
        nodes_dir (str|Path): folder that would contain all nodes of the written config network
        nodes (list): list of dictionaries containing 'nodes_file' and 'populations', or
            list of paths to nodes files in the format ``path:population1,population2,...``.
        node_sets (str|Path): 'node_sets_file' of the written config
        edges_dir (str|Path): folder that would contain all edges of the written config network
        edges_suffix (str): suffix to append to edges files, used only if edges is a list of paths.
        edges (list): list of dictionaries containing 'edges_file' and 'populations', or
            list of paths to edges files in the format ``path:population1,population2,...``.
        output_path (str|Path): path to a file where to write the config.
    """

    # pylint: disable=too-many-arguments
    def _to_dict(s, key, default_filename, main_separator=":", populations_separator=","):
        path, _, populations = s.partition(main_separator)
        if not populations:
            raise ValueError(
                f"Populations for {key} must be specified in the format `path:pop1,pop2...`"
            )
        populations = populations.split(populations_separator)
        if not path.endswith(".h5"):
            path = str(Path(path, default_filename))
        return {key: path, "populations": {pop: {} for pop in populations}}

    _to_nodes_dict = partial(_to_dict, key="nodes_file", default_filename="nodes.h5")
    _to_edges_dict = partial(_to_dict, key="edges_file", default_filename=f"edges{edges_suffix}.h5")
    nodes = [_to_nodes_dict(i) if isinstance(i, str) else i for i in nodes]
    edges = [_to_edges_dict(i) if isinstance(i, str) else i for i in edges]
    components = {
        "morphologies_dir": morph_dir,
        "biophysical_neuron_models_dir": emodel_dir,
    }
    config = make_network_config_bbp(
        base_dir=base_dir,
        nodes=nodes,
        edges=edges,
        node_sets=node_sets,
        components=components,
        nodes_dir=nodes_dir,
        edges_dir=edges_dir,
    )
    dump_json(output_path, config)
