# SPDX-License-Identifier: Apache-2.0
"""Split a SONATA node/edge population into sub-populations"""

import collections
import copy
import itertools as it
import logging
import os
from pathlib import Path

import bluepysnap
import h5py
import libsonata
import numpy as np
import pandas as pd
import voxcell
from joblib import Parallel, delayed

from brainbuilder import utils
from brainbuilder.utils.sonata import curate

L = logging.getLogger(__name__)

# So as not to exhaust memory, the edges files are loaded/written in chunks of this size
H5_READ_CHUNKSIZE = 500_000_000
# Name of the unique expected group in sonata nodes and edges files
GROUP_NAME = "0"
# Sentinel to mark an edge file being empty
DELETED_EMPTY_EDGES_FILE = "DELETED_EMPTY_EDGES_FILE"


def _create_chunked_slices(length, chunk_size):
    """return `slices` each of size `chunk_size`, that cover `length`"""
    return (slice(start, start + chunk_size) for start in range(0, length, chunk_size))


def _isin_worker(elements, test_elements, sl, invert):
    """worker for parallelized version of nump.isin"""
    return np.isin(elements[sl], test_elements, invert=invert)


def _isin(elements, test_elements, invert=False):
    """parallelized version of nump.isin"""
    h5_chunksize = _h5_get_read_chunk_size()

    if len(elements) < h5_chunksize:
        return np.isin(elements, test_elements, invert=invert)

    parallel = Parallel(
        backend="loky",
        n_jobs=-2,
        # verbose=51,
    )

    # arbitrary chunk_size; 1e6 with the default H5_READ_CHUNKSIZE seems about right
    chunk_size = max(500, int(h5_chunksize / 500))
    ret = parallel(
        delayed(_isin_worker)(elements, test_elements, sl, invert)
        for sl in _create_chunked_slices(len(elements), chunk_size)
    )

    ret = np.concatenate(ret)

    return ret


def _get_population_name(src, dst, synapse_type="chemical"):
    """Return the population name based off `src` and `dst` node population names."""
    return src if src == dst else f"{src}__{dst}__{synapse_type}"


def _get_edge_file_name(new_pop_name):
    """Return the name of the edge file split by population."""
    return f"edges_{new_pop_name}.h5"


def _get_node_file_name(new_pop_name):
    """Return the name of the node file split by population."""
    return f"nodes_{new_pop_name}.h5"


def _get_unique_population(parent):
    """Return the h5 unique population, raise an exception if not unique."""
    population_names = list(parent)
    if len(population_names) != 1:
        raise ValueError(f"Single population is supported only, found {population_names}")
    return population_names[0]


def _get_unique_group(parent):
    """Return the h5 group 0, raise an exception if non present."""
    if GROUP_NAME not in parent:
        raise ValueError(f"Single group {GROUP_NAME!r} is required")
    return parent[GROUP_NAME]


def _load_sonata_nodes(nodes_path):
    """Load nodes from a sonata file and return it as dataframe (0-based IDs).

    Note: the returned dataframe contains the orientation matrices, but it does not contain
    the information about the original orientation format (quaternions or eulers).
    """
    df = voxcell.CellCollection.load_sonata(nodes_path).as_dataframe()
    # CellCollection returns 1-based IDs but we need 0-based IDs
    df.index -= 1
    return df


def _save_sonata_nodes(nodes_path, df, population_name):
    """Save a dataframe of nodes (0-based IDs) to sonata file.

    Note: using voxcell >= 2.7.1 to load the dataframe and save the result to sonata,
    CellCollection will save the orientation using the default format (quaternions).
    """
    # CellCollection expects 1-based IDs
    df.index += 1
    cell_collection = voxcell.CellCollection.from_dataframe(df)
    cell_collection.population_name = population_name
    cell_collection.save_sonata(nodes_path, mode="a")
    # restore the original index
    df.index -= 1
    return nodes_path


def _init_edge_group(orig_group, new_group):
    """Copy the empty datasets from orig_group to new_group.

    Args:
        orig_group (h5py.Group): original group, e.g. /edges/default/0
        new_group (h5py.Group): new group, e.g. /edges/L2_X__L6_Y__chemical/0
    """
    for name, attr in orig_group.items():
        if isinstance(attr, h5py.Dataset):
            utils.create_appendable_dataset(new_group, name, attr.dtype)
        elif isinstance(attr, h5py.Group) and name == "dynamics_params":
            new_group.create_group(name)
            for k, values in attr.items():
                assert isinstance(values, h5py.Dataset), f"dynamics_params has an h5 subgroup: {k}"
                utils.create_appendable_dataset(new_group[name], k, values.dtype)
        else:
            raise ValueError('Only "dynamics_params" group is expected')


def _populate_edge_group(orig_group, new_group, sl, mask):
    """Populate the datasets from orig_group to new_group.

    Args:
        orig_group (h5py.Group): original group, e.g. /edges/default/0
        new_group (h5py.Group): new group, e.g. /edges/L2_X__L6_Y__chemical/0
        sl (slice): slice used to select the dataset range
        mask (np.ndarray): mask used to filter the dataset
    """
    for name, attr in orig_group.items():
        if isinstance(attr, h5py.Dataset):
            utils.append_to_dataset(new_group[name], attr[sl][mask])
        elif isinstance(attr, h5py.Group) and name == "dynamics_params":
            for k, values in attr.items():
                if isinstance(values, h5py.Dataset):
                    utils.append_to_dataset(new_group[name][k], values[sl][mask])
        else:
            raise ValueError('Only "dynamics_params" group is expected')


def _finalize_edges(new_edges):
    """add datasets for `new_edges` so they fulfil SONATA spec"""
    edge_count = len(new_edges["source_node_id"])
    new_edges["edge_type_id"] = np.full(edge_count, -1)
    new_edges["edge_group_id"] = np.full(edge_count, 0)
    new_edges["edge_group_index"] = np.arange(edge_count, dtype=np.uint64)


def _h5_get_read_chunk_size():
    """get the desired H5 read size, either from default of from env var"""
    return int(os.environ.get("H5_READ_CHUNKSIZE", H5_READ_CHUNKSIZE))


def _copy_edge_attributes(  # pylint: disable=too-many-arguments
    h5in,
    h5out,
    src_node_name,
    dst_node_name,
    src_edge_name,
    dst_edge_name,
    src_mapping,
    dst_mapping,
    h5_read_chunk_size=None,
):
    """Copy the attributes from the original edges into the new edge populations"""
    # pylint: disable=too-many-locals
    if h5_read_chunk_size is None:
        h5_read_chunk_size = _h5_get_read_chunk_size()

    orig_edges = h5in["edges"][src_edge_name]
    orig_group = _get_unique_group(orig_edges)
    new_edges = h5out.create_group("edges/" + dst_edge_name)
    new_group = new_edges.create_group(GROUP_NAME)

    utils.create_appendable_dataset(new_edges, "source_node_id", np.uint64)
    utils.create_appendable_dataset(new_edges, "target_node_id", np.uint64)

    new_edges["source_node_id"].attrs["node_population"] = src_node_name
    new_edges["target_node_id"].attrs["node_population"] = dst_node_name

    _init_edge_group(orig_group, new_group)

    sgids_new = src_mapping.index.to_numpy()
    tgids_new = dst_mapping.index.to_numpy()
    assert (sgids_new >= 0).all(), "Source population ids must be positive."
    assert (tgids_new >= 0).all(), "Target population ids must be positive."

    for sl in _create_chunked_slices(len(orig_edges["source_node_id"]), h5_read_chunk_size):
        sgids = orig_edges["source_node_id"][sl]
        tgids = orig_edges["target_node_id"][sl]
        sgid_mask = _isin(sgids, sgids_new)
        tgid_mask = _isin(tgids, tgids_new)

        mask = sgid_mask & tgid_mask

        if np.any(mask):
            utils.append_to_dataset(
                new_edges["source_node_id"], src_mapping.loc[sgids[mask]].new_id.to_numpy()
            )
            utils.append_to_dataset(
                new_edges["target_node_id"], dst_mapping.loc[tgids[mask]].new_id.to_numpy()
            )
            _populate_edge_group(orig_group, new_group, sl, mask)

    _finalize_edges(new_edges)


def _get_node_counts(h5out, new_pop_name):
    """for `h5out`, return the `edge_count`, `source_node_count`, and `target_node_count`"""
    source_node_count = target_node_count = 0
    new_edges = h5out["edges"][new_pop_name]
    edge_count = len(new_edges["source_node_id"])
    if edge_count > 0:
        # add 1 because IDs are 0-based
        source_node_count = int(np.max(new_edges["source_node_id"]) + 1)
        target_node_count = int(np.max(new_edges["target_node_id"]) + 1)
    return edge_count, source_node_count, target_node_count


def _write_indexes(edge_file_name, new_pop_name, source_node_count, target_node_count):
    """ibid"""
    libsonata.EdgePopulation.write_indices(
        edge_file_name, new_pop_name, source_node_count, target_node_count
    )


def _check_all_edges_used(h5in, written_edges):
    """Verify that the number of written edges matches the number of initial edges."""
    orig_edges = h5in["edges"][_get_unique_population(h5in["edges"])]
    expected_edges = len(orig_edges["source_node_id"])
    if expected_edges != written_edges:
        raise RuntimeError(
            f"Written edges mismatch: expected={expected_edges}, actual={written_edges}"
        )


def _write_edges(
    output,
    edges_path,
    id_mapping,
    h5_read_chunk_size=None,
    expect_to_use_all_edges=True,
):
    """create all new edge populations in separate files"""
    with h5py.File(edges_path, "r") as h5in:
        written_edges = 0
        for src_node_pop, dst_node_pop in it.product(id_mapping, id_mapping):
            edge_pop_name = _get_population_name(src_node_pop, dst_node_pop)
            edge_file_name = os.path.join(output, _get_edge_file_name(edge_pop_name))

            L.debug("Writing to  %s", edge_file_name)
            with h5py.File(edge_file_name, "w") as h5out:
                _copy_edge_attributes(
                    h5in=h5in,
                    h5out=h5out,
                    src_node_name=src_node_pop,
                    dst_node_name=dst_node_pop,
                    src_edge_name=_get_unique_population(h5in["edges"]),
                    dst_edge_name=edge_pop_name,
                    src_mapping=id_mapping[src_node_pop],
                    dst_mapping=id_mapping[dst_node_pop],
                    h5_read_chunk_size=h5_read_chunk_size,
                )
                edge_count, sgid_count, tgid_count = _get_node_counts(h5out, edge_pop_name)

            # after the h5 file is closed, it's indexed if valid, or it's removed if empty
            if edge_count > 0:
                _write_indexes(edge_file_name, edge_pop_name, sgid_count, tgid_count)
                L.debug("Wrote %s edges to %s", edge_count, edge_file_name)
                written_edges += edge_count
            else:
                os.unlink(edge_file_name)

        if expect_to_use_all_edges:
            _check_all_edges_used(h5in, written_edges)


def _write_nodes(output, split_nodes, population_to_path=None):
    """create all new node populations in separate files

    Args:
        output(str): base directory to write node files
        split_nodes(dict): new_population_name -> df
        population_to_path(dict): population_name -> output path
    """
    if population_to_path is None:
        population_to_path = {}

    ret = {}
    for new_population, df in split_nodes.items():
        df = df.reset_index(drop=True)
        nodes_path = population_to_path.get(new_population, _get_node_file_name(new_population))
        nodes_path = os.path.join(output, nodes_path)
        Path(nodes_path).parent.mkdir(parents=True, exist_ok=True)
        ret[new_population] = _save_sonata_nodes(nodes_path, df, population_name=new_population)
        L.debug("Wrote %s nodes to %s", len(df), nodes_path)

    return ret


def _get_node_id_mapping(split_nodes):
    """return a dict split_nodes.keys() -> DataFrame with index old_ids, and colunm new_id"""
    return {
        new_population: pd.DataFrame({"new_id": np.arange(len(df), dtype=np.int64)}, index=df.index)
        for new_population, df in split_nodes.items()
    }


def _split_population_by_attribute(nodes_path, attribute):
    """return a dictionary keyed on attribute values with each of the new populations

    Each of the unique attribute values becomes a new_population post split

    Args:
        nodes_path: path to SONATA nodes file
        attribute(str): attribute to split on

    Returns:
        dict: new_population -> df containing attributes for that new population
    """
    nodes = _load_sonata_nodes(nodes_path)
    L.debug("Splitting population on %s -> %s", attribute, nodes[attribute].unique())
    split_nodes = dict(tuple(nodes.groupby(attribute)))
    return split_nodes


def _write_circuit_config(output, split_nodes):
    """Write a simple circuit-config.json for all the node/edge populations created"""
    tmpl = {
        "manifest": {
            "$BASE_DIR": ".",
        },
        "networks": {
            "nodes": [],
            "edges": [],
        },
    }

    for src, dst in it.product(split_nodes, split_nodes):
        new_pop_name = _get_population_name(src, dst)
        if src == dst:
            tmpl["networks"]["nodes"].append(
                {
                    "nodes_file": os.path.join("$BASE_DIR", _get_node_file_name(new_pop_name)),
                    "node_types_file": None,
                }
            )

        if os.path.exists(os.path.join(output, _get_edge_file_name(new_pop_name))):
            tmpl["networks"]["edges"].append(
                {
                    "edges_file": os.path.join("$BASE_DIR", _get_edge_file_name(new_pop_name)),
                    "edge_types_file": None,
                }
            )

    filepath = Path(output) / "circuit_config.json"
    utils.dump_json(filepath, tmpl)
    L.debug("Written circuit config %s", filepath)


def split_population(output, attribute, nodes_path, edges_path):
    """split a single node SONATA dataset into separate populations based on attribute

    Creates a new nodeset, and the corresponding edges between nodesets for each
    value of the attribute.  For instance, if the attribute chosen is 'region', a nodeset
    will be created for all regions

    The edge file is also split, as required

    Args:
        output(str): path where files will be written
        attribute(str): attribute on which to break up into sub-populations
        nodes_path(str): path to nodes sonata file
        edges_path(str): path to edges sonata file

    """
    split_populations = _split_population_by_attribute(nodes_path, attribute)
    _write_nodes(output, split_populations)

    id_mapping = _get_node_id_mapping(split_populations)
    _write_edges(output, edges_path, id_mapping, expect_to_use_all_edges=True)

    _write_circuit_config(output, split_populations)


def _split_population_by_node_set(nodes_path, node_set_name, node_set_path):
    node_storage = libsonata.NodeStorage(nodes_path)
    node_population = node_storage.open_population(next(iter(node_storage.population_names)))

    node_sets = libsonata.NodeSets.from_file(node_set_path)
    ids = node_sets.materialize(node_set_name, node_population).flatten()

    split_nodes = {node_set_name: _load_sonata_nodes(nodes_path).loc[ids]}
    return split_nodes


def simple_split_subcircuit(output, node_set_name, node_set_path, nodes_path, edges_path):
    """Split a single subcircuit out of a set of nodes and edges, based on nodeset

    Args:
        output(str): path where files will be written
        node_set_name(str): name of nodeset to extract
        node_set_path(str): path to node_sets.json file
        nodes_path(str): path to nodes sonata file
        edges_path(str): path to edges sonata file
    """
    split_populations = _split_population_by_node_set(nodes_path, node_set_name, node_set_path)

    _write_nodes(output, split_populations)

    id_mapping = _get_node_id_mapping(split_populations)
    _write_edges(output, edges_path, id_mapping, expect_to_use_all_edges=False)


def _write_subcircuit_edges(
    output_path,
    edges_path,
    src_node_pop,
    dst_node_pop,
    src_edge_pop_name,
    dst_edge_pop_name,
    src_mapping,
    dst_mapping,
):
    """copy a population to an edge file

    If DELETED_EMPTY_EDGES_FILE is returned, the file was removed since no
    populations existed in it any more
    """
    with h5py.File(edges_path, "r") as h5in:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        is_file_empty = False

        with h5py.File(output_path, "a") as h5out:
            _copy_edge_attributes(
                h5in=h5in,
                h5out=h5out,
                src_node_name=src_node_pop,
                dst_node_name=dst_node_pop,
                src_edge_name=src_edge_pop_name,
                dst_edge_name=dst_edge_pop_name,
                src_mapping=src_mapping,
                dst_mapping=dst_mapping,
            )
            edge_count, sgid_count, tgid_count = _get_node_counts(h5out, dst_edge_pop_name)

            if edge_count == 0:
                del h5out[f"/edges/{dst_edge_pop_name}"]
                is_file_empty = len(h5out["/edges"]) == 0

        # after the h5 file is closed, it's indexed if valid, or it's removed if empty
        if edge_count > 0:
            _write_indexes(output_path, dst_edge_pop_name, sgid_count, tgid_count)
            L.debug("Wrote %s edges to %s", edge_count, output_path)
        elif is_file_empty:
            os.unlink(output_path)
            output_path = DELETED_EMPTY_EDGES_FILE

        return output_path


def _gather_layout_from_networks(networks):
    """find the layout of the nodes and edges files, return a dict of the name -> relative path"""

    # Note: we are 'prioritizing' the layout of the config over the layout of the files on disk:
    # 1) the `nodes`/`edges` network keys will still have the same number of elements
    #    after writing the new config (unless populations aren't used)
    # 2) The layout of the files may be slightly different; if the config has a single population
    #    in the dict, the output population will be writen to $population_name/$original_filename.h5
    #    if it has multiple elements, it will be written to
    #    $original_parent_dir/$original_filename.h5
    #
    # See tests for more clarity
    node_populations_to_paths, edge_populations_to_paths = {}, {}

    def _extract_population_paths(key):
        """extract populations from `network_base`; return dictionary with their file path"""
        key_name = f"{key}_file"
        ret = {}
        for stanza in networks[key]:
            filename = Path(stanza[key_name]).name
            if len(stanza["populations"]) == 1:
                population = next(iter(stanza["populations"]))
                ret[population] = str(Path(population) / filename)
            else:
                # multiple populations; need to group them into the same file
                base_path = Path(stanza[key_name]).parent.name
                for population in stanza["populations"]:
                    ret[population] = str(Path(base_path) / filename)
        return ret

    node_populations_to_paths = _extract_population_paths("nodes")
    edge_populations_to_paths = _extract_population_paths("edges")

    return node_populations_to_paths, edge_populations_to_paths


def _get_storage_path(edge):
    """Return the storage path."""
    return edge.h5_filepath


def _write_subcircuit_biological(
    output,
    circuit,
    node_pop_to_paths,
    edge_pop_to_paths,
    split_populations,
    id_mapping,
):
    """write node and edge population that belong in a subcircuit

    Args:
        output: path to output
        circuit: bluepysnap circuit
        node_pop_to_paths(dict): node name -> new relative path
        edge_pop_to_paths(dict): node name -> new relative path
        split_populations(dict): population -> node dataframe
        id_mapping(dict): population name -> df with index old_ids, and colunm new_id

    returns `new_node_files`, `new_edges_files`: the paths to node & edges files that were created
    """
    new_node_files = _write_nodes(output, split_populations, node_pop_to_paths)

    new_edges_files = {}
    for edge_pop_name, edge in circuit.edges.items():
        if edge.source.name in id_mapping and edge.target.name in id_mapping:
            output_path = output / edge_pop_to_paths[edge_pop_name]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            L.debug(
                "Writing edges %s for %s -> %s [%s]",
                edge_pop_name,
                edge.source.name,
                edge.target.name,
                output_path,
            )
            new_edges_files[edge_pop_name] = _write_subcircuit_edges(
                output_path=str(output_path),
                edges_path=_get_storage_path(edge),
                src_node_pop=edge.source.name,
                dst_node_pop=edge.target.name,
                src_edge_pop_name=edge_pop_name,
                dst_edge_pop_name=edge_pop_name,
                src_mapping=id_mapping[edge.source.name],
                dst_mapping=id_mapping[edge.target.name],
            )

    return new_node_files, new_edges_files


def _get_subcircuit_external_ids(all_sgids, all_tgids, wanted_src_ids, wanted_dst_ids):
    """get the `external` ids

    return `id_mapping` style DataFrame for connections between `all_sgids` and
    `all_tgids` where sgids are in wanted_src_ids and tgids are in `wanted_dst_ids`

    These are the 'external' ids that become 'virtual' in the extracted subcircuit
    """
    h5_read_chunk_size = _h5_get_read_chunk_size()
    ret = None
    for sl in _create_chunked_slices(len(all_sgids), h5_read_chunk_size):
        sgids = all_sgids[sl]
        tgids = all_tgids[sl]

        mask = _isin(sgids, wanted_src_ids) & _isin(tgids, wanted_dst_ids)

        if mask.any():
            needed = np.unique(sgids[mask])
            if ret is None:
                ret = pd.DataFrame({"new_id": np.arange(len(needed), dtype=np.uint)}, index=needed)
            else:
                mm = _isin(needed, ret.index.to_numpy(), invert=True)
                if mm.any():
                    needed = needed[mm]
                    start_id = int(ret.new_id.max()) + 1
                    new = pd.DataFrame(
                        {"new_id": start_id + np.arange(len(needed), dtype=np.uint)}, index=needed
                    )
                    ret = pd.concat((ret, new))

    if ret is None:
        ret = pd.DataFrame({"new_id": np.array([], dtype=np.uint)}, index=[])

    return ret.sort_index()


def _write_subcircuit_external(output, circuit, id_mapping):
    """Write external connectivity.

    returns: (new_node_files, new_edges_files); with, respectively,
    dictionaries with node and edge population_name -> path

    Warning: this writes `id_mapping` in place
    """
    new_nodes = {}

    new_edges_files = {}
    for name, edge in circuit.edges.items():
        if edge.source.type != "virtual" and edge.target.name in id_mapping:
            wanted_src_ids = circuit.nodes[edge.source.name].ids()

            if edge.source.name in id_mapping:
                wanted_src_ids = wanted_src_ids[
                    _isin(
                        wanted_src_ids, id_mapping[edge.source.name].index.to_numpy(), invert=True
                    )
                ]

            # only keep ids that are used; this is duplicating work in _copy_edge_attributes
            # but the alternative is that it keeps track of the new id_mapping; which
            # seemed less ideal
            with h5py.File(_get_storage_path(edge)) as h5:
                all_sgids = h5[f"edges/{name}/source_node_id"]
                all_tgids = h5[f"edges/{name}/target_node_id"]

                # overwrite wanted_src_ids with a DataFrame; the numpy array is not needed
                wanted_src_ids = _get_subcircuit_external_ids(
                    all_sgids,
                    all_tgids,
                    wanted_src_ids,
                    id_mapping[edge.target.name].index.to_numpy(),
                )

            if len(wanted_src_ids) == 0:
                continue

            new_name = f"external_{name}"
            output_path = output / (new_name + ".h5")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            L.debug(
                "Writing edges %s for %s -> %s [%s]",
                name,
                edge.source.name,
                edge.target.name,
                output_path,
            )
            new_edges_files[new_name] = _write_subcircuit_edges(
                output_path=str(output_path),
                edges_path=_get_storage_path(edge),
                src_node_pop=edge.source.name,
                dst_node_pop=edge.target.name,
                src_edge_pop_name=name,
                dst_edge_pop_name=new_name,
                src_mapping=wanted_src_ids,
                dst_mapping=id_mapping[edge.target.name],
            )
            new_nodes[new_name] = wanted_src_ids

            id_mapping[new_name] = wanted_src_ids

    new_node_files = {}
    for population_name, wanted_src_ids in new_nodes.items():
        if len(wanted_src_ids) == 0:
            continue
        # pylint: disable=protected-access
        new_node_files[population_name] = curate._create_source_nodes(
            population_name, len(wanted_src_ids), output
        )

    return new_node_files, new_edges_files


def _write_subcircuit_virtual(output, circuit, edge_populations_to_paths, id_mapping):
    """write all node/edge populations that have virtual nodes as source

    Note: the id_mapping dictionary is updated with the used virtual nodes
    """
    # pylint: disable=too-many-locals
    new_node_files, new_edges_files = {}, {}

    virtual_populations = {
        name: edge
        for name, edge in circuit.edges.items()
        if (edge.source.type == "virtual" and edge.target.name in id_mapping)
    }

    # gather the ids of the virtual populations that are used; within a circuit
    # it's possible that a virtual population points to multiple target populations
    pop_used_source_node_ids = collections.defaultdict(list)
    for name, edge in virtual_populations.items():
        target_node_ids = id_mapping[edge.target.name].index.to_numpy()
        target_node_ids = bluepysnap.circuit_ids.CircuitNodeIds.from_dict(
            {edge.target.name: target_node_ids}
        )

        pop_used_source_node_ids[edge.source.name].append(
            edge.afferent_nodes(target_node_ids, unique=True)
        )

    pop_used_source_node_ids = {
        name: np.unique(np.concatenate(ids)) for name, ids in pop_used_source_node_ids.items()
    }

    # update the mappings with the virtual nodes
    for name, ids in pop_used_source_node_ids.items():
        id_mapping[name] = pd.DataFrame({"new_id": range(len(ids))}, index=ids)

    # write the edges that have the virtual populations as source
    for edge_pop_name, edge in virtual_populations.items():
        new_edges_files[edge_pop_name] = _write_subcircuit_edges(
            output_path=os.path.join(output, edge_populations_to_paths[edge_pop_name]),
            edges_path=_get_storage_path(edge),
            src_node_pop=edge.source.name,
            dst_node_pop=edge.target.name,
            src_edge_pop_name=edge_pop_name,
            dst_edge_pop_name=edge_pop_name,
            src_mapping=id_mapping[edge.source.name],
            dst_mapping=id_mapping[edge.target.name],
        )

    # write virtual nodes based on virtual populations
    for population_name, ids in pop_used_source_node_ids.items():
        df = circuit.nodes[population_name].get(ids).reset_index(drop=True)
        nodes_path = os.path.join(output, population_name, "nodes.h5")
        Path(nodes_path).parent.mkdir(parents=True, exist_ok=True)
        new_node_files[population_name] = _save_sonata_nodes(nodes_path, df, population_name)

    return new_node_files, new_edges_files


def _update_config_with_new_paths(output, config, new_population_files, type_):
    """Update config file with the new paths

    Args:
        output: path to output
        config(dict): SONATA config
        new_population_files(dict): population -> path mapping of updated populations
        type_(str): 'nodes' or 'edges'
    """
    assert type_ in (
        "nodes",
        "edges",
    ), f'{type_} must be "nodes" or "edges"'
    output = str(output)

    config = copy.deepcopy(config)
    config["manifest"] = {"$BASE_DIR": "./"}

    def _strip_base_path(path):
        assert path.startswith(output), f"missing output path ({output}) in {path}"
        path = path[len(output) :]
        if path.startswith("/"):
            path = path[1:]
        return path

    new_population_files = copy.deepcopy(new_population_files)
    removed_populations = set()

    for el in config["networks"][type_]:
        for population in el["populations"]:
            if population not in new_population_files:
                removed_populations.add(population)
                continue

            population_path = new_population_files[population]
            del new_population_files[population]
            if population_path == DELETED_EMPTY_EDGES_FILE:
                removed_populations.add(population)
                continue

            updated_path = _strip_base_path(population_path)

            assert str(Path(output) / updated_path) == population_path, (
                "new population file was created unnecessarily "
                f"{Path(output) / updated_path} != {population_path}"
            )

            el[f"{type_}_file"] = os.path.join("$BASE_DIR", updated_path)

    assert all(
        f.startswith("external") for f in new_population_files
    ), f"Did not use all populations: {new_population_files}"

    for el in config["networks"][type_]:
        for population in list(el["populations"]):
            if population in removed_populations:
                del el["populations"][population]

    config["networks"][type_] = [el for el in config["networks"][type_] if el["populations"]]

    return config


def _update_node_sets(node_sets, id_mapping):
    """using the `id_mapping`, update impacted `node_sets`

    Note: impacted means they have a 'population', and they have 'node_ids' that changed
    """
    ret = {}
    for name, rule in node_sets.items():
        # Note: 'node_id' predicates without 'population' aren't copied, since it
        # doesn't makes sense to pick a node_id without specifying its population
        if isinstance(rule, dict) and "node_id" in rule:
            if "population" not in rule:
                L.warning("No population key in nodeset %s, ignoring", name)
                continue

            if rule["population"] not in id_mapping:
                continue

            mapping = id_mapping[rule["population"]]
            ret[name] = rule
            ret[name]["node_id"] = (
                mapping.loc[mapping.index.intersection(rule["node_id"])]
                .new_id.sort_values()
                .to_list()
            )
        else:
            ret[name] = rule

    return ret


def _write_mapping(output, id_mapping):
    """write the id mappings between the old and new populations for future analysis"""
    mapping = {}
    for population, df in id_mapping.items():
        mapping[population] = {
            "old_id": df.index.to_list(),
            "new_id": df.new_id.to_list(),
        }
    utils.dump_json(output / "id_mapping.json", mapping)


def split_subcircuit(output, node_set_name, circuit_config_path, do_virtual, create_external):
    """Split a single subcircuit out of circuit, based on nodeset

    Args:
        output(str): path where files will be written
        node_set_name(str): name of nodeset to extract
        circuit_config_path(str): path to circuit_config sonata file
        do_virtual(bool): whether to split out the virtual nodes that target the cells
            contained in the specified nodeset
        create_external(bool): whether to create new virtual populations of all the
            incoming connections
    """
    # pylint: disable=too-many-locals
    output = Path(output)

    circuit = bluepysnap.Circuit(circuit_config_path)

    node_pop_to_paths, edge_pop_to_paths = _gather_layout_from_networks(circuit.config["networks"])

    # TODO: remove backward compatibility with snap 1.0, when the dependency can be updated.
    #  In snap 2.0 it's possible to simplify:
    #    pop.get(pop.ids(node_set_name, raise_missing_property=False))
    #  with:
    #    pop.get(node_set_name, raise_missing_property=False)
    split_populations = {
        pop_name: pop.get(pop.ids(node_set_name, raise_missing_property=False))
        for pop_name, pop in circuit.nodes.items()
    }
    split_populations = {pop_name: df for pop_name, df in split_populations.items() if not df.empty}

    id_mapping = _get_node_id_mapping(split_populations)

    # TODO: should function `_write_subcircuit_biological`,
    # `_write_subcircuit_external`, `_write_subcircuit_virtual`
    # handle node updates and config updates?

    new_node_files, new_edge_files = _write_subcircuit_biological(
        output, circuit, node_pop_to_paths, edge_pop_to_paths, split_populations, id_mapping
    )

    if create_external:
        new_virtual_node_files, new_virtual_edge_files = _write_subcircuit_external(
            output, circuit, id_mapping
        )

        new_node_files.update(new_virtual_node_files)
        new_edge_files.update(new_virtual_edge_files)

    if do_virtual:
        new_virtual_node_files, new_virtual_edge_files = _write_subcircuit_virtual(
            output, circuit, edge_pop_to_paths, id_mapping
        )
        new_node_files.update(new_virtual_node_files)
        new_edge_files.update(new_virtual_edge_files)

    _write_mapping(output, id_mapping)

    config = copy.deepcopy(circuit.config)

    node_sets = _update_node_sets(utils.load_json(config["node_sets_file"]), id_mapping)
    utils.dump_json(output / "node_sets.json", node_sets)
    config["node_sets_file"] = "$BASE_DIR/node_sets.json"

    # update circuit_config
    config = _update_config_with_new_paths(output, config, new_node_files, type_="nodes")
    config = _update_config_with_new_paths(output, config, new_edge_files, type_="edges")
    utils.dump_json(output / "circuit_config.json", config)
