# SPDX-License-Identifier: Apache-2.0
"""Collection of functions to curate/edit SONATA circuits."""

import logging

import h5py
import morphio
import numpy as np
import voxcell
from bluepysnap.schemas import schemas

L = logging.getLogger(__name__)


def get_population_names(h5_file):
    """Gets the list of population names of SONATA file.

    Args:
        h5_file (str/Path): SONATA file. Can be edges or nodes file.

    Returns:
        list: list of population names
    """
    with h5py.File(h5_file, "r") as h5f:
        assert ("nodes" in h5f) ^ ("edges" in h5f), f'"edges" or "nodes" must be presented in {h5f}'
        return list(h5f["nodes"]) if "nodes" in h5f else list(h5f["edges"])


def get_population_name(h5_file, population_name=None):
    """Gets population name from SONATA file.

    If population_name is provided then it validates that it is presented in h5_file.

    Args:
        h5_file (str/Path): SONATA file. Can be edges or nodes file.
        population_name (str|None): population name

    Returns:
        str: population name
    """
    names = get_population_names(h5_file)
    if population_name is None:
        assert (
            len(names) == 1
        ), f"{h5_file} must have a single population only, if one is not specified"
        return names[0]
    elif population_name not in names:
        raise ValueError(f'"{population_name}" population does not exist in {h5_file}')
    return population_name


def _rename_population(file, root, new_name, old_name=None):
    """Renames population of SONATA file.

    Args:
        file (str/Path): SONATA file. Can be edges or nodes file.
        root (str): either 'edges' or 'nodes'.
        new_name (str): new population name
        old_name (str): old population name
    """
    with h5py.File(file, "r+") as h5f:
        population_names = list(h5f[root])
        if len(population_names) == 0:
            raise ValueError(f"No populations in {file}")
        if old_name is None:
            if len(population_names) > 1:
                raise ValueError(
                    f"Multiple populations:{population_names} in {file},"
                    f"specify the exact one to rename."
                )
            old_name = population_names[0]
        elif old_name not in population_names:
            raise ValueError(f"No {old_name} population in {file}")
        h5f.move(f"/{root}/{old_name}", f"/{root}/{new_name}")


def rename_node_population(nodes_file, new_name, old_name=None):
    """Renames population of SONATA nodes file.

    Args:
        nodes_file (str/Path): SONATA nodes file
        new_name (str): new population name
        old_name (str): old population name
    """
    L.debug("renaming node population: %s to %s", old_name, new_name)
    _rename_population(nodes_file, "nodes", new_name, old_name)


def rename_edge_population(edges_file, new_name, old_name=None):
    """Renames population of SONATA edges file.

    Args:
        edges_file (str/Path): SONATA nodes file
        new_name (str): new population name
        old_name (str): old population name
    """
    L.debug("renaming edge population: %s to %s", old_name, new_name)
    _rename_population(edges_file, "edges", new_name, old_name)


def add_edge_type_id(edges_file, population_name):
    """Adds 'edge_type_id' field to edge population if it does not exist.

    Args:
        edges_file (str/Path): SONATA edges file
        population_name (str): edge population name
    """
    with h5py.File(edges_file, "r+") as h5f:
        group = h5f[f"edges/{population_name}"]
        var = list(group["0"])[0]
        size = group["0"][f"{var}"].size
        if "edge_type_id" not in group:
            group.create_dataset(
                "edge_type_id", data=np.full((size,), fill_value=-1, dtype=np.int32)
            )


def set_group_attribute(file, root, population, group, attr_name, attr_value, overwrite=False):
    """Sets group attribute. Either for nodes or edges group.

    Args:
        file (str/Path): SONATA nodes or edges file
        root (str): either 'edges' or 'nodes'.
        population (str): population name
        group (str): group name
        attr_name (str): attribute name
        attr_value (any): attribute default value. If it is of string type then '@library' group
            will be used to store it.
    """
    # pylint: disable=too-many-locals
    group_path = f"{root}/{population}/{group}"
    L.debug('_set_group_attribute of "%s" %s -> (%s)', group_path, attr_name, attr_value)
    with h5py.File(file, "r+") as h5f:
        assert group_path in h5f, f"no such path {group_path} in {file}"
        group_h5 = h5f[group_path]
        any_ds = None
        for key in group_h5:
            if isinstance(group_h5[key], h5py.Dataset):
                any_ds = group_h5[key]
        count = len(any_ds)
        if isinstance(attr_value, str):
            if "@library" not in group_h5:
                group_h5.create_group("@library")
            lib = group_h5["@library"]
            if attr_name in lib and overwrite:
                del lib[attr_name]
            str_dt = h5py.special_dtype(vlen=str)
            lib.create_dataset(
                attr_name,
                data=np.array([attr_value], dtype=object),
                dtype=str_dt,
            )
            attr_value = 0
        # TODO: should we check the number of unique values before creating an @library style
        #  lookup?
        if attr_name in group_h5 and overwrite:
            del group_h5[attr_name]
        # to correctly detect dtype otherwise always float array is created
        dt = np.array([attr_value]).dtype
        group_h5.create_dataset(attr_name, fillvalue=attr_value, shape=(count,), dtype=dt)


def rewire_edge_population(
    edges_file,
    source_nodes_file,
    target_nodes_file,
    syn_type,
    edge_population_name=None,
    source_nodes_population_name=None,
    target_nodes_population_name=None,
):
    """Renames edge population according to the rule:

    New name = {source_nodes_population_name}__{target_nodes_population_name}__{syn_type}.
    Names of 'source_node_id' and 'source_node_id' of edge population change correspondingly. For
    details see `Circuit Documentation
    <https://bbpteam.epfl.ch/documentation/projects/circuit-documentation/latest/sonata.html#populations>`__

    Args:
        edges_file (str/Path): SONATA edges file
        source_nodes_file (str/Path): source nodes file to bind to
        target_nodes_file (str/Path): target nodes file to bind to
        syn_type (str): synapse type of edges
        edge_population_name (str/None): edges population name. If not provided then the first
            population is taken.
        source_nodes_population_name (str/None): source nodes population name. If not provided then
            the first population is taken.
        target_nodes_population_name (str/None): target nodes population name. If not provided then
            the first population is taken.
    """
    source_nodes_population_name = get_population_name(
        source_nodes_file, source_nodes_population_name
    )
    target_nodes_population_name = get_population_name(
        target_nodes_file, target_nodes_population_name
    )
    edge_population_name = get_population_name(edges_file, edge_population_name)
    with h5py.File(edges_file, "r+") as h5f:
        new_name = (
            f"/edges/{source_nodes_population_name}__{target_nodes_population_name}__{syn_type}"
        )
        L.debug("rewire_edge_population: %s", new_name)
        h5f.move(f"/edges/{edge_population_name}", new_name)

        h5f[new_name]["source_node_id"].attrs["node_population"] = source_nodes_population_name
        h5f[new_name]["target_node_id"].attrs["node_population"] = target_nodes_population_name
        return new_name


def _create_source_nodes(population_name, size, output_dir):
    """Creates source node population of virtual nodes.

    Args:
        population_name (str): nodes population name
        size (int): size of nodes
        output_dir (Path): directory where to save them

    Returns:
        Path: filepath to created nodes
    """
    L.debug("_create_source_nodes: %s(%d)", population_name, size)
    cells = voxcell.CellCollection()
    cells.properties["model_type"] = ["virtual"] * size
    # '' stands for NULL value in SONATA https://github.com/AllenInstitute/sonata/issues/122
    cells.properties["model_template"] = [""] * size
    cells.population_name = population_name
    output = output_dir / f"nodes_{population_name}.h5"
    cells.save(output)
    return output


def correct_source_nodes_offset(edges_file, edge_population_name, offset):
    """Corrects source nodes index offset of edges population.

    Args:
        edges_file (str/Path): SONATA edges file
        edge_population_name (str): edges population name
        offset (int): correction offset
    """
    L.debug("correct_source_nodes_offset: %s(%d)", edge_population_name, offset)
    with h5py.File(edges_file, "r+") as h5f:
        population = h5f["edges"][edge_population_name]
        ids = population["source_node_id"][:] - offset
        source_nodes_name = population["source_node_id"].attrs["node_population"]
        del population["source_node_id"]
        population["source_node_id"] = ids
        population["source_node_id"].attrs["node_population"] = source_nodes_name

        # need to re-number the virtual edge ids
        idx = population["indices/source_to_target/node_id_to_ranges"][offset:]
        del population["indices/source_to_target/node_id_to_ranges"]
        population["indices/source_to_target/node_id_to_ranges"] = idx


def get_source_nodes_range(edges_file, edge_population_name="default"):
    """Given edge population, gets size of its source node population.

    Args:
        edges_file (str/Path): SONATA edges file
        edge_population_name (str): edge population name

    Returns:
        (int, int, ): start index, end index
    """
    with h5py.File(edges_file, "r") as h5f:
        source_node_id = h5f[f"/edges/{edge_population_name}/source_node_id"][:]
        start = int(np.min(source_node_id))
        end = int(np.max(source_node_id))

    return start, end


def create_projection_source_nodes(
    projection_file,
    source_nodes_dir,
    source_nodes_population_name,
    fix_offset,
):
    """Create source nodes file for projection file.

    Args:
        projection_file (Path): filepath to projection in SONATA format
        source_nodes_dir (Path): directory where to store source nodes file
        source_nodes_population_name: source nodes population name
        fix_offset(bool): projections node IDs often don't start from 0; if this is
        true the offset will be removed (ie: the node IDs will start from 0)

    Returns:
        Path: path to created source nodes file.
    """
    names = get_population_names(projection_file)
    assert len(names) == 1, f"{projection_file} has multiple populations but must have only one"
    proj_population_name = next(iter(names))
    start, end = get_source_nodes_range(projection_file, proj_population_name)

    if fix_offset:
        size = end - start + 1
    else:
        if start != 0:
            L.warning(
                "%s:%s's node_ids don't start from 0 "
                "use correct_source_nodes_offset() to fix if required",
                projection_file,
                proj_population_name,
            )
        size = end + 1

    source_nodes_file = _create_source_nodes(source_nodes_population_name, size, source_nodes_dir)

    return source_nodes_file


def merge_h5_files(files, root, output_file):
    """Merges multiple h5 files into one h5 file.

    Args:
        files (dict/list/tuple): dictionary where key is a h5 filepath, and value is a list of
            population names in that h5 filepath to merge. If list or tuple of h5 filepaths is
            provided then all populations within all h5 files are merged.
        root (str): 'nodes' or 'edges' string
        output_file (Path/str): output filepath
    """
    if isinstance(files, (list, tuple)):
        files = {file: None for file in files}
    with h5py.File(output_file, "w") as out_h5f:
        out_h5f.create_group(f"{root}")
        for file, population_names in files.items():
            with h5py.File(file, "r") as h5f:
                if population_names is None:
                    population_names = list(h5f[root])
                for name in population_names:
                    h5f.copy(f"{root}/{name}/", out_h5f[f"{root}"])


def _has_sonata_ordering(morph):
    """check if morph has correct ordering

    https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md
    The soma is always section 0.
    The rest of the sections are first grouped by section type in this order:
        1 = axon, 2 = basal and 3 = apical.
    """
    REQUIRED_ORDER = {
        morphio.SectionType.axon: 0,
        morphio.SectionType.basal_dendrite: 1,
        morphio.SectionType.apical_dendrite: 2,
    }
    section_types = [s.type for s in morph.root_sections]
    ordered = sorted(section_types, key=lambda s: REQUIRED_ORDER[s])
    return ordered == section_types


def _has_unifurcations(morph):
    """check if `morph` has unifurcations"""
    return any(len(section.children) == 1 for section in morph.iter())


def check_morphology_invariants(h5_morph_dir, morph_names):
    """check if morphologies follow the SONATA spec and have no unifurcations"""

    incorrect_ordering = set()
    have_unifurcations = set()
    for name in morph_names:
        morph = morphio.Morphology(h5_morph_dir / (name + ".h5"))
        if not _has_sonata_ordering(morph):
            incorrect_ordering.add(name)

        if _has_unifurcations(morph):
            have_unifurcations.add(name)

    return incorrect_ordering, have_unifurcations


def _update_dtype(parent_h5, name, target_dtype):
    """Update dtype of the `parent_h5[name]` h5py dataset to `target_dtype`."""
    h5 = parent_h5[name]
    attrs = dict(h5.attrs)
    L.debug("convert_dtype: %s: %s -> %s", h5.name, h5.dtype, target_dtype)
    new = np.asarray(h5[:], dtype=target_dtype)
    del parent_h5[name]
    parent_h5[name] = new

    for k, v in attrs.items():
        parent_h5[name].attrs[k] = v

    return (parent_h5[name].name, target_dtype)


def update_node_dtypes(h5_file, population_name, population_type):
    """Update the datatypes of the attributes within a node population to the SONATA spec.

    Args:
        h5_file(path): to h5 file containing nodes
        population_name(str): name of the population to modify
        population_type(str): type (ex: biophysical) of the node population

    Returns:
        dict of names -> dtype of converted attributes
    """
    converted = []
    property_types, dynamics_params = schemas.nodes_schema_types(population_type)
    with h5py.File(h5_file, "r+") as h5f:
        group = h5f["nodes"][population_name]["0"]
        library = set()
        if "@library" in group:
            library = set(group["@library"])

        for attribute_name in group.keys():
            if attribute_name in ("@library", "dynamics_params"):
                continue
            if attribute_name not in property_types:
                L.info("Unknown property '%s', leaving alone", attribute_name)
                continue

            target_dtype = property_types[attribute_name]
            if target_dtype is str:
                continue

            if attribute_name in library:
                parent = group["@library"]
            else:
                parent = group

            if target_dtype != parent[attribute_name].dtype:
                converted.append(_update_dtype(parent, attribute_name, target_dtype))

        if "dynamics_params" in group:
            parent = group["dynamics_params"]
            for param in parent.keys():
                if param not in dynamics_params:
                    continue

                target_dtype = dynamics_params[param]
                if target_dtype != parent[param].dtype:
                    converted.append(_update_dtype(parent, param, target_dtype))

    return dict(converted)


def update_edge_dtypes(h5_file, population_name, population_type, virtual):
    """Update the datatypes of the attributes within an edge population to the SONATA spec.

    Args:
        h5_file(path): to h5 file containing nodes
        population_name(str): name of the population to modify
        population_type(str): type (ex: biophysical) of the node population
        virtual(bool): Whether the population is virtual

    Returns:
        dict of names -> dtype of converted attributes
    """
    property_types = schemas.edges_schema_types(population_type, virtual=virtual)
    converted = []
    with h5py.File(h5_file, "r+") as h5f:
        group = h5f["edges"][population_name]

        for name, expected in (
            ("source_node_id", np.uint64),
            ("target_node_id", np.uint64),
            ("edge_type_id", np.int64),
        ):
            if group[name].dtype != expected:
                converted.append(_update_dtype(group, name, expected))

        group = group["0"]

        for attribute_name in group.keys():
            if attribute_name not in property_types:
                L.info("Unknown property '%s', leaving alone", attribute_name)
                continue

            target_dtype = property_types[attribute_name]
            if target_dtype is str:
                continue

            if target_dtype != group[attribute_name].dtype:
                converted.append(_update_dtype(group, attribute_name, target_dtype))

    return dict(converted)
