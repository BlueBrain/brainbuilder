'''
          Copyright Adrien Devresse - 2016
 Distributed under the Boost Software License, Version 1.0.
    (See accompanying file LICENSE_1_0.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
'''
# pylint: disable=missing-docstring

import logging
import os
import sys
import time

from contextlib import contextmanager

import click
import h5py
import numpy as np

from tqdm import tqdm
from brainbuilder.app._utils import REQUIRED_PATH

L = logging.getLogger(__name__)


@click.group()
def app():
    """ Tools for working with NRN files """


def progress_print(pstr):
    if os.sys.stdout.isatty():
        sys.stdout.write("\r" + pstr)
    else:
        sys.stdout.write(pstr + "\n")
    sys.stdout.flush()


def progress_finalize():
    if os.sys.stdout.isatty():
        sys.stdout.write("\n")
    sys.stdout.flush()


def check_individual_file(nrn_file):
    first_file = f"{nrn_file}.0"
    if not os.path.exists(first_file):
        print((f">WARNING {first_file} ... does not exist ... SKIP"))
        return False

    if os.path.exists(nrn_file):
        print((f">WARNING {nrn_file} ... already exist ... SKIP"))
        return False

    return True


def list_nrnfiles(nrn_dir):
    name_files = ["nrn_positions.h5", "nrn_positions_efferent.h5",
                  "nrn.h5", "nrn_summary.h5", "nrn_extra.h5", "nrn_efferent.h5"]

    list_files = [''.join([nrn_dir, os.sep, f]) for f in name_files]
    return filter(check_individual_file, list_files)


def get_nrnfiles(nrn_dir, only):
    if only != "":
        return [''.join([nrn_dir, os.sep, only])]
    return list_nrnfiles(nrn_dir)


def count_subfiles(filename):
    count = 1
    while os.path.exists(f"{filename}.{count}"):
        count += 1
    return count


def get_all_dataset(filename, file_number, dset):
    n_filename = f"{filename}.{file_number}"
    with h5py.File(n_filename, 'r') as f:
        all_keys = f.keys()
        dset[file_number] = [key for key in all_keys if key[0] == "a"]
        return len(all_keys)


def finalize_metadata(fdesc, total_files):
    fdesc["/info"] = 0
    attrs = fdesc["/info"].attrs
    attrs.create("numberOfFiles", total_files)
    attrs.create("version", 5)


def create_merged_file(filename, link=False):
    print(">")
    print(f">> start merge for {filename}")
    total_files = count_subfiles(filename)
    print(f">> {total_files} files to merge")
    dset = {}
    n = 0
    t1 = time.time()
    progress_print(">>")
    for i in range(0, total_files):
        n += get_all_dataset(filename, i, dset)
        progress_print(f">> got all keys for file {filename}.{i}")
    progress_finalize()
    print(f">> complete listing done in {time.time() - t1:f}s")
    print(f">> total of {n} datasets to merge")
    t1 = time.time()
    progress_print(">>")
    with h5py.File(filename, 'w') as merged:
        for i in range(0, total_files):
            chunk_filename = f"{filename}.{i}"
            if link:
                progress_print(f">> create external references file {chunk_filename}")
                for k in dset[i]:
                    d_name = f"/{k}"
                    merged[d_name] = h5py.ExternalLink(chunk_filename, d_name)
            else:
                progress_print(f">> copy over data from {chunk_filename}")
                with h5py.File(chunk_filename, 'r') as chunk:
                    for k in dset[i]:
                        d_name = f"/{k}"
                        merged[d_name] = chunk[d_name][:]
        finalize_metadata(merged, (total_files if link else 1))
    progress_finalize()
    print(f">> complete merging done in {time.time() - t1:f}s")


@contextmanager
def cd(dirpath):
    old_dirpath = os.getcwd()
    new_dirpath = os.path.abspath(dirpath)
    os.chdir(new_dirpath)
    try:
        yield new_dirpath
    finally:
        os.chdir(old_dirpath)


@app.command()
@click.argument("nrn_dir")
@click.option(
    "--only", help="merge only the specified file (e.g --only=nrn_positions.h5)", default=""
)
@click.option(
    "--link", is_flag=True, help="make symbolic links instead of copying datasets"
)
def merge(nrn_dir, only, link):
    """
    Merge utility tool for nrn.h5 Blue Brain synapse file format.

    This tool creates a single file presenting the content of
    a group of nrn_*.h5.N generated by the Functionalizer
    """
    with cd(nrn_dir):
        # chdir to NRN folder to make NRN links relative
        for filename in get_nrnfiles('.', only):
            create_merged_file(filename, link=link)


SYN2_NAME_2_NRN_COLUMN_MAP = {
    'delay': 1,
    'morpho_section_id_post': 2,
    'morpho_segment_id_post': 3,
    'morpho_offset_segment_post': 4,
    'morpho_section_id_pre': 5,
    'morpho_segment_id_pre': 6,
    'morpho_offset_segment_pre': 7,
    'conductance': 8,
    'u_syn': 9,
    'depression_time': 10,
    'facilitation_time': 11,
    'decay_time': 12,
    'syn_type_id': 13,
    'n_rrp_vesicles': 17,
}

# unfortunately, we have SONATA files that have syn2 naming, so we
# re-use the syn2 naming, and add other potential mappings.  If two
# data sets exist in the same file, they better contain the same data...
SONATA_NAME_2_NRN_COLUMN_MAP = SYN2_NAME_2_NRN_COLUMN_MAP.copy()
SONATA_NAME_2_NRN_COLUMN_MAP.update(
    {'afferent_section_id': 2,
     'afferent_segment_id': 3,
     #  'afferent_section_type':
     #  'afferent_section_pos':
     'afferent_segment_offset': 4,

     'efferent_section_id': 5,
     # 'efferent_section_pos':
     # 'efferent_section_type':
     'efferent_segment_id': 6,
     'efferent_segment_offset': 7,
     })


def _make_nrn_h5_properties(mapping, properties, range_):
    '''create the 19 column dataset expected in the nrn.h5 file
    '''
    dst = np.full((len(range_), 19), fill_value=-1, dtype=np.float64)

    for prop in properties.keys():
        if prop in mapping:
            dst[:, mapping[prop]] = properties[prop][range_]
    return dst


def _write_nrn(output, index1, index2, mapping, properties, pre_synaptic_ids):
    '''write nrn.h5 to `output` directory

    Args:
        output(str): output directory
        index1(np.array-like): afferent neuron_to_id_range
        index2(np.array-like): afferent range_to_synapse_id
        mapping(dict): mapping of property names to column numbers
        properties(h5group): where the property names are stored
        pre_synaptic_ids(np.array-like): the presynaptic ids
    '''
    missing = set(mapping) - set(properties)
    if missing:
        L.warning('Properties %s are missing in h5', missing)

    missing = set(properties) - set(mapping)
    if missing:
        L.warning('Extra unknown properties %s in h5', missing)

    with h5py.File(os.path.join(output, 'nrn.h5'), 'w') as dst:
        dst['info'] = []
        dst['info'].attrs['version'] = [5]
        for gid, range1 in tqdm(enumerate(index1), total=len(index1)):
            r1_start, r1_end = range1.astype(int)

            if r1_start == r1_end:  # empty range
                continue

            if r1_start > r1_end:
                raise RuntimeError(f'Start index ({r1_start}) > than end index ({r1_end})')

            assert r1_start == r1_end - 1, 'With postsynaptic sorting, should only have 1 row'

            range2 = range(index2[r1_start][0], index2[r1_end - 1][1])

            columns = _make_nrn_h5_properties(mapping, properties, range2)

            # NRN is 1 based for pre and post gids
            columns[:, 0] = pre_synaptic_ids[range2] + 1
            assert np.all(columns[:-1, 0] <= columns[1:, 0]), 'Must be postsynaptically sorted'
            dst[f'a{gid + 1}'] = columns


@app.command()
@click.argument("syn2", type=REQUIRED_PATH)
@click.option("-o", "--output", help="Path to output NRN folder", required=True)
def from_syn2(syn2, output):
    """Convert SYN2 file to partial nrn.h5"""
    with h5py.File(syn2, 'r') as h5f:
        assert len(h5f['synapses']) == 1
        src = next(iter(h5f['synapses'].values()))
        properties = src['properties']
        index1 = src['indexes/connected_neurons_post/neuron_id_to_range']
        index2 = src['indexes/connected_neurons_post/range_to_synapse_id']
        pre_synaptic_ids = properties['connected_neurons_pre']
        _write_nrn(
            output, index1, index2, SYN2_NAME_2_NRN_COLUMN_MAP, properties, pre_synaptic_ids)


@app.command()
@click.argument("sonata", type=REQUIRED_PATH)
@click.option("-o", "--output", help="Path to output NRN folder", required=True)
def from_sonata(sonata, output):
    """Convert SONATA file to partial nrn.h5"""
    with h5py.File(sonata, 'r') as h5f:
        assert len(h5f['edges']) == 1
        src = next(iter(h5f['edges'].values()))
        properties = src['0']
        index1 = src['indices/target_to_source/node_id_to_ranges']
        index2 = src['indices/target_to_source/range_to_edge_id']
        pre_synaptic_ids = src['source_node_id']
        _write_nrn(
            output, index1, index2, SONATA_NAME_2_NRN_COLUMN_MAP, properties, pre_synaptic_ids)
