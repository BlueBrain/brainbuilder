''' Tools for working with SYN2 '''
import os
import logging

import click
import h5py
import numpy as np


L = logging.getLogger(__name__)

REQUIRED_PATH = click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True)

DEFAULT_PATH = '/synapses/default/'
PROPERTIES_PATH = os.path.join(DEFAULT_PATH, 'properties')

DEFAULT_CHECK_PROPERTIES = ','.join(['conductance',
                                     'connected_neurons_post',
                                     'connected_neurons_pre',
                                     'decay_time',
                                     'delay',
                                     'depression_time',
                                     'facilitation_time',
                                     'morpho_offset_segment_post',
                                     'morpho_section_id_post',
                                     'morpho_segment_id_post',
                                     'n_rrp_vesicles',
                                     'syn_type_id',
                                     'u_syn'])


@click.group()
def app():
    """ Tools for working with SYN2 """


def _create_appendable_dataset(h5_root, name, dtype, chunksize=1000):
    '''create an h5 appendable dataset at `h5_root` w/ `name`'''
    h5_root.create_dataset(name,
                           dtype=dtype,
                           chunks=(chunksize, ),
                           shape=(0, ),
                           maxshape=(None, ))


def _append_to_dataset(dset, values):
    '''append `values` to `dset`, which should be an appendable dataset'''
    dset.resize(dset.shape[0] + len(values), axis=0)
    dset[-len(values):] = values


def _get_property_dtypes(path):
    '''get the dtypes of all the properties in `path`

    Returns:
        dict: property_name -> dtype
    '''
    ret = {}
    with h5py.File(path, 'r') as h5:
        prop = h5[PROPERTIES_PATH]
        for p in prop.keys():
            ret[p] = prop[p].dtype
    return ret


def _concat_h5(output, sources):
    '''create `output` from `sources`, uses first source as source of truth for properties'''
    with h5py.File(output, 'w') as h5o:
        output_properties = h5o.create_group(PROPERTIES_PATH)

        properties = _get_property_dtypes(sources[0])
        for name, dtype in properties.items():
            _create_appendable_dataset(output_properties, name, dtype)

        for source in sources:
            L.debug('Opening source: %s', source)
            with h5py.File(source, 'r') as h5:
                prop = h5[PROPERTIES_PATH]
                for name in properties:
                    L.debug('Copying property[%s] %s', source, name)
                    _append_to_dataset(output_properties[name], prop[name])


def _check_syn2_invariants(path, population, expected_properties, afferent_index=True):
    '''check known syn2 invariants'''
    # pylint: disable=too-many-locals
    assert afferent_index, 'checking for efferent indexing not supported'

    def _h5_path(path):
        assert path in h5, 'missing "%s" dataset' % path
        return h5[path]

    with h5py.File(path, 'r') as h5:
        _h5_path('/synapses')
        population_path = '/synapses/%s' % population
        population = _h5_path(population_path)
        props = _h5_path(population_path + '/properties')

        for expected in expected_properties:
            assert expected in props, 'missing "%s" dataset in properties dataset' % expected

        primary_sort = 'connected_neurons_%s' % ('post' if afferent_index else 'pre')
        secondary_sort = 'connected_neurons_%s' % ('pre' if afferent_index else 'post')

        primary_gids = props[primary_sort]
        diff = primary_gids[1:] - primary_gids[:-1]
        assert np.all(diff >= 0), '%s must be sorted' % primary_sort

        secondary_gids = props[secondary_sort]
        start = 0
        groups = np.nonzero(diff)[0]
        for end in groups:
            end += 1
            assert np.all(secondary_gids[start:end - 1] <= secondary_gids[start + 1:end]), \
                '%s must be sorted' % secondary_sort
            start = end

        assert 'indexes' in population, 'missing "/synapses/default/indexes" dataset'

        _h5_path('/synapses/default/indexes/connected_neurons_post')
        _h5_path('/synapses/default/indexes/connected_neurons_pre')

        if afferent_index:
            start = 0
            idx_post = population['indexes']['connected_neurons_post']
            for end in groups:
                end += 1
                gid = primary_gids[start]
                ranges = idx_post['neuron_id_to_range'][gid]
                assert len(ranges) == 2, 'if properly sorted, should only have 2 positions in range'
                assert idx_post['range_to_synapse_id'][ranges[0]][0] == start, \
                    'start index position is wrong for gid == %s' % gid
                assert idx_post['range_to_synapse_id'][ranges[0]][1] == end, \
                    'end index position is wrong for gid == %s' % gid
                start = end


@app.command()
@click.option('-o', '--output', required=True,
              help='Path to output HDF5')
@click.argument('sources', nargs=-1, type=REQUIRED_PATH)
def concat(output, sources):
    '''concatenate multiple syn2 files'''
    _concat_h5(output, sources)
    click.echo(click.style('Indices not created; please use syn-tool for that', fg='red'))


@app.command()
@click.option("--population", default="default", show_default=True,
              help="Population name")
@click.option("--properties", default=DEFAULT_CHECK_PROPERTIES, show_default=True,
              help="Properties concated")
@click.argument('test_file', type=REQUIRED_PATH)
def check(population, properties, test_file):
    '''check syn2 invariants'''
    properties = [p.strip() for p in properties.split(',')]
    _check_syn2_invariants(test_file, population, properties)
    click.echo(click.style('syn2 file appears to pass the invariants', fg='green'))
