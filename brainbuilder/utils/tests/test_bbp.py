import os
from nose.tools import eq_

import numpy as np
import pandas as pd
from numpy.testing import assert_equal
from pandas.util.testing import assert_frame_equal
from voxcell import core
from brainbuilder.utils import bbp

mhd = {'voxel_dimensions': (25, 25, 25)}


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def test_map_regions_to_layers_unknown_area():
    h = core.Hierarchy({"id": 997, "name": "mybrain", "children": []})
    region_layers_map = bbp.map_regions_to_layers(h,
                                                  'Primary somatosensory area, barrel field')

    eq_(region_layers_map, {})


def test_map_regions_to_layers_complex():
    hierarchy = core.Hierarchy.load(os.path.join(DATA_PATH, 'hierarchy.json'))

    region_layers_map = bbp.map_regions_to_layers(hierarchy,
                                                  'Primary somatosensory area, barrel field')

    eq_(region_layers_map,
        {1062: (6,), 201: (2, 3), 1070: (5,), 1038: (6,), 981: (1,), 1047: (4,)})


def test_load_recipe_as_layer_distributions_complex():
    layer_dists = bbp.load_recipe(
        os.path.join(DATA_PATH, 'builderRecipeAllPathways.xml'))

    eq_(list(layer_dists.keys()),
        ['layer', 'mtype', 'etype', 'mClass', 'synapse_class', 'percentage'])

    eq_(len(layer_dists.values), 9)


def test_load_recipe_density_0():
    annotation_raw = np.array([1, 2])
    density = bbp.load_recipe_density(
        os.path.join(DATA_PATH, 'builderRecipeAllPathways.xml'),
        core.VoxelData(annotation_raw, voxel_dimensions=(25,)),
        {1: (1,), 2: (2,)})

    assert_equal(density.raw, np.array([0.1, 0.9], dtype=np.float32))


def test_load_recipe_density_unknown_layer_0():
    annotation_raw = np.array([1, 2, 777])
    density = bbp.load_recipe_density(
        os.path.join(DATA_PATH, 'builderRecipeAllPathways.xml'),
        core.VoxelData(annotation_raw, voxel_dimensions=(25,)),
        {1: (1,), 2: (2,), 999: (888,)})

    assert_equal(density.raw, np.array([0.1, 0.9, 0.], dtype=np.float32))


def test_load_recipe_density_no_voxels():
    annotation_raw = np.array([1, 1])
    density = bbp.load_recipe_density(
        os.path.join(DATA_PATH, 'builderRecipeAllPathways.xml'),
        core.VoxelData(annotation_raw, voxel_dimensions=(25,)),
        {1: (1,), 2: (2,)})

    assert_equal(density.raw, np.array([0.5, 0.5], dtype=np.float32))


def test_transform_into_spatial_distribution():
    annotation_raw = np.ones(shape=(3, 3, 3))
    annotation_raw[1, 1, 1] = 2

    layer_distributions = pd.DataFrame([
        {'layer': 21, 'percentage': 0.5},
        {'layer': 22, 'percentage': 0.5},
        {'layer': 23, 'percentage': 0.5}
    ])

    region_layers_map = {
        1: (21, 22),
        2: (23,)
    }

    sdist = bbp.transform_recipe_into_spatial_distribution(
        core.VoxelData(annotation_raw, **mhd),
        layer_distributions,
        region_layers_map)

    assert_frame_equal(sdist.distributions,
                       pd.DataFrame({1: [0.5,  0.5, 0.],
                                     2: [0., 0., 1.]}))

    assert_frame_equal(sdist.traits, layer_distributions)

    expected_field = np.ones(shape=(3, 3, 3))
    expected_field[1, 1, 1] = 2
    assert_equal(sdist.field.raw, expected_field)


def test_load_neurondb_v4():
    morphs = bbp.load_neurondb_v4(os.path.join(DATA_PATH, 'neuronDBv4.dat'))
    eq_(len(morphs), 6)


def test_transform_neurondb_into_spatial_distribution_empty():
    sd = bbp.transform_neurondb_into_spatial_distribution(
        core.VoxelData(np.ones(shape=(3, 3), dtype=np.int), **mhd),
        pd.DataFrame(), {},
        np.zeros(shape=(3, 3)),
        percentile=0.0)

    assert sd.traits.empty
    assert sd.distributions.empty
    assert_equal(sd.field.raw, np.ones(shape=(3, 3)) * -1)


def test_transform_neurondb_into_spatial_distribution():

    annotation = np.ones(shape=(3, 3), dtype=np.int) * 10
    annotation[1, 1] = 20

    region_layers_map = {
        10: (1,),
        20: (1, 2, 3)
    }

    neurondb = pd.DataFrame([
        {'name': 'a', 'layer': 1, 'etype': 1, 'mtype': 1, 'placement_hints': (1,)},
        {'name': 'b', 'layer': 1, 'etype': 2, 'mtype': 2, 'placement_hints': (1,)},
        {'name': 'c', 'layer': 2, 'etype': 3, 'mtype': 3, 'placement_hints': (1,)},
        {'name': 'd', 'layer': 3, 'etype': 4, 'mtype': 4, 'placement_hints': (1,)},
    ])

    sd = bbp.transform_neurondb_into_spatial_distribution(
        core.VoxelData(annotation, **mhd), neurondb, region_layers_map,
        np.zeros(shape=(3, 3)),
        percentile=0.0)

    assert_frame_equal(sd.traits, neurondb)

    print sd.distributions.values

    expected = pd.DataFrame({0: [0.5, 0.5, 0, 0],
                             1: [0.25, 0.25, 0.25, 0.25]})

    print expected.values
    assert_frame_equal(sd.distributions, expected)

    expected_field = np.zeros_like(annotation)
    expected_field[1, 1] = 1
    assert_equal(sd.field.raw, expected_field)


def test_get_region_distributions_from_placement_hints_0():

    neurondb = pd.DataFrame([
        {'name': 'a', 'layer': 1, 'etype': 1, 'mtype': 1, 'placement_hints': (1, 2)},     # 1 1 1 2 2 2
        {'name': 'b', 'layer': 1, 'etype': 2, 'mtype': 2, 'placement_hints': (2, 1)},     # 2 2 2 1 1 1
        {'name': 'c', 'layer': 2, 'etype': 3, 'mtype': 3, 'placement_hints': (1,)},       # 1 1 1 1 1 1
        {'name': 'd', 'layer': 2, 'etype': 4, 'mtype': 4, 'placement_hints': (1, 2, 1)},  # 1 1 2 2 1 1
    ])

    region_layers_map = {
        0: (1, 2)
    }

    res = bbp.get_region_distributions_from_placement_hints(neurondb, region_layers_map, 0.0)

    eq_(res.keys(), [(0,)])

    expected = pd.DataFrame({
        0: [2., 1., 1., 1.],
        1: [2., 1., 1., 1.],
        2: [2., 1., 1., 2.],
        3: [1., 2., 1., 2.],
        4: [1., 2., 1., 1.],
        5: [1., 2., 1., 1.],
    })
    expected /= expected.sum()
    assert_frame_equal(res[(0,)], expected)


def test_get_region_distributions_from_placement_hints_unknown_region():
    # we have data about neurons in layers that we don't know how to map to atlas regions
    # these should be just ignored

    neurondb = pd.DataFrame([
        {'name': 'a', 'layer': 0, 'etype': 1, 'mtype': 1, 'placement_hints': (1,)},
        {'name': 'b', 'layer': 999, 'etype': 2, 'mtype': 2, 'placement_hints': (1,)}
    ])

    region_layers_map = {
        0: (0,)
    }

    res = bbp.get_region_distributions_from_placement_hints(neurondb, region_layers_map, 0.0)
    eq_(res.keys(), [(0,)])
    assert_frame_equal(res[(0,)], pd.DataFrame({0: [1.]}))


def test_transform_neurondb_into_spatial_distribution_with_placement_hints_0():

    neurondb = pd.DataFrame([
        {'name': 'a', 'layer': 1, 'etype': 1, 'mtype': 1, 'placement_hints': (1, 2)},     # 1 1 1 2 2 2
        {'name': 'b', 'layer': 1, 'etype': 2, 'mtype': 2, 'placement_hints': (2, 1)},     # 2 2 2 1 1 1
        {'name': 'c', 'layer': 2, 'etype': 3, 'mtype': 3, 'placement_hints': (1,)},       # 1 1 1 1 1 1
        {'name': 'd', 'layer': 2, 'etype': 4, 'mtype': 4, 'placement_hints': (1, 2, 1)},  # 1 1 2 2 1 1
    ])

    region_layers_map = {
        1: (1, 2)
    }

    annotation = np.array([0] + [1] * 6)

    sd = bbp.transform_neurondb_into_spatial_distribution(
        core.VoxelData(annotation, **mhd), neurondb, region_layers_map,
        np.arange(len(annotation)),
        percentile=0.0)

    assert_frame_equal(sd.traits, neurondb)

    expected = pd.DataFrame({
        0: [2., 1., 1., 1.],
        1: [2., 1., 1., 1.],
        2: [2., 1., 1., 2.],
        3: [1., 2., 1., 2.],
        4: [1., 2., 1., 1.],
        5: [1., 2., 1., 1.],
    })
    expected /= expected.sum()
    assert_frame_equal(sd.distributions, expected)

    assert_equal(sd.field.raw, np.array([-1, 0, 1, 2, 3, 4, 5]))


def test_transform_neurondb_into_spatial_distribution_with_placement_hints_1():

    neurondb = pd.DataFrame([
        {'name': 'a', 'layer': 1, 'etype': 1, 'mtype': 1, 'placement_hints': (1, 2)},     # 1 1 1 2 2 2
        {'name': 'b', 'layer': 1, 'etype': 2, 'mtype': 2, 'placement_hints': (2, 1)},     # 2 2 2 1 1 1
        {'name': 'c', 'layer': 2, 'etype': 3, 'mtype': 3, 'placement_hints': (1,)},       # 1 1 1 1 1 1
        {'name': 'd', 'layer': 2, 'etype': 4, 'mtype': 4, 'placement_hints': (1, 2, 1)},  # 1 1 2 2 1 1
    ])

    region_layers_map = {
        1: (1, 2)
    }

    annotation = np.array([0, 0] + [1] * 12)

    sd = bbp.transform_neurondb_into_spatial_distribution(
        core.VoxelData(annotation, **mhd), neurondb, region_layers_map,
        np.array([0, 0] + range(12)),
        percentile=0.0)

    assert_frame_equal(sd.traits, neurondb)

    expected = pd.DataFrame({
        0: [2., 1., 1., 1.],
        1: [2., 1., 1., 1.],
        2: [2., 1., 1., 2.],
        3: [1., 2., 1., 2.],
        4: [1., 2., 1., 1.],
        5: [1., 2., 1., 1.],
    })
    expected /= expected.sum()
    assert_frame_equal(sd.distributions, expected)

    assert_equal(sd.field.raw, np.array([-1, -1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]))


def test_transform_neurondb_into_spatial_distribution_with_placement_hints_order():
    # bins are numbered from the bottom (further from pia) of the layer
    # towards the top (closer to pia)
    neurondb = pd.DataFrame([
        {'name': 'a', 'layer': 1, 'etype': 1, 'mtype': 1, 'placement_hints': (.25,  # bottom
                                                                              .5,  # middle
                                                                              .75)},  # top

        {'name': 'b', 'layer': 1, 'etype': 1, 'mtype': 1, 'placement_hints': (.75,  # bottom
                                                                              .5,  # middle
                                                                              .25)},  # top
    ])

    region_layers_map = {
        1: (1,)
    }

    annotation = np.array([1] * 3 + [0])  # [bottom, middle, top, pia]

    sd = bbp.transform_neurondb_into_spatial_distribution(
        core.VoxelData(annotation, **mhd), neurondb, region_layers_map,
        np.array([3, 2, 1, 0]),
        percentile=0.0)

    assert_frame_equal(sd.traits, neurondb)

    a = neurondb[neurondb.name == 'a'].index[0]
    eq_(sd.distributions[sd.field.raw[0]][a], .25)  # bottom
    eq_(sd.distributions[sd.field.raw[1]][a],  .5)  # middle
    eq_(sd.distributions[sd.field.raw[2]][a], .75)  # top

    b = neurondb[neurondb.name == 'b'].index[0]
    eq_(sd.distributions[sd.field.raw[0]][b], .75)  # bottom
    eq_(sd.distributions[sd.field.raw[1]][b],  .5)  # middle
    eq_(sd.distributions[sd.field.raw[2]][b], .25)  # top

    assert_equal(sd.field.raw, np.array([2,  # bottom
                                         1,  # middle
                                         0,  # top
                                         -1]))  # pia

    expected = pd.DataFrame({
        #     a    b
        2: [.25, .75],  # bottom
        1: [.5,   .5],  # midle
        0: [.75, .25],  # top
    })
    assert_frame_equal(sd.distributions, expected)


def test_transform_neurondb_into_spatial_distribution_with_placement_hints_undivisable():

    neurondb = pd.DataFrame([
        {'name': 'a', 'layer': 1, 'etype': 1, 'mtype': 1, 'placement_hints': (1, 2)},     # 1 1 1 2 2 2
        {'name': 'b', 'layer': 1, 'etype': 2, 'mtype': 2, 'placement_hints': (2, 1)},     # 2 2 2 1 1 1
        {'name': 'c', 'layer': 2, 'etype': 3, 'mtype': 3, 'placement_hints': (1,)},       # 1 1 1 1 1 1
        {'name': 'd', 'layer': 2, 'etype': 4, 'mtype': 4, 'placement_hints': (1, 2, 1)},  # 1 1 2 2 1 1
    ])

    region_layers_map = {
        1: (1, 2)
    }

    annotation = np.array([0, 0] + [1] * 10)

    sd = bbp.transform_neurondb_into_spatial_distribution(
        core.VoxelData(annotation, **mhd), neurondb, region_layers_map,
        np.array([0, 0] + range(10)),
        percentile=0.0)

    assert_frame_equal(sd.traits, neurondb)

    expected = pd.DataFrame({
        0: [2., 1., 1., 1.],
        1: [2., 1., 1., 1.],
        2: [2., 1., 1., 2.],
        3: [1., 2., 1., 2.],
        4: [1., 2., 1., 1.],
        5: [1., 2., 1., 1.],
    })
    expected /= expected.sum()
    assert_frame_equal(sd.distributions, expected)

    assert_equal(sd.field.raw, np.array([-1, -1, 0, 0, 1, 2, 2, 3, 4, 4, 5, 5]))


def test_get_region_distributions_from_placement_hints_multiple_regions():

    neurondb = pd.DataFrame([
        {'name': 'a', 'layer': 1, 'etype': 1, 'mtype': 1, 'placement_hints': (1, 2)},     # 1 2
        {'name': 'b', 'layer': 1, 'etype': 2, 'mtype': 2, 'placement_hints': (2, 1)},     # 2 1
        {'name': 'c', 'layer': 2, 'etype': 3, 'mtype': 3, 'placement_hints': (1,)},       # 1 1 1
        {'name': 'd', 'layer': 2, 'etype': 4, 'mtype': 4, 'placement_hints': (1, 2, 1)},  # 1 2 1
    ])

    region_layers_map = {
        1: (1,),
        2: (2,)
    }

    res = bbp.get_region_distributions_from_placement_hints(neurondb, region_layers_map, 0.0)

    eq_(res.keys(), [(2,), (1,)])
    assert_frame_equal(res[(1,)], pd.DataFrame({0: [2/3., 1/3.],
                                                1: [1/3., 2/3.]}))

    assert_frame_equal(res[(2,)], pd.DataFrame({0: [1/2., 1/2.],
                                                1: [1/3., 2/3.],
                                                2: [1/2., 1/2.]}, index=[2, 3]))


def test_get_region_distributions_from_placement_hints_percentile_selection():

    neurondb = pd.DataFrame([
        {'name': 'a', 'layer': 1, 'etype': 1, 'mtype': 1, 'placement_hints': (1, 2)},     # 1 2
        {'name': 'b', 'layer': 1, 'etype': 1, 'mtype': 1, 'placement_hints': (2, 1)},     # 2 1
        {'name': 'c', 'layer': 1, 'etype': 1, 'mtype': 1, 'placement_hints': (3,)},       # 3 3
        {'name': 'd', 'layer': 2, 'etype': 2, 'mtype': 2, 'placement_hints': (1, 2, 1)},  # 1 2 1
    ])

    region_layers_map = {
        1: (1,),
        2: (2,)
    }

    res = bbp.get_region_distributions_from_placement_hints(neurondb, region_layers_map, 0.5)

    eq_(res.keys(), [(2,), (1,)])

    # .5 percentile in of [3, 2, 1] includes 2,
    # so where mtype & etype are grouped (ie: (1, 1))
    # only [3, 2, 0] should survive the culling by percentile
    assert_frame_equal(res[(1,)], pd.DataFrame({0: [2/5., 0/5., 3/5.],
                                                1: [0/5., 2/5., 3/5.],
                                                }))
    assert_frame_equal(res[(2,)], pd.DataFrame({0: [1., ],
                                                1: [1., ],
                                                2: [1., ],
                                                }, index=[3]))

    res = bbp.get_region_distributions_from_placement_hints(neurondb, region_layers_map, 0.51)
    # .51 percentile in of [3, 2, 1] doesn't includes 2,
    # so where mtype & etype are grouped (ie: (1, 1))
    # only [3, 0, 0] should survive the culling by percentile
    assert_frame_equal(res[(1,)], pd.DataFrame({0: [0/3., 0/3., 3/3.],
                                                1: [0/3., 0/3., 3/3.],
                                                }))
    assert_frame_equal(res[(2,)], pd.DataFrame({0: [1., ],
                                                1: [1., ],
                                                2: [1., ],
                                                }, index=[3]))


def test_transform_neurondb_into_spatial_distribution_with_placement_hints_multiple_regions():

    neurondb = pd.DataFrame([
        {'name': 'a', 'layer': 1, 'etype': 1, 'mtype': 1, 'placement_hints': (1, 2)},     # 1 2
        {'name': 'b', 'layer': 1, 'etype': 2, 'mtype': 2, 'placement_hints': (2, 1)},     # 2 1
        {'name': 'c', 'layer': 2, 'etype': 3, 'mtype': 3, 'placement_hints': (1,)},       # 1 1 1
        {'name': 'd', 'layer': 2, 'etype': 4, 'mtype': 4, 'placement_hints': (1, 2, 1)},  # 1 2 1
    ])

    region_layers_map = {
        1: (1,),
        2: (2,)
    }

    annotation = np.array([0, 0] + [1] * 4 + [2] * 6)

    sd = bbp.transform_neurondb_into_spatial_distribution(
        core.VoxelData(annotation, **mhd), neurondb, region_layers_map,
        np.array([0, 0] + range(10)),
        percentile=0.0)

    assert_frame_equal(sd.traits, neurondb)

    assert_frame_equal(sd.distributions, pd.DataFrame({
        0: [0.,     0.,  1/2., 1/2.],
        1: [0.,     0.,  1/3., 2/3.],
        2: [0.,     0.,  1/2., 1/2.],
        3: [2/3., 1/3.,    0.,   0.],
        4: [1/3., 2/3.,    0.,   0.],
    }))

    assert_equal(sd.field.raw, np.array([-1, -1, 3, 3, 4, 4, 0, 0, 1, 1, 2, 2]))


def test_assign_distributions_to_voxels_empty():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([]), 1),
                 np.array([]))


def test_assign_distributions_to_voxels_single():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([0, 1, 2]), 1),
                 np.array([0, 0, 0]))


def test_assign_distributions_to_voxels_more_dists_than_voxels():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([0]), 3),
                 np.array([1]))


def test_assign_distributions_to_voxels_ascending():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([0, 1, 2, 3]), 2),
                 np.array([0, 0, 1, 1]))


def test_assign_distributions_to_voxels_descending():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([3, 2, 1, 0]), 2),
                 np.array([1, 1, 0, 0]))


def test_assign_distributions_to_voxels_unordered():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([3, 1, 2, 0]), 2),
                 np.array([1, 0, 1, 0]))


def test_clip_columns_to_percentile():
    dists = pd.DataFrame([
        {'0': 0, '1': 4, '2': 0, '3': 1},
        {'0': 1, '1': 3, '2': 0, '3': 1},
        {'0': 2, '1': 2, '2': 0, '3': 1},
        {'0': 3, '1': 1, '2': 4, '3': 1},
        {'0': 4, '1': 0, '2': 4, '3': 1},
    ])
    # should have no change in distribution
    ret = bbp.clip_columns_to_percentile(dists.copy(), 0.0)
    assert_equal(dists.values, ret.values)

    ret = bbp.clip_columns_to_percentile(dists.copy(), 1.0)
    assert_equal(np.sum(ret.values, axis=0), np.array((4, 4, 8, 5)))

    ret = bbp.clip_columns_to_percentile(dists.copy(), 0.90)
    # 1st two columns, only 4 will be saved, last 2 all elements should be
    assert_equal(np.sum(ret.values, axis=0), np.array((4, 4, 8, 5)))

    ret = bbp.clip_columns_to_percentile(dists.copy(), 0.75)
    assert_equal(np.sum(ret.values, axis=0), np.array((7, 7, 8, 5)))

    ret = bbp.clip_columns_to_percentile(dists.copy(), 0.50)
    assert_equal(np.sum(ret.values, axis=0), np.array((9, 9, 8, 5)))


def test_parse_mvd2():
    data = bbp.parse_mvd2(os.path.join(DATA_PATH, 'circuit.mvd2'))
    eq_(len(data['CircuitSeeds']), 1)
    eq_(len(data['ElectroTypes']), 3)
    eq_(len(data['MorphTypes']), 5)
    eq_(len(data['MicroBox Data']), 1)
    eq_(len(data['MiniColumnsPosition']), 5)
    eq_(len(data['Neurons Loaded']), 5)


def test_load_mvd2():
    cells = bbp.load_mvd2(os.path.join(DATA_PATH, 'circuit.mvd2'))

    eq_(cells.positions.shape, (5, 3))

    eq_(cells.orientations.shape, (5, 3, 3))

    eq_(set(cells.properties.columns),
        set(['etype', 'morphology', 'mtype', 'synapse_class',
             'layer', 'minicolumn']))

    eq_(list(cells.properties.synapse_class.unique()),
        ['inhibitory', 'excitatory'])

    eq_(list(cells.properties.mtype.unique()),
        ['L1_DLAC', 'L23_PC', 'L4_NBC', 'L5_TTPC1', 'L6_LBC'])

    eq_(list(cells.properties.etype.unique()),
        ['cNAC', 'cADpyr', 'dNAC'])

    eq_(list(cells.properties.synapse_class),
        ['inhibitory', 'excitatory', 'inhibitory', 'excitatory', 'inhibitory'])