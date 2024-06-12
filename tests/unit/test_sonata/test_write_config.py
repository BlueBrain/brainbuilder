# SPDX-License-Identifier: Apache-2.0
import json

from bluepysnap.utils import load_json

from brainbuilder.utils.sonata import write_config as test_module


def test_make_network_config_bbp():
    expected_config = json.loads(
        """
{
  "version": 2,
  "manifest": {
    "$BASE_DIR": "/base/dir",
    "$COMPONENTS_DIR": "$BASE_DIR/components",
    "$NETWORK_NODES_DIR": "$BASE_DIR/nodes/dir",
    "$NETWORK_EDGES_DIR": "$BASE_DIR/edges/dir"
  },
  "components": {
    "morphologies_dir": "/morph/dir",
    "alternate_morphologies": {
      "neurolucida-asc": "$COMPONENTS_DIR/asc",
      "h5v1" : "/h5v1/dir"
    }
  },
  "node_sets_file": "/node_sets/file",
  "networks": {
    "nodes": [
      {
        "nodes_file": "$NETWORK_NODES_DIR/All/nodes.h5",
        "populations": {"All": {}}
      },
      {
        "nodes_file": "/A_nodes.h5",
        "populations": {
          "population_a": {"type": "virtual"},
          "population_b": {
            "type": "biophysical",
            "alternate_morphologies": {"h5v1": "$COMPONENTS_DIR/h5v1"}
          },
          "population_c": {
            "type": "point_neuron",
            "point_neuron_models_dir": "/point/dir"
          }
        }
      }
    ],
    "edges": [
      {
        "edges_file": "$NETWORK_EDGES_DIR/All/edges.h5",
        "populations": {"All": {}}
      },
      {
        "edges_file": "$NETWORK_EDGES_DIR/A_edges.h5",
        "populations": {
          "population_e": {
            "type": "chemical_synapse",
            "end_feet_area": "/endfoot/dir"
          }
        }
      }
    ]
  }
}
    """
    )

    actual_config = test_module.make_network_config_bbp(
        "/base/dir",
        [
            {
                "nodes_file": "All/nodes.h5",
                "populations": {"All": {}},
            },
            {
                "nodes_file": "/A_nodes.h5",
                "populations": {
                    "population_a": {"type": "virtual"},
                    "population_b": {
                        "type": "biophysical",
                        "alternate_morphologies": {"h5v1": "h5v1"},
                    },
                    "population_c": {
                        "type": "point_neuron",
                        "point_neuron_models_dir": "/point/dir",
                    },
                },
            },
        ],
        [
            {
                "edges_file": "All/edges.h5",
                "populations": {"All": {}},
            },
            {
                "edges_file": "A_edges.h5",
                "populations": {
                    "population_e": {"type": "chemical_synapse", "end_feet_area": "/endfoot/dir"}
                },
            },
        ],
        "/node_sets/file",
        {
            "morphologies_dir": "/morph/dir",
            "alternate_morphologies": {
                "neurolucida-asc": "asc",
                "h5v1": "/h5v1/dir",
            },
        },
        "nodes/dir",
        "edges/dir",
        "components",
    )
    assert expected_config == actual_config


def test_write_network_config(tmp_path):
    expected_config = json.loads(
        """
{
  "version": 2,
  "manifest": {
    "$BASE_DIR": "/base/dir",
    "$COMPONENTS_DIR": "$BASE_DIR",
    "$NETWORK_NODES_DIR": "$BASE_DIR/nodes/dir",
    "$NETWORK_EDGES_DIR": "$BASE_DIR/edges/dir"
  },
  "components": {
    "morphologies_dir": "$COMPONENTS_DIR/morph/dir",
    "biophysical_neuron_models_dir": "/emodel/dir"
  },
  "node_sets_file": "$BASE_DIR/node_sets/file",
  "networks": {
    "nodes": [
      {
        "nodes_file": "/nodes/file/nodes.h5",
        "populations": {"nodes_pop1": {}, "nodes_pop2": {}}
      }
    ],
    "edges": [
      {
        "edges_file": "$NETWORK_EDGES_DIR/edges/file/edges_suffix.h5",
        "populations": {"edges_pop1": {}}
      }
    ]
  }
}
    """
    )
    filepath = tmp_path / "circuit_config.json"
    test_module.write_network_config(
        base_dir="/base/dir",
        morph_dir="morph/dir",
        emodel_dir="/emodel/dir",
        nodes_dir="nodes/dir",
        nodes=["/nodes/file:nodes_pop1,nodes_pop2"],
        node_sets="node_sets/file",
        edges_dir="edges/dir",
        edges_suffix="_suffix",
        edges=["edges/file:edges_pop1"],
        output_path=str(filepath),
    )
    actual_config = load_json(filepath)
    assert expected_config == actual_config
