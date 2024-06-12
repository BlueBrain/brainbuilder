# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

from click.testing import CliRunner

from brainbuilder.app import sonata as test_module


@patch("brainbuilder.utils.sonata.convert", create=True)
def test_from_mvd3(mock_module, tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(tmp_path):
        result = runner.invoke(
            test_module.from_mvd3,
            [
                "--output",
                "output_dir",
                "--model-type",
                "biophysical",
                "cells.mvd3",
            ],
        )
    assert result.exit_code == 0, f"Output:\n {result.output}"
    assert mock_module.provide_me_info.call_count == 1


@patch("brainbuilder.utils.sonata.convert", create=True)
def test_provide_me_info(mock_module, tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(tmp_path):
        result = runner.invoke(
            test_module.provide_me_info,
            [
                "--output",
                "output_dir",
                "--model-type",
                "biophysical",
                "nodes.h5",
            ],
        )
    assert result.exit_code == 0, f"Output:\n {result.output}"
    assert mock_module.provide_me_info.call_count == 1


@patch("brainbuilder.utils.sonata.convert", create=True)
def test_from_syn2(mock_module, tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(tmp_path):
        result = runner.invoke(
            test_module.from_syn2,
            [
                "--output",
                "output_dir",
                "edges.h5",
            ],
        )
    assert result.exit_code == 0, f"Output:\n {result.output}"
    assert mock_module.write_edges_from_syn2.call_count == 1


@patch("brainbuilder.utils.sonata.write_config", create=True)
def test_network_config(mock_module, tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(tmp_path):
        result = runner.invoke(
            test_module.network_config,
            [
                "--base-dir",
                ".",
                "--morph-dir",
                "path/to/morph",
                "--emodel-dir",
                "path/to/emodel",
                "--nodes-dir",
                "nodes",
                "--nodes",
                "path:pop1,pop2",
                "--node-sets",
                "node_sets.json",
                "--edges-dir",
                "edges",
                "--edges",
                "path:pop1,pop2",
                "--output",
                "circuit_config.json",
            ],
        )
    assert result.exit_code == 0, f"Output:\n {result.output}"
    assert mock_module.write_network_config.call_count == 1


@patch("brainbuilder.utils.sonata.convert", create=True)
def test_node_set_from_targets(mock_module, tmp_path):
    target_file_1 = tmp_path / "user1.target"
    target_file_2 = tmp_path / "user2.target"
    target_file_1.touch()
    target_file_2.touch()
    runner = CliRunner()
    with runner.isolated_filesystem(tmp_path):
        result = runner.invoke(
            test_module.node_set_from_targets,
            [
                "--target-file",
                str(target_file_1),
                "--target-file",
                str(target_file_2),
                "--cells-path",
                "nodes.h5",
                "--output",
                "node_sets.json",
            ],
        )
    assert result.exit_code == 0, f"Output:\n {result.output}"
    assert mock_module.write_node_set_from_targets.call_count == 1


@patch("brainbuilder.utils.bbp", create=True)
@patch("brainbuilder.utils.sonata.curate", create=True)
def test_check_morphologies(mock_module_curate, mock_module_bbp, tmp_path):
    mock_module_curate.check_morphology_invariants.return_value = (set(), set())
    h5_morphs = tmp_path / "h5"
    h5_morphs.mkdir()
    morphdb = tmp_path / "extNeuronDB.dat"
    morphdb.touch()
    runner = CliRunner()
    with runner.isolated_filesystem(tmp_path):
        result = runner.invoke(
            test_module.check_morphologies,
            [
                "--h5-morphs",
                str(h5_morphs),
                "--morphdb",
                str(morphdb),
            ],
        )
    assert result.exit_code == 0, f"Output:\n {result.output}"
    assert mock_module_bbp.load_extneurondb.call_count == 1
    assert mock_module_curate.check_morphology_invariants.call_count == 1


@patch(f"{test_module.__name__}.dump_json")
@patch("brainbuilder.utils.sonata.reindex", create=True)
def test_update_morphologies(mock_module, mock_dump_json, tmp_path):
    h5_morphs = tmp_path / "h5"
    h5_morphs.mkdir()
    output = tmp_path / "output"
    runner = CliRunner()
    with runner.isolated_filesystem(tmp_path):
        result = runner.invoke(
            test_module.update_morphologies,
            [
                "--h5-morphs",
                str(h5_morphs),
                "--output",
                str(output),
            ],
            # catch_exceptions=False
        )
    assert result.exit_code == 0, f"Output:\n {result.output}"
    assert mock_module.generate_h5_updates.call_count == 1
    assert mock_module.write_new_h5_morphs.call_count == 1
    assert mock_dump_json.call_count == 1


@patch("voxcell.CellCollection")
@patch("brainbuilder.utils.sonata.reindex", create=True)
def test_update_edge_population(mock_module, mock_cell_collection, tmp_path):
    h5_updates = tmp_path / "h5_updates.json"
    h5_updates.write_text("{}")
    nodes = tmp_path / "nodes.h5"
    nodes.touch()
    runner = CliRunner()
    with runner.isolated_filesystem(tmp_path):
        result = runner.invoke(
            test_module.update_edge_population,
            [
                "--h5-updates",
                str(h5_updates),
                "--nodes",
                str(nodes),
                "edges1.h5",
                "edges2.h5",
            ],
        )
    assert result.exit_code == 0, f"Output:\n {result.output}"
    assert mock_module.apply_edge_updates.call_count == 2
    assert mock_cell_collection.load.call_count == 1


@patch("voxcell.CellCollection")
@patch("brainbuilder.utils.sonata.reindex", create=True)
def test_update_edge_pos(mock_module, mock_cell_collection, tmp_path):
    morph_path = tmp_path / "h5"
    morph_path.mkdir()
    nodes = tmp_path / "nodes.h5"
    nodes.touch()
    runner = CliRunner()
    with runner.isolated_filesystem(tmp_path):
        result = runner.invoke(
            test_module.update_edge_pos,
            [
                "--morph-path",
                str(morph_path),
                "--nodes",
                str(nodes),
                "edges.h5",
            ],
        )
    assert result.exit_code == 0, f"Output:\n {result.output}"
    assert mock_module.write_sonata_pos.call_count == 1
    assert mock_cell_collection.load.call_count == 1


@patch("brainbuilder.utils.sonata.split_population", create=True)
def test_split_population(mock_module, tmp_path):
    nodes = tmp_path / "nodes.h5"
    nodes.touch()
    edges = tmp_path / "edges.h5"
    edges.touch()
    output = tmp_path / "output"
    output.mkdir()
    runner = CliRunner()
    with runner.isolated_filesystem(tmp_path):
        result = runner.invoke(
            test_module.split_population,
            [
                "--attribute",
                "mtype",
                "--nodes",
                str(nodes),
                "--edges",
                str(edges),
                "--output",
                str(output),
            ],
        )
    assert result.exit_code == 0, f"Output:\n {result.output}"
    assert mock_module.split_population.call_count == 1


@patch("brainbuilder.utils.sonata.split_population", create=True)
def test_simple_split_subcircuit(mock_module, tmp_path):
    nodes = tmp_path / "nodes.h5"
    nodes.touch()
    edges = tmp_path / "edges.h5"
    edges.touch()
    output = tmp_path / "output"
    output.mkdir()
    nodeset_path = tmp_path / "node_sets.json"
    nodeset_path.touch()
    runner = CliRunner()
    with runner.isolated_filesystem(tmp_path):
        result = runner.invoke(
            test_module.simple_split_subcircuit,
            [
                "--nodeset",
                "my_nodeset",
                "--nodeset-path",
                str(nodeset_path),
                "--nodes",
                str(nodes),
                "--edges",
                str(edges),
                "--output",
                str(output),
            ],
        )
    assert result.exit_code == 0, f"Output:\n {result.output}"
    assert mock_module.simple_split_subcircuit.call_count == 1


@patch("brainbuilder.utils.sonata.split_population", create=True)
def test_split_subcircuit(mock_module, tmp_path):
    circuit = tmp_path / "circuit_config.json"
    circuit.touch()
    output = tmp_path / "output"
    output.mkdir()
    runner = CliRunner()
    with runner.isolated_filesystem(tmp_path):
        result = runner.invoke(
            test_module.split_subcircuit,
            [
                "--nodeset",
                "my_nodeset",
                "--circuit",
                str(circuit),
                "--output",
                str(output),
            ],
        )
    assert result.exit_code == 0, f"Output:\n {result.output}"
    assert mock_module.split_subcircuit.call_count == 1


@patch("brainbuilder.utils.sonata.clip", create=True)
def test_clip_morphologies(mock_module, tmp_path):
    circuit = tmp_path / "circuit_config.json"
    circuit.touch()
    output = tmp_path / "output"
    output.mkdir()
    runner = CliRunner()
    with runner.isolated_filesystem(tmp_path):
        result = runner.invoke(
            test_module.clip_morphologies,
            [
                "--circuit",
                str(circuit),
                "--output",
                str(output),
                "--population-name",
                "default",
            ],
        )
    assert result.exit_code == 0, f"Output:\n {result.output}"
    assert mock_module.morphologies.call_count == 1
