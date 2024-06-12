# SPDX-License-Identifier: Apache-2.0
from click.testing import CliRunner

from brainbuilder.app import __main__ as test_module


def test_main_help():
    runner = CliRunner()
    result = runner.invoke(test_module.main, ["--help"])
    assert "Collection of tools for circuit building" in result.output
    assert result.exit_code == 0
