# SPDX-License-Identifier: Apache-2.0
"""
Test assignment of mini frequencies to the circuit MVD3.
"""

from pathlib import Path

import pandas as pd

from brainbuilder.app.cells import _assign_mini_frequencies, load_mini_frequencies

DATA_PATH = Path(Path(__file__).parent, "data")


def test_mini_frequencies_input():
    """
    Mini frequencies input data must be able to read two columns from a TSV
    file.
    1. exc_mini_frequency
    2. inh_mini_frequency

    and the layer information should be in the index.
    """
    mini_freqs = load_mini_frequencies(DATA_PATH / "mini_frequencies.tsv")
    assert mini_freqs.index.name == "layer"
    assert "exc_mini_frequency" in mini_freqs.columns
    assert "inh_mini_frequency" in mini_freqs.columns


def test_mini_frequencies_assignment():
    """
    Mini frequencies must be assigned to cells.
    """
    cells = pd.DataFrame(
        data={
            "subregion": ["plf", "IF", "im", "6b"],
        }
    )

    mini_freqs = pd.DataFrame(
        data={
            "exc_mini_frequency": [0.63, 0.26, 0.122, 0.072],
            "inh_mini_frequency": [0.012, 0.012, 0.012, 0.012],
        },
        index=["plf", "IF", "im", "6b"],
    )
    _assign_mini_frequencies(cells, mini_freqs)

    expected = pd.DataFrame(
        data={
            "subregion": ["plf", "IF", "im", "6b"],
            "exc_mini_frequency": [0.63, 0.26, 0.122, 0.072],
            "inh_mini_frequency": [0.012, 0.012, 0.012, 0.012],
        }
    )

    pd.testing.assert_frame_equal(cells, expected)
