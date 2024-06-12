# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

import brainbuilder.app.cells as test_module

DATA_PATH = Path(__file__).resolve().parent / "data"


def test_add_subregion_property_consistent_layer():
    cells = pd.DataFrame(
        data={
            "layer": ["6b", "im", None, "plf"],
            "x": [8.09745916, 24.39533068, 16.80548999, 17.63244536],
            "y": [4.87049681, 24.92992631, 46.92833477, 52.18625258],
            "z": [8.90753754, 11.33143069, 5.58402908, 4.60417839],
        }
    )

    lookup_input = np.array(
        [
            [8.09745916, 4.87049681, 8.90753754],
            [24.39533068, 24.92992631, 11.33143069],
            [16.80548999, 46.92833477, 5.58402908],
            [17.63244536, 52.18625258, 4.60417839],
        ]
    )
    lookup_output = np.array([16.0, 14.0, 12.0, 11.0])
    region_map = pd.DataFrame(
        data={"acronym": pd.Series(["plf", "IF", "im", "6b"], index=[11, 12, 14, 16])}
    )

    brain_regions_mock = Mock()
    brain_regions_mock.lookup.return_value = lookup_output
    region_map_mock = Mock()
    region_map_mock.as_dataframe.return_value = region_map

    test_module._assign_subregions(cells, brain_regions_mock, region_map_mock)

    brain_regions_mock.lookup.assert_called_once()
    assert (brain_regions_mock.lookup.call_args[0][0] == lookup_input).all()

    assert "subregion" in cells.columns
    assert list(cells["subregion"]) == ["6b", "im", "IF", "plf"]
