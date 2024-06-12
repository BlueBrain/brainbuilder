# SPDX-License-Identifier: Apache-2.0
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from voxcell import CellCollection

from brainbuilder.exceptions import BrainBuilderError
from brainbuilder.utils import bbp

DATA_PATH = Path(__file__).resolve().parent / "data"


def test_load_cell_composition_v1():
    with pytest.raises(ValueError, match="Use cell composition file of version 2"):
        bbp.load_cell_composition(DATA_PATH / "cell_composition_v1.yaml")


def test_load_cell_composition_v2():
    content = bbp.load_cell_composition(DATA_PATH / "cell_composition_v2.yaml")
    assert content == {
        "version": "v2.0",
        "neurons": [
            {
                "density": 68750,
                "region": "mc0;Rt",
                "traits": {
                    "layer": "Rt",
                    "mtype": "Rt_RC",
                    "etype": {"cNAD_noscltb": 0.43, "cAD_noscltb": 0.57},
                },
            },
            {
                "density": "{L23_MC}",
                "region": "mc1;Rt",
                "traits": {
                    "layer": "Rt",
                    "mtype": "Rt_RC",
                    "etype": {"cNAD_noscltb": 0.43, "cAD_noscltb": 0.57},
                },
            },
        ],
    }


def test_load_neurondb():
    actual = bbp.load_neurondb(DATA_PATH / "neuronDBv2.dat")
    expected = pd.DataFrame(
        {
            "morphology": ["morph-a", "morph-b"],
            "layer": ["L1", "L2"],
            "mtype": ["L1_DAC", "L23_PC"],
        }
    )
    assert_frame_equal(actual, expected, check_like=True)


def test_load_extneurondb_as_neurondb():
    actual = bbp.load_neurondb(DATA_PATH / "extneuronDB.dat")
    expected = pd.DataFrame(
        {
            "morphology": ["morph-a", "morph-b"],
            "layer": ["L1", "L2"],
            "mtype": ["L1_DAC", "L23_PC"],
        }
    )
    assert_frame_equal(actual, expected, check_like=True)


def test_load_invalid_neurondb():
    with pytest.raises(ValueError, match="Invalid NeuronDB format"):
        bbp.load_neurondb(DATA_PATH / "hierarchy.json")


def test_load_extneurondb():
    actual = bbp.load_extneurondb(DATA_PATH / "extneuronDB.dat")
    expected = pd.DataFrame(
        {
            "morphology": ["morph-a", "morph-b"],
            "layer": ["L1", "L2"],
            "mtype": ["L1_DAC", "L23_PC"],
            "etype": ["bNAC", "dNAC"],
            "me_combo": ["me-combo-a", "me-combo-b"],
        }
    )
    assert_frame_equal(actual, expected, check_like=True)


def test_load_mecombo_emodel_as_extneurondb():
    actual = bbp.load_extneurondb(DATA_PATH / "mecombo_emodel.dat")
    expected = pd.DataFrame(
        {
            "morphology": ["morph-a", "morph-b"],
            "layer": ["1", "2"],
            "mtype": ["L1_DAC", "L23_PC"],
            "etype": ["bNAC", "dNAC"],
            "me_combo": ["emodel-a", "emodel-b"],
        }
    )
    assert_frame_equal(actual, expected, check_like=True)


def test_load_invalid_extneurondb():
    with pytest.raises(ValueError, match="Invalid ExtNeuronDB format"):
        bbp.load_extneurondb(DATA_PATH / "neuronDBv2.dat")


def test_load_mecombo_emodel():
    actual = bbp.load_mecombo_emodel(DATA_PATH / "mecombo_emodel.dat")
    expected = pd.DataFrame(
        {
            "morphology": ["morph-a", "morph-b"],
            "layer": ["1", "2"],
            "mtype": ["L1_DAC", "L23_PC"],
            "etype": ["bNAC", "dNAC"],
            "emodel": ["emodel-a", "emodel-b"],
            "me_combo": ["me-combo-a", "me-combo-b"],
            "threshold_current": [1.0, 1.0],
            "holding_current": [2.0, 2.0],
        }
    )
    assert_frame_equal(actual, expected, check_like=True)


def test_load_invalid_mecombo_emodel():
    with pytest.raises(ValueError, match="Invalid mecombo_emodel format"):
        bbp.load_mecombo_emodel(DATA_PATH / "extneuronDB.dat")


def test_gid2str():
    actual = bbp.gid2str(42)
    assert actual == "a42"


def test_write_target():
    out = StringIO()
    bbp.write_target(out, "test", gids=[1, 2], include_targets=["A", "B"])
    actual = out.getvalue()
    expected = "\n".join(["", "Target Cell test", "{", "  a1 a2", "  A B", "}", ""])
    assert actual == expected


def test_write_property_targets():
    cells = pd.DataFrame({"prop-a": ["A", "B", "A"], "prop-b": ["X", "X", "Y"]}, index=[1, 2, 3])
    out = StringIO()
    bbp.write_property_targets(out, cells, "prop-a")
    bbp.write_property_targets(out, cells, "prop-b", mapping=lambda x: "z" + x)
    actual = out.getvalue()
    expected = "\n".join(
        [
            "",
            "Target Cell A",
            "{",
            "  a1 a3",
            "}",
            "",
            "Target Cell B",
            "{",
            "  a2",
            "}",
            "",
            "Target Cell zX",
            "{",
            "  a1 a2",
            "}",
            "",
            "Target Cell zY",
            "{",
            "  a3",
            "}",
            "",
        ]
    )
    assert actual == expected


def test_assign_emodels():
    cells = CellCollection()
    cells.properties = pd.DataFrame(
        [
            ("morph-A", "layer-A", "mtype-A", "etype-A", "prop-A"),
            ("morph-B", "layer-B", "mtype-B", "etype-B", "prop-B"),
        ],
        columns=["morphology", "layer", "mtype", "etype", "prop"],
    )
    morphdb = pd.DataFrame(
        [
            ("morph-A", "layer-A", "mtype-A", "etype-A", "me_combo-A"),
            ("morph-B", "layer-B", "mtype-B", "etype-B", "me_combo-B"),
        ],
        columns=["morphology", "layer", "mtype", "etype", "me_combo"],
    )
    actual = bbp.assign_emodels(cells, morphdb).properties
    expected = pd.DataFrame(
        [
            ("morph-A", "layer-A", "mtype-A", "etype-A", "prop-A", "me_combo-A"),
            ("morph-B", "layer-B", "mtype-B", "etype-B", "prop-B", "me_combo-B"),
        ],
        columns=["morphology", "layer", "mtype", "etype", "prop", "me_combo"],
    )
    assert_frame_equal(actual, expected, check_like=True)


def test_assign_emodels_multiple_choice():
    np.random.seed(0)
    cells = CellCollection()
    cells.properties = pd.DataFrame(
        [
            ("morph-A", 1, "mtype-A", "etype-A", "prop-A"),
        ],
        columns=["morphology", "layer", "mtype", "etype", "prop"],
    )
    morphdb = pd.DataFrame(
        [
            ("morph-A", 1, "mtype-A", "etype-A", "me_combo-A1"),
            ("morph-A", 1, "mtype-A", "etype-A", "me_combo-A2"),
        ],
        columns=["morphology", "layer", "mtype", "etype", "me_combo"],
    )
    actual = bbp.assign_emodels(cells, morphdb).properties
    expected = pd.DataFrame(
        [
            ("morph-A", 1, "mtype-A", "etype-A", "prop-A", "me_combo-A2"),
        ],
        columns=["morphology", "layer", "mtype", "etype", "prop", "me_combo"],
    )
    assert_frame_equal(actual, expected, check_like=True)


def test_assign_emodels_overwrite():
    cells = CellCollection()
    cells.properties = pd.DataFrame(
        [
            ("morph-A", 1, "mtype-A", "etype-A", "prop-A", "me_combo-A0"),
        ],
        columns=["morphology", "layer", "mtype", "etype", "prop", "me_combo"],
    )
    morphdb = pd.DataFrame(
        [
            ("morph-A", 1, "mtype-A", "etype-A", "me_combo-A1"),
        ],
        columns=["morphology", "layer", "mtype", "etype", "me_combo"],
    )
    actual = bbp.assign_emodels(cells, morphdb).properties
    expected = pd.DataFrame(
        [
            ("morph-A", 1, "mtype-A", "etype-A", "prop-A", "me_combo-A1"),
        ],
        columns=["morphology", "layer", "mtype", "etype", "prop", "me_combo"],
    )
    assert_frame_equal(actual, expected, check_like=True)


def test_assign_emodels_raises():
    with pytest.raises(BrainBuilderError):
        cells = CellCollection()
        cells.properties = pd.DataFrame(
            [
                ("morph-A", 1, "mtype-A", "etype-A", "prop-A"),
            ],
            columns=["morphology", "layer", "mtype", "etype", "prop"],
        )
        morphdb = pd.DataFrame(
            [
                ("morph-A", 1, "mtype-A", "etype-A1", "me_combo-A1"),
            ],
            columns=["morphology", "layer", "mtype", "etype", "me_combo"],
        )
        bbp.assign_emodels(cells, morphdb)


def test_assign_emodels_subregion_fallback():
    cells = CellCollection()
    cells.properties = pd.DataFrame(
        [
            ("morph-A", "subregion-A", "mtype-A", "etype-A", "prop-A"),
            ("morph-B", "subregion-B", "mtype-B", "etype-B", "prop-B"),
        ],
        columns=["morphology", "subregion", "mtype", "etype", "prop"],
    )
    morphdb = pd.DataFrame(
        [
            ("morph-A", "subregion-A", "mtype-A", "etype-A", "me_combo-A"),
            ("morph-B", "subregion-B", "mtype-B", "etype-B", "me_combo-B"),
        ],
        columns=["morphology", "layer", "mtype", "etype", "me_combo"],
    )
    actual = bbp.assign_emodels(cells, morphdb).properties
    expected = pd.DataFrame(
        [
            ("morph-A", "subregion-A", "mtype-A", "etype-A", "prop-A", "me_combo-A"),
            ("morph-B", "subregion-B", "mtype-B", "etype-B", "prop-B", "me_combo-B"),
        ],
        columns=["morphology", "subregion", "mtype", "etype", "prop", "me_combo"],
    )
    assert_frame_equal(actual, expected, check_like=True)
