# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from brainbuilder import BrainBuilderError
from brainbuilder.utils.sonata import clip as test_module

DATA_PATH = (Path(__file__).parent / "../data/sonata/clip").resolve()


def test__format_missing():
    missing = [f"{r}.asc" for r in range(20)]
    ret = test_module._format_missing(missing, max_to_show=0)
    assert "Missing 20 files" in ret

    ret = test_module._format_missing(missing, max_to_show=3)
    assert "0.asc" in ret
    assert "4.asc" not in ret

    ret = test_module._format_missing(missing, max_to_show=10)
    assert "9.asc" in ret
    assert "10.asc" not in ret


def test__copy_files_with_extension(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir(parents=True, exist_ok=True)

    at_root = [str(r) for r in range(3)]
    subdir = [f"{r}/{r}/{r}" for r in range(3)]

    names = at_root + subdir
    for name in names:
        (source / Path(name).parent).mkdir(parents=True, exist_ok=True)
        open(source / f"{name}.asc", "w").close()

    names += ["missing0", "missing1"]

    missing = test_module._copy_files_with_extension(source, dest, names, extension="asc")

    assert missing == ["missing0", "missing1"]

    for ext in ("asc", "swc", "h5"):
        for name in names:
            assert (source / f"{name}.{ext}").exists


def test_morphologies(tmp_path):
    circuit_config = DATA_PATH / "circuit_config.json"

    with pytest.raises(BrainBuilderError):
        test_module.morphologies(tmp_path, circuit_config, "missing_population")

    test_module.morphologies(tmp_path, circuit_config, "A")

    for ext in ("asc", "swc", "h5"):
        tmp_path / ext / f"0/0/0.{ext}"
        tmp_path / ext / f"1.{ext}"
        tmp_path / ext / f"2.{ext}"
