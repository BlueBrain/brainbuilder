# SPDX-License-Identifier: Apache-2.0
import subprocess
from pathlib import Path

from numpy.testing import assert_equal

from brainbuilder.utils import load_json

TEST_DATA_PATH = Path(__file__).parent.parent / "data"


def assert_h5_dirs_equal(actual_dir, expected_dir, pattern="*.h5"):
    """Verify that the H5 files contained in the given directories have the same content.

    h5diff from hdf5 tools must be installed.
    """
    actual_files = sorted(Path(actual_dir).glob(pattern))
    expected_files = sorted(Path(expected_dir).glob(pattern))
    assert_equal(
        {f.name for f in actual_files},
        {f.name for f in expected_files},
        err_msg=f"comparing with {expected_dir}",
    )
    assert len(expected_files) > 0, f"No files match {pattern!r} in {expected_dir}"
    for actual_file, expected_file in zip(actual_files, expected_files):
        assert_h5_files_equal(actual_file, expected_file)


def assert_h5_files_equal(actual_path, expected_path):
    """Verify that two H5 files have identical content.

    h5diff from hdf5 tools must be installed.
    """
    # From h5diff documentation:
    #  exit code: 0 if no differences, 1 if differences found, 2 if error.
    #  --use-system-epsilon: Print difference if (|a-b| > EPSILON), EPSILON is system defined value.
    #    If the system epsilon is not defined, one of the following predefined values will be used:
    #            FLT_EPSILON = 1.19209E-07 for floating-point type
    #            DBL_EPSILON = 2.22045E-16 for double precision type
    cmd = ["h5diff", "--use-system-epsilon", actual_path, expected_path]
    exit_code = subprocess.call(cmd)
    if exit_code != 0:
        msg = f"H5 file different from what expected: {expected_path} [code: {exit_code}]"
        raise AssertionError(msg)


def assert_json_files_equal(actual_path, expected_path):
    assert load_json(actual_path) == load_json(expected_path)
