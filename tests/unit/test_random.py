# SPDX-License-Identifier: Apache-2.0
import pytest

import brainbuilder.utils.random as test_module


def test_parse_distr_unif():
    test_module.parse_distr(("unif", {"a": 1, "b": 2}))
    test_module.parse_distr(("uniform", {"low": 1, "high": 2}))


def test_parse_distr_norm():
    test_module.parse_distr(("norm", {"mean": 1, "sd": 2}))
    test_module.parse_distr(("normal", {"loc": 1, "scale": 2}))


def test_parse_distr_truncnorm():
    test_module.parse_distr(("truncnorm", {"mean": 1, "sd": 2, "a": 0, "b": 1}))
    test_module.parse_distr(("truncnormal", {"loc": 1, "scale": 2, "low": 0, "high": 1}))


def test_parse_distr_scipy_fallback():
    test_module.parse_distr(("gamma", {"a": 1}))


def test_parse_distr_string():
    test_module.parse_distr('["uniform", {"a": 1, "b": 2}]')


def test_parse_distr_raises():
    with pytest.raises(KeyError):
        test_module.parse_distr(("norm", {"loc": 2}))
    with pytest.raises(AttributeError):
        test_module.parse_distr(("foo", None))
