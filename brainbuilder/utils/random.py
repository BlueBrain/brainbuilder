# SPDX-License-Identifier: Apache-2.0
""" Utilities for random sampling. """

import json

import scipy.stats


def _get_value(mapping, keys):
    """Return the value of the first key from `keys` found in `mapping`."""
    for k in keys:
        if k in mapping:
            return mapping[k]
    raise KeyError(keys)


def parse_distr(value):
    """
    Convert distribution config into `scipy.stats` distribution object.

    `value` can be either:
        - a tuple (<distribution name>, <distribution parameters>)
        - a string with JSON serialization of such a tuple

    See also:
    https://bbpteam.epfl.ch/project/spaces/display/BBPNSE/Defining+distributions+in+config+files
    """
    if isinstance(value, str):
        value = json.loads(value)
    func, params = value
    if func in ("norm", "normal"):
        loc = _get_value(params, ("mean", "loc"))
        scale = _get_value(params, ("sd", "scale"))
        return scipy.stats.norm(loc=loc, scale=scale)
    elif func in ("truncnorm", "truncnormal"):
        loc = _get_value(params, ("mean", "loc"))
        scale = _get_value(params, ("sd", "scale"))
        a = _get_value(params, ("a", "low"))
        b = _get_value(params, ("b", "high"))
        return scipy.stats.truncnorm(a=a, b=b, loc=loc, scale=scale)
    elif func in ("unif", "uniform"):
        a = _get_value(params, ("a", "low"))
        b = _get_value(params, ("b", "high"))
        loc = min(a, b)
        scale = max(a, b) - loc
        return scipy.stats.uniform(loc=loc, scale=scale)
    else:
        # Try to instantiate a distribution directly from `scipy.stats`, w/o parameter guessing
        return getattr(scipy.stats, func)(**params)
