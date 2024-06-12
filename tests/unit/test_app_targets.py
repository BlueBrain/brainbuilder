# SPDX-License-Identifier: Apache-2.0
import pandas as pd

import brainbuilder.app.targets as tested


def test__add_occupied_hierarchy():
    region_map_df = pd.DataFrame(
        columns=["id", "acronym", "parent_id"],
        data=[
            [997, "root", -1],
            [8, "grey", 997],
            [688, "CTX", 8],
            [695, "CTXpl", 688],
            [315, "Isocortex", 695],
            [184, "FRP", 315],
            [68, "FRP1", 184],
            [667, "FRP2/3", 184],
        ],
    ).set_index("id")

    occupied_regions = {"FRP1", "FRP2/3", "CTXpl"}
    result = {"CTXpl": {"region": "CTXpl"}}
    tested._add_occupied_hierarchy(region_map_df, occupied_regions, result)
    expected = {
        "FRP": ["FRP1", "FRP2/3"],
        "Isocortex": ["FRP"],
        "CTXpl": ["CTXpl-only", "Isocortex"],
        "CTXpl-only": {
            "region": "CTXpl",
        },
        "CTX": ["CTXpl"],
        "grey": ["CTX"],
        "root": ["grey"],
    }

    assert result == expected
