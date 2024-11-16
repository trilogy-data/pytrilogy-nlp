FILTERING_EXAMPLE = {
    "root": {
        "boolean": "and",
        "values": [
            {
                "operator": ">",
                "left": {
                    "name": "avg_monthly_rainfall",
                    "calculation": {
                        "operator": "AVG",
                        "arguments": [{"name": "country.monthly_rainfall"}],
                        "over": [{"name": "country.name"}],
                    },
                },
                "right": {
                    "name": "continent_avg_monthly_rainfall_2x",
                    "calculation": {
                        "operator": "MULTIPLY",
                        "arguments": [
                            {"value": "2", "type": "integer"},
                            {
                                "name": "continent_avg_monthly_rainfall",
                                "calculation": {
                                    "operator": "AVG",
                                    "arguments": ["country.monthly_rainfall"],
                                    "over": [{"name": "country.continent"}],
                                },
                            },
                        ],
                    },
                },
            }
        ],
    }
}

COLUMN_DESCRIPTION = """A Column Object is json with two fields:
    -- name: the field being referenced or a new derived name. If there is a calculation, this should always be a new derived name you came up with. 
    -- calculation: An optional calculation object.
"""
