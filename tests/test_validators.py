from trilogy_nlp.llm_interface.validation import (
    validate_response,
    Column,
    FilterResultV2,
)
from trilogy import Environment


INVALID_FUNCTION = {
    "output_columns": [
        {"name": "store_sales.store.state"},
        {
            "name": "customer_count",
            "calculation": {
                "operator": "COUNT",
                "arguments": [{"name": "store_sales.customer.id"}],
            },
        },
    ],
    "filtering": {
        "root": {
            "values": [
                {
                    "left": {
                        "name": "store_sales.sales_price",
                        "calculation": {
                            "operator": ">=",
                            "arguments": [
                                {
                                    "name": "average_category_sales_price",
                                    "calculation": {
                                        "operator": "MULTIPLY",
                                        "arguments": [
                                            {"value": "1.2", "type": "float"},
                                            {
                                                "name": "avg_sales_price_by_category",
                                                "calculation": {
                                                    "operator": "AVG",
                                                    "arguments": [
                                                        {"name": "item.list_price"}
                                                    ],
                                                    "over": [{"name": "item.category"}],
                                                },
                                            },
                                        ],
                                    },
                                }
                            ],
                        },
                    },
                    "right": {"value": "2001-01-01", "type": "string"},
                    "operator": "like",
                }
            ],
            "boolean": "and",
        }
    },
    "order": [
        {"column_name": "customer_count", "order": "asc"},
        {"column_name": "store_sales.store.state", "order": "asc"},
    ],
    "limit": -1,
}


def test_validate_response_invalid_function():
    # check to make sure that the invalid function is detected
    filtering = FilterResultV2.model_validate(INVALID_FUNCTION["filtering"])
    columns = [Column.model_validate(x) for x in INVALID_FUNCTION["output_columns"]]
    env = Environment()
    env.parse(
        """
key store_sales.store.id int;
key store_sales.customer.id int;
key store_sales.store.state string;"""
    )
    response, ir = validate_response(
        environment=env,
        output_columns=columns,
        filtering=filtering,
        order=None,
        limit=100,
        prompt="shenanigans",
    )

    assert response["status"] == "invalid", response
    errors = response["errors"]
    assert "does not use a valid function" in str(errors), errors


TEST_INVALID_FUNCTION_TWO = {
    "output_columns": [
        {"name": "store_sales.customer.state"},
        {
            "name": "customer_count",
            "calculation": {
                "operator": "COUNT",
                "arguments": [{"name": "store_sales.customer.id"}],
            },
        },
    ],
    "filtering": {
        "root": {
            "values": [
                {
                    "left": {
                        "name": "store_sales.item.current_price",
                        "calculation": {
                            "operator": "GREATER_THAN_OR_EQUAL",
                            "arguments": [
                                {
                                    "name": "average_sales_price_in_category",
                                    "calculation": {
                                        "operator": "MULTIPLY",
                                        "arguments": [
                                            {"value": "1.2", "type": "float"},
                                            {
                                                "name": "average_sales_price_in_category",
                                                "calculation": {
                                                    "operator": "AVG",
                                                    "arguments": [
                                                        {
                                                            "name": "store_sales.item.current_price"
                                                        }
                                                    ],
                                                },
                                            },
                                        ],
                                    },
                                }
                            ],
                        },
                    },
                    "right": {"value": "0", "type": "int"},
                    "operator": ">",
                },
                {
                    "left": {"name": "store_sales.date.date.year"},
                    "right": {"value": "2001", "type": "int"},
                    "operator": "=",
                },
                {
                    "left": {"name": "store_sales.date.date.month"},
                    "right": {"value": "1", "type": "int"},
                    "operator": "=",
                },
            ],
            "boolean": "and",
        }
    },
    "order": [
        {"column_name": "customer_count", "order": "asc"},
        {"column_name": "store_sales.customer.state", "order": "asc"},
    ],
    "limit": -1,
}


def test_validate_response_invalid_function_two():
    # check to make sure that the invalid function is detected
    filtering = FilterResultV2.model_validate(TEST_INVALID_FUNCTION_TWO["filtering"])
    columns = [
        Column.model_validate(x) for x in TEST_INVALID_FUNCTION_TWO["output_columns"]
    ]
    env = Environment()
    env.parse(
        """
key store_sales.store.id int;
key store_sales.customer.id int;
key store_sales.customer.state string;
key store_sales.store.state string;"""
    )
    response, ir = validate_response(
        environment=env,
        output_columns=columns,
        filtering=filtering,
        order=None,
        limit=100,
        prompt="shenanigans",
    )

    assert response["status"] == "invalid", response
    errors = response["errors"]
    assert "does not use a valid function" in str(errors), errors


INVALID_FIELD = """{
    "action": "submit_answer",
    "action_input": {
        "output_columns": [
            {"name": "store_sales.customer.state"},
            {"name": "customer_count", 
                "calculation": {
                    "operator": "COUNT",
                    "arguments": [
                        {"name": "store_sales.customer.id"}
                    ]
                }
            }
        ],
        "filtering": {
            "root": {
                "values": [
                    {
                        "operator": ">",
                        "left": {
                            "name": "store_sales.item.current_price"
                        },
                        "right": {
                            "value": {
                                "operator": "MULTIPLY",
                                "arguments": [
                                    {
                                        "name": "average_item_price_by_category"
                                    },
                                    {
                                        "value": "1.2",
                                        "type": "float"
                                    }
                                ]
                            },
                            "type": "float"
                        }
                    },
                    {
                        "operator": "=",
                        "left": {"name": "store_sales.date.date.year"},
                        "right": {"value": "2001", "type": "integer"}
                    },
                    {
                        "operator": "=",
                        "left": {"name": "store_sales.date.date.month"},
                        "right": {"value": "1", "type": "integer"}
                    },
                    {
                        "operator": "is not",
                        "left": {"name": "store_sales.item.category"},
                        "right": {"value": "null", "type": "null"}
                    },
                    {
                        "operator": ">=",
                        "left": {
                            "name": "customer_count"
                        },
                        "right": {
                            "value": "10",
                            "type": "integer"
                        }
                    }
                ],
                "boolean": "and"
            }
        },
        "order": [
            {"column_name": "customer_count", "order": "asc nulls first"},
            {"column_name": "store_sales.customer.state", "order": "asc nulls first"}
        ],
        "limit": -1
    },
    "reasoning": "I corrected the syntax error in the filtering clause by ensuring that the right-hand side of the comparison is correctly formatted as a Calculation object with a specified type. Now, I will submit this response."
}"""

from pathlib import Path
import json


def test_validate_response_invalid_field():
    # check to make sure that the invalid function is detected
    PARSED_INVALID = json.loads(INVALID_FIELD)
    filtering = FilterResultV2.model_validate(
        PARSED_INVALID["action_input"]["filtering"]
    )
    columns = [
        Column.model_validate(x)
        for x in PARSED_INVALID["action_input"]["output_columns"]
    ]

    env = Environment()
    env.add_file_import(
        Path(__file__).parent / "tpc_ds_duckdb" / "store_sales.preql", "store_sales"
    )

    response, ir = validate_response(
        environment=env,
        output_columns=columns,
        filtering=filtering,
        order=None,
        limit=100,
        prompt="shenanigans",
    )

    assert response["status"] == "invalid", response
    errors = response["errors"]
    assert "is not a valid field or previously defined by you." in str(errors), errors
