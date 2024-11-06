from trilogy_nlp.llm_interface.validation import (
    validate_response,
    Column,
    FilterResultV2,
)
from trilogy import Environment


INVALID_FUNCTION = {
    "columns": [
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
    columns = [Column.model_validate(x) for x in INVALID_FUNCTION["columns"]]
    response = validate_response(
        environment=Environment(),
        columns=columns,
        filtering=filtering,
        order=None,
        limit=100,
    )

    assert response["status"] == "invalid", response
    errors = response["error"]
    assert "does not use a valid function" in errors, errors


TEST_INVALID_FUNCTIONT_TWO = {
    "columns": [
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
    filtering = FilterResultV2.model_validate(TEST_INVALID_FUNCTIONT_TWO["filtering"])
    columns = [Column.model_validate(x) for x in TEST_INVALID_FUNCTIONT_TWO["columns"]]
    response = validate_response(
        environment=Environment(),
        columns=columns,
        filtering=filtering,
        order=None,
        limit=100,
    )

    assert response["status"] == "invalid", response
    errors = response["error"]
    assert "does not use a valid Function" in errors, errors
