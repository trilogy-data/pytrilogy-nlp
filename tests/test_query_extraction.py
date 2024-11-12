from trilogy_nlp.main import ir_to_query
from trilogy_nlp.llm_interface.models import InitialParseResponseV2
import json
from pathlib import Path
from trilogy import Environment

ORDERING_TEST_CASE = """{
    "action": "Final Answer",
    "action_input": {
        "columns": [
            {
                "name": "store_sales.customer.state"
            },
            {
                "name": "customer_count",
                "calculation": {
                    "operator": "COUNT",
                    "arguments": [
                        {
                            "name": "store_sales.customer.id"
                        }
                    ],
                    "over": [
                        {
                            "name": "store_sales.customer.state"
                        }
                    ]
                }
            },
            {
                "name": "sales_price_filtered",
                "calculation": {
                    "operator": "MULTIPLY",
                    "arguments": [
                        {
                            "name": "average_price_by_category"
                        },
                        {
                            "value": "1.2",
                            "type": "float"
                        }
                    ]
                }
            },
            {
                "name": "average_price_by_category",
                "calculation": {
                    "operator": "AVG",
                    "arguments": [
                        {
                            "name": "item.current_price"
                        }
                    ],
                    "over": [
                        {
                            "name": "item.category"
                        }
                    ]
                }
            }
        ],
        "filtering": {
            "root": {
                "values": [
                    {
                        "operator": ">=",
                        "left": {
                            "name": "customer_count"
                        },
                        "right": {
                            "value": "10",
                            "type": "int"
                        }
                    },
                    {
                        "operator": ">=",
                        "left": {
                            "name": "store_sales.item.current_price"
                        },
                        "right": {
                            "name": "sales_price_filtered"
                        }
                    },
                    {
                        "operator": "=",
                        "left": {
                            "name": "store_sales.date.date.year"
                        },
                        "right": {
                            "value": "2001",
                            "type": "int"
                        }
                    },
                    {
                        "operator": "=",
                        "left": {
                            "name": "store_sales.date.date.month"
                        },
                        "right": {
                            "value": "1",
                            "type": "int"
                        }
                    }
                ],
                "boolean": "and"
            }
        },
        "order": [
            {"column_name": "customer_count", "order": "asc"},
            {"column_name": "store_sales.customer.state", "order": "asc"}
        ],
        "limit": -1
    },
    "reasoning": "The query captures all states and their respective customer counts for customers who bought items with a sales price at least 1.2 times the average sales price of all other items in the same category during January 2001, filtering for those states with at least 10 customers."
}"""


def test_ordering_resolution():
    loaded = json.loads(ORDERING_TEST_CASE)
    validated = InitialParseResponseV2.model_validate(loaded["action_input"])
    environment = Environment(working_path=Path(__file__).parent / "tpc_ds_duckdb")
    environment.add_file_import("store_sales", "store_sales")
    environment.add_file_import("item", "item")
    environment.parse("MERGE store_sales.item.id INTO ~item.id;")
    ir = ir_to_query(validated, input_environment=environment, debug=False)


TEST_OBJECT_PROMOTION = """{
    "action": "Final Answer",
    "action_input": {
        "columns": [
            {
                "name": "store_sales.customer.state"
            },
            {
                "name": "customer_count",
                "calculation": {
                    "operator": "COUNT",
                    "arguments": [
                        {
                            "name": "store_sales.customer.id"
                        }
                    ]
                }
            },
            {
                "name": "average_sales_price_by_category",
                "calculation": {
                    "operator": "AVG",
                    "arguments": [
                        {
                            "name": "item.current_price"
                        }
                    ],
                    "over": [
                        {
                            "name": "item.category"
                        }
                    ]
                }
            },
            {
                "name": "filtered_sales_price",
                "calculation": {
                    "operator": "MULTIPLY",
                    "arguments": [
                        {
                            "name": "average_sales_price_by_category"
                        },
                        {
                            "value": "1.2",
                            "type": "float"
                        }
                    ]
                }
            }
        ],
        "filtering": {
            "root": {
                "values": [
                    {
                        "operator": ">=",
                        "left": {
                            "name": "customer_count"
                        },
                        "right": {
                            "value": "10",
                            "type": "int"
                        }
                    },
                    {
                        "operator": "=",
                        "left": {
                            "name": "store_sales.date.date.year"
                        },
                        "right": {
                            "value": "2001",
                            "type": "int"
                        }
                    },
                    {
                        "operator": "=",
                        "left": {
                            "name": "store_sales.date.date.month"
                        },
                        "right": {
                            "value": "1",
                            "type": "int"
                        }
                    },
                    {
                        "operator": ">=",
                        "left": {
                            "name": "store_sales.sales_price"
                        },
                        "right": {
                            "name": "filtered_sales_price"
                        }
                    }
                ],
                "boolean": "and"
            }
        },
        "order": [
            {"column_name": "customer_count", "order": "asc"},
            {"column_name": "store_sales.customer.state", "order": "asc"}
        ],
        "limit": -1
    },
    "reasoning": "The query calculates the average sales price by category, applies a filter for sales prices, counts customers per state, filters for customer counts of at least 10, and orders the results as specified."
}"""


def test_having_promotion():

    loaded = json.loads(TEST_OBJECT_PROMOTION)
    validated = InitialParseResponseV2.model_validate(loaded["action_input"])
    environment = Environment(working_path=Path(__file__).parent / "tpc_ds_duckdb")
    environment.add_file_import("store_sales", "store_sales")
    environment.add_file_import("item", "item")
    environment.parse("MERGE store_sales.item.id INTO ~item.id;")
    ir = ir_to_query(validated, input_environment=environment, debug=False)
    # print(Renderer().to_string(ir))
    assert "store_sales.sales_price" in [x.content.output.address for x in ir.selection]
    # assert 0 == 1
