from trilogy_nlp.main import ir_to_query
from trilogy_nlp.llm_interface.models import InitialParseResponseV2
import json
from pathlib import Path
from trilogy import Environment

ORDERING_TEST_CASE = """{
    "action": "Final Answer",
    "action_input": {
        "output_columns": [
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
                        "right":    {
                "name": "sales_price_filtered",
                "calculation": {
                    "operator": "MULTIPLY",
                    "arguments": [
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
            },
                        {
                            "value": "1.2",
                            "type": "float"
                        }
                    ]
                }
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


HAVING_WHERE_SPLIT = """{
    "action": "submit_answer",
    "action_input": {
        "output_columns": [
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
            }
            
        ],
        "filtering": {
            "root": {
                "values": [
                    {
                        "operator": ">=",
                        "left": {
                            "value": "10",
                            "type": "int"
                        },
                        "right": {
                            "name": "customer_count"
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
                        "operator": ">",
                        "left": {
                            "name": "store_sales.item.current_price"
                        },
                        "right": {
                "name": "filtered_current_price",
                "calculation": {
                    "operator": "MULTIPLY",
                    "arguments": [
                        {
                            "value": "1.2",
                            "type": "float"
                        },
                         {
                "name": "average_price_by_category",
                "calculation": {
                    "operator": "AVG",
                    "arguments": [
                        {
                            "name": "store_sales.item.current_price"
                        }
                    ],
                    "over": [
                        {
                            "name": "store_sales.item.category"
                        }
                    ]
                }
            }
                    ]
                }
            }
                    }
                ],
                "boolean": "and"
            }
        },
        "order": [
            {
                "column_name": "customer_count",
                "order": "asc"
            },
            {
                "column_name": "store_sales.customer.state",
                "order": "asc"
            }
        ],
        "limit": -1
    },
    "reasoning": "I have constructed the final query to include the necessary columns, filtering conditions, and ordering as specified in the prompt. This will return the desired output for states and customer counts for those purchasing items that meet the sales price criteria."
}"""


def test_having_where_split():
    loaded = json.loads(HAVING_WHERE_SPLIT)
    validated = InitialParseResponseV2.model_validate(loaded["action_input"])
    environment = Environment(working_path=Path(__file__).parent / "tpc_ds_duckdb")
    environment.add_file_import("store_sales", "store_sales")
    environment.add_file_import("item", "item")
    environment.parse("MERGE store_sales.item.id INTO ~item.id;")
    ir = ir_to_query(validated, input_environment=environment, debug=False)
    assert environment.concepts["customer_count"].address in [
        c.address for c in ir.having_clause.concept_arguments
    ]


OR_PARENTHETICAL = """{
    "action": "submit_answer",
    "action_input": {
        "output_columns": [
            {
                "name": "store_sales.item.name"
            },
            {
                "name": "average_quantity_sold",
                "calculation": {
                    "operator": "AVG",
                    "arguments": [
                        {
                            "name": "store_sales.quantity"
                        }
                    ]
                }
            },
            {
                "name": "average_list_price",
                "calculation": {
                    "operator": "AVG",
                    "arguments": [
                        {
                            "name": "store_sales.list_price"
                        }
                    ]
                }
            },
            {
                "name": "average_coupon_amount",
                "calculation": {
                    "operator": "AVG",
                    "arguments": [
                        {
                            "name": "store_sales.coupon_amt"
                        }
                    ]
                }
            },
            {
                "name": "average_sales_price",
                "calculation": {
                    "operator": "AVG",
                    "arguments": [
                        {
                            "name": "store_sales.sales_price"
                        }
                    ]
                }
            }
        ],
        "filtering": {
            "root": {
                "values": [
                    {
                        "operator": "=",
                        "left": {
                            "name": "store_sales.customer.demographics.gender"
                        },
                        "right": {
                            "value": "M",
                            "type": "string"
                        }
                    },
                    {
                        "operator": "=",
                        "left": {
                            "name": "store_sales.customer.demographics.marital_status"
                        },
                        "right": {
                            "value": "S",
                            "type": "string"
                        }
                    },
                    {
                        "operator": "=",
                        "left": {
                            "name": "store_sales.customer.demographics.education_status"
                        },
                        "right": {
                            "value": "College",
                            "type": "string"
                        }
                    },
                    {
                        "operator": "=",
                        "left": {
                            "name": "store_sales.date.date.year"
                        },
                        "right": {
                            "value": "2000",
                            "type": "integer"
                        }
                    },
                    {
                        "boolean": "or",
                        "values": [
                            {
                                "operator": "=",
                                "left": {
                                    "name": "store_sales.promotion.channel_event"
                                },
                                "right": {
                                    "value": "N",
                                    "type": "string"
                                }
                            },
                            {
                                "operator": "=",
                                "left": {
                                    "name": "store_sales.promotion.channel_email"
                                },
                                "right": {
                                    "value": "N",
                                    "type": "string"
                                }
                            }
                        ]
                    }
                ],
                "boolean": "and"
            }
        },
        "order": [],
        "limit": -1
    },
    "reasoning": "The response has been validated and is now ready for submission. It includes the necessary output columns and filtering conditions based on the provided prompt."
}"""


def test_parenthetical_or():
    loaded = json.loads(OR_PARENTHETICAL)
    validated = InitialParseResponseV2.model_validate(loaded["action_input"])
    environment = Environment(working_path=Path(__file__).parent / "tpc_ds_duckdb")
    environment.add_file_import("store_sales", "store_sales")
    from trilogy.parsing.render import Renderer

    ir = ir_to_query(validated, input_environment=environment, debug=False)

    query = Renderer().to_string(ir)
    assert (
        "(store_sales.promotion.channel_event = 'N' or store_sales.promotion.channel_email = 'N')"
        in query
    ), query


AGG_GRAIN_CHECK = """{
    "action": "submit_answer",
    "action_input": {
        "output_columns": [
            {
                "name": "store_sales.item.name"
            },
            {
                "name": "average_quantity_sold",
                "calculation": {
                    "operator": "AVG",
                    "arguments": [
                        {
                            "name": "store_sales.quantity"
                        }
                    ]
                }
            },
            {
                "name": "average_list_price",
                "calculation": {
                    "operator": "AVG",
                    "arguments": [
                        {
                            "name": "store_sales.list_price"
                        }
                    ]
                }
            },
            {
                "name": "average_coupon_amount",
                "calculation": {
                    "operator": "AVG",
                    "arguments": [
                        {
                            "name": "store_sales.coupon_amt"
                        }
                    ]
                }
            },
            {
                "name": "average_sales_price",
                "calculation": {
                    "operator": "AVG",
                    "arguments": [
                        {
                            "name": "store_sales.sales_price"
                        }
                    ]
                }
            }
        ],
        "filtering": {
            "root": {
                "values": [
                    {
                        "operator": "=",
                        "left": {
                            "name": "store_sales.customer.demographics.gender"
                        },
                        "right": {
                            "value": "M",
                            "type": "string"
                        }
                    },
                    {
                        "operator": "=",
                        "left": {
                            "name": "store_sales.customer.demographics.marital_status"
                        },
                        "right": {
                            "value": "S",
                            "type": "string"
                        }
                    },
                    {
                        "operator": "=",
                        "left": {
                            "name": "store_sales.customer.demographics.education_status"
                        },
                        "right": {
                            "value": "College",
                            "type": "string"
                        }
                    },
                    {
                        "operator": "=",
                        "left": {
                            "name": "store_sales.date.date.year"
                        },
                        "right": {
                            "value": "2000",
                            "type": "integer"
                        }
                    },
                    {
                        "boolean": "or",
                        "values": [
                            {
                                "operator": "=",
                                "left": {
                                    "name": "store_sales.promotion.channel_event"
                                },
                                "right": {
                                    "value": "N",
                                    "type": "string"
                                }
                            },
                            {
                                "operator": "=",
                                "left": {
                                    "name": "store_sales.promotion.channel_email"
                                },
                                "right": {
                                    "value": "N",
                                    "type": "string"
                                }
                            }
                        ]
                    }
                ],
                "boolean": "and"
            }
        },
        "order": [
            {
                "column_name": "store_sales.item.name",
                "order": "asc"
            }
        ],
        "limit": -1
    },
    "reasoning": "The query has been constructed to satisfy the requirements of the prompt, ensuring it calculates the averages for the specified fields for the appropriate customer demographics while filtering out those who came from event or email promotions. The results will be sorted by item name in ascending order."
}"""


def test_aggregate_grain():
    loaded = json.loads(AGG_GRAIN_CHECK)
    validated = InitialParseResponseV2.model_validate(loaded["action_input"])
    environment = Environment(working_path=Path(__file__).parent / "tpc_ds_duckdb")
    environment.add_file_import("store_sales", "store_sales")
    from trilogy.parsing.render import Renderer

    ir = ir_to_query(validated, input_environment=environment, debug=False)
    for x in ir.output_components:
        if x.address == "store_sales.item.name":
            continue
        assert x.grain == ir.grain, x.grain
