from trilogy import Environment
from pathlib import Path
from trilogy_nlp.main_v2 import ir_to_query
from trilogy_nlp.llm_interface.models import InitialParseResponseV2
from trilogy.core.models import SelectStatement
from trilogy_nlp.llm_interface.parsing import (
    parse_filtering,
    parse_order,
    create_column,
    generate_having_and_where
)
from trilogy.parsing.render import Renderer

INPUT = """{
        "columns": [
            {"name": "store_returns.customer.text_id"},
            {"name": "total_return_amount", 
                "calculation": {
                    "operator": "SUM",
                    "arguments": [{"name": "store_returns.return_amount"}]
                }
            }
        ],
        "filtering": {
            "root": {
                "values": [
                    {
                        "operator": ">",
                        "left": {
                            "name": "total_return_amount"
                        },
                        "right": {
                            "name": "1.2_times_avg_return_per_store",
                            "calculation": {
                                "operator": "MULTIPLY",
                                "arguments": [
                                    {"value": "1.2", "type": "float"},
                                    {
                                        "name": "avg_return_per_store",
                                        "calculation": {
                                            "operator": "AVG",
                                            "arguments": [{"name": "store_returns.return_amount"}],
                                            "over": [{"name": "store_returns.store.id"}]
                                        }
                                    }
                                ]
                            }
                        }
                    },
                    {
                        "operator": "=",
                        "left": {
                            "name": "store_returns.return_date.date.year"
                        },
                        "right": {
                            "value": "2000",
                            "type": "int"
                        }
                    },
                    {
                        "operator": "=",
                        "left": {
                            "name": "store_returns.store.state"
                        },
                        "right": {
                            "value": "TN",
                            "type": "string"
                        }
                    }
                ],
                "boolean": "and"
            }
        },
        "order": [
            {
                "column_name": "store_returns.customer.text_id",
                "order": "asc"
            }
        ],
        "limit": 100
    }"""


def test_ir_parsing():
    # we'll use the tpc_ds model for most of these
    env = Environment(working_path=Path(__file__).parent.parent / "tpc_ds_duckdb") 
    env.add_file_import("store_returns", "store_returns")
    
    parsed = InitialParseResponseV2.model_validate_json(INPUT)

    filtering =  parse_filtering(parsed.filtering, env)
    
    where, having = generate_having_and_where(filtering)

    assert str(having.conditional) == 'local.total_return_amount<local.total_return_amount> > local.1.2_times_avg_return_per_store<store_returns.store.id>', str(having)

    response:SelectStatement = ir_to_query(intermediate_results = parsed, input_environment = env, )


    # assert response.having_clause is not None

    query = Renderer().to_string(response)

    print(query)
    
    assert 1 == 0


OUTPUT_TWO = {
    "action": "Final Answer",
    "action_input": {
        "columns": [
            {
                "name": "store_returns.customer.text_id"
            },
            {
                "name": "total_returns",
                "calculation": {
                    "operator": "SUM",
                    "arguments": [
                        {
                            "name": "store_returns.return_amount"
                        }
                    ]
                }
            },
            {
                "name": "avg_returns_by_store",
                "calculation": {
                    "operator": "AVG",
                    "arguments": [
                        {
                            "name": "total_returns"
                        }
                    ],
                    "over": [
                        {
                            "name": "store_returns.store.id"
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
                            "name": "store_returns.return_date.year"
                        },
                        "right": {
                            "value": "2000",
                            "type": "integer"
                        }
                    },
                    {
                        "operator": "=",
                        "left": {
                            "name": "store_returns.store.state"
                        },
                        "right": {
                            "value": "TN",
                            "type": "string"
                        }
                    },
                    {
                        "operator": ">",
                        "left": {
                            "name": "total_returns"
                        },
                        "right": {
                            "name": "avg_returns_by_store",
                            "calculation": {
                                "operator": "MULTIPLY",
                                "arguments": [
                                    {
                                        "value": "1.2",
                                        "type": "float"
                                    },
                                    {
                                        "name": "avg_returns_by_store"
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
                "column_name": "store_returns.customer.text_id",
                "order": "asc"
            }
        ],
        "limit": 100
    },
    "reasoning": "The query calculates the total returns for each customer and store, then calculates the average returns by store. It filters the results to the state of Tennessee in the year 2000 and only shows rows where the total returns for a customer are more than 1.2 times the average returns by store. Results are sorted by customer text_id in ascending order."
}

def test_output_two():
    env = Environment(working_path=Path(__file__).parent.parent / "tpc_ds_duckdb") 
    env.add_file_import("store_returns", "store_returns")
    
    parsed = InitialParseResponseV2.model_validate_json(OUTPUT_TWO)

    filtering =  parse_filtering(parsed.filtering, env)
    
    where, having = generate_having_and_where(filtering)

    assert str(having.conditional) == 'local.total_return_amount<local.total_return_amount> > local.1.2_times_avg_return_per_store<store_returns.store.id>', str(having)

    response:SelectStatement = ir_to_query(intermediate_results = parsed, input_environment = env, )


    # assert response.having_clause is not None

    query = Renderer().to_string(response)

    print(query)
    
    assert 1 == 0