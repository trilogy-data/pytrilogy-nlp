from pathlib import Path

from trilogy import Environment
from trilogy.core.models import SelectStatement

from trilogy_nlp.llm_interface.models import InitialParseResponseV2
from trilogy_nlp.main import ir_to_query

INPUT = """{
        "output_columns": [
            {"name": "store_returns.customer.text_id"},
            {"name": "total_return_amount", 
                "calculation": {
                    "operator": "SUM",
                    "arguments": [{"name": "store_returns.return_amount"}]
                }
            },
            {
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
                            "name": "1.2_times_avg_return_per_store"
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

    query: SelectStatement = ir_to_query(parsed, input_environment=env)

    assert (
        str(query.having_clause.conditional)
        == "local.total_return_amount<store_returns.item.id,store_returns.ticket_number> > local.1.2_times_avg_return_per_store<store_returns.store.id>"
    ), str(query.having_clause)

    _ = ir_to_query(
        intermediate_results=parsed,
        input_environment=env,
    )
