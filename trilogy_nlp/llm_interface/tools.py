from typing import Optional, Union
from langchain.agents import create_structured_chat_agent, AgentExecutor
from trilogy_nlp.main import safe_limit
from trilogy.core.models import (
    Concept,
    Environment,
    ProcessedQuery,
    SelectStatement,
    Comparison,
    Conditional,
    OrderBy,
    OrderItem,
    WhereClause,
    AggregateWrapper,
    Function,
    Metadata,
    SelectItem,
    HavingClause,
)
from typing import List
from trilogy.core.query_processor import process_query
from langchain.tools import Tool, StructuredTool
import json
from pydantic import BaseModel, field_validator, ValidationError
from trilogy.core.enums import ComparisonOperator, Ordering, Purpose, BooleanOperator
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from trilogy_nlp.tools import get_wiki_tool, get_today_date
from trilogy.parsing.common import arbitrary_to_concept
from trilogy.core.models import DataType
from langchain_core.tools import ToolException
from trilogy.core.exceptions import UndefinedConceptException
from trilogy.core.processing.utility import (
    is_scalar_condition,
    decompose_condition,
    sort_select_output,
)

from trilogy_nlp.llm_interface.parsing import (
    parse_filtering,
    parse_order,
    parse_col,
    create_column,
)
from trilogy_nlp.llm_interface.models import (
    InitialParseResponseV2,
    Column,
    NLPComparisonOperator,
    NLPConditions,
    NLPComparisonGroup,
    FilterResultV2,
    OrderResultV2,
)
from trilogy_nlp.llm_interface.tools import sql_agent_tools
from trilogy_nlp.llm_interface.examples import FILTERING_EXAMPLE
from trilogy_nlp.llm_interface.constants import COMPLICATED_FUNCTIONS

# from trilogy.core.constants import
from trilogy.core.enums import (
    FunctionType,
    FunctionClass,
    InfiniteFunctionArgs,
)
from trilogy.parsing.common import arg_to_datatype
from enum import Enum
from trilogy_nlp.llm_interface.examples import COLUMN_DESCRIPTION
from trilogy_nlp.llm_interface.validation import (
    validate_response,
    ValidateResponseInterface,
)


def get_model_description(query: str, environment: Environment):
    """
    Get the description of the dataset.
    """
    datasources = [{"name": x.identifier} for x in environment.datasources.values()]
    return json.dumps(
        {
            "description": f"Contains the following datasources: {datasources}. Use the get_fields tool with a name to get more specific information about any given one."
        }
    )


def concept_to_string(concept: Concept) -> str:
    return concept.address


def get_fields(environment: Environment, search: str, *args, **kwargs) -> str:
    return json.dumps(
        {
            "fields": [
                (
                    {
                        "name": concept_to_string(x),
                        "description": x.metadata.description,
                    }
                    if x.metadata.description
                    else {
                        "name": concept_to_string(x),
                    }
                )
                for x in environment.concepts.values()
                if "__preql_internal" not in x.address
                and not x.address.endswith(".count")
            ]
        }
    )
    if search in environment.datasources:
        return json.dumps(
            {
                "fields": [
                    (
                        {
                            "name": concept_to_string(x),
                            "description": x.metadata.description,
                        }
                        if x.metadata.description
                        else {
                            "name": concept_to_string(x),
                        }
                    )
                    for x in environment.datasources[search].output_concepts
                    if "__preql_internal" not in x.address
                    and not x.address.endswith(".count")
                ]
            }
        )
    return f"Invalid search; valid options {environment.datasources.keys()}"


def sql_agent_tools(environment, prompt: str):
    def validate_response_wrapper(**kwargs):
        return validate_response(
            environment=environment,
            prompt=prompt,
            **kwargs,
        )

    tools = [
        # Tool.from_function(
        #     func=lambda x: get_model_description(x, environment),
        #     name="get_database_description",
        #     description="""
        #    Share a directory of folders to look in for exact fields. Takes no arguments. These groupings are never referenced directly.""",
        #    handle_tool_error='Call with empty string "", not {{}}.'
        # ),
        StructuredTool(
            name="validate_response",
            description="""
            Check that a response is formatted properly and accurate before your final answer. Always call this with the complete final response before reporting a Final Answer!
            """,
            func=validate_response_wrapper,
            args_schema=ValidateResponseInterface,
            handle_tool_error=True,
        ),
        # Tool.from_function(
        #     func=lambda x: validate_query(x, environment),
        #     name="validate_response",
        #     description="""
        #     Check that a pure json string response is formatted properly and accurate before your final answer. Always call this!
        #     """,
        # ),
        Tool.from_function(
            func=lambda x: get_fields(environment, x),
            name="get_fields",
            description="""
            Array of json objects containing the names of actual fields that can be referenced, with a description if it exists. Fields always need to be referenced by exact name. Queries operate on fields only, by exact name.
            """,
        ),
        Tool.from_function(
            func=get_today_date,
            name="get_today_date",
            description="""
            Useful to get the date of today.
            """,
        ),
    ]
    return tools
