from trilogy.core.models import (
    Concept,
    Environment,
)
from langchain.tools import Tool, StructuredTool
import json
from trilogy_nlp.tools import get_today_date


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
            If the response is not correct, the "valid" argument will be false and it will return an array of errors. If it is correct, it will return "true" for the valid argument.
            This validation checks for syntactic issues, but you will need to check for semantic issues yourself.
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
            Returns array of json objects containing the names of actual fields that can be referenced, with a description if it exists. Fields always need to be referenced by exact name. Queries operate on fields only, by exact name.
            """,
        ),
        Tool.from_function(
            func=get_today_date,
            name="get_today_date",
            description="""
            Use to get the date of today.
            """,
        ),
    ]
    return tools
