from trilogy.core.models import (
    Concept,
    Environment,
)
from langchain.tools import Tool, StructuredTool
import json
from trilogy_nlp.tools import get_today_date
from trilogy_nlp.exceptions import ValidationPassedException

from trilogy_nlp.llm_interface.validation import (
    validate_response,
    ValidateResponseInterface,
    VALID_STATUS,
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
                        "description": x.metadata.description
                        + f" A {x.datatype} field.",
                    }
                    if x.metadata.description
                    else {
                        "name": concept_to_string(x),
                        "description": f"A {x.datatype} field.",
                    }
                )
                for x in environment.concepts.values()
                if not x.address.startswith("_") and not x.address.endswith(".count")
            ]
        }
    )


def get_help(x):
    raise SyntaxError(x)


def submit_response(environment, prompt, **kwargs):
    response, ir = validate_response(
        environment=environment,
        prompt=prompt,
        **kwargs,
    )
    if response["status"] == VALID_STATUS:
        raise ValidationPassedException(ir=ir)
    return response


def sql_agent_tools(environment, prompt: str):
    def validate_response_wrapper(**kwargs):
        response, ir = validate_response(
            environment=environment,
            prompt=prompt,
            **kwargs,
        )
        return response

    def submit_wrapper(**kwargs):
        return submit_response(
            environment=environment,
            prompt=prompt,
            **kwargs,
        )

    tools = [
        StructuredTool(
            name="validate_response",
            description="""
            Check that a response is formatted properly and accurate before your final answer. Always call this with the complete final response before reporting a Final Answer!
            If the response is not correct, the "valid" argument will be false and it will return an array of errors. If it is correct, it will return "true" for the valid argument.
            You must fix all errors returned by validation before submitting. If you are not sure what the error means, ask for help using the get_validation_help tool.
            """,
            func=validate_response_wrapper,
            args_schema=ValidateResponseInterface,
            handle_tool_error=True,
        ),
        StructuredTool(
            name="submit_answer",
            description="""
            Submit your final answer. It will be validated, so this function may return errors to correct, similar to the validate_response function. If it passes, the answer will be submitted.
            """,
            func=submit_wrapper,
            args_schema=ValidateResponseInterface,
        ),
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
