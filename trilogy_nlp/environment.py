from difflib import get_close_matches
from pathlib import Path

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import StructuredTool, Tool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from trilogy import Environment
from trilogy.core.enums import Modifier, Purpose
from trilogy.core.exceptions import InvalidSyntaxException

from trilogy_nlp.constants import logger
from trilogy_nlp.helpers import is_relevent_concept
from trilogy_nlp.instrumentation import EventTracker
from trilogy_nlp.prompts_v2.query_environment import BASE_1


class AddImportResponse(BaseModel):
    """result of initial parse request"""

    namespaces: list[str]
    reasoning: str | None = None


def get_environment_possible_imports(env: Environment) -> str:
    raw = Path(env.working_path).glob("*.preql")
    return "Observation: " + str([x.stem for x in raw if "query" not in x.stem])


def get_environment_detailed_values(env: Environment, input: str):
    new = Environment(working_path=env.working_path)
    try:
        new.parse(f"import {input} as {input};")

    except (ImportError, InvalidSyntaxException):
        suggestions = get_close_matches(input, get_environment_possible_imports(env))
        base = "Observation: This is an invalid database. Refer back to the list from the list_databases tool. Names must match exactly."
        if suggestions:
            base += f" Did you mean {suggestions[0]}?"
        return base
    # TODO: does including a description work better?
    return "Observation:" + str(
        {
            k.split(".", 1)[1]
            for k, v in new.concepts.items()
            # skipp hidden values
            if is_relevent_concept(v)
        }
    )


def validate_response(
    namespaces: list[str],
    environment: Environment,
    reasoning: str | None = None,
    event_tracker: EventTracker | None = None,
):
    event_tracker = event_tracker or EventTracker()
    possible = get_environment_possible_imports(environment)
    if not all(x in possible for x in namespaces):
        event_tracker.count(event_tracker.etype.ENVIRONMENT_VALIDATION_FAILED)
        return {
            "status": "invalid",
            "error": f"Not all of those namespaces exist! You must pick from {possible}",
        }
    event_tracker.count(event_tracker.etype.ENVIRONMENT_VALIDATION_PASSED)
    return {
        "status": "valid",
    }


def environment_agent_tools(environment, event_tracker: EventTracker | None = None):
    def get_import_wrapper(*args, **kwargs):
        return get_environment_possible_imports(environment)

    def validate_response_wrapper(namespaces: list[str], reasoning: str | None = None):
        return validate_response(
            namespaces=namespaces,
            reasoning=reasoning,
            environment=environment,
            event_tracker=event_tracker,
        )

    tools = [
        # Tool.from_function(
        #     func=get_import_wrapper,
        #     name="list_databases",
        #     description="""
        #     Describe the databases you can look at. Takes empty argument string.""",
        #     handle_tool_error='Argument is an EMPTY STRING, rendered as "". {} is not valid. Do not call with {} ',
        # ),
        Tool.from_function(
            func=lambda x: get_environment_detailed_values(environment, x),
            name="get_namespace_description",
            description="""
           Describe the namespace and general groupings of fields available. Call with a namespace name.""",
            handle_tool_error=True,
        ),
        StructuredTool(
            name="validate_response",
            description="""
            Check that a response is formatted properly and accurate before your final answer. Always call this!
            """,
            func=validate_response_wrapper,
            args_schema=AddImportResponse,
        ),
    ]
    return tools


def llm_loop(
    input_text: str,
    input_environment: Environment,
    llm: BaseLanguageModel,
    debug: bool = False,
    event_tracker: EventTracker | None = None,
) -> AddImportResponse:
    system = BASE_1
    human = """{input}

    {agent_scratchpad}

    (reminder to respond in a JSON blob no matter what)"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", human),
        ]
    )
    prompt = prompt.partial(
        namespaces=get_environment_possible_imports(input_environment)
    )

    tools = environment_agent_tools(input_environment, event_tracker=event_tracker)
    chat_agent = create_structured_chat_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(
        agent=chat_agent, tools=tools, verbose=debug, handle_parsing_errors=True  # type: ignore
    )

    result = agent_executor.invoke({"input": input_text})
    output = result["output"]
    if "Agent stopped due to iteration limit" in output:
        raise TimeoutError(output)
    if isinstance(output, str):
        return AddImportResponse.model_validate_json(output)
    elif isinstance(output, dict):
        return AddImportResponse.model_validate(output)
    else:
        raise ValueError(f"Unable to parse LLM response {type(output)} {output}")


def select_required_import(
    input_text: str,
    environment: Environment,
    llm: BaseLanguageModel,
    debug: bool = False,
    event_tracker: EventTracker | None = None,
) -> AddImportResponse:
    exception = None
    MAX_ATTEMPTS = 3
    attempts = 0
    while attempts < MAX_ATTEMPTS:
        try:
            return llm_loop(
                input_text, environment, llm, debug=debug, event_tracker=event_tracker
            )
        except Exception as e:
            if (
                "The model produced invalid content. Consider modifying your prompt if you are seeing this error persistently"
                in str(e)
            ):
                continue
            if debug:
                raise e
            logger.error("Error in select_required_import: %s", e)
            exception = e
            attempts += 1
    if exception:
        raise exception
    raise ValueError(
        f"Unable to get parseable response after {MAX_ATTEMPTS} attempts; "
    )


def build_env_and_imports(
    input_text: str,
    working_path: Path,
    llm: BaseLanguageModel,
    debug: bool = False,
    event_tracker: EventTracker | None = None,
):
    base = Environment(working_path=working_path)
    response: AddImportResponse = select_required_import(
        input_text, base, llm=llm, debug=debug, event_tracker=event_tracker
    )

    for x in response.namespaces:
        base.parse(f"""import {x} as {x};""")

    # automatically merge concepts between the imported domains
    for k, concept in base.concepts.items():
        if not concept.namespace:
            continue
        # only merge on keys automatically
        if not concept.purpose == Purpose.KEY:
            continue
        # don't merge on date
        # TODO: be less hacky in terms of what default merges we should do
        if "date" in concept.namespace:
            continue
        if "__preql_internal" in k:
            continue
        pre_namespace_root = k.split(".", 1)[1]
        if "." not in pre_namespace_root:
            continue
        if pre_namespace_root in base.concepts:
            base.merge_concept(
                concept, base.concepts[pre_namespace_root], modifiers=[Modifier.PARTIAL]
            )

    return base
