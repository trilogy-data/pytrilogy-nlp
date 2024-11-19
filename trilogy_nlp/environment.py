from pathlib import Path
from langchain_core.language_models import BaseLanguageModel
from trilogy import Environment
from pydantic import BaseModel
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.tools import Tool, StructuredTool
from trilogy.core.enums import Purpose
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from trilogy_nlp.tools import get_wiki_tool
from trilogy.core.enums import Modifier
from difflib import get_close_matches
from trilogy.core.exceptions import InvalidSyntaxException
from trilogy_nlp.constants import logger
from trilogy_nlp.prompts_v2.query_environment import BASE_1

# from trilogy.core.exceptions import Import


class AddImportResponse(BaseModel):
    """result of initial parse request"""

    namespaces: list[str]
    reasoning: str | None = None


def get_environment_possible_imports(env: Environment) -> list[str]:
    raw = Path(env.working_path).glob("*.preql")
    return [x.stem for x in raw if "query" not in x.stem]


def get_environment_detailed_values(env: Environment, input: str):
    new = Environment(working_path=env.working_path)
    try:
        new.parse(f"import {input} as {input};")

    except (ImportError, InvalidSyntaxException):
        suggestions = get_close_matches(input, get_environment_possible_imports(env))
        base = "This is an invalid database. Refer back to the list from the get_database_list tool. Names must match exactly."
        if suggestions:
            base += f" Did you mean {suggestions[0]}?"
        return base
    # TODO: does including a description work better?
    return {
        k.split(".", 1)[1]
        for k, v in new.concepts.items()
        # skipp hidden values
        if not v.name.startswith("_") and not k.endswith(".count")
    }


def validate_response(namespaces: list[str], reasoning: str, environment: Environment):
    possible = get_environment_possible_imports(environment)
    if not all(x in possible for x in namespaces):
        return {"status": "invalid", "error": "Not all of those databases exist!"}

    return {
        "status": "valid",
    }


def environment_agent_tools(environment):
    def get_import_wrapper(*args, **kwargs):
        return get_environment_possible_imports(environment)

    def validate_response_wrapper(namespaces: list[str], reasoning: str):
        return validate_response(
            namespaces=namespaces, reasoning=reasoning, environment=environment
        )

    tools = [
        Tool.from_function(
            func=get_import_wrapper,
            name="list_databases",
            description="""
           Describe the databases you can look at. Takes empty argument string.""",
            handle_tool_error='Argument is an EMPTY STRING, rendered as "". {} is not valid. Do not call with {} ',
        ),
        Tool.from_function(
            func=lambda x: get_environment_detailed_values(environment, x),
            name="get_database_description",
            description="""
           Describe the database and general groupings of fields available. Call with a database name.""",
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
    input_text: str, input_environment: Environment, llm: BaseLanguageModel
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

    tools = environment_agent_tools(input_environment) + [get_wiki_tool()]
    chat_agent = create_structured_chat_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(
        agent=chat_agent, tools=tools, verbose=True, handle_parsing_errors=True  # type: ignore
    )

    result = agent_executor.invoke({"input": input_text})
    output = result["output"]
    print(output)
    if isinstance(output, str):
        return AddImportResponse.model_validate_json(output)
    elif isinstance(output, dict):
        return AddImportResponse.model_validate(output)
    else:
        raise ValueError(f"Unable to parse LLM response {type(output)} {output}")


def select_required_import(
    input_text: str, environment: Environment, llm: BaseLanguageModel
) -> AddImportResponse:
    exception = None
    ATTEMPTS = 3
    for attempts in range(ATTEMPTS):
        try:
            return llm_loop(input_text, environment, llm)
        except Exception as e:
            logger.error("Error in select_required_import: %s", e)
            exception = e
    if exception:
        raise exception
    raise ValueError(f"Unable to get parseable response after {ATTEMPTS} attempts; ")


def build_env_and_imports(
    input_text: str,
    working_path: Path,
    llm: BaseLanguageModel,
):
    base = Environment(working_path=working_path)
    response: AddImportResponse = select_required_import(input_text, base, llm=llm)

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
