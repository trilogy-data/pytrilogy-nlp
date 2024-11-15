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

# from trilogy.core.exceptions import Import


class AddImportResponse(BaseModel):
    """result of initial parse request"""

    namespaces: list[str]
    reasoning: str


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
        if "__preql_internal" not in k and not k.endswith(".count")
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
    system = """Thought Process: You are a data analyst assistant. Your job is to identify the datasource most relevant to answering a business question. 
    You should select the minimum number of databases that covers information. Some databases will reference others; if that's the case, 
    eg sales.customer; you do not need to explicitly include the customer database as well when looking for customer information about store sales.

    Example: if you are asked for "orders by customer", you ONLY return the orders database.

    If you need additional contextual information to understand the query that would not be in the database, use the wikimedia tool to get it.

    Sometimes you may need multiple sources:

    Example: If you are asked for "address of all employees, and how many orders they've placed" you would return employees and orders, because not all employees may have placed orders. 

    If the question suggests you use a specific datasource, eg "using ocean shipment data", assume that is sufficient.

    The output to the analyst should be a VALID JSON blob with the following keys and values followed by a stopword: <EOD>:
    - namespaces: a list of databases to use
    - reasoning: a string explaining why all are required. Remember, each additional database you include costs money!

    To start, pick a database and call the get_database_description tool on it. This will give you a description of the database and its fields. 
    Continue until you believe you have found all fields required to answer the question.

    You should always call the the validate_response tool on what you think is the final answer before returning the "Final Answer" action.
    You have access to the following tools:

    {tools}

    Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input). 
    You will get valuable information from using tools before producing your final answer.

    Use as many tools as needed before producing the "Final Answer" action.

    Valid "action" values: any of {tool_names} and, for your last result "Final Answer". 
    Only return "Final Answer" when you are done with all work. Nnever set the action to 
    "Final Answer" before you are done. 

    You should always call the the validate_response tool with your final answer before sending it to the CEO.

    Provide only ONE action per $JSON_BLOB, as shown:

    ```
    {{
        "action": $TOOL_NAME,
        "action_input": $INPUT
    }}
    ```

    Follow this format in responses:

    Question: input question to answer
    Thought: consider previous and subsequent steps
    Action:
    ```
    $JSON_BLOB
    ```
    Observation: action result
    ... (repeat Thought/Action/Observation N times)

    Action input is in JSON format, not as a JSON string blob (No escaping!)

    An example series:

    Question: Get all customers who live in US zip code 10245 and how many orders they placed in 2000
    Thought: I should get the description of of the orders dataset
    Action:
    ```
    {{
        "action": "get_database_description",
        "action_input": "orders"
    }}

    Observation: <some description>
    Thought: I should check my answer
    Action:
    ```
    {{
        "action": "validate_response",
        "action_input": "{{
        "namespaces": [
            "orders", "customers"
        ]
    }}"
    }}

    Once you have used any tools (listed below) as needed, you will produce your final result in this format. After producing your
    final answer, you cannot take any more steps.

    A final answer would look like this:
    {{
        "action": "Final Answer",
        "action_input": <VALID_JSON_SPEC_DEFINED_ABOVE>
    }}

    Begin! Reminder to ALWAYS respond with a valid json blob of an action. Always use tools. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation"""

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
    if isinstance(output, str):
        return AddImportResponse.model_validate_json(output)
    elif isinstance(output, dict):
        return AddImportResponse.model_validate(output)
    else:
        raise ValueError("Unable to parse LLM response")


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
    raise ValueError(f"Unable to get parseable response after {ATTEMPTS} attempts")


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
