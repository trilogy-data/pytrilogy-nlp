from typing import Optional
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
)
from typing import List
from trilogy.core.query_processor import process_query
from langchain.tools import Tool, StructuredTool
from datetime import datetime
import json
from pydantic import BaseModel, field_validator
from trilogy.core.enums import ComparisonOperator, Ordering, Purpose, BooleanOperator
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from trilogy.parsing.common import arbitrary_to_concept
from trilogy.core.models import Function
from trilogy.core.enums import FunctionType, Purpose
from pydantic import BaseModel
from typing import List, Optional, Dict


langchain_chat_kwargs = {
    "temperature": 0,
    "max_tokens": 4000,
    "verbose": True,
}
chat_openai_model_kwargs = {
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": -1,
}


class FilterResultV2(BaseModel):
    """The result of the filter prompt"""

    column: str
    values: list[str | int | float | bool]
    operator: ComparisonOperator

    @field_validator("values", mode="plain")
    @classmethod
    def values_validation(cls, v):
        if isinstance(v, list):
            return v
        return [v]


class OrderResultV2(BaseModel):
    """The result of the order prompt"""

    column: str
    order: Ordering


class Column(BaseModel):
    name: str
    function: Optional[str] = None

    def __hash__(self):
        return hash(self.name + str(self.function))


def create_column(c: Column, environment: Environment):
    if not c.function:
        return environment.concepts[c.name]

    root = environment.concepts[c.name]
    return arbitrary_to_concept(
        Function(
            operator=FunctionType(c.function),
            output_datatype=root.datatype,
            output_purpose=Purpose.METRIC,
            arguments=[environment.concepts[c.name]],
        ),
        namespace="local",
        name=f"{c.name}_{c.function}",
    )


class InitialParseResponseV2(BaseModel):
    """The result of the initial parse"""

    columns: list[Column]
    limit: Optional[int] = 100
    order: Optional[list[OrderResultV2]] = None
    filtering: Optional[list[FilterResultV2]] = None

    @property
    def selection(self) -> list[str]:
        filtering = [f.column for f in self.filtering] if self.filtering else []
        order = [x.column for x in self.order] if self.order else []
        return list(set(self.columns + filtering + order))

    @field_validator("filtering", mode="plain")
    @classmethod
    def filtering_validation(cls, v):
        if isinstance(v, dict):
            return [FilterResultV2.model_validate(v)]
        return [FilterResultV2.model_validate(x) for x in v]

    @field_validator("order", mode="plain")
    @classmethod
    def order_validation(cls, v):
        if isinstance(v, dict):
            return [OrderResultV2.model_validate(v)]
        return [OrderResultV2.model_validate(x) for x in v]


def parse_order(
    input_concepts: List[Concept], ordering: List[OrderResultV2] | None
) -> OrderBy:
    default_order = [
        OrderItem(expr=c, order=Ordering.DESCENDING)
        for c in input_concepts
        if c.purpose == Purpose.METRIC
    ]
    if not ordering:
        return OrderBy(items=default_order)
    final = []
    for order in ordering:
        concept = [x for x in input_concepts if x.address == order.column][0]
        final.append(OrderItem(expr=concept, order=order.order))
    return OrderBy(items=final)


def parse_filter(
    input_concepts: List[Concept], input: FilterResultV2
) -> Comparison | None:
    try:
        concept = [
            x
            for x in input_concepts
            if x.address == input.column or x.name == input.column
        ][0]
    except IndexError:
        raise ValueError(
            f"Invalid filtering response {input}, could not be matched to concepts."
        )
    return Comparison(
        left=concept,
        right=input.values[0] if len(input.values) == 1 else input.values,
        operator=input.operator,
    )


def parse_filtering(
    input_concepts: List[Concept], filtering: List[FilterResultV2]
) -> WhereClause | None:
    base = []
    for item in filtering:
        parsed = parse_filter(input_concepts, item)
        if parsed:
            base.append(parsed)
    if not base:
        return None
    if len(base) == 1:
        return WhereClause(conditional=base[0])
    left: Conditional | Comparison = base.pop()
    while base:
        right = base.pop()
        new = Conditional(left=left, right=right, operator=BooleanOperator.AND)
        left = new
    return WhereClause(conditional=left)


def run_query_save_results(executor, query: str):
    executor.execute_query(query)


def validate_query(query: dict, environment: Environment):
    parsed = InitialParseResponseV2.model_validate(query)
    for x in parsed.columns:
        if x.name not in environment.concepts:
            return {"status": "invalid", "error": f"{x.name} in fields not in concepts; check that you are using a fully qualified path"}
        if x.function and x.function.lower() not in FunctionType.__members__:
            return {"status": "invalid", "error": f"{x.function} in function not in FunctionType; check that you are using a valid option from {FunctionType.__members__}"}
    if parsed.order:
        for y in parsed.order:
            if y.column not in environment.concepts:
                return {
                    "status": "invalid",
                    "error": f"{y.column} in order not in concepts; check that you are using a fully qualified path",
                }
    if parsed.filtering:
        for z in parsed.filtering:
            if z.column not in environment.concepts:
                return {
                    "status": "invalid",
                    "error": f"{z.column} in filtering not in concepts; check that you are using a fully qualified path",
                }
    return {"status": "valid"}


def get_model_description(query: str, environment: Environment):
    """
    Get the description of the dataset.
    """
    datasources = ", ".join(
        set([x.identifier for x in environment.datasources.values()])
    )
    return json.dumps(
        {
            "description": f"database contains information about: {datasources}. No more specific information is available. Use the fields tool to get more specific information."
        }
    )


def get_today_date(query: str) -> str:
    """
    Useful to get the date of today.
    """
    # Getting today's date in string format
    today_date_string = datetime.now().strftime("%Y-%m-%d")
    return today_date_string


def concept_to_string(concept: Concept):
    return concept.address


def get_fields(environment: Environment, search: str, *args, **kwargs) -> str:
    return json.dumps(
        {
            "fields": {
                concept_to_string(x): x.metadata.description
                for x in environment.concepts.values()
                if search.lower() in x.address.lower()
                and "__preql_internal" not in x.address
            }
        }
    )


class Filter(BaseModel):
    column: str
    values: List
    operator: str


class ValidateResponseInput(BaseModel):
    columns: List[Column]
    filtering: Optional[list[FilterResultV2]] = None
    order: Optional[list[OrderResultV2]] = None
    limit: Optional[int] = None



def validate_response(
    environment: Environment,
    columns: List[Column],
    filtering: List[Filter] = None,
    order: Optional[list[OrderResultV2]] = None,
    limit: int = None,
):
    # Your validation logic here, e.g., checking if all values are valid
    # This is a dummy implementation for illustration
    base = {"columns": columns}
    if filtering:
        base["filtering"] = filtering
    if order:
        base["order"] = order
    if limit:
        base["limit"] = limit
    return validate_query(
        base,
        environment=environment,
    )


def sql_agent_tools(environment):
    def validate_response_wrapper(**kwargs):
        return validate_response(
            environment=environment,
            **kwargs,
        )

    tools = [
        Tool.from_function(
            func=lambda x: get_model_description(x, environment),
            name="get_database_description",
            description="""
           Describe the database and general groupings of fields available.""",
        ),
        StructuredTool(
            name="validate_response",
            description="""
            Check that a response is formatted properly and accurate before your final answer. Always call this!
            """,
            func=validate_response_wrapper,
            args_schema=ValidateResponseInput,
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
            The list of fields that can be selected. When looking for an aggregate (count, etc) prefer concepts with that string in the name. You must use the full name, including all namespaces.
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


def parse_query(
    input_text: str,
    input_environment: Environment,
    llm: BaseLanguageModel,
    debug: bool = False,
    log_info: bool = True,
):
    system = """Thought Process: You are a data analyst assistant. Your job is to get questions from 
    the analyst and tell them how to write a 
    SQL query to answer them in a step by step fashion. 

    You can get information on the columns available and cannot create any new ones;
    do not worry about tables, the analyst will join them.

    If you need additional contextual information that would not be in the database, use the wikimedia tool to get it.

    Do your best to get to the most complete answer possible using all tools. 

    Your goal will be to create a summary of steps in JSON format for your analyst.
    The final information should be a VALID JSON blob with the following keys and values followed by a stopword: <EOD>:
    - columns: a list of columns
    -- name: the name of the column to return or to pass into a function
    -- function (optional): the function to apply to the column, one of [SUM, AVG, COUNT, MAX, MIN]. DO not include the function in the name. 
    - limit: a number of records to limit the results to, -1 if none specified
    - order: a list of columns to order the results by, with the option to specify ascending or descending
    -- column: a column name to order by
    -- order: the direction of ordering, "asc" or "desc"
    - filtering: a list of all objects to filter the results on, where each object has the following keys:
    -- column: a column to filter on
    -- values: the value the column is filtered to
    -- operator: the comparison operator, one of "=", "in", "<", ">", "<=", "like", or ">=". A range, or between, should be expressed as two inequalities. 
    


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


    An example series:

    Question: Get orders by date and revenue for zip code 10245 in 2000
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
        "columns": [
            {{"name": "store.order.id"}},
            {{"name": "store.order.revenue", "function": "SUM"}}
        ],
        "filtering": [
            {{
                "column": "store.order.customer.zip_code",
                "values": [10245],
                "operator": "="
            }},
            {{
                "column": "store.order.date.year",
                "values": [2000],
                "operator": "="
            }}
        ],
        "limit": 100
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
    wiki = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())  # type: ignore
    wiki.description = (
        "Look up information on a specific string from Wikipedia. Use to get context"
    )
    tools = sql_agent_tools(input_environment) + [wiki]
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
        intermediate_results = InitialParseResponseV2.model_validate_json(output)
    elif isinstance(output, dict):
        intermediate_results = InitialParseResponseV2.model_validate(output)
    else:
        raise ValueError("Unable to parse LLM response")

    def create_column(c: Column, environment: Environment):
        if not c.function:
            return environment.concepts[c.name]

        from trilogy.parsing.common import arbitrary_to_concept
        from trilogy.core.models import Function
        from trilogy.core.enums import FunctionType, Purpose

        root = environment.concepts[c.name]
        new = arbitrary_to_concept(
            Function(
                operator=FunctionType(c.function.lower()),
                output_datatype=root.datatype,
                output_purpose=Purpose.METRIC,
                arguments=[environment.concepts[c.name]],
            ),
            namespace="local",
            name=f"{c.name}_{c.function}".lower(),
        )
        environment.add_concept(new)
        return new

    selection = [
        create_column(x, input_environment) for x in intermediate_results.columns
    ]
    order = parse_order(selection, intermediate_results.order or [])

    filtering = (
        parse_filtering(
            list(input_environment.concepts.values()), intermediate_results.filtering
        )
        if intermediate_results.filtering
        else None
    )

    if debug:
        print("Concepts found")
        for c in intermediate_results.columns:
            print(c)
        if intermediate_results.order:
            print("Ordering")
            for o in intermediate_results.order:
                print(o)
    query = SelectStatement(
        selection=selection,
        limit=safe_limit(intermediate_results.limit),
        order_by=order,
        where_clause=filtering,
    )
    return query


def build_query(
    input_text: str,
    input_environment: Environment,
    llm: BaseLanguageModel,
    debug: bool = False,
    log_info: bool = True,
) -> ProcessedQuery:
    query = parse_query(
        input_text,
        input_environment,
        debug=debug,
        llm=llm,
        log_info=log_info,
    )
    return process_query(statement=query, environment=input_environment)
