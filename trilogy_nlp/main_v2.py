from langchain.agents import create_structured_chat_agent, AgentExecutor
from trilogy_nlp.main import safe_limit
from trilogy.core.models import (
    Environment,
    ProcessedQuery,
    SelectStatement,
    Comparison,
    Conditional,
    WhereClause,
    SelectItem,
    HavingClause,
)
from trilogy.core.query_processor import process_query
from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from trilogy_nlp.tools import get_wiki_tool
from trilogy.core.processing.utility import (
    is_scalar_condition,
    decompose_condition,
)

from trilogy_nlp.llm_interface.parsing import (
    parse_filtering,
    parse_order,
    create_column,
)
from trilogy_nlp.llm_interface.models import InitialParseResponseV2
from trilogy_nlp.llm_interface.tools import sql_agent_tools

# from trilogy.core.constants import


def llm_loop(
    input_text: str, input_environment: Environment, llm, retries: int = 2
) -> InitialParseResponseV2:
    attempts = 0
    exceptions = []
    while attempts < retries:
        try:
            return _llm_loop(input_text, input_environment=input_environment, llm=llm)
        except Exception as e:
            # logger.info("Failed attempted llm loop")
            exceptions.append(e)
            raise e
        attempts += 1
    raise ValueError(
        f"Was unable to process query, exceptions {[str(x) for x in exceptions]}"
    )


def _llm_loop(
    input_text: str,
    input_environment: Environment,
    llm: BaseLanguageModel,
    additional_context: str | None = None,
) -> InitialParseResponseV2:
    system = """You are a data analyst assistant. Your job is to get questions from 
    the analyst and tell them how to write a 
    SQL query to answer them in a step by step fashion. 

    You can get information on the fields available and can use functions to derive new ones.
    do not worry about tables, the analyst will join them.

    Your goal will be to create a final output in JSON format for your analyst. Do your best to get to the most complete answer possible using all tools. 

    A key structure used in your responses will be a Column, a recursive json structure containing a name and an optional calculation sub-structure.
    If the Column does not have a calculation, the name must reference a name provided in the database already. 

    A Column Object is json with two fields:
    -- name: the field being referenced or a new derived name. If there is a calculation, this should always be a new derived name you came up with. That name must be unique; a calculation cannot reference an input with the same name as the output concept.
    -- calculation: An optional calculation object. Only include a calculation if you need to create a new column because there is not a good match from the existing field list. 

    A Literal Object is json with these fields:
    -- value: the literal value ('1', 'abc', etc), expressed as a string
    -- type: the type of the value ('float', 'string', 'int', 'bool'), expressed as a string

    A Calculation Object is json with three fields:
    -- arguments: a list of Column or Literal objects
    -- operator: a function to call with those arguments. [SUM, AVG, COUNT, MAX, MIN, etc], expressed as a string. A calculation object MUST have an operator. This cannot be a comparison operator.
    -- over: an optional list of Column objects used when a calculation needs to happen over other columns (sum of revenue by state, for example)

    A Comparison object is JSON with three fields:
    -- left: A Column or Literal object
    -- right: A Column or Literal object
    -- operator: the comparison operator, one of "=", "in", "<", ">", "<=", "like", or ">=". Use two comparisons to represent a between

    A ConditionGroup object is JSON with two fields used to create boolean filtering constructs. You can nest ConditionGroups to create complex filtering conditions.
    -- values: a list if Comparison Objects or ConditionGroups
    -- boolean: 'and' or 'or' (lowercase, no quotes)

    The final information should be a VALID JSON blob with the following keys and values followed by a stopword: <EOD>:
    - columns: a list of columns as Column objects
    - limit: a number of records to limit the results to, -1 if none specified
    - order: a list of columns to order the results by, with the option to specify ascending or descending
        -- column_name: a column name to order by; must reference value in columns
        -- order: the direction of ordering, "asc" or "desc"
    - filtering: an object with a single argument
        -- root: a ConditionGroup object

    
    You should always call the the validate_response tool on what you think is the final answer before returning the "Final Answer" action.
    You have access to the following tools:

    {tools}

    Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input). 
    

    You will get valuable information from using tools before producing your final answer.

    Use as many tools as needed, and always validate, before producing the "Final Answer" action.

    Valid "action" values: any of {tool_names} and, for your last result "Final Answer". 
    Only return "Final Answer" when you are done with all work. Never set the action to 
    "Final Answer" before you are done, and never set the action to final answer without some columns returned.

    You should always call the the validate_response tool with your candidate answer before declaring it the final answer. Getting this right is crucial to the analyst keeping their job.

    Provide only ONE action per $JSON_BLOB, as shown:

    ```
    {{
        "action": $TOOL_NAME,
        "action_input": $INPUT,
        "reasoning": "Your thinking"
    }}
    ```

    Action input is in JSON format, not as a JSON string blob (No escaping!)

    Follow this format in responses:

    Question: input question to answer
    Action:
    ```
    $JSON_BLOB
    ```
    <action result>
    ... (repeat Thought/Action/Observation N times)

    An example series:

    Question: Get the total revenue by order and customer id for stores in the zip code 1025 in the year 2000 where the total sales price of the items in the order was more than 100 dollars
    Action:
    ```
    {{
        "action": "get_database_description",
        "action_input": ""
        "reasoning": "I should get the available fields in the database."
    }}
    <some result>
    Action:
    ```
    {{
        "action": "validate_response",
        "action_input": {{
        "columns": [
            {{"name": "store.order.id"}},
            {{"name": "store.order.customer.id"}},
            {{"name": "revenue_sum", 
                "calculation": {{"operator":"SUM", "arguments": [{{ "name": "store.order.revenue"}}]
            }} }}
        ],
        "filtering": {{
            "root": {{
                "values": [{{
                    "left": {{"name": "store.zip_code"}},
                    "right": {{"value":"10245", "type":"integer"}},
                    "operator": "="
                }},
                {{
                    "left": {{"name": "store.order.date.year" }},
                    "right": {{"value":"2000", "type":"integer"}},
                    "operator": "="
                }},
                {{
                    "left": {{"name": "sales_price_sum_by_store", 
                    "calculation": {{"operator":"SUM", 
                        "arguments": [{{ "name": "item.sales_price"}}],
                        "over": [{{ "name": "store.order.id"}}]
                        }}
                    }},
                    "right": {{"value":"100.0", "type":"float"}},
                    "operator": ">"
                    
                }}
                ],
                "boolean": "and"
        }},
        "order": [
            {{"column_name": "revenue_sum", "order": "desc"}}
        ],
        "limit": 100
        }}, 
        "reasoning": "I can return order id, customer id, and the total order revenue. Order Id and customer Id are scalar values, while the total order revenue will require a calculation. I can filter to the zip code and the year, and then restrict to where the sales price over the order id is more than 100, which will require a calculation. Before submitting my answer, I need to validate my answer."
    }}
    }}

    Nested Column objects with calculations can create complex derivations. This can be useful for filtering. 

    Make sure to give the output concept a unique, descriptive name. 

    You don't need to use an over clause for a top level calculation if it's over the other columns you've selected.

    Ex: for customer, total_revenue, just sum revenue. 

    For example, to create a filter condition for "countries with an average monthly rainfall of 2x the average on their continent", the filtering clause might look like.

    {{
            "root": {{
                "values": [
                    {{
                    "left": {{
                        "name": "avg_monthly_rainfall",
                        "calculation": {{
                            "operator": "AVG",
                            "arguments": [{{"name": "country.monthly_rainfall"}}],
                            "over": [{{"name": "country.name"}}]
                                }}
                            }},
                    "right":  {{
                        "name": "continent_avg_monthly_rainfall_2x",
                        "calculation": {{
                            "operator": "MULTIPLY",
                            "arguments": [
                                    {{"value":"2", "type":"integer"}}, 
                                    {{"name": "continent_avg_monthly_rainfall",
                                        "calculation" : {{
                                            "operator": "AVG",
                                            "arguments": ["country.monthly_rainfall"],
                                            "over": [{{"name": "country.continent"}}]
                                        }}
                                    }}],

                                }}
                            }},
                    "operator": ">"
                }},
                ],
                "boolean": "and"
        }}
    }}
    To filter over a calculation with a scalar, multiple that in 
    Once you have used any tools (listed below) as needed, you will produce your final result in this format. After producing your
    final answer, you cannot take any more steps.

    A final answer would look like this:
    {{
        "action": "Final Answer",
        "action_input": <VALID_JSON_SPEC_DEFINED_ABOVE>,
        "reasoning": "<description of your logic>"
    }}

    You will only see the last few steps.

    Begin! Reminder to ALWAYS respond with a valid json blob for an action. Always use tools. The conversation consistes of messages of 'Action:```$JSON_BLOB```' and then a response to react to with another action."""

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

    tools = sql_agent_tools(input_environment, input_text) + [get_wiki_tool()]
    chat_agent = create_structured_chat_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(
        agent=chat_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,  # type: ignore,
        trim_intermediate_steps=3,
    )
    attempts = 0
    if additional_context:
        input_text += additional_context
    while attempts < 2:
        result = agent_executor.invoke({"input": input_text})
        output = result["output"]
        print("OUTPUT WAS")
        print(output)
        try:
            if isinstance(output, str):
                return InitialParseResponseV2.model_validate_json(output)
            elif isinstance(output, dict):
                return InitialParseResponseV2.model_validate(output)
            else:
                raise ValueError("Unable to parse LLM response")
        except ValidationError as e:
            # Here, `validation_error.errors()` will have the full info
            for x in e.errors():
                print(x)
            # inject in new context on failed answer
            raw_error = str(e)
            # TODO: better pydantic error parsing
            if "filtering.root." in raw_error:
                raw_error = (
                    "Syntax error in your filtering clause. Confirm it matches the required format. Comparisons need a left and right, etc, and Columns and Literal formats are very specific. Full error:"
                    + raw_error
                )
            input_text += f"IMPORTANT: this is your second attempt - your last attempt errored parsing your final answer: {raw_error}. Remember to use the validation tool to check your work!"
        attempts += 1
    raise ValueError(f"Unable to get parseable response after {attempts} attempts")


def parse_query(
    input_text: str,
    input_environment: Environment,
    llm: BaseLanguageModel,
    debug: bool = False,
    log_info: bool = True,
):
    intermediate_results = llm_loop(input_text, input_environment, llm=llm)

    selection = [
        create_column(x, input_environment) for x in intermediate_results.columns
    ]
    order = parse_order(selection, intermediate_results.order or [])

    filtering = (
        parse_filtering(intermediate_results.filtering, input_environment)[0]
        if intermediate_results.filtering
        else None
    )

    if debug:
        print("Concepts found")
        for c in intermediate_results.columns:
            print(c)
        if intermediate_results.filtering:
            print("filtering")
            print(str(intermediate_results.filtering))
        if intermediate_results.order:
            print("Ordering")
            for o in intermediate_results.order:
                print(o)
    where: Conditional | Comparison | None = None
    having: Conditional | Comparison | None = None
    materialized = {x.output for x in selection}
    if filtering:
        if is_scalar_condition(filtering.conditional, materialized=materialized):
            where = filtering.conditional
        else:
            components = decompose_condition(filtering.conditional)
            for x in components:
                if is_scalar_condition(x, materialized=materialized):
                    where = where + x if where else x
                else:
                    having = having + x if having else x
    print(is_scalar_condition(filtering.conditional))
    print(where)
    print(having)

    query = SelectStatement(
        selection=[SelectItem(content=x) for x in selection],
        limit=safe_limit(intermediate_results.limit),
        order_by=order,
        where_clause=WhereClause(conditional=where) if where else None,
        having_clause=HavingClause(conditional=having) if having else None,
    )
    replacements = {}
    if having:
        for x in having.concept_arguments:
            # rewrite these with our description
            if x.metadata.description == MAGIC_GENAI_DESCRIPTION:
                new = x.with_select_context(grain=query.grain)
                input_environment.add_concept(new, force=True)
                replacements[new.address] = new

    # remap selection
    selection = [SelectItem(content=replacements.get(x.address, x)) for x in selection]
    query.selection = selection

    from trilogy.parsing.render import Renderer

    print("RENDERED QUERY")
    print(Renderer().to_string(query))
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
