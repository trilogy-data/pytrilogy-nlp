from langchain.agents import create_structured_chat_agent, AgentExecutor
from trilogy_nlp.main import safe_limit
from trilogy.core.models import (
    Environment,
    ProcessedQuery,
    SelectStatement,

    SelectItem,
    Concept,
    ConceptTransform
)
from trilogy.core.query_processor import process_query
from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from trilogy_nlp.tools import get_wiki_tool

from trilogy_nlp.llm_interface.parsing import (
    parse_filtering,
    parse_order,
    create_column,
    generate_having_and_where
)
from trilogy_nlp.llm_interface.models import InitialParseResponseV2
from trilogy_nlp.llm_interface.tools import sql_agent_tools
from trilogy_nlp.llm_interface.constants import MAGIC_GENAI_DESCRIPTION


def is_local_derived(x:Concept)->bool:
    return x.metadata.description == MAGIC_GENAI_DESCRIPTION


def llm_loop(
    input_text: str,
    input_environment: Environment,
    llm: BaseLanguageModel,
    additional_context: str | None = None,
) -> SelectStatement:
    system = """You are a data analyst assistant. Your job is to turn unstructured business questions into structured queries against a database with a known schema.

    Your goal will be to create a final output in a JSON spec defined below. Do your best to get to the most complete answer possible using all tools. 

    OUTPUT STRUCTURE:
    The key structure in your output will be a Column, a recursive json structure containing a name and an optional calculation sub-structure.
    If the Column does not have a calculation, the name must reference a name provided in the database already or previously defined by a Column object.

    A Column Object is json with two fields:
    -- name: the field being referenced or a new derived name created in a previous Column object with a calculation. If there is a calculation, this should always be a new derived name you came up with. That name must be unique; a calculation cannot reference an input with the same name as the output concept.
    -- calculation: An optional calculation object. Only include a calculation if you need to create a new column because there is not a good match from the existing field list. 

    If the user requests something that would require two levels of aggregation to express in a language such as SQL - like an "average" of a "sum" - use nested calculations or references to previously defined columns to express the concept. Ensure
    each level of calculation uses the by clause to define the level to group to. For example, to get the average customer revenue by store, you would first sum the revenue by customer, then average that sum by store.

    Examples:
    # basic column
            {{
                "name": "store_id"
            }}

    # column with calculation over all output
            {{
                "name": "total_returns",
                "calculation": {{
                    "operator": "SUM",
                    "arguments": [
                        {{
                            "name": "store_returns.return_value"
                        }}
                    ]
                }}
            }}
    # column with a calculation off the previous definition, do a different granularity
            {{
                "name": "average_return_by_store",
                "calculation": {{
                    "operator": "AVG",
                    "arguments": [
                        {{
                            "name": "total_returns"
                        }}
                    ],
                    "over": [
                        {{"name": "store_id"}}
                    ]
                }}
            }}


    A Literal Object is json with these fields:
    -- value: the literal value ('1', 'abc',  1.0, etc), expressed as a string, or a Calculation Object
    -- type: the type of the value ('float', 'string', 'int', 'bool'), expressed as a string

    Examples: 
    # with constant
        {{"value": "1.2", "type": "float"}},
    # with calculation
        {{"value": {{
                        "operator": "MULTIPLY",
                        "arguments": [
                            {{"value": "1.2", "type": "float"}},
                            {{"name": "average_return_by_store"}}
                        ]
                    }},
            "type" : "float"
        }}

    A Calculation Object is json with three fields:
    -- operator: a function to call with those arguments. [SUM, AVG, COUNT, MAX, MIN, etc], expressed as a string. A calculation object MUST have an operator. This cannot be a comparison operator.
    -- arguments: a list of Column or Literal objects. If there is an operator, there MUST be arguments
    -- over: an optional list of Column objects used when an aggregate calculation needs to group over other columns (sum of revenue by state and county, for example)

    A Comparison object is JSON with three fields:
    -- operator: the comparison operator, one of "=", "in", "<", ">", "<=", "like", or ">=". Use two comparisons to represent a between
    -- left: A Column or Literal object
    -- right: A Column or Literal object

    A ConditionGroup object is JSON with two fields used to create boolean filtering constructs. You can nest ConditionGroups to create complex filtering conditions.
    -- values: a list if Comparison Objects or ConditionGroups
    -- boolean: 'and' or 'or' (lowercase, no quotes)

    All together, the input for validation and final submission should be a VALID JSON blob with the following keys and values followed by a stopword: <EOD>:
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

    You will get essential information from using tools before producing your final answer.

    Use as many tools as needed, and always validate, before producing the "Final Answer" action.

    Valid "action" values: any of {tool_names} and, for your last result "Final Answer". 

    Only return "Final Answer" when you are done with all work. Never set the action to 

    "Final Answer" before you are done, and never set the action to final answer without some columns returned.

    You should always call the the validate_response tool with your candidate answer before declaring it the final answer.

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
    Thought: consider previous and subsequent steps
    Action:
    ```
    $JSON_BLOB
    ```
    <action result>
    ... (repeat Thought/Action/Result N times)

    An example series:

    Question: Get the total revenue dollars by order and customer id for stores in the zip code 1025 in the year 2000 where the total sales price of the items in the order was more than 100 dollars and the total revenue of the order was more than 10 dollars?
    Thought: I should start by using any available tools to inform my action.
    Action:
    ```
    {{
        "action": "get_database_description",
        "action_input": ""
        "reasoning": "I should get the available fields in the database."
    }}
    {{"fields": [<a list if fields in format {{"name": "field_name", <optional description>]}} }}
    Action:
    ```
    {{
        "action": "validate_response",
        "action_input": {{
        "columns": [
            {{"name": "store.order.id"}},
            {{"name": "store.order.customer.id"}},
            {{"name": "revenue_sum", 
                "calculation": {{
                    "operator":"SUM", 
                    "arguments": [
                            {{
                            "name": "revenue_dollars",
                            "calculation" : {{
                                "operator": "MULTIPLY",
                                "arguments": [
                                    {{ "name": "store.order.revenue_cents" }}
                                    ]
                                }}
                            }}
                        ]
                    }}
            }}
        ],
        "filtering": {{
            "root": {{
                "values": [
                {{
                    "operator": "="
                    "left": {{"name": "store.zip_code"}},
                    "right": {{"value":"10245", "type":"integer"}},
                    
                }},
                {{
                    "operator": "="
                    "left": {{"name": "store.order.date.year" }},
                    "right": {{"value":"2000", "type":"integer"}},
                    
                }},
                {{
                    "operator": ">"
                    "left": {{"name": "revenue_sum" }},
                    "right": {{"value":"10", "type":"float"}},
                    
                }},
                {{  
                    "operator": ">"
                    "left": {{
                        "name": "sales_price_sum_by_store", 
                        "calculation": {{"operator":"SUM", 
                            "arguments": [
                                {{ "name": "item.sales_price"}}
                                ],
                            "over": [
                                {{ "name": "store.order.id"}}, 
                                {{ "name": "store.id"}}
                            ]
                            }}
                        }},
                    "right": {{"value":"100.0", "type":"float"}},
                    
                    
                }}
                ],
                "boolean": "and"
                }},
        }}
        "order": [
            {{"column_name": "revenue_sum", "order": "desc"}}
        ],
        "limit": 100
        }}, 
        "reasoning": "I can return order id, customer id, and the total order revenue. Order Id and customer Id are scalar values, while the total order revenue will require a calculation. I can filter to the zip code and the year, and then restrict to where the sales price over the store and order id is more than 100, which will require a calculation. Before submitting my answer, I need to validate my answer."
    }}

    Nested Column objects with calculations can create complex derivations. This can be useful for filtering. 

    Note: You don't need to use an over clause for an aggregate calculated columm you're outputting if it's over the other columns you've selected - that's implicit.

        Example: to get total revenue by customer - just select the customer id and sum(total_revenue). 
        Example: to get the average revenue customer by store, return store idand avg(sum(total_revenue) by customer_id) (in appropriate JSON format)

    IMPORTANT: don't trust that the answer formatted a literal for filtering appropriately. For example, if the prompt asks for 'the first month of the year', you may need to filter to
    1, January, or Jan. Field descriptions will contain formatting hints that can be used for this. 

    Filtering can also leverage calculations - for example, to create a filter condition for "countries with an average monthly rainfall of 2x the average on their continent", 
    the filtering clause might look like.

    "filtering": {{
            "root": {{
                "values": 
                    [
                    {{
                    "operator": ">",
                    "left": {{
                        "name": "avg_monthly_rainfall",
                        "calculation": {{
                            "operator": "AVG",
                            "arguments": [{{"name": "country.monthly_rainfall"}}],
                            "over": [
                                {{"name": "country.name"}},
                                {{"name": "date.rainfall"}}
                                ]
                            }}
                        }},
                    "right":  {{
                        "name": "continent_avg_monthly_rainfall_2x",
                        "calculation": {{
                            "operator": "MULTIPLY",
                            "arguments": [
                                    {{  "value":"2", "type":"integer"}}, 
                                    {{  "name": "continent_avg_monthly_rainfall",
                                        "calculation" : {{
                                            "operator": "AVG",
                                            "arguments": ["country.monthly_rainfall"],
                                            "over": [
                                                {{"name": "country.continent"}},
                                                {{"name": "date.rainfall"}}
                                            ]
                                        }}
                                    }}
                                    ],

                                }}
                        }}
                    }}
                    ],
                "boolean": "and"
        }}
    }}

    Once you have used any tools (listed below) as needed, you will produce your final result in this format. If your final answer is wrong,
    you'll receive the prompt again with a hint that you got it wrong. You can see the output of your last three actions only.

    A final response could look like this:

    Thought: I have my final, validated answer!
    Action:
    ```
    {{
        "action": "Final Answer",
        "action_input": <VALID_JSON_WITH_SPEC_DEFINED_ABOVE>,
        "reasoning": "<description of your logic>"
    }}
    ```
    
    Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation.

    """

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
        handle_parsing_errors="The JSON blob response you provided in between the ``` ``` was improperly formatted. Double check it's valid JSON by reviewing your last submission. Include a description of the edits you made in the reasoning of your next submission.",  # type: ignore,
        # trim_intermediate_steps=5,
    )
    attempts = 0
    if additional_context:
        input_text += additional_context
    error = None
    while attempts < 1:
        result = agent_executor.invoke({"input": input_text})
        output = result["output"]
        try:
            if isinstance(output, str):
                ir = InitialParseResponseV2.model_validate_json(output)
            elif isinstance(output, dict):
                ir = InitialParseResponseV2.model_validate(output)
            else:
                raise ValueError("Unable to parse LLM response")
            return ir_to_query(ir, input_environment=input_environment, debug=True)
        except ValidationError as e:
            error = e
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
        except NotImplementedError as e:
            error = e
            raise e

        except Exception as e:
            error = e
            print("Failed to parse LLM response")
            print(e)
            input_text += f"IMPORTANT: this is your second attempt - your last attempt errored parsing your final answer: {str(e)}. Remember to use the validation tool to check your work!"
        attempts += 1
    if error:
        raise error
    raise ValueError(f"Unable to get parseable response after {attempts} attempts")

def ir_to_query(intermediate_results:InitialParseResponseV2, input_environment:Environment, debug:bool = True):
    
    selection = [
        create_column(x, input_environment) for x in intermediate_results.columns
    ]
    order = parse_order(selection, intermediate_results.order or [])

    filtering = (
        parse_filtering(intermediate_results.filtering, input_environment)
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

    where, having = generate_having_and_where(filtering)
    query = SelectStatement(
        selection=[ConceptTransform(function=x.lineage, output=x) if is_local_derived(x) else SelectItem(content=x) for x in selection],
        limit=safe_limit(intermediate_results.limit),
        order_by=order,
        where_clause=where,
        having_clause=having
    )
    if filtering:
        def append_child_concepts(xes:list[Concept]):

            def get_address(z):
                if isinstance(z, Concept):
                    return z.address
                elif isinstance(z, ConceptTransform):
                    return z.output.address
            for x in xes:
                if not any(x.address ==  get_address(item.content) for item in query.selection):
                    if is_local_derived(x):
                        content = ConceptTransform(function=x.lineage, output=x)
                        query.selection.append(SelectItem(content=content))
                        append_child_concepts(x.lineage.concept_arguments)
        append_child_concepts(filtering.concept_arguments)
        
                
    for item in query.selection:
        # we don't know the grain of an aggregate at assignment time
        # so rebuild at this point in the tree
        # TODO: simplify
        if isinstance(item.content, ConceptTransform):
            new_concept = item.content.output.with_select_context(
                query.grain,
                conditional=None,
                environment=input_environment,
            )
            input_environment.add_concept(new_concept)
            item.content.output = new_concept
        elif isinstance(item.content, Concept):
            # Sometimes cached values here don't have the latest info
            # but we can't just use environment, as it might not have the right grain.
            item.content = input_environment.concepts[
                item.content.address
            ].with_grain(item.content.grain)
    print('select debug')
    for x in query.selection:
        print(type(x.content))


    from trilogy.parsing.render import Renderer

    print("RENDERED QUERY")
    print(Renderer().to_string(query))
    return query

def parse_query(
    input_text: str,
    input_environment: Environment,
    llm: BaseLanguageModel,
    debug: bool = False,
    log_info: bool = True,
)->SelectStatement:
    return llm_loop(input_text, input_environment, llm=llm)



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
