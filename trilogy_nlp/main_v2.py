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
)
from typing import List
from trilogy.core.query_processor import process_query
from langchain.tools import Tool, StructuredTool
from datetime import datetime
import json
from pydantic import BaseModel, field_validator, ValidationError
from trilogy.core.enums import ComparisonOperator, Ordering, Purpose, BooleanOperator
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from trilogy_nlp.tools import get_wiki_tool
from trilogy.parsing.common import arbitrary_to_concept
from trilogy.core.models import DataType
from langchain_core.tools import ToolException
from trilogy.core.exceptions import UndefinedConceptException

# from trilogy.core.constants import
from trilogy.core.enums import (
    FunctionType,
    FunctionClass,
    InfiniteFunctionArgs,
)
from trilogy.parsing.common import arg_to_datatype
from enum import Enum


# llm will not know how to use these; better not to include
COMPLICATED_FUNCTIONS = ['ALIAS', "CUSTOM", "GROUP"]

class NLPComparisonOperator(Enum):
    BETWEEN = "between"

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, list) and " " in str(value):
            value = str(value).split()
        if isinstance(value, list):
            processed = [str(v).lower() for v in value]
            if processed == ["not", "in"]:
                return ComparisonOperator.NOT_IN
            if processed == ["is", "not"]:
                return ComparisonOperator.IS_NOT
            if value == ["in"]:
                return ComparisonOperator.IN
        return super()._missing_(str(value).lower())


# hacky, but merge two enums
NLPComparisonOperator._member_map_.update(ComparisonOperator._member_map_)


class OrderResultV2(BaseModel):
    """The result of the order prompt"""

    column_name: str
    order: Ordering


class Literal(BaseModel):
    value: str
    type: str


class Calculation(BaseModel):
    arguments: list[Union["Column", Literal]]
    operator: str
    over: list["Column"] | None = None


class Column(BaseModel):
    name: str
    calculation: Optional[Calculation] = None

    def __hash__(self):
        return hash(self.name + str(self.calculation))


Calculation.model_rebuild()


# class FilterResultV2(BaseModel):
#     """The result of the filter prompt"""

#     column: Column
#     values: list[Literal | Column]
#     operator: NLPComparisonOperator


class NLPConditions(BaseModel):
    left: Column | Literal
    right: Column | Literal
    operator: ComparisonOperator


class NLPComparisonGroup(BaseModel):
    values: list[Union[NLPConditions, "NLPComparisonGroup"]]
    boolean: BooleanOperator


class FilterResultV2(BaseModel):
    """The result of the filter prompt"""

    root: NLPComparisonGroup


NLPComparisonGroup.model_rebuild()


def parse_object(ob, environment: Environment):
    if isinstance(ob, Column):
        return create_column(ob, environment)
    return create_literal(ob, environment)


def parse_datatype(dtype: str):
    mapping = {item.value.lower(): item for item in DataType}
    mapping["integer"] = DataType.INTEGER
    if dtype.lower() in mapping:
        return mapping[dtype]
    return DataType.STRING


def create_literal(l: Literal, environment:Environment) -> str | float | int | bool:
    # LLMs might get formats mixed up; if they gave us a column, hydrate it here.
    # and carry on
    if l.value in environment.concepts:
        return create_column(Column(name=l.value), environment)

    dtype = parse_datatype(l.type)

    if dtype == DataType.STRING:
        return l.value
    if dtype == DataType.INTEGER:
        return int(l.value)
    if dtype == DataType.FLOAT:
        return float(l.value)
    if dtype == DataType.BOOL:
        return bool(l.value)
    return l.value


def create_column(c: Column, environment: Environment):
    if not c.calculation:
        return environment.concepts[c.name]

    operator = FunctionType(c.calculation.operator.lower())
    if operator in FunctionClass.AGGREGATE_FUNCTIONS.value:
        purpose = Purpose.METRIC
    else:
        purpose = Purpose.PROPERTY

    args = [parse_object(c, environment) for c in c.calculation.arguments]
    base_name = c.name
    # LLMs tend to reference the same name for the output of a calculation
    # if that's so, force the outer concept a new name
    if any(isinstance(z, Concept) and z.name == base_name for z in args ):
        base_name = f'{c.name}_deriv'
    # TODO: use better helpers here
    # this duplicates a bit of pytrilogy logic
    derivation = Function(
        operator=FunctionType(c.calculation.operator.lower()),
        output_datatype=arg_to_datatype(args[0]),
        output_purpose=purpose,
        arguments=args,
        arg_count=InfiniteFunctionArgs,
    )
    if c.calculation.over:
        if purpose != Purpose.METRIC:
            raise ValueError("Can only use over with aggregate functions.")
        derivation = AggregateWrapper(
            function=derivation,
            by=[parse_object(c, environment) for c in c.calculation.over],
        )
    
    new = arbitrary_to_concept(
        derivation,
        namespace="local",
        name=f"{c.name}".lower(),
    )
    environment.add_concept(new)
    return new


class InitialParseResponseV2(BaseModel):
    """The result of the initial parse"""

    columns: list[Column]
    limit: Optional[int] = 100
    order: Optional[list[OrderResultV2]] = None
    filtering: Optional[FilterResultV2] = None

    @field_validator("filtering", mode="plain")
    @classmethod
    def filtering_validation(cls, v):
        if isinstance(v, dict):
            return FilterResultV2.model_validate(v)
        return FilterResultV2.model_validate(v)

    # @field_validator("order", mode="plain")
    # @classmethod
    # def order_validation(cls, v):
    #     if v is None:
    #         return None
    #     if isinstance(v, dict):
    #         return [OrderResultV2.model_validate(v)]
    #     return [OrderResultV2.model_validate(x) for x in v]


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
        possible_matches = [
            x
            for x in input_concepts
            if x.address == order.column_name or x.name == order.column_name
        ]
        if not possible_matches:
            has_lineage = [x for x in input_concepts if x.lineage]
            possible_matches = [
                x
                for x in has_lineage
                if any(
                    [
                        y.address == order.column_name
                        for y in x.lineage.concept_arguments
                    ]
                )
            ]
        if possible_matches:
            final.append(OrderItem(expr=possible_matches[0], order=order.order))
    return OrderBy(items=final)


def parse_filter_obj(
    inp: NLPComparisonGroup | NLPConditions | Column | Literal, environment: Environment
):
    if isinstance(inp, NLPComparisonGroup):
        print('handlign comparison group')
        children = [parse_filter_obj(x, environment) for x in inp.values]
        print(children)
        def generate_conditional(list: list, operator:BooleanOperator):
            if not list:
                return True
            left = list.pop(0)
            print(left)
            right = generate_conditional(list, operator)
            print(right)
            return Conditional(left=left, right=right, operator=operator)

        return generate_conditional(children, operator=inp.boolean)
    elif isinstance(inp, NLPConditions):
        return Comparison(
            left=parse_filter_obj(inp.left, environment),
            right=parse_filter_obj(inp.right, environment),
            operator=inp.operator,
        )
    elif isinstance(inp, (Column, Literal)):
        return parse_object(inp, environment)
    else:
        raise SyntaxError(inp)


def parse_filter(
    input: FilterResultV2, environment: Environment
) -> Comparison | Conditional | None:
    return parse_filter_obj(input.root, environment)


def parse_filtering(
    filtering: FilterResultV2, environment: Environment
) -> WhereClause | None:
    base = []
    parsed = parse_filter(filtering, environment=environment)
    print('parsed debug')
    print(parsed)
    return WhereClause(conditional=parsed)
    if filtering.root and not parsed:
        raise SyntaxError
    if parsed:
        print(parsed)
        base.append(parsed)
    if not base:
        return None
    print('filtering debug')
    print(base)
    if len(base) == 1:
        return WhereClause(conditional=base[0])
    left: Conditional | Comparison = base.pop()
    while base:
        right = base.pop()
        new = Conditional(left=left, right=right, operator=BooleanOperator.AND)
        left = new
    return WhereClause(conditional=left)


COLUMN_DESCRIPTION = """    A Column Object is json with two fields:
    -- name: the field being referenced or a new derived name. If there is a calculation, this should always be a new derived name you came up with. 
    -- calculation: An optional calculation object.
"""


def is_valid_function(name: str):
    return name.lower() in [item.value for item in FunctionType]


def invalid_operator_message(operator: str) -> str | None:
    try:
        operator = ComparisonOperator(operator)
    except Exception as e:
        return str(e)
    return None


class QueryContext(Enum):
    SELECT = "SELECT"
    FILTER = "FILTER"
    ORDER = "ORDER"


def validate_query(query: dict, environment: Environment):
    try:
        parsed = InitialParseResponseV2.model_validate(query)
    except ValidationError as e:
        return {"status": "invalid", "error": validation_error_to_string(e)}
    errors = []
    select = {col.name for col in parsed.columns}

    def validate_column(col: Column, context: QueryContext):
        if (
            col.name not in environment.concepts
            and not col.calculation
            and (context == QueryContext.SELECT or col.name not in select)
        ):
            try:
                environment.concepts[col.name]
            except UndefinedConceptException as e:
                recommendations = e.suggestions
            
            if recommendations:
                errors.append(
                f"{col.name} in {context} is not a valid field in the database; check that you are using the full exact column name (including an prefixes). Did you mean one of {recommendations}?",
              
                )
            else:
                errors.append(
                    f"{col.name} in {context} is not a valid field in the database; check that you are using the full exact column name (including an prefixes). If you want to apply a function, use a calculation - do not include it in the field name. Format reminder: {COLUMN_DESCRIPTION}. You may need to list fields again if you are not sure of the correct value.",
                )
        if col.calculation and not is_valid_function(col.calculation.operator):
            errors.append(
                f"{col.name} Column definition does not use a valid operator; check that you are using ONLY a valid option from this list: {[x for x in FunctionType.__members__.keys() if x not in COMPLICATED_FUNCTIONS] }. If the column requires no transformation, drop the calculation field.",
            )
        if col.calculation:
            for arg in col.calculation.arguments:
                if isinstance(arg, Column):
                    validate_column(arg, context)

    for x in parsed.columns:
        validate_column(x, QueryContext.SELECT)

    if parsed.order:
        for y in parsed.order:
            if y.column_name not in select:
                errors.append(
                    f"{y.column_name} in order not in select; check that you are using only values in the top level of the select.",
                )
    if parsed.filtering:
        root = parsed.filtering.root
        for val in root.values:
            if isinstance(val, Column):
                validate_column(val, QueryContext.FILTER)
            elif isinstance(val, NLPConditions):
                for subval in [val.left, val.right]:
                    if isinstance(subval, Column):
                        validate_column(subval, QueryContext.FILTER)

        # elif isinstance(root.right, )
        # for z in parsed.filtering:
        #     validate_column(z.column, "filtering")
        #     if not z.values:
        #         errors.append("A filtering argument must always have values")
        #     for y in z.values:
        #         if isinstance(y, Column):
        #             validate_column(y, "filtering values")
        #     operator_check = invalid_operator_message(z.operator)
        #     if operator_check:
        #         errors.append(
        #             f"{z} operator is invalid: {operator_check}; make sure it is a valid option from {ComparisonOperator.__members__}",
        #         )
    if errors:
        return {"status": "invalid", "error": str(errors)}
    return {"status": "valid"}


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


def get_today_date(query: str) -> str:
    """
    Useful to get the date of today.
    """
    # Getting today's date in string format
    today_date_string = datetime.now().strftime("%Y-%m-%d")
    return today_date_string


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


class Filter(BaseModel):
    column: str
    values: List
    operator: str

def validation_error_to_string(e:ValidationError):
        # Here, `validation_error.errors()` will have the full info
    for x in e.errors():
        print(x)
    # inject in new context on failed answer
    raw_error = str(e)
    #TODO: better pydantic error parsing
    if 'filtering.root.' in raw_error:
        raw_error = 'Syntax error in your filtering clause. Confirm it matches the required format and is valid JSON with brackets in right locations. Comparisons need a left, right, operator, etc, and Columns and Literal formats are very specific'
    return raw_error
def validate_response(
    environment: Environment,
    columns: List[Column],
    filtering: Optional[FilterResultV2] = None,
    order: Optional[list[OrderResultV2]] = None,
    limit: int = None,
):

    base = {"columns": columns}
    if not columns:
        raise ToolException(
            "A answer to validate should include at least one column. Call this tool with your entire predicted final answer."
        )
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


class ValidateResponseInterface(BaseModel):
    # deliberately permissive interface
    # so that we can handle the error inside the tool
    columns: List[dict]
    filtering: Optional[dict] = (None,)
    order: Optional[list[dict]] = (None,)
    limit: int = None


def sql_agent_tools(environment):
    def validate_response_wrapper(**kwargs):
        return validate_response(
            environment=environment,
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
    input_text: str, input_environment: Environment, llm: BaseLanguageModel
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
        "action_input": $INPUT
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
    Observation: action result
    ... (repeat Thought/Action/Observation N times)

    An example series:

    Question: Get the total revenue by order and customer id for stores in the zip code 1025 in the year 2000 where the total sales price of the items in the order was more than 100 dollars
    Thought: I should get the description of of the orders dataset
    Action:
    ```
    {{
        "action": "get_database_description",
        "action_input": ""
    }}

    Observation: <some description>
    Thought: I should check my answer
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
                    "left": {{"name": "total_sales_price", 
                    "calculation": {{"operator":"SUM", 
                        "arguments": [{{ "name": "item.sales_price"}}],
                        "over": [{{ "name": "store.order.id"}}]
                        }},
                    "right": {{"value":"100.0", "type":"float"}},
                    "operator": ">"
                        }}
                }}
                ],
                "boolean": "and"
        }},
        "order": [
            {{"column_name": "revenue_sum", "order": "desc"}},
        ],
        "limit": 100
        }}
    }}

    Nested Column objects with calculations can create complex derivations. This can be useful for filtering.

    For example, to create a filter condition for "countries with an average monthly rainfall of 2x the average on their continent", the filtering clause might look like.

    You don't need to use an over clause for a top level calculation if it's over the other columns you've selected.

    Ex: for customer, total_revenue, just sum revenue. 

    {{
            "root": {{
                "values": [
                    {{
                    "left": {{
                        "name": "avg_monthly_rainfall",
                        "calculation: {{
                            "operator": "AVG",
                            "arguments: [{{"name": "country.monthly_rainfall"}}],
                            "over": [{{"name": "country.name"}}]
                                }}
                            }},
                    "right":  {{
                        "name": "continent_2x",
                        "calculation: {{
                            "operator": "MULTIPLY",
                            "arguments: [
                                    {{"value":"2", "type":"integer"}}, 
                                    {{"name": "continent_avg_monthly"
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
    To filter over a calculation with a scalar, multiple that in 
    Once you have used any tools (listed below) as needed, you will produce your final result in this format. After producing your
    final answer, you cannot take any more steps.

    A final answer would look like this:
    {{
        "action": "Final Answer",
        "action_input": <VALID_JSON_SPEC_DEFINED_ABOVE>
    }}

    You will only see the last few steps.

    Begin! Reminder to ALWAYS respond with a valid json blob for an action. Always use tools. Format is Action:```$JSON_BLOB```then Observation"""

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

    tools = sql_agent_tools(input_environment) + [get_wiki_tool()]
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
            #TODO: better pydantic error parsing
            if 'filtering.root.' in raw_error:
                raw_error = 'Syntax error in your filtering clause. Confirm it matches the required format. Comparisons need a left and right, etc, and Columns and Literal formats are very specific'
            input_text += f"Error on your final answer: {raw_error}. Try Again!"
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
        parse_filtering(intermediate_results.filtering, input_environment)
        if intermediate_results.filtering
        else None
    )


    if debug:
        print("Concepts found")
        for c in intermediate_results.columns:
            print(c)
        if intermediate_results.filtering:
            print('filtering')
            print(str(intermediate_results.filtering))
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
