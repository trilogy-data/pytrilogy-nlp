from typing import Sequence

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.agents.agent import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from networkx import DiGraph, topological_sort
from trilogy.authoring import (
    AggregateWrapper,
    Concept,
    ConceptDeclarationStatement,
    ConceptRef,
    ConceptTransform,
    Environment,
    FilterItem,
    Function,
    SelectItem,
    SelectStatement,
    WindowItem,
)
from trilogy.core.query_processor import process_query
from trilogy.core.statements.execute import ProcessedQuery
from trilogy.parsing.render import Renderer

from trilogy_nlp.config import DEFAULT_CONFIG
from trilogy_nlp.constants import logger
from trilogy_nlp.exceptions import ValidationPassedException
from trilogy_nlp.helpers import safe_limit
from trilogy_nlp.instrumentation import EventTracker
from trilogy_nlp.llm_interface.constants import MAGIC_GENAI_DESCRIPTION
from trilogy_nlp.llm_interface.models import Column, InitialParseResponseV2
from trilogy_nlp.llm_interface.parsing import (
    create_column,
    generate_having_and_where,
    parse_filtering,
    parse_order,
)
from trilogy_nlp.llm_interface.tools import sql_agent_tools
from trilogy_nlp.prompts_v2.query_system import BASE_1
from trilogy_nlp.tools import get_wiki_tool


def is_local_derived(x: Concept) -> bool:
    return x.metadata.description == MAGIC_GENAI_DESCRIPTION


def handle_parsing_error(x: OutputParserException):

    return """
The expected action format was below; please try again with the correct format.
Note that the action is the name of the tool you want to use, and the action_input is the input you want to provide to that tool, and must be valid JSON.

Action:
```
{{
    "action": $TOOL_NAME,
    "action_input": $INPUT,
    "reasoning": "Your thinking"
}}
```
Observation:
"""


def llm_loop(
    input_text: str,
    input_environment: Environment,
    llm: BaseLanguageModel,
    additional_context: str | None = None,
    debug: bool = False,
    event_tracker: EventTracker | None = None,
) -> SelectStatement:

    human = """{input}

    {agent_scratchpad}

    (reminder to respond in a JSON blob no matter what)"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", BASE_1),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", human),
        ]
    )

    tools = sql_agent_tools(input_environment, input_text, event_tracker) + [
        get_wiki_tool()
    ]
    chat_agent = create_structured_chat_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(
        agent=chat_agent,
        tools=tools,
        verbose=True if debug else False,
        max_iterations=DEFAULT_CONFIG.LLM_VALIDATION_ATTEMPTS,
        early_stopping_method="force",
        handle_parsing_errors=handle_parsing_error,
    )
    attempts = 0
    if additional_context:
        input_text += additional_context
    error = None
    while attempts < 4:
        output = None
        try:
            output = agent_executor.invoke({"input": input_text})
            return ir_to_query(
                InitialParseResponseV2.model_validate(output["output"]),
                input_environment,
                debug=debug,
            )
        except ValidationPassedException as e:
            ir = e.ir
            return ir_to_query(ir, input_environment=input_environment, debug=True)

        except Exception as e:
            error = e
            logger.error(
                f"Error in main execution llm loop with output {output}: {str(e)}"
            )
            if (
                "peer closed connection without sending complete message body (incomplete chunked read)"
                in str(e)
            ):
                attempts + 1
                continue
            # don't retry an unknown error
            break
    if error:
        raise error
    raise ValueError(f"Unable to get parseable response after {attempts} attempts")


def determine_ordering(columns: list[Column]):
    edges = []

    def handle_column(column: Column):
        calculation = column.calculation
        base_name = column.name
        if not calculation:
            return
        for arg in calculation.arguments:
            if isinstance(arg, Column) and arg.name == base_name:
                base_name = base_name + "_deriv"
        for arg in calculation.arguments:
            if not isinstance(arg, Column):
                continue
            edges.append((arg.name, base_name))
            handle_column(arg)
        for arg in calculation.over or []:
            if not isinstance(arg, Column):
                continue
            edges.append((arg.name, base_name))
            handle_column(arg)

    for ic in columns:
        handle_column(ic)
    graph: DiGraph = DiGraph(edges)
    base = list(topological_sort(graph))
    for column in columns:
        if column.name not in base:
            base.append(column.name)
    return base


def sort_by_name_list(arr: list[Column], reference: list[str]):
    order_map = {value: index for index, value in enumerate(reference)}
    return sorted(arr, key=lambda x: order_map.get(x.name, float("inf")))


def get_address(z: Concept | ConceptTransform | ConceptRef):
    if isinstance(z, Concept):
        return z.address
    elif isinstance(z, ConceptTransform):
        return z.output.address
    elif isinstance(z, ConceptRef):
        return z.address
    raise ValueError(f"Unknown type {type(z)}")


def ir_to_query(
    intermediate_results: InitialParseResponseV2,
    input_environment: Environment,
    debug: bool = False,
):
    ordering = determine_ordering(intermediate_results.output_columns)
    selection = [
        create_column(x, input_environment)
        for x in sort_by_name_list(intermediate_results.output_columns, ordering)
    ]
    order = parse_order(selection, intermediate_results.order or [])

    filtering = (
        parse_filtering(intermediate_results.filtering, input_environment)
        if intermediate_results.filtering
        else None
    )

    if debug:
        print("Output Concepts")
        for c in intermediate_results.output_columns:
            print(c)
        if intermediate_results.filtering:
            print("Filtering")
            print(str(intermediate_results.filtering))
        if intermediate_results.order:
            print("Ordering")
            for o in intermediate_results.order:
                print(o)
    normalized_select = [x if isinstance(x, Concept) else x.output for x in selection]
    where, having = generate_having_and_where(
        [x.address for x in normalized_select],
        environment=input_environment,
        filtering=filtering,
    )
    query: SelectStatement = SelectStatement.from_inputs(
        selection=[
            (
                SelectItem(content=ConceptTransform(function=x.lineage, output=x))
                if is_local_derived(x)
                and x.lineage
                and isinstance(
                    x.lineage, (Function, FilterItem, WindowItem, AggregateWrapper)
                )
                else SelectItem(content=x.reference)
            )
            for x in normalized_select
        ],
        limit=safe_limit(intermediate_results.limit),
        order_by=order,
        where_clause=where,
        environment=input_environment,
        having_clause=having,
    )

    if having:
        # append anything filtered in the having clause to the output of the select
        def append_child_concepts(xes: Sequence[ConceptRef]):
            for ref in xes:
                x = input_environment.concepts[ref.address]
                if not any(
                    x.address == get_address(item.content) for item in query.selection
                ):
                    if (
                        is_local_derived(x)
                        and x.lineage
                        and isinstance(
                            x.lineage,
                            (Function, FilterItem, WindowItem, AggregateWrapper),
                        )
                    ):
                        content = ConceptTransform(function=x.lineage, output=x)
                        query.selection.append(SelectItem(content=content))
                        if x.lineage:
                            append_child_concepts(x.lineage.concept_arguments)

        append_child_concepts(having.concept_arguments)
    renderer = Renderer()
    print("RENDERED QUERY")
    print("---------")
    for new_c in input_environment.concepts.values():
        if is_local_derived(new_c):
            print(renderer.to_string(ConceptDeclarationStatement(concept=new_c)))
    print(renderer.to_string(query))
    print("---------")
    query.validate_syntax(environment=input_environment)
    return query


def parse_query(
    input_text: str,
    input_environment: Environment,
    llm: BaseLanguageModel,
    debug: bool = False,
    log_info: bool = True,
    event_tracker: EventTracker | None = None,
) -> SelectStatement:
    return llm_loop(
        input_text, input_environment, llm=llm, debug=debug, event_tracker=event_tracker
    )


def build_query(
    input_text: str,
    input_environment: Environment,
    llm: BaseLanguageModel,
    debug: bool = False,
    log_info: bool = True,
    event_tracker: EventTracker | None = None,
) -> ProcessedQuery:
    query = parse_query(
        input_text,
        input_environment,
        debug=debug,
        llm=llm,
        log_info=log_info,
        event_tracker=event_tracker,
    )
    return process_query(statement=query, environment=input_environment)
