from langchain.agents import create_structured_chat_agent, AgentExecutor
from trilogy.core.models import (
    Environment,
    ProcessedQuery,
    SelectStatement,
    SelectItem,
    Concept,
    ConceptTransform,
    ConceptDeclarationStatement
)
from trilogy.core.query_processor import process_query
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from trilogy_nlp.tools import get_wiki_tool

from trilogy_nlp.llm_interface.parsing import (
    parse_filtering,
    parse_order,
    create_column,
    generate_having_and_where,
)
from trilogy_nlp.llm_interface.models import InitialParseResponseV2, Column
from trilogy_nlp.llm_interface.tools import sql_agent_tools
from trilogy_nlp.llm_interface.constants import MAGIC_GENAI_DESCRIPTION
from trilogy_nlp.prompts_v2.query_system import BASE_1
from trilogy_nlp.helpers import safe_limit
from trilogy_nlp.exceptions import ValidationPassedException


def is_local_derived(x: Concept) -> bool:
    return x.metadata.description == MAGIC_GENAI_DESCRIPTION


def llm_loop(
    input_text: str,
    input_environment: Environment,
    llm: BaseLanguageModel,
    additional_context: str | None = None,
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
        max_iterations=5,
        early_stopping_method="force",
        # handle_parsing_errors="The JSON blob response you provided in between the ``` ``` was improperly formatted. Double check it's valid JSON by reviewing your last submission. Include a description of the edits you made in the reasoning of your next submission.",  # type: ignore,
        # trim_intermediate_steps=5,
    )
    attempts = 0
    if additional_context:
        input_text += additional_context
    error = None
    while attempts < 1:
        try:
            result = agent_executor.invoke({"input": input_text})
        except ValidationPassedException as e:
            ir = e.ir
            return ir_to_query(ir, input_environment=input_environment, debug=True)
        attempts += 1
    if error:
        raise error
    raise ValueError(f"Unable to get parseable response after {attempts} attempts")


from networkx import DiGraph, topological_sort


def determine_ordering(columns: list[Column]):
    edges = []

    def handle_column(column: Column):
        calculation = column.calculation
        if not calculation:
            return
        for arg in calculation.arguments:
            if not isinstance(arg, Column):
                continue
            edges.append((arg.name, column.name))
            handle_column(arg)
        for arg in calculation.over or []:
            if not isinstance(arg, Column):
                continue
            edges.append((arg.name, column.name))
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


def ir_to_query(
    intermediate_results: InitialParseResponseV2,
    input_environment: Environment,
    debug: bool = True,
):
    ordering = determine_ordering(intermediate_results.columns)
    selection = [
        create_column(x, input_environment)
        for x in sort_by_name_list(intermediate_results.columns, ordering)
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

    where, having = generate_having_and_where([x.address for x in selection], filtering)
    query = SelectStatement(
        selection=[
            (
                ConceptTransform(function=x.lineage, output=x)
                if is_local_derived(x)
                else SelectItem(content=x)
            )
            for x in selection
        ],
        limit=safe_limit(intermediate_results.limit),
        order_by=order,
        where_clause=where,
        having_clause=having,
    )

    # if having:

    #     def append_child_concepts(xes: list[Concept]):

    #         def get_address(z):
    #             if isinstance(z, Concept):
    #                 return z.address
    #             elif isinstance(z, ConceptTransform):
    #                 return z.output.address

    #         for x in xes:
    #             if not any(
    #                 x.address == get_address(item.content) for item in query.selection
    #             ):
    #                 # if is_local_derived(x):
    #                 if x.lineage:
    #                     content = ConceptTransform(function=x.lineage, output=x)
    #                     query.selection.append(SelectItem(content=content))
    #                     append_child_concepts(x.lineage.concept_arguments)
    #                 else:
    #                     query.selection.append(SelectItem(content=x))

    #     append_child_concepts(having.concept_arguments)

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
            item.content = input_environment.concepts[item.content.address].with_grain(
                item.content.grain
            )
    from trilogy.parsing.render import Renderer
    renderer = Renderer()
    print("RENDERED QUERY")
    for c in input_environment.concepts.values():
        if is_local_derived(c):
            print(renderer.to_string(ConceptDeclarationStatement(concept=c)))
    print(renderer.to_string(query))
    query.validate_syntax()
    return query


def parse_query(
    input_text: str,
    input_environment: Environment,
    llm: BaseLanguageModel,
    debug: bool = False,
    log_info: bool = True,
) -> SelectStatement:
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
