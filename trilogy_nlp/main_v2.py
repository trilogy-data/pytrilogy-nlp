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
from trilogy_nlp.prompts_v2.query_system import BASE_1

def is_local_derived(x:Concept)->bool:
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
