import uuid
from typing import List, Union

from preql.core.enums import BooleanOperator, Purpose
from preql.core.models import (
    Comparison,
    Concept,
    Conditional,
    Environment,
    OrderBy,
    Ordering,
    OrderItem,
    ProcessedQuery,
    Select,
    WhereClause,
    unique,
)
from preql.core.query_processor import process_query_v2

from preql_nlp.constants import DEFAULT_LIMIT, logger
from preql_nlp.models import (
    ConceptSelectionResponse,
    FilterResult,
    FinalFilterResult,
    FinalOrderResult,
    InitialParseResponse,
    IntermediateParseResults,
    OrderResult,
    # TokenInputs,
    SemanticTokenResponse,
)
from preql_nlp.prompts import (
    FilterRefinementCase,
    SelectionPromptCase,
    SemanticExtractionPromptCase,
    SemanticToTokensPromptCase,
    run_prompt,
)
from preql_nlp.tokenization import build_token_list_by_purpose, tokens_to_concept


def get_phrase_from_x(x: Union[str, FilterResult, OrderResult]):
    if isinstance(x, str):
        return x
    elif isinstance(x, FilterResult):
        return x.concept
    elif isinstance(x, OrderResult):
        return x.concept
    raise ValueError


def tokenize_phrases(
    phrase_list: List[str], tokens: List[str], session_uuid, log_info: bool
) -> SemanticTokenResponse:
    phrase_tokens: SemanticTokenResponse = run_prompt(  # type: ignore
        SemanticToTokensPromptCase(phrases=phrase_list, tokens=tokens),
        debug=True,
        session_uuid=session_uuid,
        log_info=log_info,
    )
    phrase_tokens.__root__ = [
        x for x in phrase_tokens.__root__ if x.phrase in phrase_list
    ]
    return phrase_tokens


def concept_names_from_token_response(
    phrase_tokens: SemanticTokenResponse, concepts: dict[str, Concept]
) -> list[str]:
    token_universe = []
    output: list[str] = []

    for mapping in phrase_tokens:
        token_universe += mapping.tokens
    for mapping in phrase_tokens:
        found = False
        concepts_str_matches = tokens_to_concept(
            mapping.tokens,
            [c for c in concepts.keys()],
            limits=5,
            universe=token_universe,
        )
        if concepts_str_matches:
            logger.info(f"For phrase {mapping.phrase} got {concepts_str_matches}")
            output += concepts_str_matches
            found = True
            break
        if not found:
            raise ValueError(
                f"Could not find concept for input {mapping.phrase} with tokens {mapping.tokens} and concepts {list(concepts.keys())}"
            )
    return output


def enrich_filter(input: FinalFilterResult, log_info: bool, session_uuid):
    if not (input.concept.metadata and input.concept.metadata.description):
        return input
    input.values = run_prompt(  # type: ignore
        FilterRefinementCase(
            values=input.values,
            description=input.concept.metadata.description,
        ),
        session_uuid=session_uuid,
        log_info=log_info,
    ).new_values


def discover_inputs(
    input_text: str,
    input_environment: Environment,
    debug: bool = False,
    log_info: bool = True,
) -> IntermediateParseResults:
    # the core logic flow

    # DETERMINSTIC: setup logging
    # LLM: semantic extraction
    # LLM: tokenization of all strings (reduce search space over all concepts)
    # DETERMINISTIC: concept candidates from tokens (map reduced search space to candidates)
    # LLM: final selection of concepts (final get the concept result we want)
    # DETERMINISTIC: map phrases to final concepts
    # LLM: enrich filter values
    # DETERMINISTIC: return results

    # DETERMINISTIC: setup logging
    session_uuid = uuid.uuid4()
    env_concepts = input_environment.concepts

    # LLM: semantic extraction
    parsed: InitialParseResponse = run_prompt(  # type:ignore
        SemanticExtractionPromptCase(input_text),
        debug=debug,
        log_info=log_info,
        session_uuid=session_uuid,
    )

    # LLM: tokenization of all strings (reduce search space over all concepts)
    concept_candidates = []
    global_phrase_token_map = {}
    for semantic_category, valid_purposes in {
        "metrics": [Purpose.METRIC],
        "dimensions": [Purpose.KEY, Purpose.PROPERTY, Purpose.CONSTANT],
        "filtering": list(Purpose),
        "order": list(Purpose),
    }.items():
        category_tokens = build_token_list_by_purpose(env_concepts, valid_purposes)
        local_phrases = [
            get_phrase_from_x(x) for x in getattr(parsed, semantic_category)
        ]
        # skip if we have no relevant phrases
        if not local_phrases:
            continue
        token_mapping = tokenize_phrases(
            local_phrases, category_tokens, log_info=log_info, session_uuid=session_uuid
        )

        for item in token_mapping:
            if not all([x in category_tokens] for x in item.tokens):
                invalid = [x for x in item.tokens if x not in category_tokens]
                raise ValueError(
                    f"Phrase {item.phrase} return invalid tokens {invalid}"
                )
            global_phrase_token_map[item.phrase] = item.tokens
        concept_candidates += concept_names_from_token_response(
            token_mapping, env_concepts
        )

    # DETERMINISTIC: concept candidates from tokens (map reduced search space to candidates)
    selections: ConceptSelectionResponse = run_prompt(  # type: ignore
        SelectionPromptCase(concept_names=concept_candidates, question=input_text),
        debug=debug,
        session_uuid=session_uuid,
        log_info=log_info,
    )
    selected_concepts = list(set(selections.matches))

    for selection in selected_concepts:
        if selection not in env_concepts:
            raise ValueError(
                f"Returned concept {selection} that does not actually exist in environment!"
            )

    # DETERMINISTIC: map phrases to final concepts
    phrase_to_concept_map: dict[str, Concept] = {}
    for key, tokens in global_phrase_token_map.items():
        mapped = tokens_to_concept(concepts=selected_concepts, tokens=tokens, limits=1)
        if mapped:
            phrase_to_concept_map[key] = env_concepts[mapped[0]]

    final_ordering = [
        FinalOrderResult(
            concept=phrase_to_concept_map[order.concept], order=order.order
        )
        for order in parsed.order
        if phrase_to_concept_map[order.concept]
    ]

    final_filters_pre = [
        FinalFilterResult(
            concept=phrase_to_concept_map[filter.concept],
            operator=filter.operator,
            values=filter.values,
        )
        for filter in parsed.filtering
    ]

    # LLM: enrich filter values
    for item in final_filters_pre:
        enrich_filter(item, log_info=log_info, session_uuid=session_uuid)
    # DETERMINISTIC: return results
    return IntermediateParseResults(
        select=[env_concepts[c] for c in selected_concepts],
        limit=parsed.limit or 20,
        order=final_ordering,
        filtering=final_filters_pre,
    )


def safe_limit(input: int | None) -> int:
    if not input:
        return DEFAULT_LIMIT
    if input in (-1, 0):
        return DEFAULT_LIMIT
    return input


def parse_order(
    input_concepts: List[Concept], ordering: List[FinalOrderResult] | None
) -> OrderBy:
    default_order = [
        OrderItem(expr=c, order=Ordering.DESCENDING)
        for c in input_concepts
        if c.purpose == Purpose.METRIC
    ]
    if not ordering:
        return OrderBy(default_order)
    final = []
    for order in ordering:
        final.append(OrderItem(expr=order.concept, order=order.order))
    return OrderBy(items=final)


def parse_filter(input: FinalFilterResult) -> Comparison | None:
    return Comparison(
        left=input.concept,
        right=input.values[0] if len(input.values) == 1 else input.values,
        operator=input.operator,
    )


def parse_filtering(filtering: List[FinalFilterResult]) -> WhereClause | None:
    base = []
    for item in filtering:
        parsed = parse_filter(item)
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


def parse_query(
    input_text: str,
    input_environment: Environment,
    debug: bool = False,
    log_info: bool = True,
):
    intermediate_results = discover_inputs(
        input_text, input_environment, debug=debug, log_info=log_info
    )
    order = parse_order(intermediate_results.select, intermediate_results.order)

    filtering = parse_filtering(intermediate_results.filtering)
    # from preql.core.models import unique
    # concepts = unique(concepts, 'address')
    if debug:
        print("Concepts found")
        for c in intermediate_results.select:
            print(c.address)
    query = Select(
        selection=unique(intermediate_results.select, "address"),
        limit=safe_limit(intermediate_results.limit),
        order_by=order,
        where_clause=filtering,
    )
    return query


def build_query(
    input_text: str,
    input_environment: Environment,
    debug: bool = False,
    log_info: bool = True,
) -> ProcessedQuery:
    query = parse_query(input_text, input_environment, debug=debug, log_info=log_info)
    return process_query_v2(statement=query, environment=input_environment)
