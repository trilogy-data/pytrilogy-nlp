from preql.core.models import (
    Select,
    ProcessedQuery,
    Concept,
    OrderItem,
    OrderBy,
    Ordering,
)
from preql.core.query_processor import process_query_v2
from preql.core.enums import Purpose
from typing import Iterable, List
from collections import defaultdict

from preql.core.models import Environment
from preql_nlp.prompts import (
    run_prompt,
    SemanticToTokensPromptCase,
    SelectionPromptCase,
    SemanticExtractionPromptCase,
)
from preql_nlp.constants import logger, DEFAULT_LIMIT

from dataclasses import dataclass
from typing import Any

import re
import uuid

def split_to_tokens(input_text: str) -> list[str]:
    return list(set(re.split("\.|\_", input_text)))


def build_token_list_by_purpose(concepts, purposes: Iterable[Purpose]):
    concepts = {k: v for k, v in concepts.items() if v.purpose in purposes}

    unique = set(concepts.keys())
    final = set()
    # if a concept has another concept as a substring, remove
    # to help restrict the size of the search space
    for concept in unique:
        # matched = [comparison  for comparison in unique if comparison in concept]
        # filtered = [m for m in matched if m !=concept]
        # if not filtered:
        #     final.append(concept)
        for x in split_to_tokens(concept):
            final.add(x)
    return ", ".join(list(final))


def tokens_to_concept(
    tokens: list[str],
    concepts: List[str],
    limits: int = 5,
    universe: list[str] | None = None,
):
    tokens = list(set(tokens))
    universe_list = list(set(universe or []))
    mappings = {x: split_to_tokens(x) for x in concepts}
    candidates = [x for x in concepts if all(token in mappings[x] for token in tokens)]
    pickings = defaultdict(list)

    if not candidates:
        return None
    tiers = set()
    for candidate in candidates:
        tier = sum(
            [1 if token in mappings[candidate] else 0 for token in universe_list]
        )
        pickings[tier].append(candidate)
        tiers.add(tier)
    tier_list = sorted(list(tiers), key=lambda x: -x)

    found: List[str] = []
    while len(found) < limits and tier_list:
        stier = tier_list.pop(0)
        candidates = sorted(pickings[stier], key=lambda x: len(x))
        required = limits - len(found)
        found += candidates[:required]
    return found


@dataclass
class IntermediateParseResults:
    select: List[str]
    limit: int
    order: List[str]
    # filter: List[FilterItem]


def coerce_list_dict(input: Any) -> List[dict]:
    if not isinstance(input, list):
        raise ValueError("Input must be a list")
    for x in input:
        if not isinstance(x, dict):
            raise ValueError("Input must be a list of dicts")
    return input


def coerce_list_str(input: Any) -> List[str]:
    if not isinstance(input, list):
        raise ValueError("Input must be a list")
    for x in input:
        if not isinstance(x, str):
            raise ValueError("Input must be a list of dicts")
    return input


def discover_inputs(
    input_text: str, input_environment: Environment, debug: bool = False, log_info: bool = True
) -> IntermediateParseResults:
    # we work around prompt size issues and hallucination by doing a two phase discovery
    # first we parse the question into metrics/dimensions
    # then for each one, we match those to a token list
    # and then we deterministically map the tokens back to the most relevant concept
    # TODO: turn the third pass into a prompt
    concepts = input_environment.concepts
    debug = True
    final_concept_list = list(concepts.keys())
    metrics = build_token_list_by_purpose(concepts, (Purpose.METRIC,))
    dimensions = build_token_list_by_purpose(
        concepts, (Purpose.KEY, Purpose.PROPERTY, Purpose.CONSTANT)
    )

    session_uuid = uuid.uuid4()

    parsed = coerce_list_dict(
        run_prompt(SemanticExtractionPromptCase(input_text), debug=debug, log_info=log_info, session_uuid=session_uuid)
    )[0]
    order = parsed.get("order", [])
    token_inputs = {"metrics": metrics, "dimensions": dimensions}

    output: List[str] = []
    for field in [
        "metrics",
        "dimensions",
    ]:
        local_phrases = [x for x in parsed[field]]
        phrase_tokens = coerce_list_dict(
            run_prompt(
                SemanticToTokensPromptCase(
                    phrases=local_phrases, tokens=token_inputs[field]
                ),
                debug=True,
                session_uuid=session_uuid,
                log_info=log_info
            )
        )
        token_universe = []
        for mapping in phrase_tokens:
            for k, v in mapping.items():
                token_universe += v
        for mapping in phrase_tokens:
            for k, v in mapping.items():
                concepts = tokens_to_concept(
                    v,
                    [c for c in final_concept_list],
                    limits=5,
                    universe=token_universe,
                )
                logger.info(f"For phrase {k} got {concepts}")
                if concepts:
                    output += concepts
                else:
                    raise ValueError(
                        f"Could not find concept for input {k} with tokens {v}"
                    )
    selections = coerce_list_dict(
        run_prompt(SelectionPromptCase(concepts=output, question=input_text), debug=debug, session_uuid=session_uuid, log_info=log_info)
    )[0]
    final = list(set(selections.get("matches", [])))

    return IntermediateParseResults(
        select=final, limit=parsed.get("limit", 20), order=order
    )


def safe_limit(input: int | None) -> int:
    if not input:
        return DEFAULT_LIMIT
    if input in (-1, 0):
        return DEFAULT_LIMIT
    return input


def safe_parse_order_item(
    input: str, input_concepts: List[Concept]
) -> OrderItem | None:
    concept, direction = input.split(" ", 1)
    matched = [c for c in input_concepts if concept in c.address]
    if not matched:
        return None
    try:
        order = Ordering(direction)
    except Exception:
        return None
    return OrderItem(expr=matched[0], order=order)


def parse_order(input_concepts: List[Concept], ordering: List[str] | None) -> OrderBy:
    default_order = [
        OrderItem(expr=c, order=Ordering.DESCENDING)
        for c in input_concepts
        if c.purpose == Purpose.METRIC
    ]
    if not ordering:
        return OrderBy(default_order)
    final = []
    for order in ordering:
        parsed = safe_parse_order_item(order, input_concepts)
        if parsed:
            final.append(parsed)
    return OrderBy(items=final)


def parse_query(
    input_text: str,
    input_environment: Environment,
    debug: bool = False,
    log_info: bool=True
):
    results = discover_inputs(input_text, input_environment, debug=debug, log_info=log_info)
    concepts = [input_environment.concepts[x] for x in results.select]
    order = parse_order(concepts, results.order)
    if debug:
        print("Concepts found")
        for c in concepts:
            print(c.address)
    query = Select(selection=concepts, limit=safe_limit(results.limit), order_by=order)
    return query



def build_query(
    input_text: str,
    input_environment: Environment,
    debug: bool = False,
    log_info: bool = True
) -> ProcessedQuery:
    query = parse_query(input_text, input_environment, debug=debug, log_info=log_info)
    return process_query_v2(statement=query, environment=input_environment)
