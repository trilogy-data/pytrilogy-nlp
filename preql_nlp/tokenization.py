from typing import Iterable, Dict, List
from preql.core.models import Concept
from preql.core.enums import Purpose
import re
from collections import defaultdict


class Token(str):
    def __init__(*args):
        super.__init__(*args)


def split_to_tokens(input_text: str) -> list[str]:
    return list(set(re.split("\.|\_", input_text)))


def build_token_list_by_purpose(
    concepts: Dict[str, Concept], purposes: Iterable[Purpose]
) -> list[str]:
    concepts = {k: v for k, v in concepts.items() if v.purpose in purposes}

    unique = set(concepts.keys())
    final = set()
    for concept in unique:
        # matched = [comparison  for comparison in unique if comparison in concept]
        # filtered = [m for m in matched if m !=concept]
        # if not filtered:
        #     final.append(concept)
        for x in split_to_tokens(concept):
            final.add(x)
    return list(final)


def tokens_to_concept(
    tokens: list[str],
    concepts: List[str],
    limits: int = 5,
    universe: list[str] | None = None,
) -> list[str] | None:
    tokens = list(set(tokens))
    universe_list = list(set(universe or []))
    mappings = {x: split_to_tokens(x) for x in concepts}
    candidates = [x for x in concepts if any(token in mappings[x] for token in tokens)]
    pickings = defaultdict(list)

    if not candidates:
        return None
    tiers = set()

    for candidate in candidates:
        score = 0
        candidate_tokens = mappings[candidate]
        # exact match to concept is +2
        for x in tokens:
            if x in candidate_tokens:
                # print(f'+2 score from token {x} in candidate list')
                score += 10
        # universe context is + 1
        for y in universe_list:
            if y in candidate_tokens:
                # print(f"+1 score from token {y} in universe")
                score += 1
        pickings[score].append(candidate)
        tiers.add(score)
    tier_list = sorted(list(tiers), key=lambda x: -x)
    found: List[str] = []
    while len(found) < limits and tier_list:
        stier = tier_list.pop(0)
        # sort within list by length ascending, then alphabetical
        # for consistency
        candidates = sorted(pickings[stier], key=lambda x: (len(x), x))
        required = limits - len(found)
        found += candidates[:required]
    return found
