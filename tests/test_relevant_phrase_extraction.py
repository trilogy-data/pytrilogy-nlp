from preql_nlp.prompts.prompt_executor import SemanticExtractionPromptCase
from preql_nlp.models import InitialParseResponse, FilterResult, OrderResult
from tests.utility import generate_test_case, evaluate_cases
from preql.core.enums import Ordering, ComparisonOperator
from pydantic import BaseModel

from typing import List


def flatten_arg_list(obj, args):
    output = []
    for arg in args:
        output += getattr(obj, arg)
    return output


def coerce_list(item):
    if not isinstance(item, list):
        return [item]
    return item


def validator_factory(key, test_values):
    def validator(input: InitialParseResponse, lkey=key, test_values=test_values):
        lkey = key
        ltest = coerce_list(test_values)
        field_values: list[str] = [x for x in coerce_list(getattr(input, lkey))]
        # if all of the values appear in at least one of the sub phrases
        check = all(
            [
                any(
                    (test == partial or (isinstance(partial, str) and test in partial))
                    for partial in field_values
                )
                for test in ltest
            ]
        )
        if not check:
            raise ValueError(
                "could not find ",
                test_values,
                " in ",
                key,
                " with ",
                field_values,
                "from",
                str(input),
            )
        return check

    return validator


def gen_validate_initial_parse_result(**kwargs):
    outputs = []
    for key, test_values in kwargs.items():
        # CRITICAL
        # we need to assign the values here in the loop to defaults on the lambda
        # to avoid lazy binding
        outputs.append(
            lambda x, test_values=test_values, key=key: validator_factory(
                key, test_values
            )(x)
        )
    return outputs


def test_extraction_prompt(test_logger):
    case1 = generate_test_case(
        SemanticExtractionPromptCase,
        question="How many questions are asked per year? Order results by year desc",
        tests=gen_validate_initial_parse_result(
            selection=["question", "year", "count"],
            order=[OrderResult(concept="year", order=Ordering.DESCENDING)],
        ),
    )

    case2 = generate_test_case(
        SemanticExtractionPromptCase,
        question="How many questions were asked in the year 2020?",
        tests=gen_validate_initial_parse_result(
            selection=["question"],
            filtering=[
                FilterResult(
                    concept="year", values=["2020"], operator=ComparisonOperator.EQ
                )
            ],
        ),
    )

    case3 = generate_test_case(
        SemanticExtractionPromptCase,
        question="50 most common names by count in the state of VT in the year 2010?",
        tests=gen_validate_initial_parse_result(
            selection=[
                "name",
                "count",
                "state",
            ],
            limit=50,
            filtering=[
                FilterResult(concept="year", values=["2010"], operator="="),
                FilterResult(concept="state", values=["VT"], operator="="),
            ],
        ),
    )

    case4 = generate_test_case(
        SemanticExtractionPromptCase,
        question="What were the 50 most common names by count in Vermont in the year 2010?",
        tests=gen_validate_initial_parse_result(
            selection=["name", "count", "state", "year"],
            limit=50,
            filtering=[
                FilterResult(
                    concept="year", values=["2010"], operator=ComparisonOperator.EQ
                ),
                FilterResult(
                    concept="state", values=["Vermont"], operator=ComparisonOperator.EQ
                ),
            ],
        ),
    )
    evaluate_cases([case1, case2, case3, case4])


def test_like_predicates():
    case1 = generate_test_case(
        SemanticExtractionPromptCase,
        question="SO answers where the body contains the word jaguar?",
        tests=gen_validate_initial_parse_result(
            selection=["body"],
            filtering=[
                MultiModeFilterMatch(
                    valid=[
                        FilterResult(
                            concept="body",
                            values=["%jaguar%"],
                            operator=ComparisonOperator.LIKE,
                        ),
                        FilterResult(
                            concept="post body",
                            values=["%jaguar%"],
                            operator=ComparisonOperator.LIKE,
                        ),
                    ]
                )
            ],
        ),
    )
    evaluate_cases([case1])


class MultiModeFilterMatch(BaseModel):
    valid: List[FilterResult]

    def __eq__(self, x):
        return any([x == v for v in self.valid])


def test_abstract_terms():
    case1 = generate_test_case(
        SemanticExtractionPromptCase,
        question="Shoe sales on christmas day?",
        tests=gen_validate_initial_parse_result(
            selection=["product", "sale"],
            filtering=[
                MultiModeFilterMatch(
                    valid=[
                        FilterResult(
                            concept="day",
                            values=["Christmas"],
                            operator=ComparisonOperator.EQ,
                        ),
                        FilterResult(
                            concept="day",
                            values=["Christmas Day"],
                            operator=ComparisonOperator.EQ,
                        ),
                        FilterResult(
                            concept="day of week",
                            values=["Christmas Day"],
                            operator=ComparisonOperator.EQ,
                        ),
                        FilterResult(
                            concept="day of year",
                            values=["December 25"],
                            operator=ComparisonOperator.EQ,
                        ),
                        FilterResult(
                            concept="day of year",
                            values=["12/25"],
                            operator=ComparisonOperator.EQ,
                        ),
                    ]
                )
            ],
        ),
    )
    evaluate_cases([case1])
