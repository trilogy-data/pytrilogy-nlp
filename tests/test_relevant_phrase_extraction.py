from preql_nlp.prompts.prompt_executor import SemanticExtractionPromptCase
from preql_nlp.models import InitialParseResponse, FilterResult, OrderResult
from tests.utility import generate_test_case, evaluate_cases
from preql.core.enums import Ordering, ComparisonOperator

def flatten_arg_list(obj, args):
    output = []
    for arg in args:
        output += getattr(obj, arg)
    return output


def gen_validate_initial_parse_result(**kwargs):
    outputs = []
    for key, test_values in kwargs.items():

        def validator(input: InitialParseResponse):
            # coerce any of the object results
            # like filtering
            # to a string for simplicity
            #
            field_values: list[str] = [x for x in getattr(input, key)]
            # if all of the values appear in at least one of the sub phrases
            check = all(
                [
                    any(
                        (test in partial or test == partial) for partial in field_values
                    )
                    for test in test_values
                ]
            )
            if not check:
                print(
                    "could not find ", test_values, " in ", key, " with ", field_values
                )
            return check

        outputs.append(validator)
    return outputs


def test_extraction_prompt(test_logger):
    case1 = generate_test_case(
        SemanticExtractionPromptCase,
        question="How many questions are asked per year? Order results by year desc",
        tests=gen_validate_initial_parse_result(
            selection=[
                "questions",
                "asked",
                "per",
                "year",
            ],
            order=[OrderResult(concept="year", order=Ordering.DESCENDING)],
        ),
    )

    case2 = generate_test_case(
        SemanticExtractionPromptCase,
        question="How many questions were asked in the year 2020?",
        tests=gen_validate_initial_parse_result(
            selection=["question", "count"],
            filtering=[FilterResult(concept="year", values=["2020"], operator =ComparisonOperator.EQ)],
        ),
    )
    evaluate_cases([case1, case2])
