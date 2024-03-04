from preql_nlp.prompts.prompt_executor import FilterRefinementCase
from preql_nlp.models import FilterRefinementResponse
from tests.utility import generate_test_case, evaluate_cases
from preql.core.enums import DataType


def gen_select_test(word: list[str]):
    def select_test(x: FilterRefinementResponse):
        print(x.new_values, word)
        return x.new_values == word

    return [select_test]


def test_filter_refinement():
    case1 = generate_test_case(
        FilterRefinementCase,
        values=[
            "California",
        ],
        datatype=DataType.STRING,
        description="The two character code for a state, such as MA for Massachusetts or CT for Connecticut.",
        tests=gen_select_test(
            [
                "CA",
            ],
        ),
    )

    case2 = generate_test_case(
        FilterRefinementCase,
        values=[
            "95%",
        ],
        datatype=DataType.FLOAT,
        description="Field storing a float representing the percentage of the population that likes coconuts",
        tests=gen_select_test(
            [
                0.95,
            ],
        ),
    )
    evaluate_cases([case1, case2])
