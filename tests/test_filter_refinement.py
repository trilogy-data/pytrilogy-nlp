from preql_nlp.prompts.prompt_executor import FilterRefinementCase
from preql_nlp.models import FilterRefinementResponse
from tests.utility import generate_test_case, evaluate_cases


def gen_select_test(word):
    def select_test(x: FilterRefinementResponse):
        return x.new_value == word

    return [select_test]

def test_filter_refinement():
    case1 = generate_test_case(
        FilterRefinementCase,
        values=["California",],
        description="Field storing two digit state codes, ex MA for Massachusetts",
        tests=gen_select_test(["CA",],
            
        ),
    )

    case2 = generate_test_case(
        FilterRefinementCase,
        values=["95%",],
        description="Field storing a float representing the percentage of the population that likes coconuts",
        tests=gen_select_test([".95",],
            
        ),
    )
    evaluate_cases([case1, case2])