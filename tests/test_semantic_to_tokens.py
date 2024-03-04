from preql_nlp.prompts import SemanticToTokensPromptCase
from preql_nlp.models import SemanticTokenMatch
from tests.utility import generate_test_case, evaluate_cases


def gen_semantic_tests(match_dict):
    output = []
    for key, tokens in match_dict.items():

        def test_case(input: list[SemanticTokenMatch]):
            candidate = [x for x in input if x.phrase == key]
            if not candidate:
                return False
            matched: SemanticTokenMatch = candidate[0]
            return set(tokens) == set(matched.tokens)

        output.append(test_case)
    return output


def test_structured_input():
    first_phrase = "number of questions per year"
    test1 = generate_test_case(
        SemanticToTokensPromptCase,
        tests=gen_semantic_tests({first_phrase: ["question", "year", "count"]}),
        phrases=[first_phrase],
        tokens=["question", "year", "count", "albatross", "answer", "month", "quarter"],
        purpose="calculated metrics",
    )

    evaluate_cases([test1])
