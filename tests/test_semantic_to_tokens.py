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


output    # 347938152690149153

    test1 = generate_test_case(
        SemanticToTokensPromptCase,
        tests=gen_semantic_tests({"number of questions per year": ["question", "year", "count"]}),
        phrases=["questions per year"],
        tokens=["question", "year", "count", "albatross", "answer", "month", "quarter"],
    )

    evaluate_cases([test1])
