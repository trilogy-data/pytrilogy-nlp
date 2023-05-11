from preql_nlp.prompts.prompt_executor import SelectionPromptCase
from preql_nlp.models import ConceptSelection
from tests.utility import generate_test_case, evaluate_cases


def gen_select_test(words):
    def select_test(x: ConceptSelection):
        return set(x.matches) == set(words)

    return [select_test]


def test_selection_prompt(test_logger):
    test1 = generate_test_case(
        SelectionPromptCase,
        tests=gen_select_test(["question.creation_date.year", "question.id.count"]),
        question="How many questions are asked per year?",
        concepts=[
            "question.creation_date.year",
            "question.id",
            "question.id.count",
            "question.author",
        ],
    )

    test2 = generate_test_case(
        SelectionPromptCase,
        tests=gen_select_test(["question.author", "question.id"]),
        question="Author of question id 2?",
        concepts=[
            "question.creation_date.year",
            "question.id",
            "question.id.count",
            "question.author",
        ],
    )
    evaluate_cases([test1, test2])
