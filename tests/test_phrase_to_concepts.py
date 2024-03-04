from preql_nlp.prompts.prompt_executor import SelectionPromptCase
from preql_nlp.models import FinalParseResponse
from tests.utility import generate_test_case, evaluate_cases


def gen_select_test(words, filters: list[str] | None = None):
    def select_test(x: FinalParseResponse):
        return set(x.selection) == set(words)

    if filters:

        def filter_test(x: FinalParseResponse):
            return set([z.concept for z in x.filtering]) == set(filters)  # type: ignore

        return [select_test, filter_test]
    return [select_test]


def test_selection_prompt(test_logger):
    test1 = generate_test_case(
        SelectionPromptCase,
        tests=gen_select_test(["question.creation_date.year", "question.id.count"]),
        question="How many questions are asked per year?",
        concept_names=[
            "question.creation_date.year",
            "question.id",
            "question.id.count",
            "question.author",
        ],
    )

    test2 = generate_test_case(
        SelectionPromptCase,
        tests=gen_select_test(["question.author"], ["question.id"]),
        question="Author of question id 2?",
        concept_names=[
            "question.creation_date.year",
            "question.id",
            "question.id.count",
            "question.author",
        ],
    )
    evaluate_cases([test1, test2])
