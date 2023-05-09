from preql_nlp.prompts.prompt_executor import SemanticExtractionPromptCase
from tests.utility import generate_test_case, evaluate_cases


def test_extraction_prompt(test_logger):

    case1 = generate_test_case(
        "How many questions are asked per year? Order results by year desc",
        SemanticExtractionPromptCase,
        test_logger,
        select=["questions", "asked", "per", "year",],
        order=["year", "desc"],
    )


    case2 = generate_test_case(
        "How many questions where asked in the year 2020?",
        SemanticExtractionPromptCase,
        test_logger,
        select=["question",],
        where=["year", "2020"],
    )

    evaluate_cases([case1, case2])
