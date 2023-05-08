from promptimize.prompt_cases import PromptCase
from promptimize import evals
from promptimize.suite import Suite
from promptimize.reports import Report
from preql_nlp.prompts.final_selection import gen_selection_v1
from preql_nlp.prompts.prompt_executor import SelectionPromptCase


def test_selection_prompt(test_logger):
    concepts = [
        "question.creation_date.year",
        "question.id",
        "question.id.count",
        "question.author",
    ]
    prompt = SelectionPromptCase(
        concepts=concepts, question="How many questions are asked per year?",
        evaluators=[
            lambda x: evals.all_words(
                x, ["question.creation_date.year", "question.id.count"]
            )
        ],
    )

    suite = Suite([prompt])
    output = suite.execute(
        verbose=True  # verbose,
        # style=style,
        # silent=silent,
        # report=report,
        # dry_run=dry_run,
        # keys=key,
        # force=force,
        # repair=repair,
        # human=human,
        # shuffle=shuffle,
        # limit=limit,
    )
    for prompt in suite.prompts.values():
        # print(prompt.response)
        test_logger.info(prompt.response)
    if output:
        output_report = Report.from_suite(suite)
        # print results to log
        test_logger.info(output_report)
        # actually fail the test if something doesnt pass
        for key in output_report.failed_keys:
            raise Exception(f"Failed key: {key}")
