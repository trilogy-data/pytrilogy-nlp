from promptimize.prompt_cases import PromptCase
from promptimize import evals
from promptimize.suite import Suite
from promptimize.reports import Report
from preql_nlp.prompts.query_semantic_extraction import gen_extraction_prompt_v1


def test_extraction_prompt():
    prompt = gen_extraction_prompt_v1(input="How many questions are asked per year?")

    case = PromptCase(
        user_input=prompt,
        evaluators=[
            lambda x: evals.all_words(
                x, ["questions", "asked", "per", "year", "How", "many", "are", "?"]
            )
        ],
    )

    suite = Suite([case])
    output = suite.execute(
        # verbose=verbose,
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
    if output:
        output_report = Report.from_suite(suite)
        # print results to log
        print(output_report)
        # actually fail the test if something doesnt pass
        for key in output_report.failed_keys:
            raise Exception(f"Failed key: {key}")
