from promptimize.prompt_cases import PromptCase
from promptimize import evals
from promptimize.suite import Suite
from promptimize.reports import Report
from preql_nlp.prompts import SemanticToTokensPromptCase


def test_structured_input():
    phrases = ["questions per year"]
    tokens = ["question", "year", "count"]
    prompt = SemanticToTokensPromptCase(phrases=phrases, tokens=tokens)


    suite = Suite([prompt])
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
