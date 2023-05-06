from promptimize.prompt_cases import PromptCase
from promptimize import evals
from promptimize.suite import Suite
from promptimize.reports import Report
from preql_nlp.prompts.query_semantic_extraction import gen_extraction_prompt_v1
from promptimize.utils import extract_json_objects
from typing import List


def validate_object(input: str, field: str, matches: List[str]):
    jobject = extract_json_objects(input)[0]
    field = jobject.get(field, "")
    return all([x in field for x in matches])


def test_extraction_prompt(test_logger):
    prompt = gen_extraction_prompt_v1(
        input="How many questions are asked per year? Order results by year desc"
    )

    case = PromptCase(
        user_input=prompt,
        evaluators=[
            lambda x: evals.all_words(x, ["questions", "asked", "per", "year"]),
            lambda x: validate_object(x, "order_by", ["year", "desc"]),
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
        test_logger.info("reporting output")
        output_report = Report.from_suite(suite)
        # print results to log
        test_logger.info(output_report)
        # actually fail the test if something doesnt pass
        for key in output_report.failed_keys:
            raise Exception(f"Failed key: {key}")
