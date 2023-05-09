from promptimize import evals
from promptimize.suite import Suite
from promptimize.reports import Report
from preql_nlp.prompts.prompt_executor import SemanticExtractionPromptCase
from promptimize.utils import extract_json_objects
from typing import List


def validate_object(input: str, field: str, matches: List[str]):
    jobject = extract_json_objects(input)[0]
    field = jobject.get(field, "")
    return all([x in field for x in matches])


def generate_test_case(
    question: str,  test_logger, select:list[str] | None =None, where:list[str] | None= None, 
    order:list[str] | None =None, **kwargs
):  
    evaluators = []
    if select:
        evaluators.append(            lambda x: validate_object(x, "order_by", ["year", "desc"]),)
    if order:
        evaluators.append(            lambda x: validate_object(x, "order_by", ["year", "desc"]),)
    if where:
        evaluators.append(            lambda x: validate_object(x, "order_by", ["year", "desc"]),)
    case = SemanticExtractionPromptCase(
        question=question,
        evaluators=evaluators,
        **kwargs,
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

def test_extraction_prompt(test_logger):
    prompt = "How many questions are asked per year? Order results by year desc"

    case = SemanticExtractionPromptCase(
        question=prompt,
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
