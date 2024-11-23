from typing import Callable, List

from promptimize.reports import Report
from promptimize.suite import Suite
from promptimize.utils import serialize_object
from pydantic import BaseModel

from trilogy_nlp.prompts.prompt_executor import BasePreqlPromptCase


def generate_test_case(
    prompt: BasePreqlPromptCase,
    tests: List[Callable[[BaseModel], bool]],
    llm,
    **kwargs,
):
    evaluators = []
    evaluators.append(lambda x: prompt.parse_response(x) is not None)
    for idx, test in enumerate(tests):
        evaluators.append(lambda x, idx=idx, test=test: test(prompt.parse_response(x)))  # type: ignore

    case = prompt(**kwargs, evaluators=evaluators, llm=llm)
    return case


def evaluate_cases(cases: List[BasePreqlPromptCase]):
    suite = Suite(cases)
    suite.execute(
        silent=True
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
    output_report = Report.from_suite(suite)
    # print results to log
    print(
        serialize_object(output_report.data.to_dict(), highlighted=False, style="yaml")
    )
    print(output_report.print_summary())
    # actually fail the test if something doesnt pass
    for key in output_report.failed_keys:
        raise Exception(f"Failed key: {key}")
