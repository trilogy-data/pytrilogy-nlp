from promptimize.suite import Suite
from promptimize.reports import Report
from preql_nlp.prompts.prompt_executor import BasePreqlPromptCase
from promptimize.utils import serialize_object
from typing import List, Callable
from pydantic import BaseModel


def validate_model(input:BaseModel, accessor:Callable[[BaseModel], List[str]], matches: List[str])->bool:
    field_vals = accessor(input)
    success = all([any(x in field for field in field_vals) for x in matches])
    return success

def generate_test_case(
    prompt: BasePreqlPromptCase,
    tests:List[Callable[[BaseModel], bool]],
    **kwargs,
):
    evaluators = []
    evaluators.append(lambda x: prompt.parse_model.parse_raw(x) is not None)
    for test in tests:
        evaluators.append(lambda x: test(prompt.parse_model.parse_raw(x)) )
    case = prompt(
        **kwargs,
        evaluators=evaluators,
    )
    return case

def evaluate_cases(cases:List[BasePreqlPromptCase]):
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
    print(serialize_object(output_report.data.to_dict(), highlighted=False, style="yaml"))
    # print(output_report.print_summary())
    # actually fail the test if something doesnt pass
    for key in output_report.failed_keys:
        raise Exception(f"Failed key: {key}")
