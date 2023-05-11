from promptimize.suite import Suite
from promptimize.reports import Report
from preql_nlp.prompts.prompt_executor import BasePreqlPromptCase
from promptimize.utils import serialize_object
from typing import List, Callable, Type
from pydantic import BaseModel


# def validate_object(input: str, fields: str | list[str], matches: List[str]) -> bool:
#     if isinstance(fields, str):
#         fields = [fields]
#     field_vals: list[str] = []
#     jobject = extract_json_objects(input)[0]
#     print(jobject)
#     for field in fields:
#         print('getting')
#         print(field)
#         field_vals+=jobject.get(field, [])
#     print('for matches', matches)
#     print(field_vals)
#     output = all([any(x in field for field in field_vals) for x in matches])
#     print('BOOLEAN')
#     print(output)
#     return output

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
