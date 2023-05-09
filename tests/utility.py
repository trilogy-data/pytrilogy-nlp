from promptimize.suite import Suite
from promptimize.reports import Report
from promptimize.utils import extract_json_objects
from promptimize.utils import serialize_object
from typing import List


def validate_object(input: str, fields: str | list[str], matches: List[str]) -> bool:
    if isinstance(fields, str):
        fields = [fields]
    field_vals: list[str] = []
    jobject = extract_json_objects(input)[0]
    print(jobject)
    for field in fields:
        print('getting')
        print(field)
        field_vals+=jobject.get(field, [])
    print('for matches', matches)
    print(field_vals)
    output = all([any(x in field for field in field_vals) for x in matches])
    print('BOOLEAN')
    print(output)
    return output


def generate_test_case(
    question: str,
    prompt,
    test_logger,
    select: list[str] | None = None,
    where: list[str] | None = None,
    order: list[str] | None = None,
    **kwargs,
):
    evaluators = []
    if select:
        select_check = select or []
        evaluators.append(
            lambda x: validate_object(x, ["dimensions", "metrics"], select_check),
        )

    if order:
        order_check = order or []
        evaluators.append(lambda x: validate_object(x, "order", order_check))
    if where:
        where_check = where or []
        evaluators.append(
            lambda x: validate_object(x, "filtering", where_check),
        )
    case = prompt(
        question=question,
        evaluators=evaluators,
        **kwargs,
    )
    return case

def evaluate_cases(cases):
    suite = Suite(cases)
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


    output_report = Report.from_suite(suite)
    # print results to log
    print(serialize_object(output_report.data.to_dict(), highlighted=False, style="yaml"))
    # print(output_report.print_summary())
    # actually fail the test if something doesnt pass
    for key in output_report.failed_keys:
        raise Exception(f"Failed key: {key}")
