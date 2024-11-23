import difflib
import json
from collections import defaultdict
from enum import Enum
from typing import List, Optional

from langchain_core.tools import ToolException
from pydantic import BaseModel
from pydantic_core import ErrorDetails, ValidationError

# from trilogy.core.constants import
from trilogy.core.enums import (
    ComparisonOperator,
    FunctionType,
)
from trilogy.core.exceptions import UndefinedConceptException
from trilogy.core.models import (
    DataType,
    Environment,
)

from trilogy_nlp.constants import logger
from trilogy_nlp.helpers import is_relevent_concept
from trilogy_nlp.instrumentation import EventTracker
from trilogy_nlp.llm_interface.constants import COMPLICATED_FUNCTIONS
from trilogy_nlp.llm_interface.models import (
    Calculation,
    Column,
    FilterResultV2,
    InitialParseResponseV2,
    Literal,
    NLPConditions,
    OrderResultV2,
)

VALID_STATUS = "valid"


def is_valid_function(name: str):
    return name.lower() in [item.value for item in FunctionType]


def invalid_operator_message(operator: str) -> str | None:
    try:
        ComparisonOperator(operator)
    except Exception as e:
        return str(e)
    return None


VALIDATION_CACHE: dict[str, InitialParseResponseV2] = {}


class QueryContext(Enum):
    SELECT = "COLUMN"
    FILTER = "FILTER"
    ORDER = "ORDER"

    def __str__(self) -> str:
        if self == QueryContext.SELECT:
            return "output column declaration"
        return f"{self.value.lower()} declaration"


def validate_query(
    query: dict,
    environment: Environment,
    prompt: str,
    event_tracker: EventTracker | None = None,
) -> tuple[dict, InitialParseResponseV2 | None]:
    tracker = event_tracker or EventTracker()
    try:
        parsed = InitialParseResponseV2.model_validate(query)
    except ValidationError as e:
        tracker.count(tracker.etype.INITIAL_VALIDATION_PARSING_FAILED)
        return {"status": "invalid", "error": validation_error_to_string(e)}, None
    errors = []
    # assume to start that all our select calculations are valid
    select = {col.name for col in parsed.output_columns if col.calculation}
    top_select = {col.name for col in parsed.output_columns}
    filtered_on = set()

    def validate_calculation(calc: Calculation, context: QueryContext) -> bool:
        valid = True
        if not is_valid_function(calc.operator):
            errors.append(
                f"{calc} in {context} does not use a valid function (is using {calc.operator}); check that you are using ONLY a valid option from this list: {sorted([x for x in FunctionType.__members__.keys() if x not in COMPLICATED_FUNCTIONS]) }. If the column requires no transformation, drop the calculation field.",
            )
            valid = False
            tracker.count(tracker.etype.INVALID_FUNCTION)
        if calc.over:
            for x in calc.over:
                local = validate_column(x, context)
                valid = valid and local
            if calc.operator not in ["AVG", "SUM", "MIN", "MAX", "COUNT"]:
                errors.append(
                    f"{calc} in {context} should only use 'over' when using an aggregate function [AVG, SUM, MIN, MAX, COUNT] (is using {calc.operator}); remove this 'over' (make sure it's included in any parent aggregate if needed)",
                )
                tracker.count(tracker.etype.OVER_CLAUSE_WITHOUT_AGGREGATE)
                valid = False

        for arg in calc.arguments:
            if isinstance(arg, Column):
                if arg.name in environment.concepts:
                    dtype = environment.concepts[arg.name].datatype
                    if dtype == DataType.STRING:
                        errors.append(
                            f"Aggregate function {calc.operator} in {context} is being used on a string field {arg.name}; if you need this field in the output, just return it without a calculation. (Don't forget to set the full name {arg.name} when you drop the calculation!)",
                        )
                        valid = False
                        tracker.count(tracker.etype.STRING_FIELD_WITH_AGGREGATE)
            if isinstance(arg, Column):
                local = validate_column(arg, context)
                valid = valid and local
        return valid

    def validate_literal(lit: Literal, context: QueryContext) -> bool:
        if isinstance(lit.value, str) and lit.value in environment.concepts:
            return True
        elif isinstance(lit.value, Calculation):
            return validate_calculation(lit.value, context)
        return True

    def validate_column(col: Column, context: QueryContext) -> bool:
        valid = False
        if (
            col.name not in select
            and col.name not in environment.concepts
            and not col.calculation
        ):
            recommendations = None
            tracker.count(tracker.etype.INVALID_COLUMN_NAME_NO_CALCULATION)
            try:
                environment.concepts[col.name]
            except UndefinedConceptException as e:
                recommendations = e.suggestions
            if context == QueryContext.FILTER:
                hint = "add another nested column with a calculation to define it in this filter"
            else:
                hint = "add an output column to calculate it"
            if recommendations:
                errors.append(
                    f"'{col.name}' in {context} is not a valid preexisting field returned by the get fields tool or previously defined by you; if this is a new calculated column you need to answer the question, {hint}. Did you want to use one of these? {recommendations}?",
                )
            else:
                errors.append(
                    f"'{col.name}' in {context} is not a valid preexisting field returned by the get fields tool or previously defined by you. If this is a new calculated column you need to answer the question, {hint}. If you just misspelled an existing field, maybe it should be one of {difflib.get_close_matches(col.name, [k for k,v in environment.concepts.items() if is_relevent_concept(v)], 3, 0.4)}",
                )
        elif col.name in environment.concepts:
            valid = True
            if col.calculation:
                tracker.count(tracker.etype.CALCULATION_WITH_PREDEFINED_FIELD)
                errors.append(
                    f"{col.name} in {context} is a predefined field and should not have a calculation; check that you are using the field as is, without any additional transformations.",
                )
                valid = False
        elif col.calculation:
            local = validate_calculation(col.calculation, context)
            valid = valid and local
        if valid:
            if context == QueryContext.SELECT:
                select.add(col.name)
            elif context == QueryContext.FILTER:
                filtered_on.add(col.name)
        return valid

    for x in parsed.output_columns:
        validate_column(x, QueryContext.SELECT)

    if parsed.order:
        for y in parsed.order:
            if y.column_name not in top_select:
                tracker.count(tracker.etype.ORDER_BY_NOT_SELECTED)
                errors.append(
                    f"{y.column_name} being ordered by is not in output_columns, add if needed.",
                )
    if parsed.filtering:
        root = parsed.filtering.root
        for val in root.values:
            if isinstance(val, Column):
                validate_column(val, QueryContext.FILTER)

            elif isinstance(val, NLPConditions):
                for subval in [val.left, val.right]:
                    if isinstance(subval, Column):
                        validate_column(subval, QueryContext.FILTER)
                    elif isinstance(subval, Literal):
                        validate_literal(subval, QueryContext.FILTER)
                if isinstance(val.left, Column) and isinstance(val.right, Column):
                    if val.left.name == val.right.name:
                        errors.append(
                            f"Comparison {val} has the same column name on both sides; this is a meaningless comparison. Check that you are comparing two different fields.",
                        )

    if errors:
        return {
            "status": "invalid",
            "errors": {f"Error {idx+1}: {error}" for idx, error in enumerate(errors)},
        }, parsed
    tracker.count(tracker.etype.QUERY_VALIDATION_PASSED)
    tips = [
        f'Congrats! No validation errors - looking good! A few final checks - if you believe these do not apply, go ahead and submit this response as your final answer! (or make a tweak or two!) First, check that you have all fields you think are required in the output. Even better; keep it clean - if an output column is added only to be used in filtering and not explicitly asked for, you might be able to move it out of the select (move the entire definition into a nested reference).  The original prompt was: "{prompt}"!'
    ]
    for all_address in select.union(filtered_on):
        if all_address in environment.concepts:
            concept = environment.concepts[all_address]
            if concept.metadata.description:
                tips.append(
                    f'And another one - for the field {all_address}, reminder that the field description is "{concept.metadata.description}". Double check that any filtering on this field matches the described format! (this is a reminder, not an error)'
                )
    VALIDATION_CACHE[prompt] = parsed
    return {"status": VALID_STATUS, "tips": tips}, parsed


def validation_error_to_string(err: ValidationError):
    # inject in new context on failed answer
    raw_error = str(err)
    errors = err.errors()
    logger.error(f"Validation error: {errors}")
    if "filtering.root." in raw_error:
        missing = []
        path_freq: dict[str, int] = defaultdict(lambda: 0)
        path_map: dict[str, list[str]] = defaultdict(lambda: [])
        for validation_error in errors:
            e: ErrorDetails = validation_error
            if e["type"] == "missing":
                path = "".join([str(x) for x in e["loc"][:-1]])
                path_freq[path] += 1
                key = e["loc"][-1]
                message = f'Missing "{e["loc"][-1]}" in this JSON object you provided: {json.dumps(e["input"], indent=4)}. You may also be using the wrong object. Double check Literal and Column formats.'
                if key == "type" and "value" in e["input"]:
                    nested = e["input"]["value"]
                    if "type" in nested:
                        message = f'Missing "{e["loc"][-1]}". You have have overly nested a literal in {json.dumps(e["input"], indent=4)}. Flatten the inner object {json.dumps(nested, indent=4)} out into the parent.'
                path_map[path].append(message)
                missing.append(message)
            elif e["type"] == "extra_forbidden":
                path = "".join([str(x) for x in e["loc"][:-1]])
                path_freq[path] += 1
                message = f'Extra invalid key "{e["loc"][-1]}" in this JSON object you provided: {json.dumps(e["input"], indent=4)}'
                path_map[path].append(message)
                missing.append(message)
            else:
                missing_path = ".".join(
                    [
                        str(v)
                        for v in e["loc"]
                        if str(v)
                        not in [
                            "NLPComparisonGroup",
                            "NLPConditions",
                            "Literal",
                            "Column",
                        ]
                    ]
                )
                missing.append(missing_path)
        min_path = min(path_freq, key=path_freq.get)  # type: ignore
        locations = path_map[min_path]
        raw_error = f"Syntax error in your filtering clause. Check that you only use valid keys for Literal, Comparison, and Column objects, and have all required keys. Look at these locations: {locations}"
    else:
        for e in errors:
            if e["type"] == "missing":
                raw_error = f'Could not parse {e["input"]}. Maybe missing {e["loc"][-1]}? You might have incorrect JSON formatting or the key is actually missing. Double check Literal and Column formats.'
    return raw_error


def validate_response(
    environment: Environment,
    prompt: str,
    output_columns: List[Column],
    filtering: Optional[FilterResultV2] = None,
    order: Optional[list[OrderResultV2]] = None,
    limit: int | None = None,
    event_tracker: EventTracker | None = None,
) -> tuple[dict, InitialParseResponseV2 | None]:

    base: dict[str, list[Column] | FilterResultV2 | list[OrderResultV2] | int] = {
        "output_columns": output_columns
    }
    if not output_columns:
        raise ToolException(
            "A request to validate should include at least one column. The input should be the exactly same as the input for your final answer. Generate your answer, call this tool again, and then submit it if it passes."
        )
    if filtering:
        base["filtering"] = filtering
    if order:
        base["order"] = order
    if limit:
        base["limit"] = limit
    return validate_query(
        base,
        environment=environment,
        prompt=prompt,
        event_tracker=event_tracker,
    )


class ValidateResponseInterface(BaseModel):
    # deliberately permissive interface
    # so that we can handle the error inside the tool
    output_columns: List[dict]
    filtering: Optional[dict] = None
    order: Optional[list[dict]] = None
    limit: int | None = None
