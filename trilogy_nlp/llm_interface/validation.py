from typing import Optional
from trilogy.core.models import (
    Environment,
)
from typing import List
from pydantic import BaseModel, ValidationError
from trilogy.core.enums import ComparisonOperator
from langchain_core.tools import ToolException
from trilogy.core.exceptions import UndefinedConceptException
from collections import defaultdict

from trilogy_nlp.llm_interface.models import (
    InitialParseResponseV2,
    Column,
    NLPConditions,
    FilterResultV2,
    OrderResultV2,
    Literal,
    Calculation,
)
from trilogy_nlp.llm_interface.constants import COMPLICATED_FUNCTIONS
import difflib

# from trilogy.core.constants import
from trilogy.core.enums import (
    FunctionType,
)
from enum import Enum
import json

VALID_STATUS = "valid"


def is_valid_function(name: str):
    return name.lower() in [item.value for item in FunctionType]


def invalid_operator_message(operator: str) -> str | None:
    try:
        operator = ComparisonOperator(operator)
    except Exception as e:
        return str(e)
    return None


VALIDATION_CACHE: dict[str, InitialParseResponseV2] = {}


class QueryContext(Enum):
    SELECT = "COLUMN"
    FILTER = "FILTER"
    ORDER = "ORDER"

    def __str__(self) -> str:
        return f"{self.value} definition"


def validate_query(
    query: dict, environment: Environment, prompt: str
) -> tuple[dict, InitialParseResponseV2 | None]:
    try:
        parsed = InitialParseResponseV2.model_validate(query)
    except ValidationError as e:
        return {"status": "invalid", "error": validation_error_to_string(e)}, None
    errors = []
    # assume to start that all our select calculations are valid
    select = {col.name for col in parsed.columns if col.calculation}
    filtered_on = set()

    def validate_calculation(calc: Calculation, context: QueryContext) -> bool:
        if not is_valid_function(calc.operator):
            errors.append(
                f"{calc} in {context} does not use a valid function (is using {calc.operator}); check that you are using ONLY a valid option from this list: {sorted([x for x in FunctionType.__members__.keys() if x not in COMPLICATED_FUNCTIONS]) }. If the column requires no transformation, drop the calculation field.",
            )
            if calc.over:
                for x in calc.over:
                    local = validate_column(x, context)
                    valid = valid and local

            for arg in calc.arguments:
                if isinstance(arg, Column):
                    local = validate_column(arg, context)
                    valid = valid and local

    def validate_literal(lit: Literal, context: QueryContext) -> bool:
        if isinstance(lit.value, str) and lit.value in environment.concepts:
            return True
        elif isinstance(lit.value, Calculation):
            return validate_calculation(lit.value, context)

    def validate_column(col: Column, context: QueryContext) -> bool:
        valid = False
        if (
            col.name not in select
            and col.name not in environment.concepts
            and not col.calculation
        ):
            recommendations = None
            try:
                environment.concepts[col.name]
            except UndefinedConceptException as e:
                recommendations = e.suggestions
            if recommendations:
                errors.append(
                    f"{col.name} in {context} is not a valid field or previously defined by you; if this is a new metric you need, add a select to calculate it. Is one of these the correct spelling? {recommendations}?",
                )
            else:

                errors.append(
                    f"{col.name} in {context} is not a valid field or previously defined by you. If this is a new metric you need, add a select to calculate it. If you misspelled a field, potential matches are {difflib.get_close_matches(col.name, environment.concepts.keys(), 3, 0.4)}",
                )
        elif col.name in environment.concepts:
            valid = True
            if col.calculation:
                errors.append(
                    f"{col.name} in {context} is a predefined field and should not have a calculation; check that you are using the field as is, without any additional transformations.",
                )
        elif col.calculation:

            if not is_valid_function(col.calculation.operator):
                errors.append(
                    f"{col.name} Column definition in {context} does not use a valid function (is using {col.calculation.operator}); check that you are using ONLY a valid option from this list: {sorted([x for x in FunctionType.__members__.keys() if x not in COMPLICATED_FUNCTIONS]) }. If the column requires no transformation, drop the calculation field.",
                )
            else:
                valid = True
            if col.calculation.over:
                for x in col.calculation.over:
                    local = validate_column(x, context)
                    valid = valid and local

            for arg in col.calculation.arguments:
                if isinstance(arg, Column):
                    local = validate_column(arg, context)
                    valid = valid and local
        if valid:
            if context == QueryContext.SELECT:
                select.add(col.name)
            elif context == QueryContext.FILTER:
                filtered_on.add(col.name)
        return valid

    for x in parsed.columns:
        validate_column(x, QueryContext.SELECT)

    if parsed.order:
        for y in parsed.order:
            if y.column_name not in select:
                errors.append(
                    f"{y.column_name} being ordered by is not in select output; add if needed.",
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
            "errors": {f"Error {idx}: {error}" for idx, error in enumerate(errors)},
        }
    tips = [
        f'No validation errors - looking good! Just double check you have covered the original prompt, and submit it! (ONLY call revalidate again if you make a change after this. But if you do make a change to submission, you MUST recall validation) Prompt: "{prompt}"!'
    ]
    for x in select.union(filtered_on):
        if x in environment.concepts:
            concept = environment.concepts[x]
            if concept.metadata.description:
                tips.append(
                    f'For {x}, reminder that the field description is "{concept.metadata.description}". Double check any filtering on this field matches the described format! (this is a reminder, not an error)'
                )
    VALIDATION_CACHE[prompt] = parsed
    return {"status": VALID_STATUS, "tips": tips}, parsed


def validation_error_to_string(e: ValidationError):
    # inject in new context on failed answer
    raw_error = str(e)
    errors = e.errors()
    # TODO: better pydantic error parsing
    if "filtering.root." in raw_error:
        missing = []
        path_freq = defaultdict(lambda: 0)
        path_map = defaultdict(lambda: [])
        for e in errors:
            if e["type"] == "missing":
                path = "".join([str(x) for x in e["loc"][:-1]])
                path_freq[path] += 1
                message = f'Missing "{e["loc"][-1]}" in this JSON object you provided: {json.dumps(e["input"], indent=4)}. You may also be using the wrong object. Double check Literal and Column formats.'
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
        min_path = min(path_freq, key=path_freq.get)
        locations = path_map[min_path]
        raw_error = f"Syntax error in your filtering clause. Check that you only use valid keys for Literal, Comparison, and Column objects, and have all required keys. Look at these locations: {locations}"
    else:
        for e in errors:
            if e["type"] == "missing":
                raw_error = f'Missing {e["loc"][-1]} in {e["input"]}. You might have incorrect JSON formats or the key is actually missing. Double check Literal and Column formats.'
    return raw_error


def validate_response(
    environment: Environment,
    prompt: str,
    columns: List[Column],
    filtering: Optional[FilterResultV2] = None,
    order: Optional[list[OrderResultV2]] = None,
    limit: int = None,
) -> tuple[dict, InitialParseResponseV2 | None]:

    base = {"columns": columns}
    if not columns:
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
    )


class ValidateResponseInterface(BaseModel):
    # deliberately permissive interface
    # so that we can handle the error inside the tool
    columns: List[dict]
    filtering: Optional[dict] = (None,)
    order: Optional[list[dict]] = (None,)
    limit: int = None
