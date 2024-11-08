from typing import Optional
from trilogy.core.models import (
    Environment,
)
from typing import List
from pydantic import BaseModel, ValidationError
from trilogy.core.enums import ComparisonOperator
from langchain_core.tools import ToolException
from trilogy.core.exceptions import UndefinedConceptException

from trilogy_nlp.llm_interface.models import (
    InitialParseResponseV2,
    Column,
    NLPConditions,
    FilterResultV2,
    OrderResultV2,
    Literal,
)
from trilogy_nlp.llm_interface.examples import FILTERING_EXAMPLE
from trilogy_nlp.llm_interface.constants import COMPLICATED_FUNCTIONS

# from trilogy.core.constants import
from trilogy.core.enums import (
    FunctionType,
)
from enum import Enum
from trilogy_nlp.llm_interface.examples import COLUMN_DESCRIPTION


def is_valid_function(name: str):
    return name.lower() in [item.value for item in FunctionType]


def invalid_operator_message(operator: str) -> str | None:
    try:
        operator = ComparisonOperator(operator)
    except Exception as e:
        return str(e)
    return None


class QueryContext(Enum):
    SELECT = "SELECT"
    FILTER = "FILTER"
    ORDER = "ORDER"


def validate_query(query: dict, environment: Environment, prompt: str):
    try:
        parsed = InitialParseResponseV2.model_validate(query)
    except ValidationError as e:
        return {"status": "invalid", "error": validation_error_to_string(e)}
    errors = []
    # assume to start that all our select calculations are valid
    select = {col.name for col in parsed.columns if col.calculation}
    filtered_on = set()

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
                    f"{col.name} in {context} is not a valid field or previously defined; check that you are using the full exact column name (including any prefixes). Did you mean one of {recommendations}?",
                )
            else:
                errors.append(
                    f"{col.name} in {context} is not a valid field or previously defined; check that you are using the full exact column name (including any prefixes). If you want to apply a function, use a calculation - do not include it in the field name.",
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
                    valid = valid and validate_column(x, context)

            for arg in col.calculation.arguments:
                if isinstance(arg, Column):
                    valid = valid and validate_column(arg, context)
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
                    f"{y.column_name} in order not in select; check that you are using only values in the top level of the select.",
                )
    if parsed.filtering:
        root = parsed.filtering.root
        for val in root.values:
            if isinstance(val, Column):
                validate_column(val, QueryContext.FILTER)
            elif isinstance(val, NLPConditions):
                print(val)
                for subval in [val.left, val.right]:
                    if isinstance(subval, Column):
                        validate_column(subval, QueryContext.FILTER)
                if isinstance(val.left, Column) and isinstance(val.right, Column):
                    if val.left.name == val.right.name:
                        errors.append(
                            f"Comparison {val} has the same column name on both sides; this is a meaningless comparison. Check that you are comparing two different fields.",
                        )

    if errors:
        return {"status": "invalid", "error": errors}
    tips = [
        f'No validation errors - looking good! Just double check you have all the filters from the original prompt, validate any changes, and send it off! Prompt: "{prompt}"!'
    ]
    for x in select.union(filtered_on):
        if x in environment.concepts:
            concept = environment.concepts[x]
            if concept.metadata.description:
                tips.append(
                    f'For {x}, reminder that the field description is "{concept.metadata.description}". Make sure to double check any filtering on this field matches the described format!'
                )

    return {"status": "valid", "tips": tips}

from collections import defaultdict

def validation_error_to_string(e: ValidationError):
    # Here, `validation_error.errors()` will have the full info
    for x in e.errors():
        print(x)
    # inject in new context on failed answer
    raw_error = str(e)
    errors = e.errors()
    # TODO: better pydantic error parsing
    if "filtering.root." in raw_error:
        missing = []
        path_freq = defaultdict(lambda : 0)
        path_map = defaultdict(lambda : [])
        for e in errors:
            if e['type'] == 'missing':
                path = ''.join([str(x) for x in e['loc'][:-1]])
                path_freq[path] += 1
                message = f'Missing {e["loc"][-1]} in {e["input"]}'
                path_map[path].append(message)
                missing.append(message)
            else:
                missing_path = ".".join([str(v) for v in e["loc"] if str(v) not in ['NLPComparisonGroup', 'NLPConditions', 'Literal', 'Column']])
                missing.append(missing_path)
        min_path = min(path_freq, key=path_freq.get)
        locations = path_map[min_path]
        raw_error = f"Syntax error in your filtering clause. Comparisons need a left, right, operator, etc, and Columns and Literal formats are very specific. Hint on where to look: {locations}"
    return raw_error


def validate_response(
    environment: Environment,
    prompt: str,
    columns: List[Column],
    filtering: Optional[FilterResultV2] = None,
    order: Optional[list[OrderResultV2]] = None,
    limit: int = None,
):

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
