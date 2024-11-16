from trilogy.core.models import (
    Concept,
    Environment,
    Comparison,
    Conditional,
    OrderBy,
    OrderItem,
    WhereClause,
    AggregateWrapper,
    Function,
    Metadata,
    HavingClause,
    ConceptTransform,
    MagicConstants,
    Parenthetical,
)
from typing import List
from trilogy.core.enums import Purpose, BooleanOperator, ComparisonOperator, Ordering
from trilogy.parsing.common import arbitrary_to_concept
from trilogy.core.models import DataType

from trilogy_nlp.llm_interface.models import (
    Column,
    NLPConditions,
    NLPComparisonGroup,
    FilterResultV2,
    OrderResultV2,
    Calculation,
    Literal,
)
from trilogy_nlp.llm_interface.constants import (
    MAGIC_GENAI_DESCRIPTION,
)
from trilogy.core.processing.utility import (
    is_scalar_condition,
    decompose_condition,
)

from trilogy.core.enums import (
    FunctionType,
    FunctionClass,
    InfiniteFunctionArgs,
)
from trilogy.parsing.common import arg_to_datatype


def parse_object(ob, environment: Environment):
    if isinstance(ob, Column):
        return create_column(ob, environment)
    return create_literal(ob, environment)


def parse_datatype(dtype: str):
    mapping = {item.value.lower(): item for item in DataType}
    mapping["integer"] = DataType.INTEGER
    if dtype.lower() in mapping:
        return mapping[dtype]
    return DataType.STRING


def get_next_inline_calc_name(environment: Environment) -> str:
    return f"inline_calc_{len(environment.concepts)+1}"


def create_literal(
    literal: Literal, environment: Environment
) -> str | float | int | bool | MagicConstants | Concept | ConceptTransform:
    # LLMs might get formats mixed up; if they gave us a column, hydrate it here.
    # and carry on
    if isinstance(literal.value, Calculation):
        return create_column(
            Column(
                name=get_next_inline_calc_name(environment), calculation=literal.value
            ),
            environment,
        )
    if literal.value in environment.concepts:
        return create_column(Column(name=literal.value), environment)

    # otherwise, we really have a literal
    if literal.type == "null":
        return MagicConstants.NULL
    dtype = parse_datatype(literal.type)

    if dtype == DataType.STRING:
        return literal.value
    if dtype == DataType.INTEGER:
        return int(literal.value)
    if dtype == DataType.FLOAT:
        return float(literal.value)
    if dtype == DataType.BOOL:
        return bool(literal.value)
    return literal.value


def create_column(c: Column, environment: Environment) -> Concept | ConceptTransform:
    if not c.calculation:
        return environment.concepts[c.name]
    if c.calculation.operator.lower() not in FunctionType.__members__:
        if c.calculation.operator == "RENAME":
            c.calculation.operator = "ALIAS"

    operator = FunctionType(c.calculation.operator.lower())
    if operator in FunctionClass.AGGREGATE_FUNCTIONS.value:
        purpose = Purpose.METRIC
    else:
        purpose = Purpose.PROPERTY

    args = [parse_object(c, environment) for c in c.calculation.arguments]
    base_name = c.name
    # TAG: resiliency
    # LLM may reference the same name for the output of a calculation
    # if that's so, force the outer concept a new name
    if any(isinstance(z, Concept) and z.name == base_name for z in args):
        base_name = f"{c.name}_deriv"
    # TODO: use better helpers here
    # this duplicates a bit of pytrilogy logic
    deriv_function = Function(
        operator=FunctionType(c.calculation.operator.lower()),
        output_datatype=arg_to_datatype(args[0]),
        output_purpose=purpose,
        arguments=args,
        arg_count=InfiniteFunctionArgs,
    )
    derivation: Function | AggregateWrapper
    if c.calculation.over:
        if purpose != Purpose.METRIC:
            raise ValueError("Can only use over with aggregate functions.")
        derivation = AggregateWrapper(
            function=deriv_function,
            by=[parse_object(c, environment) for c in c.calculation.over],
        )
    else:
        derivation = deriv_function

    new = arbitrary_to_concept(
        derivation,
        namespace="local",
        name=f"{base_name}".lower(),
        metadata=Metadata(description=MAGIC_GENAI_DESCRIPTION),
    )
    environment.add_concept(new)

    return new


def parse_order(
    input_concepts: List[Concept | ConceptTransform],
    ordering: List[OrderResultV2] | None,
) -> OrderBy:
    normalized = [x if isinstance(x, Concept) else x.output for x in input_concepts]
    default_order = [
        OrderItem(expr=c, order=Ordering.DESCENDING)
        for c in normalized
        if c.purpose == Purpose.METRIC
    ]
    if not ordering:
        return OrderBy(items=default_order)
    final = []
    for order in ordering:
        possible_matches = [
            x
            for x in normalized
            if x.address == order.column_name or x.name == order.column_name
        ]
        if not possible_matches:

            possible_matches = [
                x
                for x in normalized
                if x.lineage
                and any(
                    [
                        y.address == order.column_name
                        for y in x.lineage.concept_arguments
                    ]
                )
            ]
        if possible_matches:
            final.append(OrderItem(expr=possible_matches[0], order=order.order))
    return OrderBy(items=final)


def parse_filter_obj(
    inp: NLPComparisonGroup | NLPConditions | Column | Literal, environment: Environment
):
    if isinstance(inp, NLPComparisonGroup):
        children = [parse_filter_obj(x, environment) for x in inp.values]

        def generate_conditional(list: list, operator: BooleanOperator):
            left = list.pop(0)
            if not list:
                return left
            right = generate_conditional(list, operator)
            return Conditional(left=left, right=right, operator=operator)

        base = generate_conditional(children, operator=inp.boolean)
        if inp.boolean == BooleanOperator.OR:
            return Parenthetical(content=base)
        return base
    elif isinstance(inp, NLPConditions):
        left = parse_filter_obj(inp.left, environment)
        right = parse_filter_obj(inp.right, environment)
        operator = inp.operator
        if right == MagicConstants.NULL and operator == ComparisonOperator.NE:
            operator = ComparisonOperator.IS_NOT
        return Comparison(
            left=left,
            right=right,
            operator=operator,
        )
    elif isinstance(inp, (Column, Literal)):
        return parse_object(inp, environment)
    else:
        raise SyntaxError(inp)


def parse_filter_obj_flat(
    inp: NLPComparisonGroup | NLPConditions | Column | Literal, environment: Environment
):
    if isinstance(inp, NLPComparisonGroup):
        children = [parse_filter_obj(x, environment) for x in inp.values]

        def generate_conditional(list: list, operator: BooleanOperator):
            if not list:
                return True
            left = list.pop(0)
            right = generate_conditional(list, operator)
            flat = [
                left,
            ] + right
            if operator == BooleanOperator.AND:
                return flat
            else:
                return [flat]

        return generate_conditional(children, operator=inp.boolean)
    elif isinstance(inp, NLPConditions):
        return Comparison(
            left=parse_filter_obj(inp.left, environment),
            right=parse_filter_obj(inp.right, environment),
            operator=inp.operator,
        )
    elif isinstance(inp, (Column, Literal)):
        return parse_object(inp, environment)
    else:
        raise SyntaxError(inp)


def parse_filter(
    input: FilterResultV2, environment: Environment
) -> Comparison | Conditional | None:
    return parse_filter_obj(input.root, environment)


def parse_filtering(
    filtering: FilterResultV2, environment: Environment
) -> WhereClause | None:
    parsed = parse_filter(filtering, environment=environment)
    if parsed:
        return WhereClause(conditional=parsed)
    return None


def generate_having_and_where(
    selected: list[str],
    filtering: WhereClause | None = None,
) -> tuple[WhereClause | None, HavingClause | None]:
    """If a concept is output by the select, and it is an aggregate;
    move that condition to the having clause. If it's an aggregate that is not output,
    we can keep it in where, as well as any scalars."""
    if not filtering:
        return None, None
    where: Conditional | Comparison | Parenthetical | None = None
    having: Conditional | Comparison | Parenthetical | None = None

    if is_scalar_condition(filtering.conditional):
        where = filtering.conditional
    else:
        components = decompose_condition(filtering.conditional)
        for x in components:
            if is_scalar_condition(x):
                where = where + x if where else x
            else:
                if any([c.address in selected for c in x.concept_arguments]):
                    if any(
                        (
                            (
                                isinstance(x.lineage, AggregateWrapper)
                                # and not x.lineage.by
                            )
                            or (
                                isinstance(x.lineage, Function)
                                and x.lineage.operator
                                in FunctionClass.AGGREGATE_FUNCTIONS.value
                            )
                        )
                        for x in x.concept_arguments
                    ):
                        having = having + x if having else x
                        continue
                # otherwise we end up here
                where = where + x if where else x
    return WhereClause(conditional=where) if where else None, (
        HavingClause(conditional=having) if having else None
    )
