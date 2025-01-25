from typing import Any, List

from trilogy.authoring import (
    DEFAULT_NAMESPACE,
    AggregateWrapper,
    BooleanOperator,
    CaseElse,
    CaseWhen,
    Comparison,
    ComparisonOperator,
    Concept,
    ConceptRef,
    Conditional,
    DataType,
    Environment,
    Function,
    FunctionClass,
    FunctionType,
    HavingClause,
    InfiniteFunctionArgs,
    MagicConstants,
    Metadata,
    OrderBy,
    Ordering,
    OrderItem,
    Parenthetical,
    Purpose,
    SubselectComparison,
    WhereClause,
)
from trilogy.parsing.common import arbitrary_to_concept, arg_to_datatype

from trilogy_nlp.llm_interface.constants import (
    MAGIC_GENAI_DESCRIPTION,
)
from trilogy_nlp.llm_interface.models import (
    Calculation,
    Column,
    FilterResultV2,
    Literal,
    NLPComparisonGroup,
    NLPConditions,
    OrderResultV2,
)


def is_scalar_condition(
    element: Any,
    environment: Environment,
    materialized: set[str] | None = None,
) -> bool:
    assert environment
    if isinstance(element, Parenthetical):
        return is_scalar_condition(element.content, environment, materialized)
    elif isinstance(element, SubselectComparison):
        return True
    elif isinstance(element, Comparison):
        return is_scalar_condition(
            element.left, environment, materialized
        ) and is_scalar_condition(element.right, environment, materialized)
    elif isinstance(element, Function):
        if element.operator in FunctionClass.AGGREGATE_FUNCTIONS.value:
            return False
        return all(
            [
                is_scalar_condition(x, environment, materialized)
                for x in element.arguments
            ]
        )
    elif isinstance(element, ConceptRef):
        if materialized and element.address in materialized:
            return True
        ec: Concept = environment.concepts[element.address]
        if ec.lineage and isinstance(ec.lineage, AggregateWrapper):
            return is_scalar_condition(ec.lineage, environment, materialized)
        if ec.lineage and isinstance(ec.lineage, Function):
            return is_scalar_condition(ec.lineage, environment, materialized)
        return True
    elif isinstance(element, AggregateWrapper):
        return is_scalar_condition(element.function, environment, materialized)
    elif isinstance(element, Conditional):
        return is_scalar_condition(
            element.left, environment, materialized
        ) and is_scalar_condition(element.right, environment, materialized)
    elif isinstance(element, (CaseWhen,)):
        return is_scalar_condition(
            element.comparison, environment, materialized
        ) and is_scalar_condition(element.expr, environment, materialized)
    elif isinstance(element, (CaseElse,)):
        return is_scalar_condition(element.expr, environment, materialized)
    elif isinstance(element, MagicConstants):
        return True
    return True


CONDITION_TYPES = (
    SubselectComparison,
    Comparison,
    Conditional,
    Parenthetical,
)


def decompose_condition(
    conditional: Conditional | Comparison | Parenthetical,
) -> list[SubselectComparison | Comparison | Conditional | Parenthetical]:
    chunks: list[SubselectComparison | Comparison | Conditional | Parenthetical] = []
    if not isinstance(conditional, Conditional):
        return [conditional]
    if conditional.operator == BooleanOperator.AND:
        if not (
            isinstance(conditional.left, CONDITION_TYPES)
            and isinstance(
                conditional.right,
                CONDITION_TYPES,
            )
        ):
            chunks.append(conditional)
        else:
            for val in [conditional.left, conditional.right]:
                if isinstance(val, Conditional):
                    chunks.extend(decompose_condition(val))
                else:
                    chunks.append(val)
    else:
        chunks.append(conditional)
    return chunks


def parse_object(ob, environment: Environment):
    if isinstance(ob, Column):
        return create_column(ob, environment)
    elif isinstance(ob, Calculation):
        return create_anon_calculation(ob, environment)
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
) -> (
    str
    | float
    | int
    | bool
    | MagicConstants
    | Concept
    | list[str]
    | list[int]
    | list[float]
):
    # LLMs might get formats mixed up; if they gave us a column, hydrate it here.
    # and carry on
    if isinstance(literal.value, Calculation):
        return create_column(
            Column(
                name=get_next_inline_calc_name(environment), calculation=literal.value
            ),
            environment,
        )
    if isinstance(literal.value, str) and literal.value in environment.concepts:
        return create_column(Column(name=literal.value), environment)

    # otherwise, we really have a literal
    if literal.type == "null":
        return MagicConstants.NULL
    dtype = parse_datatype(literal.type)
    if isinstance(literal.value, list):
        return literal.value
    if dtype == DataType.STRING:
        return literal.value
    if dtype == DataType.INTEGER:
        return int(literal.value)
    if dtype == DataType.FLOAT:
        return float(literal.value)
    if dtype == DataType.BOOL:
        return bool(literal.value)
    if dtype == DataType.ARRAY:
        return literal.value
    return literal.value


def create_anon_calculation(
    c: Calculation, environment: Environment
) -> Function | AggregateWrapper:
    operator = FunctionType(c.operator.lower())
    if operator in FunctionClass.AGGREGATE_FUNCTIONS.value:
        purpose = Purpose.METRIC
    else:
        purpose = Purpose.PROPERTY
    if c.operator.lower() not in FunctionType.__members__:
        if c.operator == "RENAME":
            c.operator = "ALIAS"
    args = [parse_object(c, environment) for c in c.arguments]
    deriv_function = Function(
        operator=FunctionType(c.operator.lower()),
        output_datatype=arg_to_datatype(args[0]),
        output_purpose=purpose,
        arguments=args,
        arg_count=InfiniteFunctionArgs,
    )
    derivation: Function | AggregateWrapper
    if c.over:
        if purpose != Purpose.METRIC:
            raise ValueError("Can only use over with aggregate functions.")
        derivation = AggregateWrapper(
            function=deriv_function,
            by=[parse_object(c, environment) for c in c.over],
        )
    else:
        derivation = deriv_function
    return derivation


def create_column(c: Column, environment: Environment) -> Concept:
    if not c.calculation:
        return environment.concepts[c.name]
    base_name = c.name
    # TAG: resiliency
    # LLM may reference the same name for the output of a calculation
    # if that's so, force the outer concept a new name
    derivation = create_anon_calculation(c.calculation, environment)
    if any(
        isinstance(z, Concept) and z.name == base_name
        for z in derivation.concept_arguments
    ):
        base_name = f"{c.name}_deriv"
    new = arbitrary_to_concept(
        derivation,
        namespace=DEFAULT_NAMESPACE,
        name=f"{base_name}".lower(),
        metadata=Metadata(description=MAGIC_GENAI_DESCRIPTION),
        environment=environment,
    )
    environment.add_concept(new)

    return new


def parse_order(
    input_concepts: List[Concept],
    ordering: List[OrderResultV2] | None,
) -> OrderBy:
    normalized = [x if isinstance(x, Concept) else x.output for x in input_concepts]
    default_order = [
        OrderItem(expr=c.reference, order=Ordering.DESCENDING)
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
            final.append(
                OrderItem(expr=possible_matches[0].reference, order=order.order)
            )
    return OrderBy(items=final)


def parse_filter_obj(
    inp: NLPComparisonGroup | NLPConditions | Column | Literal | list[Column | Literal],
    environment: Environment,
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
        cast_eligible = operator not in [
            ComparisonOperator.IS_NOT,
            ComparisonOperator.IS,
            ComparisonOperator.IN,
            ComparisonOperator.NOT_IN,
        ]
        if cast_eligible and arg_to_datatype(left) != arg_to_datatype(right):
            right = Function(
                operator=FunctionType.CAST,
                output_datatype=arg_to_datatype(left),
                output_purpose=Purpose.PROPERTY,
                arguments=[right, arg_to_datatype(left)],  # type: ignore
                arg_count=2,
            )
        return Comparison(
            left=left,
            right=right,
            operator=operator,
        )
    elif isinstance(inp, (Column, Literal)):
        return parse_object(inp, environment)
    elif isinstance(inp, list):
        return [parse_filter_obj(x, environment) for x in inp]
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
    if not input.root.values:
        return None
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
    environment: Environment,
    filtering: WhereClause | None = None,
) -> tuple[WhereClause | None, HavingClause | None]:
    """If a concept is output by the select, and it is an aggregate;
    move that condition to the having clause. If it's an aggregate that is not output,
    we can keep it in where, as well as any scalars."""
    assert environment
    if not filtering:
        return None, None
    where: Conditional | Comparison | Parenthetical | None = None
    having: Conditional | Comparison | Parenthetical | None = None

    if is_scalar_condition(filtering.conditional, environment=environment):
        where = filtering.conditional
    else:
        components = decompose_condition(filtering.conditional)
        for x in components:
            if is_scalar_condition(x, environment=environment):
                where = where + x if where else x
            else:
                if any([c.address in selected for c in x.concept_arguments]):

                    if any(
                        not is_scalar_condition(x, environment=environment)
                        for x in x.concept_arguments
                    ):
                        having = having + x if having else x
                        continue
                # otherwise we end up here
                where = where + x if where else x
    return WhereClause(conditional=where) if where else None, (
        HavingClause(conditional=having) if having else None
    )
