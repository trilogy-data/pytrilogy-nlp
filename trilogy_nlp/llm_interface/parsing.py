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
)
from typing import List
from trilogy.core.enums import Ordering, Purpose, BooleanOperator
from trilogy.parsing.common import arbitrary_to_concept
from trilogy.core.models import DataType

from trilogy_nlp.llm_interface.models import (
    Column,
    NLPConditions,
    NLPComparisonGroup,
    FilterResultV2,
    OrderResultV2,
    Literal,
)
from trilogy_nlp.llm_interface.constants import (
    MAGIC_GENAI_DESCRIPTION,
)
from trilogy.core.processing.utility import (
    is_scalar_condition,
    decompose_condition,
)

# from trilogy.core.constants import
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


def create_literal(l: Literal, environment: Environment) -> str | float | int | bool:
    # LLMs might get formats mixed up; if they gave us a column, hydrate it here.
    # and carry on
    if l.value in environment.concepts:
        return create_column(Column(name=l.value), environment)

    dtype = parse_datatype(l.type)

    if dtype == DataType.STRING:
        return l.value
    if dtype == DataType.INTEGER:
        return int(l.value)
    if dtype == DataType.FLOAT:
        return float(l.value)
    if dtype == DataType.BOOL:
        return bool(l.value)
    return l.value


def create_column(c: Column, environment: Environment) -> Concept | ConceptTransform:
    if not c.calculation:
        return environment.concepts[c.name]
    if c.calculation.operator.lower() not in FunctionType.__members__:
        if c.calculation.operator == 'RENAME':
            c.calculation.operator = 'ALIAS'
            
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
    derivation = Function(
        operator=FunctionType(c.calculation.operator.lower()),
        output_datatype=arg_to_datatype(args[0]),
        output_purpose=purpose,
        arguments=args,
        arg_count=InfiniteFunctionArgs,
    )
    if c.calculation.over:
        if purpose != Purpose.METRIC:
            raise ValueError("Can only use over with aggregate functions.")
        derivation = AggregateWrapper(
            function=derivation,
            by=[parse_object(c, environment) for c in c.calculation.over],
        )

    new = arbitrary_to_concept(
        derivation,
        namespace="local",
        name=f"{base_name}".lower(),
        metadata=Metadata(description=MAGIC_GENAI_DESCRIPTION),
    )
    environment.add_concept(new)

    return new


def parse_order(
    input_concepts: List[Concept], ordering: List[OrderResultV2] | None
) -> OrderBy:
    default_order = [
        OrderItem(expr=c, order=Ordering.DESCENDING)
        for c in input_concepts
        if c.purpose == Purpose.METRIC
    ]
    if not ordering:
        return OrderBy(items=default_order)
    final = []
    for order in ordering:
        possible_matches = [
            x
            for x in input_concepts
            if x.address == order.column_name or x.name == order.column_name
        ]
        if not possible_matches:
            has_lineage = [x for x in input_concepts if x.lineage]
            possible_matches = [
                x
                for x in has_lineage
                if any(
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
            if not list:
                return True
            left = list.pop(0)
            right = generate_conditional(list, operator)
            return Conditional(left=left, right=right, operator=operator)

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


def parse_filter_flat(
    input: FilterResultV2, environment: Environment
) -> list[Conditional]:
    return parse_filter_obj_flat(input.root, environment, flat=True)


def parse_filtering(
    filtering: FilterResultV2, environment: Environment
) -> WhereClause:
    base = []
    parsed = parse_filter(filtering, environment=environment)
    # flat = parse_filter_flat(filtering, environment=environment)
    return WhereClause(conditional=parsed)
    if filtering.root and not parsed:
        raise SyntaxError
    if parsed:
        print(parsed)
        base.append(parsed)
    if not base:
        return None
    print("filtering debug")
    print(base)
    if len(base) == 1:
        return WhereClause(conditional=base[0])
    left: Conditional | Comparison = base.pop()
    while base:
        right = base.pop()
        new = Conditional(left=left, right=right, operator=BooleanOperator.AND)
        left = new
    return WhereClause(conditional=left)


def generate_having_and_where(
    filtering: WhereClause | None = None,
) -> tuple[WhereClause | None, HavingClause | None]:
    if not filtering:
        return None, None
    where: Conditional | Comparison | None = None
    having: Conditional | Comparison | None = None
    if is_scalar_condition(filtering.conditional):
        where = filtering.conditional
    else:
        components = decompose_condition(filtering.conditional)
        for x in components:
            if is_scalar_condition(x):
                where = where + x if where else x
            else:
                having = having + x if having else x
    return WhereClause(conditional=where) if where else None, HavingClause(
        conditional=having
    ) if having else None