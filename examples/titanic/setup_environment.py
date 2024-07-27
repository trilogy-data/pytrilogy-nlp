import pandas as pd
from trilogy import Executor, Dialects
from trilogy.core.models import Environment
from sqlalchemy import create_engine
from trilogy.core.models import Datasource, Concept, ColumnAssignment, Grain, Function, DataType
from trilogy.core.enums import Purpose, FunctionType
from os.path import dirname
from pathlib import PurePath
from typing import Optional

from trilogy_nlp.main_v2 import build_query as build_query_v2
from logging import StreamHandler, DEBUG

from trilogy_nlp.constants import logger
from trilogy_nlp.main import build_query
from trilogy.core.functions import function_args_to_output_purpose, arg_to_datatype


def create_function_derived_concept(
    name: str,
    namespace: str,
    operator: FunctionType,
    arguments: list[Concept],
    output_type: Optional[DataType] = None,
    output_purpose: Optional[Purpose] = None,
) -> Concept:
    purpose = (
        function_args_to_output_purpose(arguments)
        if output_purpose is None
        else output_purpose
    )
    output_type = arg_to_datatype(arguments[0]) if output_type is None else output_type
    return Concept(
        name=name,
        namespace=namespace,
        datatype=output_type,
        purpose=purpose,
        lineage=Function(
            operator=operator,
            arguments=arguments,
            output_datatype=output_type,
            output_purpose=purpose,
            arg_count=len(arguments),
        ),
    )


def setup_engine() -> Executor:
    engine = create_engine(r"duckdb:///:memory:", future=True)
    csv = PurePath(dirname(__file__)) / "train.csv"
    df = pd.read_csv(csv)
    _ = df
    output = Executor(engine=engine, dialect=Dialects.DUCK_DB)

    output.execute_raw_sql("CREATE TABLE raw_titanic AS SELECT * FROM df")
    return output


def setup_titanic(env: Environment):
    namespace = "passenger"
    id = Concept(
        name="id", namespace=namespace, datatype=DataType.INTEGER, purpose=Purpose.KEY
    )
    age = Concept(
        name="age",
        namespace=namespace,
        datatype=DataType.INTEGER,
        purpose=Purpose.PROPERTY,
        keys=[id],
    )

    name = Concept(
        name="name",
        namespace=namespace,
        datatype=DataType.STRING,
        purpose=Purpose.PROPERTY,
        keys=[id],
    )

    pclass = Concept(
        name="class",
        namespace=namespace,
        purpose=Purpose.PROPERTY,
        datatype=DataType.INTEGER,
        keys=[id],
    )
    survived = Concept(
        name="survived",
        namespace=namespace,
        purpose=Purpose.PROPERTY,
        datatype=DataType.INTEGER,
        keys=[id],
    )

    survived_count = create_function_derived_concept(
        "survived_count",
        namespace,
        FunctionType.SUM,
        [survived],
        output_purpose=Purpose.METRIC,
    )

    fare = Concept(
        name="fare",
        namespace=namespace,
        purpose=Purpose.PROPERTY,
        datatype=DataType.FLOAT,
        keys=[id],
    )
    embarked = Concept(
        name="embarked",
        namespace=namespace,
        purpose=Purpose.PROPERTY,
        datatype=DataType.INTEGER,
        keys=[id],
    )
    cabin = Concept(
        name="cabin",
        namespace=namespace,
        purpose=Purpose.PROPERTY,
        datatype=DataType.STRING,
        keys=[id],
    )
    ticket = Concept(
        name="ticket",
        namespace=namespace,
        purpose=Purpose.PROPERTY,
        datatype=DataType.STRING,
        keys=[id],
    )

    last_name = Concept(
        name="last_name",
        namespace=namespace,
        purpose=Purpose.PROPERTY,
        datatype=DataType.STRING,
        keys=[id],
        lineage=Function(
            operator=FunctionType.INDEX_ACCESS,
            arguments=[
                Function(
                    operator=FunctionType.SPLIT,
                    arguments=[name, ","],
                    output_datatype=DataType.ARRAY,
                    output_purpose=Purpose.PROPERTY,
                    arg_count=2,
                ),
                1,
            ],
            output_datatype=DataType.STRING,
            output_purpose=Purpose.PROPERTY,
            arg_count=2,
        ),
    )
    all_concepts = [
        id,
        age,
        survived,
        survived_count,
        name,
        pclass,
        fare,
        cabin,
        embarked,
        ticket,
        last_name,
    ]
    for x in all_concepts:
        env.add_concept(x)

    env.add_datasource(
        Datasource(
            identifier="raw_data",
            address="raw_titanic",
            columns=[
                ColumnAssignment(alias="passengerid", concept=id),
                ColumnAssignment(alias="age", concept=age),
                ColumnAssignment(alias="survived", concept=survived),
                ColumnAssignment(alias="pclass", concept=pclass),
                ColumnAssignment(alias="name", concept=name),
                ColumnAssignment(alias="fare", concept=fare),
                ColumnAssignment(alias="cabin", concept=cabin),
                ColumnAssignment(alias="embarked", concept=embarked),
                ColumnAssignment(alias="ticket", concept=ticket),
            ],
            grain=Grain(components=[id]),
        ),
    )
    return env


def create_passenger_dimension(exec: Executor, name: str):
    exec.execute_raw_sql(f"CREATE SEQUENCE seq_{name} START 1;")
    exec.execute_raw_sql(
        f"""create table dim_{name} as 
                         SELECT passengerid id, name, age,
                          SPLIT(name, ',')[1] last_name
                            FROM raw_data

"""
    )


def create_arbitrary_dimension(exec: Executor, key: str, name: str):
    exec.execute_raw_sql(
        f"""create table dim_{name} as 
                         with tmp as 
                         (select {key}
                         from raw_data group by 1
                         )
                         SELECT  row_number() over() as id,
                         {key} as {name}
                          FROM tmp
"""
    )


def create_fact(
    exec: Executor,
    dims: Optional[list[str]] = None,
    include: Optional[list[str]] = None,
):
    exec.execute_raw_sql(
        """create table fact_titanic as 
                         SELECT 
                         row_number() OVER () as fact_key,
                         passengerid,
                         survived,
                         fare,
                         embarked,
                         b.id class_id,
                         cabin  
                         FROM raw_data a 
                         LEFT OUTER JOIN dim_class b on a.pclass=b.class
                         """
    )


def setup_normalized_engine() -> Executor:
    engine = create_engine(r"duckdb:///:memory:", future=True)
    csv = PurePath(dirname(__file__)) / "train.csv"
    df = pd.read_csv(csv)
    _ = df
    output = Executor(engine=engine, dialect=Dialects.DUCK_DB)

    output.execute_raw_sql("CREATE TABLE raw_data AS SELECT * FROM df")
    create_passenger_dimension(output, "passenger")
    create_arbitrary_dimension(output, "pclass", "class")
    create_fact(output, ["passenger"])
    return output


def setup_titanic_distributed(env: Environment):
    namespace = "passenger"
    id = Concept(
        name="id", namespace=namespace, datatype=DataType.INTEGER, purpose=Purpose.KEY
    )
    age = Concept(
        name="age",
        namespace=namespace,
        datatype=DataType.INTEGER,
        purpose=Purpose.PROPERTY,
        keys=[id],
    )

    name = Concept(
        name="name",
        namespace=namespace,
        datatype=DataType.STRING,
        purpose=Purpose.PROPERTY,
        keys=[id],
    )
    class_id = Concept(
        name="_class_id",
        namespace=namespace,
        purpose=Purpose.KEY,
        datatype=DataType.INTEGER,
        # keys=[id],
    )
    pclass = Concept(
        name="class",
        namespace=namespace,
        purpose=Purpose.PROPERTY,
        datatype=DataType.INTEGER,
        keys=[class_id],
    )
    survived = Concept(
        name="survived",
        namespace=namespace,
        purpose=Purpose.PROPERTY,
        datatype=DataType.BOOL,
        keys=[id],
    )
    survived = Concept(
        name="survived",
        namespace=namespace,
        purpose=Purpose.PROPERTY,
        datatype=DataType.BOOL,
        keys=[id],
    )

    survived_count = create_function_derived_concept(
        "survived_count", namespace, FunctionType.SUM, [survived]
    )

    fare = Concept(
        name="fare",
        namespace=namespace,
        purpose=Purpose.PROPERTY,
        datatype=DataType.FLOAT,
        keys=[id],
    )
    embarked = Concept(
        name="embarked",
        namespace=namespace,
        purpose=Purpose.PROPERTY,
        datatype=DataType.INTEGER,
        keys=[id],
    )
    cabin = Concept(
        name="cabin",
        namespace=namespace,
        purpose=Purpose.PROPERTY,
        datatype=DataType.STRING,
        keys=[id],
    )
    last_name = Concept(
        name="last_name",
        namespace=namespace,
        purpose=Purpose.PROPERTY,
        datatype=DataType.STRING,
        keys=[id],
        lineage=Function(
            operator=FunctionType.INDEX_ACCESS,
            arguments=[
                Function(
                    operator=FunctionType.SPLIT,
                    arguments=[name, ","],
                    output_datatype=DataType.ARRAY,
                    output_purpose=Purpose.PROPERTY,
                    arg_count=2,
                ),
                1,
            ],
            output_datatype=DataType.STRING,
            output_purpose=Purpose.PROPERTY,
            arg_count=2,
        ),
    )
    for x in [
        id,
        age,
        survived,
        survived_count,
        name,
        pclass,
        fare,
        cabin,
        embarked,
        last_name,
    ]:
        env.add_concept(x)

    env.add_datasource(
        Datasource(
            identifier="dim_passenger",
            address="dim_passenger",
            columns=[
                ColumnAssignment(alias="id", concept=id),
                ColumnAssignment(alias="age", concept=age),
                ColumnAssignment(alias="name", concept=name),
                ColumnAssignment(alias="last_name", concept=last_name),
                # ColumnAssignment(alias="pclass", concept=pclass),
                # ColumnAssignment(alias="name", concept=name),
                # ColumnAssignment(alias="fare", concept=fare),
                # ColumnAssignment(alias="cabin", concept=cabin),
                # ColumnAssignment(alias="embarked", concept=embarked),
            ],
            grain=Grain(components=[id]),
        ),
    )

    env.add_datasource(
        Datasource(
            identifier="fact_titanic",
            address="fact_titanic",
            columns=[
                ColumnAssignment(alias="passengerid", concept=id),
                ColumnAssignment(alias="survived", concept=survived),
                ColumnAssignment(alias="class_id", concept=class_id),
                ColumnAssignment(alias="fare", concept=fare),
                ColumnAssignment(alias="cabin", concept=cabin),
                ColumnAssignment(alias="embarked", concept=embarked),
            ],
            grain=Grain(components=[id]),
        ),
    )

    env.add_datasource(
        Datasource(
            identifier="dim_class",
            address="dim_class",
            columns=[
                ColumnAssignment(alias="id", concept=class_id),
                ColumnAssignment(alias="class", concept=pclass),
                # ColumnAssignment(alias="fare", concept=fare),
                # ColumnAssignment(alias="cabin", concept=cabin),
                # ColumnAssignment(alias="embarked", concept=embarked),
            ],
            grain=Grain(components=[class_id]),
        ),
    )

    return env


# if main gating for python
if __name__ == "__main__":
    executor = setup_engine()

    env = Environment()
    model = setup_titanic(env)
    for c in env.concepts:
        print(c)

    executor.environment = env
    environment = env
    logger.setLevel(DEBUG)
    logger.addHandler(StreamHandler())

    question = ("HOw many passengers survived in each cabin?",)
    processed_query = build_query(
        question,
        environment,
        debug=True,
    )

    processed_query_v2 = build_query_v2(
        question,
        environment,
        debug=True,
    )

    for q in [processed_query, processed_query_v2]:
        print("output_columns")
        for key in q.output_columns:
            print(key)

        print(executor.generator.compile_statement(q))
        results = executor.execute_query(q)
        for row in results:
            print(row)
