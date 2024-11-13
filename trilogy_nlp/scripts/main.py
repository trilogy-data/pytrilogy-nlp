from click import Path, argument, group, UNPROCESSED
from trilogy.dialect.enums import Dialects
from pathlib import Path as PathlibPath
import os
from sys import path as sys_path
from trilogy.parsing.render import Renderer
from datetime import datetime

from trilogy.dialect.enums import Dialects  # noqa
from trilogy.executor import Executor
from trilogy_nlp.main import build_query
from trilogy_nlp.environment import build_env_and_imports

# handles development cases
nb_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys_path.insert(0, nb_path)

renderer = Renderer()


def generate_executor(dialect):
    pass


def print_tabulate(q, tabulate):
    result = q.fetchall()
    print(tabulate(result, headers=q.keys(), tablefmt="psql"))


def run_query(
    text: str, working_path: PathlibPath, dialect: Dialects, engine: Executor, llm
):
    engine = generate_executor(dialect)
    env = build_env_and_imports(text, working_path=working_path, llm=llm)
    processed_query = build_query(
        input_text=text,
        input_environment=env,
        debug=True,
        llm=llm,
    )
    # fetch our results
    parse_start = datetime.now()
    engine.environment = env
    query = engine.generate_sql(processed_query)[-1]
    parse_time = datetime.now() - parse_start
    print(f"Parse time: {parse_time}")
    exec_start = datetime.now()
    results = engine.execute_raw_sql(query)
    print(f"Exec time: {datetime.now() - exec_start}")
    if not results:
        print("Empty results")
    try:
        import tabulate

        print_tabulate(results, tabulate.tabulate)
    except ImportError:
        print("Install tabulate (pip install tabulate) for a prettier output")
        print(", ".join(results.keys()))
        for row in results:
            print(row)
        print("---")
    return query


@group()
def main():
    """Parent CLI"""
    pass


@main.command(
    context_settings=dict(
        ignore_unknown_options=True,
    ),
)
@argument("preql", type=Path())
@argument("output_path", type=Path(exists=False))
@argument("dialect", type=str)
@argument("conn_args", nargs=-1, type=UNPROCESSED)
def simple(preql: str | Path, output_path: Path, dialect: str, conn_args):
    # _= Dialects(dialect)
    _: PathlibPath = PathlibPath(str(preql))


if __name__ == "__main__":
    main()
