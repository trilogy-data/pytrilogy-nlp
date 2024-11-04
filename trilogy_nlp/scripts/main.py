from click import Path, argument, group, UNPROCESSED
from trilogy.dialect.enums import Dialects
from pathlib import Path as PathlibPath
import os
from sys import path as sys_path
from trilogy.parsing.render import Renderer
import datetime
import tomli_w
from trilogy.dialect.enums import Dialects  # noqa
from trilogy.executor import Executor
from trilogy_nlp.environment import helper
# handles development cases
nb_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys_path.insert(0, nb_path)

renderer = Renderer()


def generate_executor(dialect):
    pass


def run_query(text:str, dialect:Dialects, engine: Executor,llm):
    engine = generate_executor(dialect)
    env, processed_query = helper(text, llm)
    # fetch our results
    parse_start = datetime.now()
    engine.environment = env
    query = engine.generate_sql(processed_query)[-1]
    parse_time = datetime.now() - parse_start
    exec_start = datetime.now()
    results = engine.execute_raw_sql(query)
    if not results:
        print('Empty results')
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
    edialect = Dialects(dialect)
    preqlt: PathlibPath = PathlibPath(str(preql))


if __name__ == "__main__":
    main()
