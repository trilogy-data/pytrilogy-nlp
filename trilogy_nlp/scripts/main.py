from click import Path, argument, option, group, pass_context, UNPROCESSED
from trilogy.dialect.enums import Dialects
from pathlib import Path as PathlibPath
import os
from sys import path as sys_path
from trilogy.parsing.render import Renderer
import datetime
# handles development cases
nb_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys_path.insert(0, nb_path)

renderer = Renderer()

def generate_executor(dialect):
    pass

def run_query(engine: Executor, idx: int, llm):

    with open(working_path / f"query{idx:02d}.prompt") as f:
        text = f.read()

    env, processed_query = helper(text, llm)
    # fetch our results
    parse_start = datetime.now()
    engine.environment = env
    query = engine.generate_sql(processed_query)[-1]
    parse_time = datetime.now() - parse_start
    exec_start = datetime.now()
    results = engine.execute_raw_sql(query)
    exec_time = datetime.now() - exec_start
    # assert results == ''
    comp_results = list(results.fetchall())
    assert len(comp_results) > 0, "No results returned"
    # run the built-in comp
    comp_start = datetime.now()
    base = engine.execute_raw_sql(f"PRAGMA tpcds({idx});")
    base_results = list(base.fetchall())
    comp_time = datetime.now() - comp_start

    # # check we got it
    if len(base_results) != len(comp_results):
        assert False, f"Row count mismatch: {len(base_results)} != {len(comp_results)}"
    for qidx, row in enumerate(base_results):
        for cell in row:
            assert (
                cell in comp_results[qidx]
            ), f"Could not find value {cell} in row {qidx} (expected row v test): {row} != {comp_results[qidx]}"

    with open(working_path / f"zquery{idx:02d}.log", "w") as f:
        f.write(
            tomli_w.dumps(
                {
                    "query_id": idx,
                    "parse_time": parse_time.total_seconds(),
                    "exec_time": exec_time.total_seconds(),
                    "comp_time": comp_time.total_seconds(),
                    "gen_length": len(query),
                    "generated_sql": query,
                },
                multiline_strings=True,
            )
        )
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
def dbt(preql: str | Path, output_path: Path, dialect: str, conn_args):
    edialect = Dialects(dialect)
    preqlt: PathlibPath = PathlibPath(str(preql))
    


if __name__ == "__main__":
    main()
