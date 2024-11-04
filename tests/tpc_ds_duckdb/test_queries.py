from pathlib import Path
import pytest
from datetime import datetime
import tomli_w
from trilogy import Executor
from trilogy_nlp.main_v2 import build_query as build_query_v2
from trilogy_nlp.environment import build_env_and_imports
from trilogy_nlp.constants import logger

working_path = Path(__file__).parent


def helper(text: str, llm):
    environment = build_env_and_imports(text, working_path=working_path, llm=llm)
    processed_query = build_query_v2(
        input_text=text,
        input_environment=environment,
        debug=True,
        llm=llm,
    )
    return environment, processed_query


def run_query(engine: Executor, idx: int, llm):

    with open(working_path / f"query{idx:02d}.prompt") as f:
        text = f.read()

    env, processed_query = helper(text, llm)
    # fetch our results
    parse_start = datetime.now()
    engine.environment = env
    query = engine.generate_sql(processed_query)[-1]
    logger.info(query)
    print(query)
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


def test_one(engine, llm):
    query = run_query(engine, 1, llm)
    assert len(query) < 9000, query


@pytest.mark.skip(reason="Is duckdb correct??")
def test_two(engine):
    run_query(engine, 2)


def test_three(engine, llm):
    run_query(engine, 3, llm)


@pytest.mark.skip(reason="Is duckdb correct??")
def test_four(engine):
    run_query(engine, 4)


@pytest.mark.skip(reason="Is duckdb correct??")
def test_five(engine):
    run_query(engine, 5)


@pytest.mark.cli
def test_six(engine, llm):
    query = run_query(engine, 6, llm)
    assert len(query) < 7100, query


@pytest.mark.skip(reason="No prompt yet")
def test_seven(engine, llm):
    run_query(engine, 7, llm)


@pytest.mark.skip(reason="No prompt yet")
def test_eight(engine, llm):
    run_query(engine, 8, llm)


@pytest.mark.skip(reason="No prompt yet")
def test_ten(engine, llm):
    query = run_query(engine, 10, llm)
    assert len(query) < 7000, query


@pytest.mark.skip(reason="No prompt yet")
def test_twelve(engine, llm):
    run_query(engine, 12, llm)


@pytest.mark.skip(reason="No prompt yet")
def test_fifteen(engine, llm):
    run_query(engine, 15, llm)


@pytest.mark.skip(reason="No prompt yet")
def test_sixteen(engine, llm):
    query = run_query(engine, 16, llm)
    # size gating
    assert len(query) < 16000, query


@pytest.mark.skip(reason="No prompt yet")
def test_twenty(engine, llm):
    _ = run_query(engine, 20, llm)
    # size gating
    # assert len(query) < 6000, query


@pytest.mark.skip(reason="No prompt yet")
def test_twenty_one(engine, llm):
    _ = run_query(engine, 21, llm)
    # size gating
    # assert len(query) < 6000, query


@pytest.mark.skip(reason="No prompt yet")
def test_twenty_four(engine, llm):
    _ = run_query(engine, 24, llm)
    # size gating
    # assert len(query) < 6000, query


@pytest.mark.skip(reason="No prompt yet")
def test_twenty_five(engine, llm):
    query = run_query(engine, 25, llm)
    # size gating
    assert len(query) < 10000, query


@pytest.mark.skip(reason="No prompt yet")
def test_twenty_six(engine, llm):
    _ = run_query(engine, 26, llm)
    # size gating
    # assert len(query) < 6000, query


def run_adhoc(number: int):
    from trilogy import Environment, Dialects
    from trilogy.hooks.query_debugger import DebuggingHook
    from logging import INFO

    env = Environment(working_path=Path(__file__).parent)
    engine: Executor = Dialects.DUCK_DB.default_executor(
        environment=env, hooks=[DebuggingHook(INFO)]
    )
    engine.execute_raw_sql(
        """INSTALL tpcds;
LOAD tpcds;
SELECT * FROM dsdgen(sf=1);"""
    )
    run_query(engine, number)


if __name__ == "__main__":
    run_adhoc(24)
