from pathlib import Path
import pytest
import tomli_w
from trilogy import Executor
from trilogy_nlp.main import build_query as build_query_v2
from trilogy_nlp.environment import build_env_and_imports
from trilogy_nlp.constants import logger
from langchain_core.language_models import BaseLanguageModel
from collections import defaultdict

import tomllib

working_path = Path(__file__).parent


class EnvironmentSetupException(Exception):
    pass


def helper(text: str, llm, imports: list[str]):
    environment = build_env_and_imports(text, working_path=working_path, llm=llm)

    if not set(imports) == set(environment.imports.keys()):
        raise EnvironmentSetupException(
            f"Mismatched imports: {imports} not same  {list(environment.imports.keys())}"
        )
    processed_query = build_query_v2(
        input_text=text,
        input_environment=environment,
        debug=True,
        llm=llm,
    )
    return environment, processed_query


# from dataclasses import dataclass

# @dataclass
# def PromptInput:

ATTEMPTS = 1

TARGET = 0.8


def matrix(
    engine: Executor,
    idx: int,
    llm: BaseLanguageModel,
    prompts: dict[str, dict[str, str]],
) -> dict[str, int]:
    output = {}
    for name, prompt_info in prompts.items():
        prompt = prompt_info["prompt"]
        imports = prompt_info["imports"]
        required = prompt_info.get("required", True)
        if not required:
            continue
        cases = []
        outputs = defaultdict(lambda: 0)
        for attempt in range(0, ATTEMPTS):
            result, reason = query_loop(prompt, imports, engine, idx, llm=llm)
            if reason:
                outputs[reason] += 1
            cases.append(result)
        ratio = sum(1 if c else 0 for c in cases) / ATTEMPTS
        output[name] = ratio
        assert sum(1 if c else 0 for c in cases) / ATTEMPTS > TARGET, outputs
    return output


def query_loop(
    prompt: str, imports: list[str], engine: Executor, idx: int, llm: BaseLanguageModel
) -> tuple[bool, str | None]:
    try:
        env, processed_query = helper(prompt, llm, imports)
    except EnvironmentSetupException as e:
        return False, str(e)
    # fetch our results
    # parse_start = datetime.now()
    engine.environment = env
    query = engine.generate_sql(processed_query)[-1]
    logger.info(query)
    print(query)
    # parse_time = datetime.now() - parse_start
    # exec_start = datetime.now()
    results = engine.execute_raw_sql(query)
    # exec_time = datetime.now() - exec_start
    # assert results == ''
    comp_results = list(results.fetchall())
    assert len(comp_results) > 0, "No results returned"
    # run the built-in comp
    # comp_start = datetime.now()
    base = engine.execute_raw_sql(f"PRAGMA tpcds({idx});")
    base_results = list(base.fetchall())
    # comp_time = datetime.now() - comp_start

    # # check we got it
    if len(base_results) != len(comp_results):

        return False, f"Row count mismatch: {len(base_results)} != {len(comp_results)}"
    for qidx, row in enumerate(base_results):
        for cell in row:
            if cell not in comp_results[qidx]:
                return (
                    False,
                    f"Could not find value {cell} in row {qidx} (expected row v test): {row} != {comp_results[qidx]}",
                )
    return True, None


def run_query(engine: Executor, idx: int, llm: BaseLanguageModel):

    with open(working_path / f"query{idx:02d}.prompt") as f:
        text = f.read()
        parsed = tomllib.loads(text)

    prompts = matrix(engine, idx, llm, parsed)

    with open(working_path / f"zquery{idx:02d}.log", "w") as f:
        f.write(
            tomli_w.dumps(
                {"query_id": idx, "model": str(type(llm)), "success_rates": prompts},
                multiline_strings=True,
            )
        )
    return 1


def test_one(engine, llm):
    run_query(engine, 1, llm)


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


def run_adhoc(number: int, text: str | None = None, comparison: str | None = None):
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
    if text:
        if comparison:
            comp_text = comparison
        else:
            comp_text = f"PRAGMA tpcds({number});"
        comp = engine.execute_raw_sql(comp_text)
        comp_results = comp.fetchall()
        results = engine.execute_text(text)
        for idx, row in enumerate(results[0].fetchall()):
            print(row)
            print(comp_results[idx])
    else:
        run_query(engine, number)


if __name__ == "__main__":
    TEST_2 = """
import store_returns as store_returns;
    WHERE
    store_returns.return_date.year = 2000 and store_returns.store.state = 'TN'
SELECT
    store_returns.customer.text_id,
    store_returns.store.id,
    sum(store_returns.return_amount) by store_returns.customer.text_id, store_returns.store.id -> customer_total_returns,
    avg(customer_total_returns) by store_returns.store.id -> avg_total_returns_by_store,
    1.2 * avg_total_returns_by_store -> inline_calc_158,
HAVING
    customer_total_returns > inline_calc_158 
ORDER BY
    store_returns.customer.text_id asc

LIMIT 100;"""
    TEST = """
import store_returns as store_returns;
WHERE
    store_returns.return_date.date.year = 2000 and store_returns.store.state = 'TN'
SELECT
    store_returns.customer.text_id,
    store_returns.store.id, 
    sum(store_returns.return_amount) -> total_return_amount,
    avg(total_return_amount) by store_returns.store.id -> avg_return_per_store,
    1.2 * avg_return_per_store -> twelve_times_avg_return_per_store,
HAVING
    total_return_amount > twelve_times_avg_return_per_store

ORDER BY
    store_returns.customer.text_id asc

LIMIT 100;"""
    run_adhoc(
        1,
        text=TEST_2,
        comparison="""WITH customer_total_return AS
  (SELECT sr_customer_sk AS ctr_customer_sk,
          sr_store_sk AS ctr_store_sk,
          sum(sr_return_amt) AS ctr_total_return
   FROM store_returns,
        date_dim
   WHERE sr_returned_date_sk = d_date_sk
     AND d_year = 2000
   GROUP BY sr_customer_sk,
            sr_store_sk)
SELECT 
    c_customer_id
FROM customer_total_return ctr1,
     store,
     customer
WHERE ctr1.ctr_total_return >
    (SELECT avg(ctr_total_return)*1.2
     FROM customer_total_return ctr2
     WHERE ctr1.ctr_store_sk = ctr2.ctr_store_sk)
  AND s_store_sk = ctr1.ctr_store_sk
  AND s_state = 'TN'
  AND ctr1.ctr_customer_sk = c_customer_sk
ORDER BY c_customer_id
LIMIT 100;""",
    )
