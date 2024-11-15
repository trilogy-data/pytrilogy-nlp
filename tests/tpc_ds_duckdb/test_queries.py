from pathlib import Path
import pytest
import tomli_w
from trilogy import Executor
from trilogy_nlp.main import build_query
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

    if not set(environment.imports.keys()).issubset(set(imports)):
        raise EnvironmentSetupException(
            f"Mismatched imports: {imports} should be a superset of selected {list(environment.imports.keys())}"
        )
    processed_query = build_query(
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
        target = prompt_info.get("target", TARGET)
        attempts = prompt_info.get("attempts", ATTEMPTS)
        if not required:
            continue
        cases = []
        outputs = defaultdict(lambda: 0)
        for _ in range(0, attempts):
            result, reason = query_loop(prompt, imports, engine, idx, llm=llm)
            if reason:
                outputs[reason] += 1
            cases.append(result)
        ratio = sum(1 if c else 0 for c in cases) / attempts
        output[name] = ratio
        assert sum(1 if c else 0 for c in cases) / attempts >= target, outputs
    return output


def query_loop(
    prompt: str, imports: list[str], engine: Executor, idx: int, llm: BaseLanguageModel
) -> tuple[bool, str | None]:
    try:
        env, processed_query = helper(prompt, llm, imports)
    except EnvironmentSetupException as e:
        return False, str(e)
    except Exception as e:
        logger.error("Error in query_loop: %s", e)
        return False, str(e)
    # fetch our results
    # parse_start = datetime.now()
    engine.environment = env
    query = engine.generate_sql(processed_query)[-1]
    logger.info(query)
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

        return (
            False,
            f"Row count mismatch: target: {len(base_results)} != test: {len(comp_results)}",
        )
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
SELECT * FROM dsdgen(sf=.5);"""
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
            print("test: ", row, " vs sql: ", comp_results[idx])

    else:
        run_query(engine, number)


if __name__ == "__main__":
    TEST = """
import store_sales as store_sales;

metric customer_count <- count(store_sales.customer.id); # local to select
metric average_item_price_by_category <- avg(store_sales.item.current_price) by store_sales.item.category; # local to select
property inline_calc_319 <- average_item_price_by_category * 1.2; # local to select
WHERE
    ((store_sales.item.current_price > inline_calc_319 and store_sales.date.date.month = 1) and store_sales.date.date.year = 2001) and store_sales.item.category is not null
SELECT
    count(store_sales.customer.id) -> customer_count,
    store_sales.customer.state,
HAVING
    customer_count >= 10 and True

ORDER BY
    customer_count asc,
    store_sales.customer.state asc

LIMIT 100;"""
    TEST = """
import store_sales as store_sales;
WHERE
    store_sales.customer.demographics.gender = 'M' 
    and store_sales.customer.demographics.marital_status = 'S' and
    store_sales.customer.demographics.education_status = 'College' 
    and store_sales.date.date.year = 2000 and (store_sales.promotion.channel_event = 'N' or store_sales.promotion.channel_email = 'N')
SELECT
    avg(store_sales.quantity) -> average_quantity_sold,
    avg(store_sales.list_price) -> average_list_price,
    avg(store_sales.coupon_amt) -> average_coupon_amount,
    avg(store_sales.sales_price) -> average_sales_price,
    store_sales.item.name,
ORDER BY
    store_sales.item.name asc

LIMIT 100;
"""

    run_adhoc(
        7,
        text=TEST,
    )
