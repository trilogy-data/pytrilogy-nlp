from pathlib import Path
import pytest
import tomli_w
from trilogy import Executor
from trilogy_nlp.main import build_query
from trilogy_nlp import NLPEngine
from trilogy_nlp.environment import build_env_and_imports
from trilogy_nlp.constants import logger
from langchain_core.language_models import BaseLanguageModel
from collections import defaultdict
from datetime import datetime

import tomllib

working_path = Path(__file__).parent

# defaults
ATTEMPTS = 1

TARGET = 0.8

GLOBAL_DEBUG: bool = True


class EnvironmentSetupException(Exception):
    pass


def helper(text: str, llm, imports: list[str]):
    environment = build_env_and_imports(
        text, working_path=working_path, llm=llm, debug=True
    )

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


def matrix(
    engine: Executor,
    idx: int,
    llm: BaseLanguageModel,
    prompts: dict[str, dict[str, str]],
    debug: bool = False,
) -> dict[str, dict[str, int | list[int]]]:
    output = {}
    output["cases"] = {}
    output["durations"] = {}
    for name, prompt_info in prompts.items():
        prompt = prompt_info["prompt"]
        imports = prompt_info["imports"]
        required = prompt_info.get("required", True)
        target = prompt_info.get("target", TARGET)
        attempts = prompt_info.get("attempts", ATTEMPTS)
        if not required:
            continue
        cases = []
        failure_reason = defaultdict(lambda: 0)
        durations = []
        for _ in range(0, attempts):
            start = datetime.now()
            result, reason = query_loop(
                prompt,
                imports,
                engine,
                idx,
                llm=llm,
                debug=debug,
            )
            if result is not True:
                failure_reason[reason] += 1
            cases.append(result)
            end = datetime.now()
            duration = end - start
            durations.append(duration.total_seconds())

        ratio = sum(1 if c else 0 for c in cases) / attempts
        output["cases"][name] = ratio
        output["durations"][name] = durations
        assert sum(1 if c else 0 for c in cases) / attempts >= target, {
            k: v for k, v in failure_reason.items()
        }
        logger.info(f"Successful run for query {idx}!")
    return output


def query_loop(
    prompt: str,
    imports: list[str],
    engine: Executor,
    idx: int,
    llm: BaseLanguageModel,
    debug: bool = False,
) -> tuple[bool, str | None]:
    try:
        env, processed_query = helper(prompt, llm, imports)
    except EnvironmentSetupException as e:
        if debug:
            raise e
        return False, str(e)
    except Exception as e:
        logger.error("Error in query_loop: %s", e)
        print(
            f"Error in query_loop: {str(e)}",
        )
        if debug:
            raise e
        return False, str(e)
    # fetch our results
    # parse_start = datetime.now()
    engine.environment = env
    query = engine.generate_sql(processed_query)[-1]
    # parse_time = datetime.now() - parse_start
    # exec_start = datetime.now()
    results = engine.execute_raw_sql(query)
    # exec_time = datetime.now() - exec_start
    # assert results == ''
    comp_results = list(results.fetchall())
    try:
        assert len(comp_results) > 0, "No results returned"
    except AssertionError as e:
        if debug:
            raise e
        return False, str(e)
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


def run_query(engine: Executor, idx: int, llm: NLPEngine, debug: bool = False):

    with open(working_path / f"query{idx:02d}.prompt") as f:
        text = f.read()
        parsed = tomllib.loads(text)

    matrix_info = matrix(engine, idx, llm.llm, parsed, debug=debug)

    with open(working_path / f"zquery{idx:02d}.log", "w") as f:
        f.write(
            tomli_w.dumps(
                {
                    "query_id": idx,
                    "model": f"{llm.provider.value}-{llm.model}",
                    **matrix_info,
                },
                multiline_strings=True,
            )
        )
    return 1


def test_one(engine, llm):
    run_query(engine, 1, llm, debug=GLOBAL_DEBUG)


@pytest.mark.skip(reason="Is duckdb correct??")
def test_two(engine):
    run_query(engine, 2)


def test_three(engine, llm):
    run_query(engine, 3, llm, debug=GLOBAL_DEBUG)


@pytest.mark.skip(reason="Is duckdb correct??")
def test_four(engine):
    run_query(engine, 4)


@pytest.mark.skip(reason="Is duckdb correct??")
def test_five(engine):
    run_query(engine, 5)


@pytest.mark.cli
def test_six(engine, llm):
    run_query(engine, 6, llm, debug=GLOBAL_DEBUG)


def test_seven(engine, llm):
    run_query(engine, 7, llm, debug=GLOBAL_DEBUG)


@pytest.mark.skip(reason="No prompt yet")
def test_eight(engine, llm):
    run_query(engine, 8, llm)


@pytest.mark.skip(reason="No prompt yet")
def test_ten(engine, llm):
    query = run_query(engine, 10, llm)
    assert len(query) < 7000, query


# @pytest.mark.skip(reason="No prompt yet")
def test_twelve(engine, llm):
    run_query(engine, 12, llm, debug=GLOBAL_DEBUG)


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
        from trilogy_nlp import NLPEngine, Provider
        from trilogy_nlp.enums import CacheType

        llm = NLPEngine(
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            cache=CacheType.SQLLITE,
            cache_kwargs={"database_path": ".tests.db"},
        ).llm
        run_query(engine, number, llm=llm, debug=True)


if __name__ == "__main__":
    TEST = """
import web_sales as web_sales;
metric total_class_external_sales <- sum(web_sales.external_sales_price) by web_sales.item.class; # local to select
property class_revenue_ratio <- total_external_sales_price / total_class_external_sales; # local to select
WHERE
    web_sales.item.category in ['Sports', 'Books', 'Home'] and (web_sales.date.date >= CAST('1999-02-22' AS date) and web_sales.date.date <= CAST('1999-03-24' AS date))
SELECT
    web_sales.item.name,
    web_sales.item.desc,
    web_sales.item.category,
    web_sales.item.class,
    web_sales.item.current_price,
    sum(web_sales.external_sales_price) by web_sales.item.name, web_sales.item.desc, web_sales.item.category, web_sales.item.class, web_sales.item.current_price -> total_external_sales_price,
    total_external_sales_price / total_class_external_sales -> class_revenue_ratio,
ORDER BY
    web_sales.item.category asc,
    web_sales.item.class asc,
    web_sales.item.name asc,
    web_sales.item.desc asc,
    class_revenue_ratio asc

LIMIT 100;"""

    run_adhoc(
        12,
        text=TEST
    )
