from trilogy import Dialects, Environment, Executor
from trilogy.dialect.config import DuckDBConfig
import pytest
from pathlib import Path

# from tests.modeling.tpc_ds_duckdb.analyze_test_results import analyze
from trilogy_nlp.constants import logger
from pytest import fixture
from logging import StreamHandler, DEBUG
from trilogy_nlp import NLPEngine, Provider
from trilogy_nlp.enums import CacheType
import os

working_path = Path(__file__).parent

SF = .5


@fixture(scope="session", autouse=True)
def test_logger():
    logger.addHandler(StreamHandler())
    logger.setLevel(DEBUG)
    yield logger


@fixture(scope="session", autouse=True)
def llm():

    # yield NLPEngine(provider=Provider.LLAMAFILE, model="na", cache=CacheType.SQLLITE, cache_kwargs={'database_path':".tests.db"}).llm
    yield NLPEngine(
        provider=Provider.OPENAI,
        model="gpt-3.5-turbo",
        cache=CacheType.SQLLITE,
        cache_kwargs={"database_path": ".tests.db"},
    ).llm
    # yield NLPEngine(provider=Provider.OPENAI, model="gpt-4").llm


@pytest.fixture(scope="session")
def engine():
    env = Environment(working_path=working_path)
    engine: Executor = Dialects.DUCK_DB.default_executor(
        environment=env,
        # hooks=[DebuggingHook(level=INFO, process_other=False, process_ctes=False)],
        conf=DuckDBConfig(),
    )
    string_sf = str(SF).replace(".", "_")
    base_path =   working_path / f"sf_{string_sf}" / "memory" 
    sentinal_file = base_path / "schema.sql"
    if Path(sentinal_file).exists():
        # TODO: Detect if loaded
        engine.execute_raw_sql(f"IMPORT DATABASE '{base_path}';")
    else:
        os.makedirs(base_path, exist_ok=True)
    results = engine.execute_raw_sql("SHOW TABLES;").fetchall()
    tables = [r[0] for r in results]
    if "store_sales" not in tables:
        engine.execute_raw_sql(
            f"""
        INSTALL tpcds;
        LOAD tpcds;
        SELECT * FROM dsdgen(sf={SF});
        EXPORT DATABASE '{base_path}' (FORMAT PARQUET);"""
        )
    yield engine


@pytest.fixture(autouse=True, scope="session")
def my_fixture():
    # setup_stuff
    yield
    # teardown_stuff
    # analyze()
