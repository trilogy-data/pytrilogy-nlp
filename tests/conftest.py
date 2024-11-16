from trilogy_nlp.constants import logger
from pytest import fixture
from logging import StreamHandler, DEBUG
from trilogy_nlp import NLPEngine, Provider


@fixture(scope="session", autouse=True)
def test_logger():
    logger.addHandler(StreamHandler())
    logger.setLevel(DEBUG)
    yield logger


@fixture(scope="session", autouse=True)
def engine():
    yield NLPEngine(
        provider=Provider.OPENAI,
        model="gpt-3.5-turbo",
    ).llm
