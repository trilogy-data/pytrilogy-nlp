from logging import DEBUG, StreamHandler

from pytest import fixture

from trilogy_nlp import NLPEngine, Provider
from trilogy_nlp.constants import logger


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
