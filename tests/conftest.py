
from preql_nlp.constants import logger
from pytest import fixture
from logging import StreamHandler, DEBUG

@fixture
def test_logger():
    logger.addHandler(StreamHandler())
    logger.setLevel(DEBUG)
    yield logger