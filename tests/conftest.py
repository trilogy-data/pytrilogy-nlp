
from preql_nlp.constants import logger
from pytest import fixture
from logging import StreamHandler, DEBUG, basicConfig

basicConfig()


@fixture
def test_logger():
    logger.addHandler(StreamHandler())
    logger.setLevel(DEBUG)
    yield logger