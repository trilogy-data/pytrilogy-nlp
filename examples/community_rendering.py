from trilogy_public_models import models
from preql import Dialects
from preql_nlp import build_query
from preql_nlp.main import parse_query
from preql.hooks.query_debugger import DebuggingHook
from logging import StreamHandler, DEBUG
from preql_nlp.constants import logger
from preql.parsing.render import render_query

logger.setLevel(DEBUG)
logger.addHandler(StreamHandler())


# grab the model we want to parse
environment = models["bigquery.usa_names"]


processed_query = parse_query(
    "Most common names in Vermont in 1990?",
    environment,
    debug=True,
)

print(render_query(processed_query))