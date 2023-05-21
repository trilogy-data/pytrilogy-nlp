from trilogy_public_models import models
from preql_nlp.main import parse_query
from logging import StreamHandler, DEBUG
from preql_nlp.constants import logger
from preql.parsing.render import render_query

logger.setLevel(DEBUG)
logger.addHandler(StreamHandler())


# grab the model we want to parse
environment = models["bigquery.stack_overflow"]


processed_query = parse_query(
    "How many questions are from authors in Germany?",
    environment,
    debug=True,
)

print(render_query(processed_query))