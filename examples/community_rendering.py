from trilogy_public_models import models
from preql_nlp.main import parse_query
from logging import StreamHandler, DEBUG
from preql_nlp.constants import logger
from preql.parsing.render import render_query

logger.setLevel(DEBUG)
logger.addHandler(StreamHandler())


# grab the model we want to parse
environment = models["bigquery.usa_names"]

environment.concepts[
    "state"
].metadata.description = "The common two character abbreviation for a state, such as MA for Massachusetts or CT for Connecticut."


processed_query = parse_query(
    "What were the most common names in Vermont in 1990?",
    environment,
    debug=True,
)

print(render_query(processed_query))
