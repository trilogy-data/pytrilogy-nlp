from trilogy_public_models import models
from preql_nlp.main import parse_query
from logging import StreamHandler, INFO
from preql_nlp.constants import logger
from preql.parsing.render import render_query

logger.setLevel(INFO)
logger.addHandler(StreamHandler())


# grab the model we want to parse
environment = models["bigquery.thelook_ecommerce"]


processed_query = parse_query(
    "what were the top 10 cities for cancelled orders that were created in 2020?  ",
    environment,
    debug=True,
)

print(render_query(processed_query))