from logging import INFO, StreamHandler

from trilogy import Dialects
from trilogy.hooks.query_debugger import DebuggingHook
from trilogy.parsing.render import render_query
from trilogy_public_models import models

from trilogy_nlp.constants import logger
from trilogy_nlp.main import parse_query

logger.setLevel(INFO)
logger.addHandler(StreamHandler())


# grab the model we want to parse
environment = models["bigquery.thelook_ecommerce"]


processed_query = parse_query(
    "Top 10 cities with the most orders with returned status in 4? ",
    environment,
    debug=True,
)

print(render_query(processed_query))


executor = Dialects.BIGQUERY.default_executor(
    environment=environment, hooks=[DebuggingHook()]
)


results = executor.execute_query(processed_query)
for row in results:
    print(row)
