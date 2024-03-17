from trilogy_public_models import models
from preql_nlp.main import parse_query
from logging import StreamHandler, INFO
from preql_nlp.constants import logger
from preql.parsing.render import render_query
from preql import Dialects
from preql.hooks.query_debugger import DebuggingHook

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
