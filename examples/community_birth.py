from trilogy_public_models import models
from preql import Dialects
from preql_nlp import build_query
from preql.hooks.query_debugger import DebuggingHook
from logging import StreamHandler, DEBUG
from preql.parsing.render import render_query

from preql_nlp.constants import logger

logger.setLevel(DEBUG)
logger.addHandler(StreamHandler())


# grab the model we want to parse
environment = models["bigquery.usa_names"]


processed_query = build_query(
    "Most common birth names in the state of California in 1990?",
    environment,
    debug=True,
)

for key in processed_query.output_columns:
    print(key)

executor = Dialects.BIGQUERY.default_executor(
    environment=environment, hooks=[DebuggingHook()]
)

print(executor.generator.compile_statement(processed_query))
results = executor.execute_query(processed_query)
for row in results:
    print(row)
