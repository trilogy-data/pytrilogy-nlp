from logging import DEBUG, StreamHandler

from trilogy import Dialects
from trilogy.hooks.query_debugger import DebuggingHook
from trilogy_public_models import models

from trilogy_nlp import build_query
from trilogy_nlp.constants import logger

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
