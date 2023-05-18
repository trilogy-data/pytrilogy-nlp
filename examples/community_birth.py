from trilogy_public_models import models
from preql import Dialects
from preql_nlp import build_query
from preql.hooks.query_debugger import DebuggingHook
from logging import StreamHandler, DEBUG
from preql_nlp.constants import logger
from preql.parsing.render import render_query

logger.setLevel(DEBUG)
logger.addHandler(StreamHandler())


# grab the model we want to parse
environment = models["bigquery.usa_names"]


processed_query = build_query(
    "Most common names in the state of VT in 1990?",
    environment,
    debug=True,
)

for key in processed_query.output_columns:
    print(key)

executor = Dialects.BIGQUERY.default_executor(
    environment=environment, hooks=[DebuggingHook()]
)

# render_query(processed_query)

# results = executor.execute_query(processed_query)
# for row in results:
#     print(row)
