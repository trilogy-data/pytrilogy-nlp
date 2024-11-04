from trilogy_public_models import models
from trilogy_nlp.main import parse_query
from logging import StreamHandler, DEBUG
from trilogy_nlp.constants import logger
from trilogy.parsing.render import render_query
from trilogy import Dialects
from trilogy.hooks.query_debugger import DebuggingHook

logger.setLevel(DEBUG)
logger.addHandler(StreamHandler())


# grab the model we want to parse
environment = models["bigquery.usa_names"]

environment.concepts["state"].metadata.description = (
    "The common two character abbreviation for a state, such as MA for Massachusetts or CT for Connecticut."
)


processed_query = parse_query(
    "What were the most common names in Vermont in 1990?",
    environment,
    debug=True,
)

print(render_query(processed_query))

executor = Dialects.BIGQUERY.default_executor(
    environment=environment, hooks=[DebuggingHook()]
)

print(executor.generator.compile_statement(processed_query))
results = executor.execute_query(
    executor.generator.generate_queries([processed_query])[0]
)
for row in results:
    print(row)
