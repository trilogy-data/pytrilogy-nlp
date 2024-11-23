from os.path import dirname
from sys import path

nb_path = __file__
root_path = dirname(dirname(__file__))

print(root_path)
path.insert(0, root_path)

from logging import DEBUG, StreamHandler

from trilogy import Dialects
from trilogy.hooks.query_debugger import DebuggingHook
from trilogy.parsing.render import render_query
from trilogy_public_models import models

from trilogy_nlp import Provider
from trilogy_nlp.constants import logger
from trilogy_nlp.core import NLPEngine
from trilogy_nlp.main import parse_query

logger.setLevel(DEBUG)
logger.addHandler(StreamHandler())


# grab the model we want to parse
environment = models["bigquery.usa_names"]

environment.concepts["state"].metadata.description = (
    "The common two character abbreviation for a state, such as MA for Massachusetts or CT for Connecticut."
)

environment = models["bigquery.thelook_ecommerce"]
open_ai = NLPEngine(provider=Provider.OPENAI, model="gpt-4o-mini")
processed_query = parse_query(
    "What were the most common names in Vermont in 1990?",
    environment,
    debug=True,
    llm=open_ai.llm,
)

print(render_query(processed_query))

executor = Dialects.BIGQUERY.default_executor(
    environment=environment, hooks=[DebuggingHook()]
)

print(executor.generator.compile_statement(processed_query))
results = executor.execute_query(
    executor.generator.generate_queries(
        environment=executor.environment, statements=[processed_query]
    )[0]
)
for row in results:
    print(row)
