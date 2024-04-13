from sys import path
from os.path import dirname

nb_path = __file__
root_path = dirname(dirname(dirname(__file__)))

print(root_path)
path.insert(0, root_path)


from trilogy_public_models import models
from preql_nlp.main import parse_query
from logging import StreamHandler, INFO
from preql_nlp.constants import logger
from preql.parsing.render import render_query
from preql_nlp.core import NLPEngine
from preql_nlp import Provider

logger.setLevel(INFO)
logger.addHandler(StreamHandler())


# grab the model we want to parse
environment = models["bigquery.thelook_ecommerce"]
open_ai = NLPEngine(provider=Provider.OPENAI, model="gpt-3.5-turbo")

processed_query = parse_query(
    input_text="How many orders were placed in 2023?",
    input_environment=environment,
    debug=True,
    llm=open_ai.llm,
)

print(render_query(processed_query))
