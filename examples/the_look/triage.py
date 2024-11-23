from os.path import dirname
from sys import path

nb_path = __file__
root_path = dirname(dirname(dirname(__file__)))

print(root_path)
path.insert(0, root_path)


from logging import INFO, StreamHandler

from trilogy.parsing.render import render_query
from trilogy_public_models import models

from trilogy_nlp import Provider
from trilogy_nlp.constants import logger
from trilogy_nlp.core import NLPEngine
from trilogy_nlp.main import parse_query

logger.setLevel(INFO)
logger.addHandler(StreamHandler())


# grab the model we want to parse
environment = models["bigquery.thelook_ecommerce"]
open_ai = NLPEngine(provider=Provider.OPENAI, model="gpt-4o-mini")

processed_query = parse_query(
    input_text="How many orders were placed by customers living in Burlington, VT in 2023?",
    input_environment=environment,
    debug=True,
    llm=open_ai.llm,
)

print(render_query(processed_query))
