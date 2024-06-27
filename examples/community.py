from trilogy_public_models import models
from trilogy import Dialects
from trilogy_nlp import build_query
from trilogy.hooks.query_debugger import DebuggingHook
from logging import StreamHandler, DEBUG
from trilogy_nlp.constants import logger

logger.setLevel(DEBUG)
logger.addHandler(StreamHandler())


# grab the model we want to parse
environment = models["bigquery.stack_overflow"]

# define a few new metrics
environment.parse("auto question.answer.count <- count(answer.id) by question.id;")

environment.parse("auto question.answer.count.avg <- answer.count/ question.count;")

processed_query = build_query(
    "How many questions are asked per year? order results by year desc",
    environment,
    debug=True,
)

for key in processed_query.output_columns:
    print(key)

executor = Dialects.BIGQUERY.default_executor(
    environment=environment, hooks=[DebuggingHook()]
)

results = executor.execute_query(processed_query)
for row in results:
    print(row)
