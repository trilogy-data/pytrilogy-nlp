## PreQL NLP

Natural language interface for generating PreQL query objects.

PreQL is easier for a large language model to interact with as it requires only extract relevant concepts from a text,
classifying them as metrics or dimensions, and mapping them to what is available in a model.

This makes it more testable and less prone to hallucination than generating SQL directly. 

Requires setting the following environment variables
- OPENAI_API_KEY
- OPENAI_MODEL

Recommended to use "gpt-3.5-turbo" or higher as the model.




## Examples


```python
from trilogy_public_models import models
from preql import Executor, default_engine, Dialect
from preql_nlp import build_query

## set up standard preql configuration
engine = default_engine(Dialect.BIGQUERY)
# grab the model we want to parse
environment = models["bigquery.stack_overflow"]

processed_query = build_query(
    "How many questions are asked per year?",
    models["bigquery.stack_overflow"],
)

executor = Executor(
    engine=engine,
    dialect=Dialects.BIGQUERY,
    environment=environment,
)

results = executor.execute_query(processed_query)
for row in results:
    print(row)
```



## Setting Up Your Environment

Recommend that you work in a virtual environment with requirements from both requirements.txt and requirements-test.txt installed. The latter is necessary to run
tests (surprise). 

Pypreql-nlp is python 3.10+
