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

### Basic BQ example


```python
from trilogy_public_models import models
from preql import Executor, Dialects
from preql_nlp import build_query

# define the model we want to parse
environment = models["bigquery.stack_overflow"]

# set up preql executor
# default bigquery executor requires local default credentials configured
executor = Dialects.BIGQUERY.default_executor(environment= environment)

# build a query off text and the selected model
processed_query = build_query(
    "How many questions are asked per year?",
    environment,
)

# make sure we got reasonable outputs
for concept in processed_query.output_columns:
    print(concept.name)

# and run that to get our answer
results = executor.execute_query(processed_query)
for row in results:
    print(row)
```



## Setting Up Your Environment

Recommend that you work in a virtual environment with requirements from both requirements.txt and requirements-test.txt installed. The latter is necessary to run
tests (surprise). 

Pypreql-nlp is python 3.10+
