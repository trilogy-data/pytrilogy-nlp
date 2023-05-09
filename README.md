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
from preql import Dialects
from preql_nlp import build_query, NlpPreqlModelClient

# define the model we want to parse
environment = models["bigquery.stack_overflow"]

# set up preql executor
# default bigquery executor requires local default credentials configured
executor = Dialects.BIGQUERY.default_executor(environment= environment)

# build an NLP client for the preql model
client = NlpPreqlModelClient(openai_model="gpt-3.5-turbo", preql_model=environment, preql_executor=executor)

# ask a data question  about the model in natural language.
question = "How many questions are asked per year?"
results = client.answer(question)

# print the results
for r in results:
    print(r)

```



## Setting Up Your Environment

Recommend that you work in a virtual environment with requirements from both requirements.txt and requirements-test.txt installed. The latter is necessary to run
tests (surprise). 

Pypreql-nlp is python 3.10+
