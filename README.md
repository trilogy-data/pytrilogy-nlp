## Trilogy NLP

Natural language interface for generating SQL queries via a Trilogy data model.

Trilogy is easier for a large language model (LLM) to interact with as it requires only identifying which
objects in the data model best match the question, rather than generating arbitrary SQL from scratch.
The extra data encoded in the semantic model for query generation reduces common sources of LLM errors. 

This makes it more testable and less prone to hallucination than generating SQL directly. 

Requires setting the following environment variables
- OPENAI_API_KEY
- OPENAI_MODEL

Recommended to use "gpt-3.5-turbo" or higher as the model.

## Examples

### Basic BQ example


```python
from trilogy_public_models import models
from trilogy import Executor, Dialects
from trilogy_nlp import build_query

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

## Don't Expecct Perfection

Results are non-determistic and may not always be accurate.

```sql
# generated from prompt: What is Taylor Swift's birthday? How many questions were asked on that day in 2020?
SELECT
    question.count,
    answer.creation_date.year,
    question.creation_date.year,
    question.creation_date,
WHERE
    question.creation_date.year = 1989
ORDER BY
    question.count desc,

    question.count desc

LIMIT 100;
```



## Setting Up Your Environment

Recommend that you work in a virtual environment with requirements from both requirements.txt and requirements-test.txt installed. The latter is necessary to run
tests (surprise). 

trilogy-nlp is python 3.10+
