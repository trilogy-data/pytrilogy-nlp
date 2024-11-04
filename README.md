## Trilogy NLP

Natural language interface for generating SQL queries via a Trilogy data model.

Most of the value in a SQL statement comes from the column selection, transformation, and filtering.

Joins, table selection, group bys are all opportunitites to introduce errors. 

Trilogy is easier SQL for humans because it separates out those parts in the language into a reusable metadata
layer; the exact same benefits apply to an LLM.

The extra data encoded in the semantic model, and the significantly reduced target space for generation reduce common sources of LLM errors. 

This makes it more testable and less prone to hallucination than generating SQL directly. 

Trilogy-NLP is built on the common NLP backend (langchain, etc) and supports configurable backends.


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

Results are non-determistic; review the generated trilogy to make sure it maches your expectations. 

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

## Open AI Config
Requires setting the following environment variables
- OPENAI_API_KEY
- OPENAI_MODEL

Recommended to use "gpt-3.5-turbo" or higher as the model.

## LlamaFile Config
