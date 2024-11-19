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

> [!TIP]
> These utilize the `trilogy-public-models` package to get predefined model.s, which can be installed with `pip install trilogy-public-models`

### Hello World

```python

from trilogy_public_models import get_executor
from trilogy_nlp import NLPEngine, Provider, CacheType

# we use this to run queries
# get a Trilogy executor preloaded with the tpc_ds schema in duckdb
# Executors run queries again a model using an engine
executor = get_executor("duckdb.tpc_ds")

# create an NLP engine
# we use this to generate queries against the model
engine = NLPEngine(
    provider=Provider.OPENAI,
    model="gpt-4o-mini",
    cache=CacheType.SQLLITE,
    cache_kwargs={"database_path": ".demo.db"},
)

# We can pass the executor to the engine
# to directly run a querie
results = engine.run_query(
    "What was the store sales for the first 5 days of January 2000 for customers in CA?",
    executor=executor,
)

for row in results:
    print(row)

# Or generate a query without executing it
query = engine.generate_query(
    "What was the store sales for the first 5 days of January 2000 for customers in CA?",
    env=executor.environment,
)

# which can compile it to SQL
# this might be multiple statements in some cases
# but here we can just grab the last one
print(executor.generate_sql(query)[-1])


```

### BQ Example
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

> [!WARNING]  
> Don't expect perfection - results are non-determistic; review the generated Trilogy to make sure it maches your expectations. Treat queries as a starting point for refinement. 

## Setting Up Your Environment

Recommend that you work in a virtual environment with requirements from both requirements.txt and requirements-test.txt installed. The latter is necessary to run
tests (surprise). 

trilogy-nlp is python 3.10+

## Open AI Config
Requires setting the following environment variables or passing them into NLPEngine creation.

- OPENAI_API_KEY
- OPENAI_MODEL

Recommended to use "gpt-4o-mini" or higher as the model.

## Gemini
Requires setting the following environment variables or passing them into NLpEngine reation

- GOOGLE_API_KEY

## LlamaFile Config

Run server locally
