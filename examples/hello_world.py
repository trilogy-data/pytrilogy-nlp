from sys import path
from os.path import dirname

nb_path = __file__
root_path = dirname(dirname(__file__))

print(root_path)
path.insert(0, root_path)

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

# Or generate a query without executing it
query = engine.generate_query(
    "What was the store sales for the first 5 days of January 2000 for customers in CA?",
    env=executor.environment,
)

# which can compile it to SQL
# this might be multiple statements in some cases
# but here we can just grab the last one
print(executor.generate_sql(query)[-1])

