
from sys import path
from os.path import dirname

nb_path = __file__
root_path = dirname(dirname(dirname(__file__)))

print(root_path)
path.insert(0, root_path)


from examples.titanic.setup_environment import setup_engine, setup_titanic

from preql_nlp.enums import Provider
from preql.core.models import Environment

from preql_nlp.main_v2 import build_query as build_query_v2
from logging import StreamHandler, DEBUG

from preql_nlp.constants import logger
from preql_nlp.main import build_query


if __name__ == "__main__":
    provider_type = Provider.GOOGLE
    executor = setup_engine()

    env = Environment()
    model = setup_titanic(env)
    for c in env.concepts:
        print(c)

    executor.environment = env
    environment = env
    logger.setLevel(DEBUG)
    logger.addHandler(StreamHandler())

    question = ("HOw many passengers survived in each cabin?",)
    processed_query = build_query(
        question,
        environment,
        debug=True,
        provider=provider_type
    )

    processed_query_v2 = build_query_v2(
        question,
        environment,
        debug=True,
        provider = Provider.GOOGLE
    )

    for q in [processed_query, processed_query_v2]:
        print("output_columns")
        for key in q.output_columns:
            print(key)

        print(executor.generator.compile_statement(q))
        results = executor.execute_query(q)
        for row in results:
            print(row)