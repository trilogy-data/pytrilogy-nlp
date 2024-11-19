from trilogy_nlp import NLPEngine, Provider, CacheType
from trilogy import Executor

def test_age_class_query_resolution(normalized_engine:Executor, test_env):
    nlp_engine = NLPEngine(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        cache=CacheType.SQLLITE,
        cache_kwargs={"database_path": ".tests.db"},
    )
    executor = normalized_engine
    env = test_env
    executor.environment = env

    for _ in range(3):
        query = nlp_engine.build_query_from_text(text = 'What was the survival rate (ratio) by class? order by the class id, ascending', env=test_env)
        results = executor.execute_query(query).fetchall()
        expected = [(1, 0.6296296296296297), (2, 0.47282608695652173), (3, 0.24236252545824846)]
        for idx, row in enumerate(results):
            assert all([y in row for y in expected[idx]]), f'Expected {expected[idx]} but got {row}'


def test_filtering(normalized_engine:Executor, test_env):
    nlp_engine = NLPEngine(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        cache=CacheType.SQLLITE,
        cache_kwargs={"database_path": ".tests.db"},
    )
    executor = normalized_engine
    env = test_env
    executor.environment = env


    query = nlp_engine.build_query_from_text(text = 'How many survivors in each cabin in 1st class?', env=test_env)

    query_text = executor.generate_sql(query)[0]

    executor.execute_query(query).fetchall()
    
    assert 'dim_class."class" = 1' in query_text