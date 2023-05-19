


from trilogy_public_models import models
from preql_nlp.main import parse_query
from preql.core.models import Select


def test_e2e_community_so(test_logger):
    # grab the model we want to parse
    environment = models["bigquery.stack_overflow"]
    location =environment.concepts['user.location']

    processed_query = parse_query(
        "Who is a user in germany?",
        environment,
        debug=True,
    )
    for x in processed_query.selection:
        print(x)
        test_logger.info(x)
    assert location in [x.content for x in processed_query.selection]
    # assert processed_query.fil
    # print(render_query(processed_query))