from preql_nlp.main import parse_query
from preql.core.models import Environment, Concept
from preql.core.enums import Purpose, DataType


def test_e2e_basic(test_logger):
    # grab the model we want to parse
    environment = Environment()
    id = Concept(
        name="id", namespace="user", datatype=DataType.INTEGER, purpose=Purpose.KEY
    )
    location = Concept(
        name="location",
        namespace="user",
        datatype=DataType.STRING,
        purpose=Purpose.PROPERTY,
        keys=[id],
    )
    name = Concept(
        name="name",
        namespace="user",
        datatype=DataType.STRING,
        purpose=Purpose.PROPERTY,
        keys=[id],
    )

    environment.add_concept(concept=id)
    environment.add_concept(concept=location)
    environment.add_concept(concept=name)

    location = environment.concepts["user.location"]

    processed_query = parse_query(
        "Which users are in germany?",
        environment,
        debug=True,
    )
    assert location in [x for x in processed_query.output_components]
    # assert processed_query.fil
    # print(render_query(processed_query))
