from trilogy import Environment
from trilogy.core.enums import Purpose
from trilogy.core.models.author import Concept
from trilogy.core.models.core import DataType

from trilogy_nlp.main import parse_query


def test_e2e_basic(engine):
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
        keys={id.address},
    )
    name = Concept(
        name="name",
        namespace="user",
        datatype=DataType.STRING,
        purpose=Purpose.PROPERTY,
        keys={id.address},
    )

    environment.add_concept(concept=id)
    environment.add_concept(concept=location)
    environment.add_concept(concept=name)

    location = environment.concepts["user.location"]

    processed_query = parse_query(
        input_text="Which users are in germany?",
        input_environment=environment,
        llm=engine,
    )
    assert location in [x for x in processed_query.where_clause.concept_arguments]
