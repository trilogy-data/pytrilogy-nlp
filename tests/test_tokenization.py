from preql_nlp.tokenization import tokens_to_concept


def test_tokens_to_concept():
    x = tokens_to_concept(
        ["date", "year", "time"],
        ["answer.creation.date", "answer.creation.date.year"],
        universe=["answer"],
        limits=1,
    )
    assert x == ["answer.creation.date.year"]

    x = tokens_to_concept(
        ["timestamp", "second", "time", "hour", "date", "minute"],
        [
            "badge.awarded_time",
            "badge.awarded_date",
            "answer.creation_date",
            "question.creation_date",
        ],
        universe=[
            "month",
            "timestamp",
            "second",
            "time",
            "hour",
            "count",
            "date",
            "minute",
            "year",
            "question",
        ],
        limits=1,
    )
    assert x == ["question.creation_date"]
