from trilogy_nlp.models import SemanticTokenResponse, InitialParseResponse


def test_phrase_token():
    test = """[
   {
      "phrase":"questions per year",
      "tokens":[
         "question",
         "year"
      ]
   }
]"""

    x = SemanticTokenResponse.model_validate_json(test)

    assert x[0].tokens == ["question", "year"]


def test_initial_parse_result():
    response = """{
"metrics":["number of questions"],
"dimensions": ["year"],
"limit": -1,
"order": [],
"filtering": [{"concept":"year", "values":["2020"], "operator":"="}]
}"""

    x = InitialParseResponse.model_validate_json(response)
    assert x.limit == -1
