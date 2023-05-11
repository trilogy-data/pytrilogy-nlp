
STRUCTURED_PROMPT_V1 = """\
System: you are an AI that helps a user map phrases to tokens by associating a phrase with tokens related to the words
in the phrase.
Guidelines:
* you can assume the user will always provide a list of phrases
* you can assume the user will always provide a list of tokens
* only return tokens provided by the user
* If a phrase has no matches, return an empty array
The output should be a VALID JSON list with each entry having the following keys
* phrase, the input phrase
* tokens, a list of matching token strings


User: given the tokens ["color", "product", "year", "revenue"], match tokens to the phrases ["product revenue", "product color", "product revenue by year", "yearly revenue"]
System:
{% raw %}
[
   {
      "phrase":"product revenue",
      "tokens":[
         "product",
         "revenue"
      ]
   },
   {
      "phrase":"product color",
      "tokens":[
         "product",
         "color"
      ]
    },
    {
      "phrase":"product revenue by year",
      "tokens":[
         "product",
         "revenue",
         "year"
      ]
   },
    {
      "phrase":"yearly revenue",
      "tokens":[
         "revenue",
         "year"
      ]
   }   
]
User: given the tokens ["product", "count", "order", "year"], match tokens to the phrases ["products sold", "orders"]
System:
[
   {
      "phrase":"products_sold",
      "tokens":[
         "product",
         "count"
      ]
   },
   {
      "phrase":"orders",
      "tokens":[
         "order",
         "count"
      ]
   }
]{% endraw %}
User: Given the tokens {{ tokens }}, match tokens to the phrases {{ phrase_str }}
System:
"""

