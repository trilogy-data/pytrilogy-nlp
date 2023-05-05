from typing import List


def gen_structured_prompt_v1(phrases:List[str], tokens:str)->str:
    phrase_str = ', '.join(phrases)
    user_input = f"Given the tokens {tokens}, match tokens to the phrases {phrase_str}"
    template = f"""\
System: you are an AI that helps a user map phrases to tokens by associating a phrase with tokens related to the words
in the phrase.
Guidelines:
* you can assume the user will always provide a list of phrases
* you can assume the user will always provide a list of tokens
* only return tokens provided by the user
The output should be a VALID JSON blob with the phrases as keys and arrays of tokens as values:
* If a phrase has no matches, return an empty array
User: given the tokens ["color", "product", "year", "revenue"], match tokens to the phrases ["product revenue", "product color", "product revenue by year", "yearly revenue"]
System:
{{
    "product revenue": ["product", "revenue"],
    "product color": ["product", "color"],
    "product revenue by year": ["product", "revenue", "year"],
    "yearly revenue": ["year", "revenue"]
}}
User: given the tokens ["product", "count", "order", "year"], match tokens to the phrases ["products sold", "orders"]
System:
{{
    "products sold": ["product", "count"],
    "orders": ["order", "count"]
}}
User: { user_input }
System:
"""  # noqa"""
    return template

