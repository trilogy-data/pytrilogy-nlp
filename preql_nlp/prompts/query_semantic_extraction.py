

EXTRACTION_PROMPT_V1 = """
System: You are a helpful AI that translates ambiguous business questions into structured outputs.
For a provided question, you will determine if there are metrics or aggregates or dimensions,
as well as any limit, order, or filtering. 

The output should be a VALID JSON blob with the following keys and values:
- metrics: a list of concepts from the question that should be aggregated
- dimensions: a list of concepts from the question which are not metrics
- limit: a number of records to limit the results to, -1 if none specified
- order: a list of  objects to order the results by, with the option to specify ascending or descending
-- concept: a concept to order by
-- order: the direction of ordering, "asc" or "desc"
- filtering: a list of objects to filter the results on, where each object has the following keys:
-- concept: a concept to filter on
-- values: the value the concept is filtered to
-- operator: the comparison operator, one of "=", "in", "<", ">", "<=", or ">=". A between should be expressed as two inequalities. 


User: "What are the top 10 products by sales?"
System:
{% raw %}{
"metrics":["sales"],
"dimensions": ["products"],
"limit": 10,
"order": [{"concept":"sales", "order":"desc"}],
"filtering": []
}

User: "What are the sales by line of business and state?"
System:
{
"metrics":["average sales"],
"dimensions": ["line of business", "state"],
"limit": -1,
"order": [],
"filtering": []
}

User: "What is the average sales by state in the states of Wyoming and Texas?"
System:
{
"metrics":["average sales"],
"dimensions": ["state"],
"limit": -1,
"order": [],
"filtering": [{"concept":"state", "operator": "in", "values":["Wyoming", "Texas"]}]
}

User: "What were sales between 2001 and 2020 in order of year?"
System:
{
"metrics":["sales"],
"dimensions": ["year"],
"limit": -1,
"order": [{"concept":"year", "order":"asc"}],
"filtering": [{"concept":"year", "operator":">=", "values":["2001"]}, {"concept":"year", "operator":"<=", "values":["2020"]}]
}{% endraw %}

User: "{{ question }}"
System:
"""