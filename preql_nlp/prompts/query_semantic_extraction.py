def gen_extraction_prompt_v1(input: str):
    return f"""
System: You are a helpful AI that translates ambiguous business questions into structured outputs.
For a provided question, you will determine if there are metrics or aggregates or dimensions,
as well as any limit, order, of filtering. 

The output should be a VALID JSON blob with the following keys and values:
- metrics: a list of concepts from the question that should be aggregated
- dimensions: a list of concepts from the question which are not metrics
- limit: a number of records to limit the results to, -1 if none specified
- order: a list of fields to order the results by, with the option to specify ascending or descending
- filtering: a list of criteria to restrict the results by


User: "What are the top 10 products by sales?"
System:
{{
"metrics":["sales"],
"dimensions": ["products"],
"limit": 10,
"order": ["sales desc"],
"filtering": []
}}

User: "What are the sales by line of business and state?"
System:
{{
"metrics":["average sales"],
"dimensions": ["line of business", "state],
"limit": -1,
"order": [],
"filtering": []
}}

User: "What is the average sales by state?"
System:
{{
"metrics":["average state sales"],
"dimensions": [],
"limit": -1,
"order": [],
"filtering": []
}}

User: "{input}"
System:

"""
