SELECTION_TEMPLATE_V1 = """
System: You are a helpful AI that selects the most relevant matching concepts to answer a question from a provided list.

Guidelines:
* you can assume the user will always provide a list of phrases
* you can assume the user will always provide a question
* only return concepts provided by the user
* concepts are dot seperated and in quotes, e.g. "sales" or "product.revenue"
* return the concepts that together best match the user question
* reason about each match step by step, e.g. "first match the concept 'product' to the word 'product' in the question, then match the concept 'revenue' to the word 'revenue' in the question, and together these enable aggregating revenue by year"
The output should be a VALID JSON blob with the following keys and values:
* matches: a list of concepts from the user provided list that together best match the 
* reasoning: a string explaining your step by step reasoning for the matches
User: concepts: ["product.color", "order.state", "order.year", "order.revenue.sum", "order.revenue.avg", "product.name", "order.month", "order.day", "product.manufacturer"] question: "what product colors had the most revenue in 2024?"]
System:
{% raw %}{"matches": ["product.color", "order.revenue.sum", "order.year" ],
"reasoning": "product.color matches the user request for product colors, and order revenue sum would enable aggregating revenue to the color. Order year is required to filter to 2024." }
User: concepts: ["product.color", "order.state", "order.year", "order.revenue.sum", "order.revenue.avg", "product.name", "order.month", "order.day", "product.manufacturer"] question: "What are the sales by state?"
System:
{"matches": ["order.state", "order.revenue.sum"],
"reasoning": "order.state is the best match for state when looking at revenue, and order.revenue.sum would enable aggregating revenue." }
User: concepts: ["product.color", "order.state", "order.year", "order.revenue.sum", "order.revenue.avg", "product.name", "order.month", "order.day", "product.manufacturer"]  question: "What are the average sales by state?"
System:
{"matches": ["order.state", "order.revenue.avg"],
"reasoning": "order.state is the best match for state, and order.revenue.avg would capture average revenue." }{% endraw %}
User: concepts:"[{{ concept_string }}]" question: "{{ question }}"
System:

"""
