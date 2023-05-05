## PreQL NLP

Natural language interface for querying PreQL models. 

PreQL is easier for a large language model to interact with as it requires only extract relevant concepts from a text,
classifying them as metrics or dimensions, and mapping them to what is available in a model.

This makes it more testable and less prone to hallucination than generating SQL directly. 

Requires setting the following environment variables
- OPENAI_API_KEY
- OPENAI_MODEL

Recommended to use "gpt-3.5-turbo" or higher as the model.




## Examples



## Setting Up Your Environment

Recommend that you work in a virtual environment with requirements from both requirements.txt and requirements-test.txt installed. The latter is necessary to run
tests (surprise). 

Pypreql is python 3.10+
