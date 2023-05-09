from pydantic import BaseModel

''''- metrics: a list of concepts from the question that should be aggregated
- dimensions: a list of concepts from the question which are not metrics
- limit: a number of records to limit the results to, -1 if none specified
- order: a list of fields to order the results by, with the option to specify ascending or descending
- filtering: a list of criteria to restrict the results by'''

class TokenInputs(BaseModel):
    """The inputs to the tokenization prompt"""
    metrics:list[str]
    dimensions:list[str]

class InitialParseResult(BaseModel):
    """The result of the initial parse"""
    metrics:list[str]
    dimensions:list[str]
    limit:int
    order:list[str]
    filtering:list[str]