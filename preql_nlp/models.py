from pydantic import BaseModel
from preql.core.enums import ComparisonOperator
from preql.core.enums import Ordering

''''- metrics: a list of concepts from the question that should be aggregated
- dimensions: a list of concepts from the question which are not metrics
- limit: a number of records to limit the results to, -1 if none specified
- order: a list of fields to order the results by, with the option to specify ascending or descending
- filtering: a list of criteria to restrict the results by'''

class TokenInputs(BaseModel):
    """The inputs to the tokenization prompt"""
    metrics:list[str]
    dimensions:list[str]
    order:list[str]
    filtering:list[str]

class FilterResult(BaseModel):
    """The result of the filter prompt"""
    concept:str
    values:list[str]
    operator:ComparisonOperator

class OrderResult(BaseModel):
    """The result of the order prompt"""
    concept:str
    order: Ordering

class InitialParseResponse(BaseModel):
    """The result of the initial parse"""
    metrics:list[str]
    dimensions:list[str]
    limit:int
    order:list[OrderResult]
    filtering:list[FilterResult]

    @property
    def selection(self)->list[str]:
        filtering = [f.concept for f in self.filtering]
        order = [x.concept for x in self.order]
        return self.metrics + self.dimensions + filtering + order


class SemanticTokenMatch(BaseModel):
    phrase: str
    tokens: list[str]

class SemanticTokenResponse(BaseModel):
    __root__:list[SemanticTokenMatch]

    def __iter__(self):
        return iter(self.__root__)

    def __getitem__(self, idx):
        return self.__root__.__getitem__(idx)

class ConceptSelectionResponse(BaseModel):
    matches:list[str]
    reasoning:str



class IntermediateParseResults(BaseModel):
    select: list[str]
    limit: int
    order: list[OrderResult]
    filtering:list[FilterResult]


class FilterRefinementResponse(BaseModel):
    value:str
    description:str