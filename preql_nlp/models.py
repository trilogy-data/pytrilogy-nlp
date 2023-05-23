from pydantic import BaseModel
from preql.core.enums import ComparisonOperator, Ordering
from preql.core.models import Concept

### Intermediate Models

class FilterResult(BaseModel):
    """The result of the filter prompt"""
    concept:str
    values:list[str]
    operator:ComparisonOperator


class FinalFilterResult(BaseModel):
    concept:Concept
    values:list[str]
    operator:ComparisonOperator

class OrderResult(BaseModel):
    """The result of the order prompt"""
    concept:str
    order: Ordering

class FinalOrderResult(BaseModel):
    """The processed result of the order prompt"""
    concept:Concept
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
        return list(set(self.metrics + self.dimensions + filtering + order))


class FinalParseResponse(BaseModel):
    """The result of the initial parse"""
    selection:list[str]
    limit:int
    order:list[OrderResult]
    filtering:list[FilterResult]


class IntermediateParseResults(BaseModel):
    select: list[Concept]
    limit: int
    order: list[FinalOrderResult]
    filtering:list[FinalFilterResult]


### Parse Result Models

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


class FilterRefinementResponse(BaseModel):
    new_values:list[str]
    old_values:list[str]
    reasoning:str