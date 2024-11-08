from typing import Optional, Union
from pydantic import BaseModel, field_validator
from trilogy.core.enums import ComparisonOperator, Ordering, BooleanOperator
from pydantic import BaseModel, Field, AliasChoices, ConfigDict

# from trilogy.core.constants import
from enum import Enum


class OrderResultV2(BaseModel):
    """The result of the order prompt"""
    model_config = ConfigDict(extra="forbid")
    column_name: str  = Field(validation_alias=AliasChoices('column_name', 'column', 'name'))
    order: Ordering


class Literal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str
    type: str


class Calculation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arguments: list[Union["Column", Literal]]
    operator: str
    over: list["Column"] | None = None


class Column(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    calculation: Optional[Calculation] = None

    def __hash__(self):
        return hash(self.name + str(self.calculation))


Calculation.model_rebuild()


class NLPComparisonOperator(Enum):
    BETWEEN = "between"

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, list) and " " in str(value):
            value = str(value).split()
        if isinstance(value, list):
            processed = [str(v).lower() for v in value]
            if processed == ["not", "in"]:
                return ComparisonOperator.NOT_IN
            if processed == ["is", "not"]:
                return ComparisonOperator.IS_NOT
            if value == ["in"]:
                return ComparisonOperator.IN
        return super()._missing_(str(value).lower())


# hacky, but merge two enums
NLPComparisonOperator._member_map_.update(ComparisonOperator._member_map_)


class NLPConditions(BaseModel):
    left: Column | Literal
    right: Column | Literal
    operator: ComparisonOperator


class NLPComparisonGroup(BaseModel):
    values: list[Union[NLPConditions, "NLPComparisonGroup"]]
    boolean: BooleanOperator


class FilterResultV2(BaseModel):
    """The result of the filter prompt"""

    root: NLPComparisonGroup


NLPComparisonGroup.model_rebuild()


class InitialParseResponseV2(BaseModel):
    """The result of the initial parse"""

    columns: list[Column]
    limit: Optional[int] = 100
    order: Optional[list[OrderResultV2]] = None
    filtering: Optional[FilterResultV2] = None

    @field_validator("filtering", mode="plain")
    @classmethod
    def filtering_validation(cls, v):
        if isinstance(v, dict):
            return FilterResultV2.model_validate(v)
        return FilterResultV2.model_validate(v)
