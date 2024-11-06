from typing import Optional, Union
from langchain.agents import create_structured_chat_agent, AgentExecutor
from trilogy_nlp.main import safe_limit
from trilogy.core.models import (
    Concept,
    Environment,
    ProcessedQuery,
    SelectStatement,
    Comparison,
    Conditional,
    OrderBy,
    OrderItem,
    WhereClause,
    AggregateWrapper,
    Function,
    Metadata,
    SelectItem,
    HavingClause,
)
from typing import List
from trilogy.core.query_processor import process_query
from langchain.tools import Tool, StructuredTool
import json
from pydantic import BaseModel, field_validator, ValidationError
from trilogy.core.enums import ComparisonOperator, Ordering, Purpose, BooleanOperator
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from trilogy_nlp.tools import get_wiki_tool, get_today_date
from trilogy.parsing.common import arbitrary_to_concept
from trilogy.core.models import DataType
from langchain_core.tools import ToolException
from trilogy.core.exceptions import UndefinedConceptException
from trilogy.core.processing.utility import (
    is_scalar_condition,
    decompose_condition,
    sort_select_output,
)

from trilogy_nlp.llm_interface.parsing import (
    parse_filtering,
    parse_order,
    parse_col,
    create_column,
)
from trilogy_nlp.llm_interface.models import (
    InitialParseResponseV2,
    Column,
    NLPComparisonOperator,
    NLPConditions,
    NLPComparisonGroup,
    FilterResultV2,
    OrderResultV2,
    Literal,
)
from trilogy_nlp.llm_interface.tools import sql_agent_tools
from trilogy_nlp.llm_interface.examples import FILTERING_EXAMPLE
from trilogy_nlp.llm_interface.constants import (
    COMPLICATED_FUNCTIONS,
    MAGIC_GENAI_DESCRIPTION,
)

# from trilogy.core.constants import
from trilogy.core.enums import (
    FunctionType,
    FunctionClass,
    InfiniteFunctionArgs,
)
from trilogy.parsing.common import arg_to_datatype
from enum import Enum
from trilogy_nlp.llm_interface.examples import COLUMN_DESCRIPTION
from trilogy_nlp.llm_interface.validation import (
    validate_response,
    ValidateResponseInterface,
)


class OrderResultV2(BaseModel):
    """The result of the order prompt"""

    column_name: str
    order: Ordering


class Literal(BaseModel):
    value: str
    type: str


class Calculation(BaseModel):
    arguments: list[Union["Column", Literal]]
    operator: str
    over: list["Column"] | None = None


class Column(BaseModel):
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
