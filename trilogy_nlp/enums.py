from enum import Enum, auto


class Provider(Enum):
    OPENAI = "openai"
    OCTO_AI = "octoai"
    LOCAL = "local"
    LLAMAFILE = "llamafile"
    GOOGLE = "google"


class CacheType(Enum):
    MEMORY = "memory"
    SQLLITE = "sqlite"


class EventType(Enum):
    OPEN_DATABASE = auto()

    ## environment
    ENVIRONMENT_VALIDATION_FAILED = auto()
    ENVIRONMENT_VALIDATION_PASSED = auto()
    ## validation
    INITIAL_VALIDATION_PARSING_FAILED = auto()

    STRING_FIELD_WITH_AGGREGATE = auto()

    OVER_CLAUSE_WITHOUT_AGGREGATE = auto()

    INVALID_FUNCTION = auto()

    INVALID_COLUMN_NAME_NO_CALCULATION = auto()

    CALCULATION_WITH_PREDEFINED_FIELD = auto()

    VALIDATION_ERROR = auto()

    ORDER_BY_NOT_SELECTED = auto()

    QUERY_VALIDATION_PASSED = auto()
