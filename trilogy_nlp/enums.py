from enum import Enum


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
    OPEN_DATABASE = "open_database"

    ## validation
    VALIDATION_ERROR = "validation_error"
