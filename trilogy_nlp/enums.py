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
