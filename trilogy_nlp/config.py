from dataclasses import dataclass


@dataclass
class Config:
    LLM_VALIDATION_ATTEMPTS: int = 7


DEFAULT_CONFIG = Config()
