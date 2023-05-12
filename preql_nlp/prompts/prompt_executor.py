# Brining some "prompt generator" classes - note that you can derive and extend those
from promptimize.prompt_cases import TemplatedPromptCase

# Bringing some useful eval function that help evaluating and scoring responses
# eval functions have a handle on the prompt object and are expected
# to return a score between 0 and 1
from preql_nlp.constants import logger
from preql_nlp.prompts.query_semantic_extraction import EXTRACTION_PROMPT_V1
from preql_nlp.prompts.semantic_to_tokens import STRUCTURED_PROMPT_V1
from preql_nlp.prompts.final_selection import SELECTION_TEMPLATE_V1
from preql_nlp.models import (
    InitialParseResponse,
    ConceptSelectionResponse,
    SemanticTokenResponse,
)
from preql_nlp.cache_providers.base import BaseCache
from preql_nlp.cache_providers.local_sqlite import SqlliteCache
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Callable, Union, Type, overload
import uuid
import json
import os


def gen_hash(obj, keys: set[str]) -> str:
    """Generate a deterministic hash for an object across multiple runs"""
    import hashlib

    m = hashlib.sha256(usedforsecurity=False)

    # we need to hash things in a deterministic order
    key_list = sorted(list(keys), key=lambda x: x)
    for key in key_list:
        s = str(getattr(obj, key))

        m.update(s.encode("utf-8"))

    return m.digest().hex()


class BasePreqlPromptCase(TemplatedPromptCase):
    parse_model: Type[BaseModel]

    def __init__(
        self,
        category: str,
        fail_on_parse_error: bool = True,
        evaluators: Optional[Union[Callable, List[Callable]]] = None,
    ):
        super().__init__(category=category, evaluators=evaluators)
        self._prompt_hash = str(uuid.uuid4())
        self.parsed = None
        self.fail_on_parse_error = fail_on_parse_error
        self.stash: BaseCache = SqlliteCache()

    def execute_prompt(self, prompt_str):
        # if we already have a local result
        # skip hitting remote
        # TODO: make the cache provider pluggable and injected
        hash_val = gen_hash(self, self.attributes_used_for_hash)
        resp = self.stash.retrieve(hash_val)
        if resp:
            self.response = resp
            return self.response
        self.response = self.prompt_executor(prompt_str)
        self.stash.stash(hash_val, self.category, self.response)
        return self.response

    def get_extra_template_context(self):
        raise NotImplementedError("This class can't be used directly.")

    def post_run(self):
        try:
            self.parsed = self.parse_model.parse_raw(self.response)
        except ValidationError as e:
            print("was unable to parse response using ", str(self.parse_model))
            print(self.response)
            if self.fail_on_parse_error:
                raise e


class SemanticExtractionPromptCase(BasePreqlPromptCase):
    template = EXTRACTION_PROMPT_V1
    parse_model = InitialParseResponse

    attributes_used_for_hash = {"category", "question", "template"}

    def __init__(
        self,
        question: str,
        evaluators: Optional[Union[Callable, List[Callable]]] = None,
    ):
        self.question = question
        super().__init__(category="semantic_extraction", evaluators=evaluators)

    def get_extra_template_context(self):
        return {"question": self.question}


class SemanticToTokensPromptCase(BasePreqlPromptCase):
    template = STRUCTURED_PROMPT_V1
    parse_model = SemanticTokenResponse

    attributes_used_for_hash = {"tokens", "phrases", "category", "template"}

    def __init__(
        self,
        tokens: List[str],
        phrases: List[str],
        evaluators: Optional[Union[Callable, List[Callable]]] = None,
    ):
        self.tokens = tokens
        self.phrases = phrases
        super().__init__(category="semantic_to_tokens", evaluators=evaluators)

    def get_extra_template_context(self):
        return {
            "tokens": self.tokens,
            "phrase_str": ", ".join([f'"{c}"' for c in self.phrases]),
        }


class SelectionPromptCase(BasePreqlPromptCase):
    template = SELECTION_TEMPLATE_V1
    parse_model = ConceptSelectionResponse

    attributes_used_for_hash = {"question", "concepts", "category", "template"}

    def __init__(
        self,
        question: str,
        concepts: List[str],
        evaluators: Optional[Union[Callable, List[Callable]]] = None,
    ):
        self.question = question
        self.concepts = sorted(list(set(concepts)), key=lambda x: x)
        super().__init__(evaluators=evaluators, category="selection")
        self.execution.score = None

    def get_extra_template_context(self):
        return {
            "concept_string": ", ".join([f'"{c}"' for c in self.concepts]),
            "question": self.question,
        }


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "log_data"
)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def log_prompt_info(prompt: TemplatedPromptCase, session_uuid: uuid.UUID):
    prompt_hash = prompt.prompt_hash
    prompt_context = prompt.jinja_context
    template = prompt.template
    category = prompt.category

    data = {
        "prompt_token": prompt_hash,
        "prompt_context": prompt_context,
        "template": template,
        "category": category,
        "session_uuid": str(session_uuid),
        "response": prompt.response,
    }
    with open(
        os.path.join(DATA_DIR, str(session_uuid), prompt_hash + ".json"), "w"
    ) as f:
        print(
            "printing to...{}".format(
                os.path.join(DATA_DIR, str(session_uuid), prompt_hash + ".json")
            )
        )
        json.dump(data, f)


@overload
def run_prompt(
    prompt: SemanticExtractionPromptCase,
    debug: bool = False,
    log_info=True,
    session_uuid: uuid.UUID | None = None,
) -> InitialParseResponse:
    ...


@overload
def run_prompt(
    prompt: SemanticToTokensPromptCase,
    debug: bool = False,
    log_info=True,
    session_uuid: uuid.UUID | None = None,
) -> SemanticTokenResponse:
    ...


@overload
def run_prompt(
    prompt: SelectionPromptCase,
    debug: bool = False,
    log_info=True,
    session_uuid: uuid.UUID | None = None,
) -> ConceptSelectionResponse:
    ...


def run_prompt(
    prompt: BasePreqlPromptCase,
    debug: bool = False,
    log_info=True,
    session_uuid: uuid.UUID | None = None,
) -> ConceptSelectionResponse | SemanticTokenResponse | InitialParseResponse:
    if not session_uuid:
        session_uuid = uuid.uuid4()

    if log_info and not os.path.exists(os.path.join(DATA_DIR, str(session_uuid))):
        os.makedirs(os.path.join(DATA_DIR, str(session_uuid)))
    if debug:
        logger.debug("prompt")
        logger.debug(prompt.__dict__)
        logger.debug(prompt.render())

    raw_output = prompt._run(dry_run=False)
    if debug:
        logger.debug("output")
        logger.debug(raw_output)

    if log_info:
        log_prompt_info(prompt, session_uuid)
    if prompt.parsed:
        return prompt.parsed
    raise ValueError("Could not parse prompt response!")
