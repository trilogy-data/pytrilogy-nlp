# Brining some "prompt generator" classes - note that you can derive and extend those
from promptimize.prompt_cases import TemplatedPromptCase

# Bringing some useful eval function that help evaluating and scoring responses
# eval functions have a handle on the prompt object and are expected
# to return a score between 0 and 1
from trilogy_nlp.constants import logger
from trilogy_nlp.models import (
    InitialParseResponse,
    ConceptSelectionResponse,
    SemanticTokenResponse,
    FilterRefinementResponse,
    FinalParseResponse,
)
from trilogy.core.models import DataType, ListType, MapType, StructType
from trilogy_nlp.cache_providers.base import BaseCache
from trilogy_nlp.cache_providers.local_sqlite import SqlliteCache
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Callable, Union, Type, overload
import uuid
import json
import os
from jinja2 import FileSystemLoader, Environment, Template
from os.path import dirname
from langchain_core.messages import HumanMessage
from trilogy_nlp.helpers import retry_with_exponential_backoff
from langchain_core.language_models import BaseLanguageModel
import sys
from pathlib import Path

PROMPT_STOPWORD = "<EOM>"

MAX_REFINEMENT_TRIES = 4


def resource_path() -> str:
    if not getattr(sys, "frozen", False):
        return dirname(__file__)
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    application_path = Path(sys._MEIPASS)  # type: ignore
    search_path = application_path / "trilogy_nlp" / "prompts"

    return str(Path(search_path))


loader = FileSystemLoader(searchpath=resource_path())
templates = Environment(loader=loader)


def gen_hash(obj, keys: set[str]) -> str:
    """Generate a deterministic hash for an object across multiple runs"""
    import hashlib

    m = hashlib.sha256(usedforsecurity=False)

    # we need to hash things in a deterministic order
    key_list = sorted(list(keys), key=lambda x: x)
    for key in key_list:
        x = getattr(obj, key)
        if isinstance(x, Template):
            s = x.render()
        else:
            s = str(x)

        m.update(s.encode("utf-8"))

    return m.digest().hex()


class BasePreqlPromptCase(TemplatedPromptCase):
    parse_model: Type[BaseModel]
    template: Template
    stopword = PROMPT_STOPWORD

    def __init__(
        self,
        category: str,
        llm: BaseLanguageModel,
        fail_on_parse_error: bool = True,
        evaluators: Optional[Union[Callable, List[Callable]]] = None,
    ):
        # this isn't actually the right way to pass through a stopword to the complete prompt
        # so we're splitting in the response
        # TODO: figure out how to do this when we figure out how to group prompts in one API call
        self.llm = llm
        super().__init__(
            category=category,
            evaluators=evaluators,
            prompt_executor_kwargs={"stopword": PROMPT_STOPWORD},
        )
        self._prompt_hash = str(uuid.uuid4())
        self.parsed = None
        self.fail_on_parse_error = fail_on_parse_error
        self.retry_prompt = "User: That response was incorrectly formatted. Please return the same content as the first response with formatting fixed (eg. valid JSON) to conform with original request. System:\n"
        self.stash: BaseCache = SqlliteCache()
        self.has_rerun = False

    @classmethod
    def parse_response(cls, response: str):
        try:
            return cls.parse_model.model_validate_json(response.split(cls.stopword)[0])
        except ValidationError as e:
            raise e

    def get_prompt_executor(self):
        if self.llm:
            return self.llm
        from langchain_openai import ChatOpenAI

        model_name = os.environ.get("OPENAI_MODEL") or "gpt-3.5-turbo"
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        return ChatOpenAI(model=model_name, openai_api_key=openai_api_key)

    @retry_with_exponential_backoff
    def execute_prompt(self, prompt_str: str, skip_cache: bool = False):
        # if we already have a local result
        # skip hitting remote
        # TODO: make the cache provider pluggable and injected
        hash_val = gen_hash(self, self.attributes_used_for_hash)
        if not skip_cache:
            resp = self.stash.retrieve(hash_val)
            if resp:
                self.response = resp
                return self.response
        self.response = self.prompt_executor(
            [
                HumanMessage(content=prompt_str),
            ]
        ).content

        return self.response

    def get_extra_template_context(self):
        return {"stopword": self.stopword}

    def rerun(self):
        self.prompt = self.prompt + self.response + "\n" + self.retry_prompt
        self.has_rerun = True
        self.execute_prompt(self.prompt, skip_cache=True)

    def post_run(self):
        try:
            self.parsed = self.parse_response(self.response)
            logger.info(
                f"Sucessfully parsed response to {self.category}: {self.parsed.model_dump_json()}"
            )
            hash_val = gen_hash(self, self.attributes_used_for_hash)
            self.stash.store(hash_val, self.category, self.response)
        except ValidationError as e:
            if not self.has_rerun:
                self.rerun()
                return self.post_run()
            print(self.render())
            print("was unable to parse response using ", str(self.parse_model))
            print(self.response)
            if self.fail_on_parse_error:
                raise e

    def render(
        self,
    ):
        return self.template.render(**self.jinja_context)


class SemanticExtractionPromptCase(BasePreqlPromptCase):
    template = templates.get_template("prompt_query_semantic_groupings.jinja2")
    parse_model = InitialParseResponse

    attributes_used_for_hash = {
        "category",
        "question",
        "template",
    }

    def __init__(
        self,
        question: str,
        llm: BaseLanguageModel,
        evaluators: Optional[Union[Callable, List[Callable]]] = None,
    ):
        self.question = question
        super().__init__(category="semantic_extraction", evaluators=evaluators, llm=llm)

    def get_extra_template_context(self):
        return {**super().get_extra_template_context(), "question": self.question}


class SemanticToTokensPromptCase(BasePreqlPromptCase):
    template = templates.get_template("prompt_semantic_to_tokens.jinja2")
    parse_model = SemanticTokenResponse

    attributes_used_for_hash = {
        "tokens",
        "phrases",
        "category",
        "template",
        "purpose",
    }

    def __init__(
        self,
        purpose: str,
        tokens: List[str],
        phrases: List[str],
        llm: BaseLanguageModel,
        evaluators: Optional[Union[Callable, List[Callable]]] = None,
    ):
        tokens = [token for token in tokens if token]
        self.tokens = sorted(tokens)
        self.purpose = purpose
        # we need to ensure that we always have a list
        # for some LLMs to understand the prompt properly
        # pad out the inputs with a random phrase
        self.padding_phrase = "democratic elections"
        self.phrases = sorted(phrases + [self.padding_phrase])
        super().__init__(category="semantic_to_tokens", llm=llm, evaluators=evaluators)
        self.retries = 0

    def get_extra_template_context(self):
        return {
            **super().get_extra_template_context(),
            "purpose": self.purpose,
            "tokens": ", ".join([f'"{c}"' for c in self.tokens if c]),
            "phrase_str": ", ".join([f'"{c}"' for c in self.phrases]),
        }

    def _process_tokens(self, input_tokens):
        input_tokens.tokens = [t for t in input_tokens.tokens if t in self.tokens]
        return input_tokens

    def post_run(self):
        super().post_run()
        self.parsed.root = [
            self._process_tokens(x)
            for x in self.parsed.root
            if x.phrase != self.padding_phrase
        ]
        # TODO: evaluate tradeoffs
        # do we ask to refine, or just silently drop invalid tokens?
        # tokens = []
        # for x in self.parsed.__root__:
        #     tokens+= x.tokens
        # missing = False
        # tokens = list(set(tokens))
        # for token in tokens:
        #     if token not in self.tokens:
        #         missing = True
        #         break
        # if missing:
        #     self.retries += 1
        #     valid = ", ".join([f'"{c}"' for c in self.tokens if c]),
        #     retry_prompt = f'User: The token "{token}" in your answer was not in the provided list, please return a new answer without any unprovided lists. Valid tokens are [{valid}] Return only the corrected JSON, do not apologize. \nSystem: '
        #     self.prompt = self.prompt + "\n" + self.response + "\n" + retry_prompt
        #     if self.retries < MAX_REFINEMENT_TRIES:
        #         self.execute_prompt(self.prompt, skip_cache=True)
        #         return self.post_run()
        #     else:
        #         print(self.prompt)
        #         raise ValueError(
        #             f"After {self.retries} attempts, LLM returned token {token} that does not exist in input tokens, cannot progress - returned {self.parsed}"
        #         )


class SelectionPromptCase(BasePreqlPromptCase):
    template = templates.get_template("prompt_final_concepts_v2.jinja2")
    parse_model = FinalParseResponse

    attributes_used_for_hash = {
        "question",
        "concept_names",
        "category",
        "template",
    }

    def __init__(
        self,
        question: str,
        concept_names: List[str],
        llm: BaseLanguageModel,
        all_concept_names: List[str] | None = None,
        evaluators: Optional[Union[Callable, List[Callable]]] = None,
    ):
        self.question = question
        self.all_concept_names_internal = all_concept_names or concept_names
        self.concept_names = sorted(list(set(concept_names)), key=lambda x: x)
        super().__init__(
            evaluators=evaluators, llm=llm, category="final_concept_selection"
        )
        self.execution.score = None
        self.retries = 0

    def get_extra_template_context(self):
        return {
            **super().get_extra_template_context(),
            "concept_string": ", ".join([f'"{c}"' for c in self.concept_names]),
            "question": self.question,
        }

    def post_run(self):
        super().post_run()

        selected_concepts: List[str] = self.parsed.selection
        for item in self.parsed.filtering:
            selected_concepts.append(item.concept)
        for item in self.parsed.order:
            selected_concepts.append(item.concept)

        missing = False
        selected_concepts = list(set(selected_concepts))
        for selection in selected_concepts:
            if selection not in self.all_concept_names_internal:
                missing = True
                break
        if missing:
            self.retries += 1
            valid = ", ".join([f'"{c}"' for c in self.concept_names])
            retry_prompt = f'User: The concept "{selection}" in your answer was not one in the provided valid list, please return a new answer with only valid concepts from this list: [{valid}] Return only the corrected JSON, do not apologize. \nSystem: '
            self.prompt = self.prompt + "\n" + self.response + "\n" + retry_prompt
            if self.retries < MAX_REFINEMENT_TRIES:
                self.execute_prompt(self.prompt, skip_cache=True)
                return self.post_run()
            else:
                raise ValueError(
                    f"LLM returned concept {selection} that does not exist in input concepts, cannot progress - returned {self.parsed}"
                )


class FilterRefinementCase(BasePreqlPromptCase):
    template = templates.get_template("prompt_refine_filter.jinja2")
    parse_model = FilterRefinementResponse

    attributes_used_for_hash = {
        "values",
        "description",
        "template",
    }

    def __init__(
        self,
        values: list[str | int | float | bool],
        description: str,
        datatype: DataType | StructType | ListType | MapType,
        llm: BaseLanguageModel,
        evaluators: Optional[Union[Callable, List[Callable]]] = None,
    ):
        self.values = values
        self.description = description
        self.datatype = datatype
        super().__init__(evaluators=evaluators, category="filter_refinement", llm=llm)

    def get_extra_template_context(self):
        return {
            **super().get_extra_template_context(),
            "values": ", ".join([f'"{x}"' for x in self.values]),
            "description": self.description,
            "datatype": self.datatype.value,
        }


class FilterRefinementErrorCase(BasePreqlPromptCase):
    template = templates.get_template("prompt_refine_filter_error.jinja2")
    parse_model = FilterRefinementResponse

    attributes_used_for_hash = {
        "values",
        "error",
        "template",
    }

    def __init__(
        self,
        values: list[str | int | float | bool],
        error: str,
        datatype: DataType | StructType | ListType | MapType,
        llm: BaseLanguageModel,
        evaluators: Optional[Union[Callable, List[Callable]]] = None,
    ):
        self.values = values
        self.error = error
        self.datatype = datatype
        super().__init__(
            evaluators=evaluators, category="filter_refinement_error", llm=llm
        )

    def get_extra_template_context(self):
        return {
            **super().get_extra_template_context(),
            "values": ", ".join([f'"{x}"' for x in self.values]),
            "error": self.error,
            "datatype": self.datatype.value,
        }


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "log_data"
)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def log_prompt_info(prompt: TemplatedPromptCase, session_uuid: uuid.UUID):
    prompt_hash = prompt.prompt_hash
    prompt_context = prompt.jinja_context
    template = prompt.template.render(**prompt.jinja_context)
    category = prompt.category

    data = {
        "prompt_token": prompt_hash,
        "prompt_context": prompt_context,
        "template": template,
        "category": category,
        "session_uuid": str(session_uuid),
        "response": prompt.response,
    }
    path = os.path.join(
        DATA_DIR, str(session_uuid), category + "-" + prompt_hash + ".json"
    )
    with open(
        path,
        "w",
    ) as f:
        logger.debug("printing to...{}".format(path))
        json.dump(data, f)


@overload
def run_prompt(  # type: ignore
    prompt: SelectionPromptCase,
    debug: bool = False,
    log_info: bool = True,
    session_uuid: uuid.UUID | None = None,
) -> ConceptSelectionResponse:
    ...


@overload
def run_prompt(  # type: ignore
    prompt: SemanticExtractionPromptCase,
    debug: bool = False,
    log_info: bool = True,
    session_uuid: uuid.UUID | None = None,
) -> InitialParseResponse:
    ...


@overload
def run_prompt(  # type: ignore
    prompt: SemanticToTokensPromptCase,
    debug: bool = False,
    log_info: bool = True,
    session_uuid: uuid.UUID | None = None,
) -> SemanticTokenResponse:
    ...


@overload
def run_prompt(  # type: ignore
    prompt: FilterRefinementCase,
    debug: bool = False,
    log_info: bool = True,
    session_uuid: uuid.UUID | None = None,
) -> FilterRefinementResponse:
    ...


def run_prompt(
    prompt: BasePreqlPromptCase,
    debug: bool = False,
    log_info: bool = True,
    session_uuid: uuid.UUID | None = None,
) -> (
    ConceptSelectionResponse
    | SemanticTokenResponse
    | InitialParseResponse
    | FilterRefinementResponse
):
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
