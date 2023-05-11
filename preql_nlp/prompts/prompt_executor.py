# Brining some "prompt generator" classes - note that you can derive and extend those
from promptimize.prompt_cases import TemplatedPromptCase

# Bringing some useful eval function that help evaluating and scoring responses
# eval functions have a handle on the prompt object and are expected
# to return a score between 0 and 1
from promptimize.utils import extract_json_objects
from preql_nlp.constants import logger
from preql_nlp.prompts.query_semantic_extraction import EXTRACTION_PROMPT_V1
from preql_nlp.prompts.semantic_to_tokens import STRUCTURED_PROMPT_V1
from preql_nlp.prompts.final_selection import SELECTION_TEMPLATE_V1
from preql_nlp.models import InitialParseResult, ConceptSelection, SemanticTokenResponse
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Callable, Union, Type
import uuid
import json
import os
import sqlite3

SQLITE_ADDRESS = "local_prompt_cache.db"

def get_result_if_cached(prompt_hash:str)->str | None:
    print('checking for cache with prompt hash ', prompt_hash)
    con = sqlite3.connect(SQLITE_ADDRESS)
    cur= con.cursor()
    cur.execute('create table if not exists prompt_cache (cache_id string, prompt_type string, response string)')
    res = cur.execute('select response, prompt_type from prompt_cache where cache_id = ?', (prompt_hash,))
    current = res.fetchone()
    if current:
        print('got cached response of type ', current[1])
        return current[0]
    return None
    

def stash_result(prompt_hash:str, category:str, result:str):
    con = sqlite3.connect(SQLITE_ADDRESS)
    cur = con.cursor()
    cur.execute('create table if not exists prompt_cache (cache_id string, prompt_type string, response string)')
    cur.execute('insert into prompt_cache select ?, ?,  ?', (prompt_hash, category, result))
    con.commit()

class BasePreqlPromptCase(TemplatedPromptCase):
    parse_model:Type[BaseModel]

    def __init__(
        self,
        category: str,
        fail_on_parse_error:bool = True,
        evaluators: Optional[Union[Callable, List[Callable]]] = None,

    ):
        super().__init__(category=category, evaluators=evaluators)
        self._prompt_hash = str(uuid.uuid4())
        self.parsed = None
        self.fail_on_parse_error = fail_on_parse_error


    def execute_prompt(self, prompt_str):
        # if we already have a local result
        # skip hitting remote
        resp = get_result_if_cached(hash(self))
        if resp:
            print('GOT CACHED RESPONSE \o/')
            self.response = resp
            return self.response
        self.response = self.prompt_executor(prompt_str)
        stash_result(hash(self), self.category, self.response)
        return self.response
    

    def get_extra_template_context(self):
        raise NotImplementedError("This class can't be used directly.")

    def post_run(self):
        try:
            self.parsed = self.parse_model.parse_raw(self.response)
        except ValidationError as e:
            print('was unable to parse response using ', str(self.parse_model))
            print(self.response)
            if self.fail_on_parse_error:
                raise e
            
class SemanticExtractionPromptCase(BasePreqlPromptCase):
    template = EXTRACTION_PROMPT_V1
    parse_model = InitialParseResult

    attributes_used_for_hash = BasePreqlPromptCase.attributes_used_for_hash | {
        "category",
        "question",
    }
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
    
    attributes_used_for_hash = {
        "tokens",
        "phrases",
        "category"
    }
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
        return {"tokens": self.tokens, "phrase_str": ",".join(self.phrases)}


class SelectionPromptCase(BasePreqlPromptCase):
    template = SELECTION_TEMPLATE_V1
    parse_model = ConceptSelection

    attributes_used_for_hash = BasePreqlPromptCase.attributes_used_for_hash | {
        "question",
        "concepts",
        "category"
    }

    def __init__(
        self,
        question: str,
        concepts: List[str],
        evaluators: Optional[Union[Callable, List[Callable]]] = None,
    ):
        self.question = question
        self.concepts = concepts
        super().__init__(evaluators=evaluators, category="selection")
        self.execution.score = None

    def get_extra_template_context(self):
        return {"concept_string": ", ".join(self.concepts), "question": self.question}


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


def run_prompt(
    prompt: TemplatedPromptCase,
    debug: bool = False,
    log_info=True,
    session_uuid: uuid.UUID | None = None,
) -> list[dict | list]:
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

    base = extract_json_objects(raw_output)
    if debug:
        logger.debug(base)
    return base
