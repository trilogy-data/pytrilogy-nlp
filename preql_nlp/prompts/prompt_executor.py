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
from preql_nlp.prompts.check_answer import CHECK_ANSWER_PROMPT_V1
from langchain.llms import OpenAI

from typing import List, Optional, Callable, Union
import uuid
import json
import os


class BasePreqlPromptCase(TemplatedPromptCase):
    def __init__(
        self,
        category: str,
        evaluators: Optional[Union[Callable, List[Callable]]] = None,
        model: Optional[str] = None
    ):
        self.model = model or "gpt-3.5-turbo"
        super().__init__(category=category, evaluators=evaluators)
        self._prompt_hash = str(uuid.uuid4())

    def get_extra_template_context(self):
        raise NotImplementedError("This class can't be used directly.")
    
    def get_prompt_executor(self):
        model_name = self.model
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.prompt_executor_kwargs = {"model_name": model_name}
        return OpenAI(model_name=model_name, openai_api_key=openai_api_key)


class SemanticExtractionPromptCase(BasePreqlPromptCase):
    template = EXTRACTION_PROMPT_V1

    def __init__(
        self,
        question: str,
        evaluators: Optional[Union[Callable, List[Callable]]] = None,
        model: Optional[str] = None
    ):
        self.question = question
        super().__init__(category="semantic_extraction", evaluators=evaluators, model=model)

    def get_extra_template_context(self):
        return {"question": self.question}


class SemanticToTokensPromptCase(BasePreqlPromptCase):
    template = STRUCTURED_PROMPT_V1

    def __init__(
        self,
        tokens: List[str],
        phrases: List[str],
        evaluators: Optional[Union[Callable, List[Callable]]] = None,
        model: Optional[str] = None
    ):
        self.tokens = tokens
        self.phrases = phrases
        super().__init__(category="semantic_to_tokens", evaluators=evaluators, model=model)

    def get_extra_template_context(self):
        return {"tokens": self.tokens, "phrase_str": ",".join(self.phrases)}


class SelectionPromptCase(BasePreqlPromptCase):
    template = SELECTION_TEMPLATE_V1

    def __init__(
        self,
        question: str,
        concepts: List[str],
        evaluators: Optional[Union[Callable, List[Callable]]] = None,
        model: Optional[str] = None
    ):
        self.question = question
        self.concepts = concepts
        super().__init__(evaluators=evaluators, category="selection", model=model)

    def get_extra_template_context(self):
        return {"concept_string": ", ".join(self.concepts), "question": self.question}


class CheckAnswerPromptCase(BasePreqlPromptCase):
    template = CHECK_ANSWER_PROMPT_V1

    def __init__(self, question: str, columns: List[str], answer: List[tuple], evaluators: Optional[Union[Callable, List[Callable]]] = None,
        model: Optional[str] = None):
        self.question = question
        self.columns = columns
        self.answer = answer
        super().__init__(evaluators=evaluators, category="check", model=model)

    def get_extra_template_context(self):
        return {"results": self.answer, "columns": self.columns, "question": self.question}
        
    

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "log_data"
)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def log_prompt_info(prompt: BasePreqlPromptCase, session_uuid: uuid.UUID):
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
        "model": prompt.model
    }
    with open(
        os.path.join(DATA_DIR, str(session_uuid), prompt_hash + ".json"), "w"
    ) as f:
        json.dump(data, f)


def run_prompt(
    prompt: BasePreqlPromptCase,
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
