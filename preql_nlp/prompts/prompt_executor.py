# Brining some "prompt generator" classes - note that you can derive and extend those
from promptimize.prompt_cases import BasePromptCase, TemplatedPromptCase

# Bringing some useful eval function that help evaluating and scoring responses
# eval functions have a handle on the prompt object and are expected
# to return a score between 0 and 1
from promptimize.utils import extract_json_objects
from preql_nlp.constants import logger
from preql_nlp.prompts.query_semantic_extraction import EXTRACTION_PROMPT_V1
from preql_nlp.prompts.semantic_to_tokens import STRUCTURED_PROMPT_V1
from preql_nlp.prompts.final_selection import SELECTION_TEMPLATE_V1

from typing import List, Optional, Callable, Union


class SemanticExtractionPromptCase(TemplatedPromptCase):

    template = EXTRACTION_PROMPT_V1

    def __init__(self, question: str, evaluators: Optional[Union[Callable, List[Callable]]] = None):
        self.question = question
        super().__init__(category="semantic_extraction", evaluators=evaluators)

    def get_extra_template_context(self):
        return {"question": self.question}


class SemanticToTokensPromptCase(TemplatedPromptCase):

    template = STRUCTURED_PROMPT_V1

    def __init__(self, tokens: List[str], phrases: List[str], evaluators: Optional[Union[Callable, List[Callable]]] = None):
        self.tokens = tokens
        self.phrases = phrases
        super().__init__(category="semantic_to_tokens", evaluators=evaluators)

    
    def get_extra_template_context(self):
        return {"tokens": self.tokens, "phrase_str": ",".join(self.phrases)}


class SelectionPromptCase(TemplatedPromptCase):

    template = SELECTION_TEMPLATE_V1

    def __init__(self, question: str, concepts: List[str], evaluators: Optional[Union[Callable, List[Callable]]] = None):
        self.question = question
        self.concepts = concepts
        super().__init__(evaluators, category="selection")
        self.execution.score = None

    def get_extra_template_context(self):
        return {"concept_string": ", ".join(self.concepts), "question": self.question}




def run_prompt(prompt: BasePromptCase, debug: bool = False) -> list[dict | list]:
    # we probably don't need to use promptimize here
    # but consistency makes it easy to optimize with the test cases
    if debug:
        logger.debug("prompt")
        logger.debug(prompt.__dict__)
        logger.debug(prompt.render())

    raw_output = prompt._run(dry_run=False)
    if debug:
        logger.debug("output")
        logger.debug(raw_output)

    base = extract_json_objects(raw_output)
    if debug:
        logger.debug(base)
    return base
