# Brining some "prompt generator" classes - note that you can derive and extend those
from promptimize.prompt_cases import PromptCase

# Bringing some useful eval function that help evaluating and scoring responses
# eval functions have a handle on the prompt object and are expected
# to return a score between 0 and 1
from promptimize.utils import extract_json_objects
from preql_nlp.constants import logger


def run_prompt(prompt: str, debug: bool = False) -> list[dict | list]:
    # we probably don't need to use promptimize here
    # but consistency makes it easy to optimize with the test cases
    if debug:
        logger.debug("prompt")
        logger.debug(prompt)
    deep_prompt = PromptCase(user_input=prompt, evaluators=None)

    raw_output = deep_prompt._run(dry_run=False)
    if debug:
        logger.debug("output")
        logger.debug(raw_output)

    base = extract_json_objects(raw_output)
    if debug:
        logger.debug(base)
    return base
