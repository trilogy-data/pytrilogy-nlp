# patched method while waiting for upstream PR to be merged
import logging
from typing import (
    Any,
    Callable,
    Union,
)
import langchain_community.llms.openai as langchain_openai

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def patch_langchain():
    def create_retry_decorator(
        llm: Union[langchain_openai.BaseOpenAI, langchain_openai.OpenAIChat]
    ) -> Callable[[Any], Any]:
        import openai

        min_seconds = 4
        max_seconds = 10
        # Wait 2^x * 1 second between each retry starting with
        # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
        return retry(
            reraise=True,
            stop=stop_after_attempt(llm.max_retries),
            wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
            retry=(
                retry_if_exception_type(openai.APITimeoutError)
                | retry_if_exception_type(openai.APIError)
                | retry_if_exception_type(openai.APIConnectionError)
            ),
            before_sleep=before_sleep_log(langchain_openai.logger, logging.WARNING),
        )

    langchain_openai._create_retry_decorator = create_retry_decorator
