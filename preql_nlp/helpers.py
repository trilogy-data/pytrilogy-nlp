# imports
import random
import time
import re

import openai
from preql.constants import logger

extract = re.compile("Please try again in ([0-9]+)s")


# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                m = extract.search(str(e))

                # if we have a delay
                # wait for that
                # but continue to backoff if lots of retries
                # as there may be concurrent requests
                if extract.match(str(e)):
                    delay = int(m.group(1)) + 1
                    for x in range(num_retries):
                        delay *= exponential_base * (1 + jitter * random.random())
                else:
                    delay *= exponential_base * (1 + jitter * random.random())

                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                logger.info(f"Retrying on rate limit error for {delay}")
                # Increment the delay
                # delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper
