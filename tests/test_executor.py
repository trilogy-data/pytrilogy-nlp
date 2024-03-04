from preql_nlp.prompts.prompt_executor import SemanticToTokensPromptCase
from pydantic import ValidationError
import pytest


def test_retry():
    test = SemanticToTokensPromptCase(
        tokens=["alphabet", "soup", "apple", "fruit", "shape", "orange"],
        phrases=["round fruits"],
        purpose="Dimension",
    )

    test.execute_prompt = (
        lambda x, skip_cache=False: test.prompt + "\nFORCED_SYNTAX_ERROR"
    )

    with pytest.raises(ValidationError):
        test._run(dry_run=False)

    assert test.retry_prompt in test.prompt
