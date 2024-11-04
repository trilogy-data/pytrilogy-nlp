from trilogy.core.models import FunctionType


def is_valid(name: str):
    return name.lower() in [item.value for item in FunctionType]


def test_function_lookup():
    assert is_valid("SUM")
