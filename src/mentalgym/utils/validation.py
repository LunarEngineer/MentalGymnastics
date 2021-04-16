"""Contains utilities used to validate data structures."""
from typing import Any

def is_function(item: Any) -> bool:
    # 1) Check for type
    if not isinstance(item, Dict):
        return False
    # 2) Check that required keys exist.
    required_keys = ['id', 'type', 'input', 'location']
    if not set(required_keys).issubset(item):
        return False

