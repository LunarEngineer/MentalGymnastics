"""Contains utilities used to validate data structures."""

def is_action(item: Any) -> bool:
    # 1) Check for type
    if not isinstance(item, Dict):
        return False
    # 2) Check that required keys exist.
    required_keys = ['function_id']
    if not set(required_keys).issubset(item):
        return False