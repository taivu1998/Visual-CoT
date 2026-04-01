"""
Explicit configuration loading and CLI override handling.
"""
from copy import deepcopy
from typing import Any, Dict, Iterable, List

import yaml


def _infer_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def parse_override_tokens(tokens: Iterable[str]) -> Dict[str, Any]:
    """
    Parse dot-notation CLI override tokens.

    Expected format:
        --section.key value --other.key value
    """
    token_list = list(tokens)
    if len(token_list) % 2 != 0:
        raise ValueError(
            "Override tokens must come in '--section.key value' pairs. "
            "Received an odd number of tokens."
        )

    overrides: Dict[str, Any] = {}
    for index in range(0, len(token_list), 2):
        key = token_list[index]
        value = token_list[index + 1]

        if not key.startswith("--"):
            raise ValueError(
                "Override keys must start with '--'. "
                f"Received '{key}'."
            )

        normalized_key = key[2:]
        if not normalized_key:
            raise ValueError("Override key cannot be empty.")

        overrides[normalized_key] = _infer_scalar(value)

    return overrides


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any], allow_new_keys: bool = False) -> Dict[str, Any]:
    """
    Apply dot-notation overrides to a config dictionary.
    """
    updated = deepcopy(config)

    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        cursor: Dict[str, Any] = updated

        for part in parts[:-1]:
            if part not in cursor:
                if not allow_new_keys:
                    raise KeyError(
                        f"Unknown override path '{dotted_key}'. "
                        f"Missing intermediate key '{part}'."
                    )
                cursor[part] = {}

            next_value = cursor[part]
            if not isinstance(next_value, dict):
                raise KeyError(
                    f"Cannot override '{dotted_key}' because '{part}' is not a mapping."
                )
            cursor = next_value

        leaf_key = parts[-1]
        if leaf_key not in cursor and not allow_new_keys:
            raise KeyError(
                f"Unknown override key '{dotted_key}'. "
                f"Leaf key '{leaf_key}' does not exist."
            )
        cursor[leaf_key] = value

    return updated


def load_config(config_path: str, override_tokens: Iterable[str] = None, allow_new_keys: bool = False) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if override_tokens:
        overrides = parse_override_tokens(override_tokens)
        config = apply_overrides(config, overrides, allow_new_keys=allow_new_keys)

    return config


def dump_config(config: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, default_flow_style=False, sort_keys=False)
