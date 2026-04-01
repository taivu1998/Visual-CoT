"""
Backward-compatible wrapper around explicit config loading utilities.
"""
import argparse
from typing import Any, Dict, Iterable, Optional

from src.config_loader import apply_overrides, load_config, parse_override_tokens


class ConfigParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="V-CoT Pipeline")
        self.parser.add_argument("--config", type=str, required=True, help="Path to YAML config")

    def load(
        self,
        config_path: Optional[str] = None,
        override_tokens: Optional[Iterable[str]] = None,
        allow_new_keys: bool = False,
    ) -> Dict[str, Any]:
        if config_path is None:
            args, unknown = self.parser.parse_known_args()
            config_path = args.config
            override_tokens = unknown if override_tokens is None else override_tokens

        return load_config(
            config_path=config_path,
            override_tokens=override_tokens or [],
            allow_new_keys=allow_new_keys,
        )

    @staticmethod
    def parse_overrides(tokens: Iterable[str]) -> Dict[str, Any]:
        return parse_override_tokens(tokens)

    @staticmethod
    def apply(config: Dict[str, Any], overrides: Dict[str, Any], allow_new_keys: bool = False) -> Dict[str, Any]:
        return apply_overrides(config, overrides, allow_new_keys=allow_new_keys)
