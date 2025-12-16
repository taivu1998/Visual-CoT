"""
Smart Configuration Parser.
Handles loading YAML files and allowing dot-notation overrides from CLI.
Example: python train.py --config c.yaml --training.learning_rate 0.001
"""
import argparse
import yaml
import sys
from typing import Any, Dict

class ConfigParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="V-CoT Pipeline")
        self.parser.add_argument("--config", type=str, required=True, help="Path to YAML config")

    def load(self) -> Dict[str, Any]:
        # 1. Parse known args to get the config file
        args, unknown = self.parser.parse_known_args()
        
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            
        # 2. Handle CLI overrides (dot notation)
        # Format: --section.key value
        for i in range(0, len(unknown), 2):
            key = unknown[i]
            if not key.startswith("--"):
                continue
                
            key = key.lstrip("-")
            val = unknown[i+1]
            
            # Type inference
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    if val.lower() == "true": val = True
                    elif val.lower() == "false": val = False
            
            # Update nested dict
            keys = key.split(".")
            ref = config
            try:
                for k in keys[:-1]:
                    ref = ref.setdefault(k, {})
                ref[keys[-1]] = val
            except Exception as e:
                print(f"Warning: Could not set override {key}={val}: {e}")

        return config