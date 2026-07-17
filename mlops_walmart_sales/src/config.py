"""Tiny config loader - reads config.yaml into a plain nested dict."""

import os
import yaml

DEFAULT_CONFIG_PATH = os.environ.get("CONFIG_PATH", "/app/config.yaml")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
