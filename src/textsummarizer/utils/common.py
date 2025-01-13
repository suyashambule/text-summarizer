import os
from box.exceptions import BoxValueError
from src.textsummarizer.logging import logging
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import yaml

@ensure_annotations
def read_yaml(path_to_yaml:Path)->ConfigBox:
    try:
        with open (path_to_yaml) as yaml_file:
            contents=yaml.safe_load(yaml_file)
            logging.INFO(f"yaml {path_to_yaml} is open successfully")
            return ConfigBox(contents)
    except BoxValueError:
        raise ValueError("yaml file is Empty")
    except Exception as e:
        raise e

