from typing import List, Dict, Any
import json
import os
import io

import base58
import dill
import hashlib


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [
            json.loads(line.strip()) for line in file.readlines() if line.strip()
        ]
    return instances


def write_jsonl(instances: List[Dict], file_path: str):
    print(f"Writing {len(instances)} instance in {file_path}")
    with open(file_path, "w") as file:
        for instance in instances:
            file.write(json.dumps(instance) + "\n")


def write_json(instance: Dict, file_path: str) -> None:
    print(f"Writing data to {file_path}")
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    print(f"Writing json in {file_path}")
    with open(file_path, "w") as file:
        file.write(json.dumps(instance) + "\n")


def read_json(file_path: str) -> Dict:
    with open(file_path, "r") as file:
        instance = json.load(file)
    return instance


def hash_object(o: Any) -> str:
    # Taken from allennlp
    """Returns a character hash code of arbitrary Python objects."""
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()
