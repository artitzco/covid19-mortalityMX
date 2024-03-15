import hashlib
import pickle
import json


def save_json(dict, file):
    with open(file, 'w', encoding='utf-8') as file:
        json.dump(dict, file, indent=4)


def load_json(file):
    with open(file, 'r') as file:
        return json.load(file)


def save_object(obj, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def load_object(file_path):
    with open(file_path, "rb") as file:
        loaded_object = pickle.load(file)
    return loaded_object


def dict_hash(dictionary, ignore_none=True):
    dictionary = {k: dict_hash(v, ignore_none=ignore_none)
                  if isinstance(v, dict) else v for k, v in dictionary.items()
                  if v is not None or not ignore_none}
    return hashlib.sha256(json.dumps(dictionary, sort_keys=True).encode()).hexdigest()


def dict_str_hash(dictionary):
    return hashlib.sha256(json.dumps(
        {k: str(v) for k, v in dictionary.items()}, sort_keys=True).encode()).hexdigest()
