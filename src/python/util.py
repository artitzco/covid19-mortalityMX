import hashlib
import pickle
import json


def save_json(dict, file):
    json.dump(dict, open(file, 'w', encoding='utf-8'), indent=4)


def load_json(file):
    return json.load(open(file, 'r'))


def save_object(obj, file):
    pickle.dump(obj, open(file, "wb"))


def load_object(file):
    return pickle.load(open(file, "rb"))


def dict_hash(dictionary, ignore_none=True):
    dictionary = {k: dict_hash(v, ignore_none=ignore_none)
                  if isinstance(v, dict) else v for k, v in dictionary.items()
                  if v is not None or not ignore_none}
    return hashlib.sha256(json.dumps(dictionary, sort_keys=True).encode()).hexdigest()


def dict_str_hash(dictionary):
    return hashlib.sha256(json.dumps(
        {k: str(v) for k, v in dictionary.items()}, sort_keys=True).encode()).hexdigest()
