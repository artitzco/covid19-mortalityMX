import hashlib
import pickle
import json


def save_json(dict, file):
    with open(file, 'w') as file:
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


def dict_search(dictionary, keypath, default=None):
    if isinstance(keypath, str):
        keypath = [keypath]
    for key in keypath[:-1]:
        if key in dictionary and isinstance(dictionary[key], dict):
            dictionary = dictionary[key]
            continue
        return default
    return dictionary.get(keypath[-1], default)


def dict_set(dictionary, keypath, value, overwrite=False):
    if isinstance(keypath, str):
        keypath = [keypath]
    for key in keypath[:-1]:
        if key in dictionary:
            if not isinstance(dictionary[key], dict):
                if overwrite:
                    dictionary[key] = {}
                else:
                    raise ValueError("The dictionary cannot be overwritten")
        else:
            dictionary[key] = {}
        dictionary = dictionary[key]
    old = dictionary.get(keypath[-1], None)
    dictionary[keypath[-1]] = value
    return old


def dict_hash(dictionary):
    return hashlib.sha256(json.dumps(dictionary, sort_keys=True).encode()).hexdigest()

