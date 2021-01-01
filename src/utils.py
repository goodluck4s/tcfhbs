# *coding:utf-8 *

import json

def load_json_file(path):
    with open(path, "r") as f:
        res = json.load(f)
    return res


def dump_json_file(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)