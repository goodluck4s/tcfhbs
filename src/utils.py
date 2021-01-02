# *coding:utf-8 *

import json
import os
import pickle

def load_json_file(path):
    with open(path, "r") as f:
        res = json.load(f)
    return res


def dump_json_file(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)



def read_label_map(task_name, folder):
    with open(os.path.join(folder, task_name + "_index2label.pkl"), "rb") as f:
        index2label = pickle.load(f)
    with open(os.path.join(folder, task_name + "_label2index.pkl"), "rb") as f:
        label2index = pickle.load(f)
    return index2label, label2index


def trunc_seq_pair(text_a, text_b, max_len):
    if text_b:
        while len(text_a) + len(text_b) > max_len - 3:
            if len(text_b) > len(text_a):
                text_b = text_b[:-1]
            else:
                text_a = text_a[:-1]
        return text_a, text_b
    else:
        text_a = text_a[:max_len - 2]
        return text_a, None