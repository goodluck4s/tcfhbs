# *coding:utf-8 *



#
#
# >>>>>>>>>>>>>>>>>>>>基本统计<<<<<<<<<<<<<<<<<<<<<<<
# 总数 35694
# 最后一行 {'id': '35693', 'text_a': '听说你最近很上进哦,哈!', 'text_b': None, 'label': 'happiness', 'task': 'oce'}
# >>>>>>>>>>>>>>>>>>>>>>>>>>>label统计<<<<<<<<<<<<<<<<<<<<<<<<<
# Counter({
# 'sadness': 12604,
# 'happiness': 8981,
# 'disgust': 4405,
# 'anger': 4115,
# 'like': 4087,
# 'surprise': 908,
# 'fear': 594})
#
#
# >>>>>>>>>>>>>>>>>>>>>>>>句子长 统计<<<<<<<<<<<<<<<<<<<<<<<<
#
# >>>>>>>>>>>>>>>>>>>>基本统计<<<<<<<<<<<<<<<<<<<<<<<
# 总数 53387
# 最后一行 {'id': '53386', 'text_a': '对,后来全翻过来聊这个了', 'text_b': '后来我们聊这个了', 'label': '0', 'task': 'ocn'}
# >>>>>>>>>>>>>>>>>>>>>>>>>>>label统计<<<<<<<<<<<<<<<<<<<<<<<<<
# Counter({
# '1': 18285,
# '0': 17726,
# '2': 17376})
#
# >>>>>>>>>>>>>>>>>>>>>>>>句子长 统计<<<<<<<<<<<<<<<<<<<<<<<<
#
# >>>>>>>>>>>>>>>>>>>>基本统计<<<<<<<<<<<<<<<<<<<<<<<
# 总数 63360
# 最后一行 {'id': '63359', 'text_a': '日本虎视眈眈“武力夺岛”, 美军向俄后院开火,普京终不再忍!', 'text_b': None, 'label': '113', 'task': 'tn'}
# >>>>>>>>>>>>>>>>>>>>>>>>>>>label统计<<<<<<<<<<<<<<<<<<<<<<<<<
# Counter({
# '109': 7044,
# '104': 6156,
# '102': 5886,
# '113': 5756,
# '107': 4909,
# '101': 4817,
# '103': 4758,
# '110': 4348,
# '108': 4083,
# '112': 4061,
# '116': 4049,
# '115': 3380,
# '106': 2485,
# '100': 1326,
# '114': 302})



import re
from collections import Counter
import json
import os
import pandas as pd
import pickle



def format_data(task_name, path, to_folfer, row_len, has_b=False):
    res = []
    with open(path, "r") as f:
        for line in f:
            s = re.split(r"[\t\n]", line)
            s = s[:-1]
            if len(s) != row_len:
                print("format error", line)
                continue
            tmp = {}
            tmp["id"] = s[0]
            tmp["text_a"] = s[1]
            tmp["text_b"] = s[2] if has_b else None
            tmp["label"] = s[3] if has_b else s[2]
            tmp["task"] = task_name
            res.append(tmp)
    print(">>>>>>>>>>>>>>>>>>>>基本统计<<<<<<<<<<<<<<<<<<<<<<<")
    print("总数", len(res))
    print("最后一行", res[-1])
    with open(os.path.join(to_folfer, task_name + "_samples.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    label_fq = Counter([i["label"] for i in res])
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>label统计<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(label_fq)

    label2index = {}
    index2label = {}
    for index, label in enumerate(label_fq.keys()):
        label2index[label] = index
        index2label[index] = label
    print("label2index", label2index)
    print("index2label", index2label)
    with open(os.path.join(to_folfer, task_name + "_label2index.pkl"), "wb") as f:
        pickle.dump(label2index, f)
    with open(os.path.join(to_folfer, task_name + "_index2label.pkl"), "wb") as f:
        pickle.dump(index2label, f)

    df_len = pd.DataFrame(
        sorted([len(t["text_a"]) + len(t["text_b"]) if has_b else len(t["text_a"]) for t in res], reverse=True))
    print(">>>>>>>>>>>>>>>>>>>>>>>>句子长 统计<<<<<<<<<<<<<<<<<<<<<<<<")
    print(df_len.describe())
    print("<125占比", len(df_len[df_len[0] < 128 - 3]) / len(df_len))
    print("<253占比", len(df_len[df_len[0] < 256 - 3]) / len(df_len))


# print("绝对路径==")
# print(os.path.abspath("OCEMOTION_train1128.csv"))
# print(os.path.abspath("."))
# print(os.getcwd())
#
# format_data("oce", "OCEMOTION_train1128.csv", "./", row_len=3, has_b=False)
# format_data("ocn", "OCNLI_train1128.csv", "./", row_len=4, has_b=True)
# format_data("tn", "TNEWS_train1128.csv", "./", row_len=3, has_b=False)



def format_test(task_name, path, to_folfer, row_len, has_b=False):
    res = []
    with open(path, "r") as f:
        for line in f:
            s = re.split(r"[\t\n]", line)
            s = s[:-1]
            if len(s) != row_len:
                print("format error", line)
                continue
            tmp = {}
            tmp["id"] = s[0]
            tmp["text_a"] = s[1]
            tmp["text_b"] = s[2] if has_b else None
            tmp["task"] = task_name
            res.append(tmp)
    print(">>>>>>>>>>>>>>>>>>>>基本统计<<<<<<<<<<<<<<<<<<<<<<<")
    print("总数", len(res))
    print("最后一行", res[-1])
    with open(os.path.join(to_folfer, task_name + "_test.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

format_test("oce", "OCEMOTION_a.csv", "./", row_len=2, has_b=False)
format_test("ocn", "OCNLI_a.csv", "./", row_len=3, has_b=True)
format_test("tn", "TNEWS_a.csv", "./", row_len=2, has_b=False)