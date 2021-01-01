# *coding:utf-8 *

import utils
from collections import Counter
from logger_module import log_obj
import numpy as np
from sklearn.model_selection import train_test_split
import properties
import copy
import math


def inspect_data(path):
    sample = utils.load_json_file(path)
    key_fq = Counter([i["label"] for i in sample])
    sorted_key_fq = sorted(key_fq.items(), key=lambda x: x[1],reverse=True)
    log_obj.info("样本总数"+str(len(sample)))
    log_obj.info("类别总数"+str(len(key_fq)))
    log_obj.info(key_fq)
    log_obj.info(sorted_key_fq)
    return sorted_key_fq



def shuffle_list(lis):
    t = np.array(lis)
    np.random.shuffle(t)
    return t.tolist()


# 重新采样与训练测试集划分
def process(path,train_to_path,dev_to_path,extend_sample_map):
    """
    :param extend_sample_map:  过采样比例
    """
    # 划分训练与验证集
    model0_sample = utils.load_json_file(path)
    sample_dic={}
    for i in model0_sample:
        if i["label"] in sample_dic:
            sample_dic[i["label"]].append(i)
        else:
            sample_dic[i["label"]]=[i]

    train_examples = []
    test_examples = []
    for l,lis in sample_dic.items():
        a, b = train_test_split(lis, test_size=properties.test_dev_size)
        train_examples.extend(a)
        test_examples.extend(b)
    print("train vs dev=",len(train_examples),len(test_examples))

    # 扩充训练集
    train_dic = {}
    for i in train_examples:
        if i["label"] in train_dic:
            train_dic[i["label"]].append(i)
        else:
            train_dic[i["label"]] = [i]


    for k,v in train_dic.items():
        print(k,len(v))
    for label,ratio in extend_sample_map.items():
        if ratio<=1:
            tmp_lis = copy.deepcopy(train_dic[label])
            tmp_lis = shuffle_list(tmp_lis)
            train_dic[label]=tmp_lis[:int(len(tmp_lis)*ratio)]
        else:
            tmp_lis = copy.deepcopy(train_dic[label])
            tmp_lis = shuffle_list(tmp_lis)
            for j in range(math.ceil(ratio)):
                train_dic[label].extend(tmp_lis)
            train_dic[label] = train_dic[label][:int(len(tmp_lis) * ratio)]
    print("重新采样后")
    for k,v in train_dic.items():
        print(k,len(v))
    train_examples = []
    for l,lis in train_dic.items():
        train_examples.extend(lis)

    train_examples = shuffle_list(train_examples)
    test_examples = shuffle_list(test_examples)
    log_obj.info("划分 训练集 : 验证集 = %s : %s" % (len(train_examples), len(test_examples)))

    utils.dump_json_file(train_to_path, train_examples)
    utils.dump_json_file(dev_to_path, test_examples)




