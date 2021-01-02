# *coding:utf-8 *

import pickle

with open("./ocn_index2label.pkl","rb") as f:
    a = pickle.load(f)

    print(a)

with open("./ocn_label2index.pkl", "rb") as f:
    a = pickle.load(f)

    print(a)
