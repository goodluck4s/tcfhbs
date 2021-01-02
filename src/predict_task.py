# *coding:utf-8 *

import os
import utils
from predict_model import Predict
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
bert_model_path = "./chinese_wwm_ext_pytorch"
ckpt_path = "./to_model/0/ckpt/model.ckpt"
from_pt = True

oce_index2label, oce_label2index = utils.read_label_map("oce", "./data/")
ocn_index2label, ocn_label2index = utils.read_label_map("ocn", "./data/")
tn_index2label, tn_label2index = utils.read_label_map("tn", "./data/")

oce_test = utils.load_json_file("data/oce_test.json")
ocn_test = utils.load_json_file("data/ocn_test.json")
tn_test = utils.load_json_file("data/tn_test.json")

# oce_test = oce_test[:3]
# ocn_test = ocn_test[:3]
# tn_test = tn_test[:3]

model = Predict(ckpt_path, bert_model_path, from_pt=from_pt,
                oce_cls_num=len(oce_label2index),
                ocn_cls_num=len(ocn_label2index),
                tn_cls_num=len(tn_label2index)
                )


def predict_test_dataset(samples, index2label, task_name):
    y_predict = model.predict(samples, task_name=task_name)
    label_predicts = [index2label[i] for i in y_predict]
    name_map = {
        "oce": "ocemotion_predict.json",
        "ocn": "ocnli_predict.json",
        "tn": "tnews_predict.json",
    }

    with open(os.path.join("./predict_res/", name_map[task_name]), "w") as f:
        for sample, label in zip(samples, label_predicts):
            tmp = {"id": sample["id"], "label": label}
            tmp = json.dumps(tmp, ensure_ascii=False)
            f.write(tmp+"\n")



predict_test_dataset(oce_test, oce_index2label, "oce")
predict_test_dataset(ocn_test, ocn_index2label, "ocn")
predict_test_dataset(tn_test, tn_index2label, "tn")
