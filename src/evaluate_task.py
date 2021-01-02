# *coding:utf-8 *

import os
import utils
from predict_model import Predict
from sklearn import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
bert_model_path = "./chinese_wwm_ext_pytorch"
ckpt_path = "./to_model/0/ckpt/model.ckpt"
from_pt = True
to_model_path = "to_model"  # 模型保存位置  空串不会保存模型

oce_index2label, oce_label2index = utils.read_label_map("oce", "./data/")
ocn_index2label, ocn_label2index = utils.read_label_map("ocn", "./data/")
tn_index2label, tn_label2index = utils.read_label_map("tn", "./data/")

oce_dev = utils.load_json_file("data/oce_dev.json")
ocn_dev = utils.load_json_file("data/ocn_dev.json")
tn_dev = utils.load_json_file("data/tn_dev.json")

oce_dev = oce_dev[:100]
ocn_dev = ocn_dev[:100]
tn_dev = tn_dev[:100]

model = Predict(ckpt_path, bert_model_path, from_pt=from_pt,
                oce_cls_num=len(oce_label2index),
                ocn_cls_num=len(ocn_label2index),
                tn_cls_num=len(tn_label2index)
                )


def evaluate_dev(dev_samples, index2label, task_name):
    y_predict = model.predict(dev_samples, task_name=task_name)
    label_predicts = [index2label[i] for i in y_predict]
    labels = [i["label"] for i in dev_samples]
    pre = metrics.precision_score(labels, label_predicts, average='macro')
    rc = metrics.recall_score(labels, label_predicts, average='macro')
    f1 = metrics.f1_score(labels, label_predicts, average='macro')
    print(pre, rc, f1)

    return f1


oce_f1 = evaluate_dev(oce_dev, oce_index2label, "oce")
ocn_f1 = evaluate_dev(ocn_dev, ocn_index2label, "ocn")
tn_f1 = evaluate_dev(tn_dev, tn_index2label, "tn")

print((oce_f1 + ocn_f1 + tn_f1) / 3)
