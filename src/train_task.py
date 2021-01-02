# *coding:utf-8 *


import os
from modeling_trm import BertTrmHeadModel
import sample_processor
import train_func
import utils
from logger_module import log_obj

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
bert_model_path = "./chinese_wwm_ext_pytorch"
do_cut_samples = True  # 测试时会只截取一个批次进行
from_pt = True
EPOCHS = 1
to_model_path="to_model"  # 模型保存位置  空串不会保存模型

oce_index2label, oce_label2index = utils.read_label_map("oce", "./data/")
ocn_index2label, ocn_label2index = utils.read_label_map("ocn", "./data/")
tn_index2label, tn_label2index = utils.read_label_map("tn", "./data/")

# sample_processor.process(
#     "data/oce_samples.json",
#     "data/oce_train.json",
#     "data/oce_dev.json",
#     extend_sample_map={'surprise': 2, 'fear': 2}
# )
# sample_processor.process(
#     "data/ocn_samples.json",
#     "data/ocn_train.json",
#     "data/ocn_dev.json",
#     extend_sample_map={}
# )
# sample_processor.process(
#     "data/tn_samples.json",
#     "data/tn_train.json",
#     "data/tn_dev.json",
#     extend_sample_map={'114': 3}
# )

oce_train = utils.load_json_file("data/oce_train.json")
oce_dev = utils.load_json_file("data/oce_dev.json")
ocn_train = utils.load_json_file("data/ocn_train.json")
ocn_dev = utils.load_json_file("data/ocn_dev.json")
tn_train = utils.load_json_file("data/tn_train.json")
tn_dev = utils.load_json_file("data/tn_dev.json")

bert_dense_model = BertTrmHeadModel(bert_model_path, from_pt=from_pt,
                                      oce_cls_num=len(oce_label2index),
                                      ocn_cls_num=len(ocn_label2index),
                                      tn_cls_num=len(tn_label2index))
log_obj.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>oce<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
train_ds, test_ds, trn, ten = train_func.make_dataset(oce_train, oce_dev,
                                                      label2index=oce_label2index,
                                                      tokenizer=bert_dense_model.tokenizer,
                                                      cut_off=do_cut_samples)
train_func.train_a_dataset(bert_dense_model, train_ds, test_ds,
                           task_name="oce", EPOCHS=EPOCHS, to_model_path=to_model_path)


log_obj.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>ocn<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
train_ds, test_ds, trn, ten = train_func.make_dataset(ocn_train, ocn_dev,
                                                      label2index=ocn_label2index,
                                                      tokenizer=bert_dense_model.tokenizer,
                                                      cut_off=do_cut_samples)
train_func.train_a_dataset(bert_dense_model, train_ds, test_ds,
                           task_name="ocn", EPOCHS=EPOCHS, to_model_path=to_model_path)


log_obj.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>tn<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
train_ds, test_ds, trn, ten = train_func.make_dataset(tn_train, tn_dev,
                                                      label2index=tn_label2index,
                                                      tokenizer=bert_dense_model.tokenizer,
                                                      cut_off=do_cut_samples)
train_func.train_a_dataset(bert_dense_model, train_ds, test_ds,
                           task_name="tn", EPOCHS=EPOCHS, to_model_path=to_model_path)

if to_model_path:
    saved_path = os.path.join(to_model_path, str("0"), "ckpt")
    os.makedirs(saved_path, exist_ok=True)
    bert_dense_model.save_weights(os.path.join(saved_path,"model.ckpt"))
    log_obj.info("模型保存成功")

log_obj.info(">>>>>>>>>>>word done<<<<<<<<<<<")