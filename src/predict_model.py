# *coding:utf-8 *


from logger_module import log_obj
import properties
from dtos import *
import tokenize_utils
import tensorflow as tf
from modeling_trm import BertTrmHeadModel
import utils


class Predict():
    def __init__(self, ckpt_path, bert_model_path):

        self.model = BertTrmHeadModel(bert_model_path, from_pt=False, oce_cls_num=7, ocn_cls_num=3, tn_cls_num=15)
        self.model.load_weights(ckpt_path)
        self.tokenizer = self.model.tokenizer

    def _make_examples(self, sample):
        examples = []
        for _, x in enumerate(sample):
            text_a, text_b = utils.trunc_seq_pair(x["text_a"], x.get("text_b", None), max_len=properties.MAX_LEN)
            examples.append(InputExample(x.get("id", None),
                                         text_a,
                                         text_b=text_b,
                                         label=x.get("label", None)))
        return examples

    def make_dataset(self, samples, cut_off=False):
        if not samples:
            return None
        examples = self._make_examples(samples)
        if cut_off:
            examples = examples[:properties.BATCH_SIZE]
        # 这里会处理label的转换
        ds = tokenize_utils.convert_examples_to_features(examples, self.tokenizer, label_map={})
        ds = ds.batch(properties.BATCH_SIZE)  # .repeat(-1)会一直Repeat下去
        return ds

    def predict(self, input_lis, task_name):
        ds = self.make_dataset(input_lis)
        if ds is None:
            return []
        y_predict = []
        for inputs in ds:
            outputs = self.model(inputs[0], task_name, training=False)
            res = tf.math.argmax(outputs, axis=-1)
            y_predict.extend(res.numpy().tolist())
        assert len(y_predict) == len(input_lis), "len not match"
        return y_predict
