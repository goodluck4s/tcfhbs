# *coding:utf-8 *

import os
import tensorflow as tf
from bert_encoder import BERTEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class BertBiGruHeadModel(tf.keras.Model):
    def __init__(self, bert_model_path, from_pt=False, oce_cls_num=7, ocn_cls_num=3, tn_cls_num=15):
        super(BertBiGruHeadModel, self).__init__()

        self.bert_encoder_obj = BERTEncoder(bert_model_path, from_pt=from_pt,do_update_weights=False)
        self.tf_bert_model = self.bert_encoder_obj.get_model()
        self.tokenizer = self.bert_encoder_obj.get_tokenizer()
        self.bert_config = self.bert_encoder_obj.get_config()

        self.oce_cls_num = oce_cls_num
        self.ocn_cls_num = ocn_cls_num
        self.tn_cls_num = tn_cls_num

        self.oce_layer_dic=bi_gru_head_builder(self.oce_cls_num,self.bert_config.hidden_size,self.bert_config.hidden_dropout_prob)
        self.ocn_layer_dic=bi_gru_head_builder(self.ocn_cls_num,self.bert_config.hidden_size,self.bert_config.hidden_dropout_prob)
        self.tn_layer_dic=bi_gru_head_builder(self.tn_cls_num,self.bert_config.hidden_size,self.bert_config.hidden_dropout_prob)


    def call(self, input_dic, task, training):
        # 本身是对TFBertMainLayer的一层封装 就是单纯的12个encoder叠加
        bert_encoder_output = self.tf_bert_model(input_dic, training=training)
        # 最后一层的原始输出[bz,sl,hz]
        sequence_output = bert_encoder_output[0]
        # 是最后一层第一个token的原始输出[bz,hz]过一次不加dropout的dense 即 pooled_output = self.dense(first_token_tensor)
        # dense也会输出hidden_size
        pooled_output = bert_encoder_output[1]

        def get_dic(task):
            layer_dic = {}
            if task == "oce":
                layer_dic = self.oce_layer_dic
            elif task == "ocn":
                layer_dic = self.ocn_layer_dic
            elif task == "tn":
                layer_dic = self.tn_layer_dic
            else:
                raise Exception("未定义这个任务")
            return layer_dic

        layer_dic = get_dic(task)
        bi_gru_out = layer_dic["bi_gru"](sequence_output)
        mid_out = layer_dic["dense_mid"](bi_gru_out)
        mid_out = layer_dic["dropout"](mid_out,training=training)
        logits = layer_dic["dense_logit"](mid_out)

        return logits


def bi_gru_head_builder(cls_num,hidden_size,dropout_prob):

    bi_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(hidden_size))
    dense_mid = tf.keras.layers.Dense(hidden_size, activation='relu')
    dense_logit = tf.keras.layers.Dense(cls_num, activation='relu')

    dropout = tf.keras.layers.Dropout(dropout_prob)

    # return (bi_gru,dense_mid,dense_logit,dropout)
    return {
        "bi_gru":bi_gru,
        "dense_mid":dense_mid,
        "dense_logit":dense_logit,
        "dropout":dropout,
    }





if __name__=="__main__":
    pt_base_model_path = "/Users/qianlai/Documents/_model/chinese_wwm_ext_pytorch"
    test_model = BertBiGruHeadModel(pt_base_model_path, from_pt=True)
    inputs = test_model.tokenizer.encode_plus("Hello, my dog is cute","hahaha", return_tensors="tf")

    res = test_model(inputs,task="ocn",training=True)
    print(res)
    print(len(test_model.trainable_variables))
    print(len(test_model.variables))


