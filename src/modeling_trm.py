# *coding:utf-8 *

import os
import tensorflow as tf
from bert_encoder import BERTEncoder
from transformers.modeling_tf_bert import TFBertLayer,TFBertPooler

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class BertTrmHeadModel(tf.keras.Model):
    def __init__(self, bert_model_path, from_pt=False, oce_cls_num=7, ocn_cls_num=3, tn_cls_num=15):
        super(BertTrmHeadModel, self).__init__()

        self.bert_encoder_obj = BERTEncoder(bert_model_path, from_pt=from_pt,do_update_weights=True)
        self.tf_bert_model = self.bert_encoder_obj.get_model()
        self.tokenizer = self.bert_encoder_obj.get_tokenizer()
        self.bert_config = self.bert_encoder_obj.get_config()

        self.oce_cls_num = oce_cls_num
        self.ocn_cls_num = ocn_cls_num
        self.tn_cls_num = tn_cls_num

        self.oce_layer_dic=head_builder(self.bert_config,oce_cls_num)
        self.ocn_layer_dic=head_builder(self.bert_config,ocn_cls_num)
        self.tn_layer_dic=head_builder(self.bert_config,tn_cls_num)


    def call(self, input_dic, task, training):

        attention_mask = input_dic["attention_mask"]

        # 本身是对TFBertMainLayer的一层封装 就是单纯的12个encoder叠加
        bert_encoder_output = self.tf_bert_model(input_dic, training=training)
        # 最后一层的原始输出[bz,sl,hz]
        sequence_output = bert_encoder_output[0]
        # 是最后一层第一个token的原始输出[bz,hz]过一次不加dropout的dense 即 pooled_output = self.dense(first_token_tensor)
        # dense也会输出hidden_size
        pooled_output = bert_encoder_output[1]

        logits = None
        if task == "oce":
            logits = compute_logit(self.oce_layer_dic,sequence_output,attention_mask,training=training)
        elif task == "ocn":
            logits = compute_logit(self.ocn_layer_dic,sequence_output,attention_mask,training=training)
        elif task == "tn":
            logits = compute_logit(self.tn_layer_dic,sequence_output,attention_mask,training=training)
        else:
            raise Exception("未定义这个任务")

        return logits


def head_builder(config,num_cls):
    additional_trm_layer = TFBertLayer(config)
    additional_pooler = TFBertPooler(config)
    dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
    classifier = tf.keras.layers.Dense(num_cls)

    return {
        "additional_trm_layer":additional_trm_layer,
        "additional_pooler":additional_pooler,
        "dropout":dropout,
        "classifier":classifier,
    }


def compute_logit(head_dic,sequence_output,attention_mask,training):
    extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
    extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    addition_layer_out_tuple = head_dic["additional_trm_layer"]([sequence_output, extended_attention_mask, None], training=training)
    pooled_output = head_dic["additional_pooler"](addition_layer_out_tuple[0])
    pooled_output = head_dic["dropout"](pooled_output, training=training)
    logits = head_dic["classifier"](pooled_output)

    return logits



if __name__=="__main__":
    pt_base_model_path = "/Users/qianlai/Documents/_model/chinese_wwm_ext_pytorch"
    test_model = BertTrmHeadModel(pt_base_model_path,from_pt=True)
    inputs = test_model.tokenizer.encode_plus("Hello, my dog is cute", return_tensors="tf")

    res = test_model(inputs,task="tn",training=False)
    print(res)
    print(len(test_model.trainable_variables))
    print(len(test_model.variables))


