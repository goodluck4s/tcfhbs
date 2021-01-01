# *coding:utf-8 *

import os
from transformers import BertTokenizer, TFBertModel,BertConfig

pt_base_model_path = "/Users/qianlai/Documents/_model/chinese_wwm_ext_pytorch"
tf_base_model_path = "/Users/qianlai/Documents/_model/chinese_L-12_H-768_A-12"



class BERTEncoder():
    def __init__(self,model_path,from_pt=False,do_update_weights=False):
        self.config = BertConfig.from_pretrained(os.path.join(model_path, "config.json"))
        self.tokenizer = BertTokenizer.from_pretrained("./")
        self.tf_bert_model = TFBertModel.from_pretrained(model_path, config=self.config, from_pt=from_pt, trainable=do_update_weights)

    def get_model(self):
        return self.tf_bert_model

    def get_tokenizer(self):
        return self.tokenizer

    def get_config(self):
        return self.config

if __name__=="__main__":
    b = BERTEncoder("./chinese_wwm_ext_pytorch",from_pt=True)
    inputs = b.tokenizer("Hello, my dog is cute", return_tensors="tf")
    res = b.tf_bert_model(inputs)
    print(res[0])