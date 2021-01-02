# *coding:utf-8 *


import os
import pickle
from logger_module import log_obj
from collections import Counter
import properties
from dtos import *
import tokenize_utils
import tensorflow as tf
from sklearn import metrics


def read_label_map(task_name, folder):
    with open(os.path.join(folder, task_name + "_index2label.pkl"), "rb") as f:
        index2label = pickle.load(f)
    with open(os.path.join(folder, task_name + "_label2index.pkl"), "rb") as f:
        label2index = pickle.load(f)
    return index2label, label2index


def trunc_seq_pair(text_a, text_b, max_len):
    if text_b:
        while len(text_a) + len(text_b) > max_len - 3:
            if len(text_b) > len(text_a):
                text_b = text_b[:-1]
            else:
                text_a = text_a[:-1]
        return text_a, text_b
    else:
        text_a = text_a[:max_len - 2]
        return text_a, None


def _make_examples(sample):
    all_labels = [x["label"] for x in sample]
    all_labels = Counter(all_labels)
    log_obj.info("label分类统计%s" % (all_labels))
    examples = []
    for _, x in enumerate(sample):
        text_a,text_b = trunc_seq_pair(x["text_a"], x.get("text_b", None),max_len=properties.MAX_LEN)
        examples.append(InputExample(x.get("id", None),
                                     text_a,
                                     text_b=text_b,
                                     label=x.get("label", None)))
    return examples


def make_dataset(sample_train, sample_dev, label2index,tokenizer, cut_off=False):
    log_obj.info(">>>>>>>>>>>>>样本数目=%s %s<<<<<<<<<<<<<<" % (len(sample_train), len(sample_dev)))
    sample_train = list(filter(lambda x: x["label"] in label2index, sample_train))  # true的留下
    sample_dev = list(filter(lambda x: x["label"] in label2index, sample_dev))  # true的留下
    log_obj.info(">>>>>>>>>>>>>过滤后样本数目=%s %s<<<<<<<<<<" % (len(sample_train), len(sample_dev)))
    train_examples = _make_examples(sample_train)
    test_examples = _make_examples(sample_dev)
    if cut_off:
        train_examples = train_examples[:properties.BATCH_SIZE]
        test_examples = test_examples[:properties.EVAL_BATCH_SIZE]

    # 这里会处理label的转换
    train_ds = tokenize_utils.convert_examples_to_features(train_examples, tokenizer, label2index)
    test_ds = tokenize_utils.convert_examples_to_features(test_examples, tokenizer, label2index)
    train_ds = train_ds.shuffle(2 * properties.BATCH_SIZE).batch(properties.BATCH_SIZE)  # .repeat(-1)会一直Repeat下去
    test_ds = test_ds.batch(properties.EVAL_BATCH_SIZE)
    return train_ds, test_ds, len(train_examples), len(test_examples)


def train_a_dataset(model,train_ds, test_ds,task_name,EPOCHS=10,to_model_path=""):
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
    loss_fuc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy("train_acc")
    dev_loss = tf.keras.metrics.Mean(name='dev_loss')
    dev_acc = tf.keras.metrics.SparseCategoricalAccuracy("dev_acc")
    dev_recall = tf.keras.metrics.Mean(name='dev_recall')
    dev_precision = tf.keras.metrics.Mean(name='dev_precision')
    dev_f1 = tf.keras.metrics.Mean(name='dev_f1')

    @tf.function
    def train_step(inputs):
        with tf.GradientTape() as tape:
            outputs = model(inputs[0],task=task_name, training=True)  # inputs 0位置3个dic,1位置label  outputs 是logits
            loss = loss_fuc(inputs[1], outputs)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_acc(inputs[1], outputs)

    def dev_step(inputs):
        outputs = model(inputs[0],task=task_name, training=False)
        loss = loss_fuc(inputs[1], outputs)

        dev_loss(loss)
        dev_acc(inputs[1], outputs)

        predict_labels = tf.argmax(outputs, axis=-1)
        labels = inputs[1]
        dev_precision(metrics.precision_score(labels.numpy(), predict_labels.numpy(), average='macro'))
        dev_recall(metrics.recall_score(labels.numpy(), predict_labels.numpy(), average='macro'))
        dev_f1(metrics.f1_score(labels.numpy(), predict_labels.numpy(), average='macro'))

    for e in range(EPOCHS):
        log_obj.info(">>>>>>epoch=%s<<<<<<" % (e))
        train_loss.reset_states()
        train_acc.reset_states()

        dev_loss.reset_states()
        dev_acc.reset_states()
        dev_precision.reset_states()
        dev_recall.reset_states()
        dev_f1.reset_states()

        for _, batch_inputs in enumerate(train_ds):
            train_step(batch_inputs)
            if _ % 800 == 0:  # 50批打印一次
                log_obj.info(
                    'train Batch={} loss={:.4f} acc={:.4f}'.format(
                        _, train_loss.result(), train_acc.result()))

        for _, batch_dev_inputs in enumerate(test_ds):
            dev_step(batch_dev_inputs)
            if _ % 300 == 0:  # 50批打印一次
                log_obj.info('dev Batch={} loss={:.4f} acc={:.4f} precision={:.4f} recall={:.4f} f1={:.4f}'.format(
                    _, dev_loss.result(), dev_acc.result(), dev_precision.result(), dev_recall.result(),
                    dev_f1.result()))
        if to_model_path:
            saved_path = os.path.join(to_model_path, str(e), "h5")
            os.makedirs(saved_path, exist_ok=True)
            model.save_pretrained(saved_path)

