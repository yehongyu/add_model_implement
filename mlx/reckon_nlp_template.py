# encoding=utf-8
import os
import numpy
import tensorflow as tf

from archer.layers.utils import get_shape_list
from archer import models
from archer.models.model_config import parse_json_model_config

from textops.tensorflow.nlp_bert_tokenize import nlp_bert_tokenize_and_look_op
from tensorflow.python.ops import lookup_ops


# 更详细的使用方式见： https://bytedance.feishu.cn/docs/doccnuhB4w3xsITTYI7YOrP3otb
# 更多模型见： https://bytedance.feishu.cn/docs/doccnjt9hDk7tVur5swfi08jYbf#
check_model_json = "triplet_on_qt70b_14Bbdu_archer_fix/model_config.json"
vocab_file = "triplet_on_qt70b_14Bbdu_archer_fix/fine_fix.txt"
model_dir = "triplet_on_qt70b_14Bbdu_archer_fix"
seq_len = 32


def test_predict(inputs):
    input_strings = tf.placeholder(tf.string, shape=[None])
    output_tensor = bert_model_with_input(input_strings, check_model_json, vocab_file)
    sess = tf.Session()

    # restore from checkpoint
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    print(saver)

    sess.run([tf.tables_initializer()])
    output_tensor_np = sess.run(output_tensor, feed_dict={input_strings: inputs})
    print(output_tensor_np.shape)
    print(output_tensor_np)


def bert_model_with_input(input_strings, model_json, vocab_file):
    batch_size = tf.shape(input_strings)[0]
    vocab = lookup_ops.index_table_from_file(vocab_file)
    input_ids = nlp_bert_tokenize_and_look_op(input_strings, vocab)

    # perform truncating & padding
    input_ids = tf.reshape(input_ids.to_tensor(), [batch_size, -1])
    input_ids = input_ids[:, :seq_len]
    pad_right = seq_len - tf.shape(input_ids)[1]
    input_ids = tf.pad(input_ids, [[0, 0], [0, pad_right]])

    token_type_ids = tf.zeros([batch_size, seq_len], dtype=tf.int32)
    attention_mask = tf.ones([batch_size, seq_len], dtype=tf.int32)

    model_config = parse_json_model_config(model_json)
    model = getattr(models, model_config.model_name)(
        config=model_config, is_training=False, batch_size=batch_size,
        seq_length=seq_len)
    output_tensor = model(
        input_ids,
        input_mask=attention_mask,
        token_type_ids=token_type_ids,
        use_fast_transformer=False,
        # ft_ckpt_path=os.path.dirname(model_json),
    )
    return output_tensor


## dump saved_model
def model_fn_builder(model_json, vocab_file):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        input_strings = features["input_strings"]
        output_tensor = bert_model_with_input(input_strings, model_json, vocab_file)

        predictions = {"last_layer_cls": output_tensor}

        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)
        return output_spec

    return model_fn


if __name__ == "__main__":
    tests = ["Hello", "World", "你好", "世界", "helloworld", "hardworking", "abcdefg" * 20]
    test_predict(tests)