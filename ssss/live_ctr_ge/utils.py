import math

import tensorflow as tf

import sail.common as S
import sail.model as M
from sail import layers
from sail import losses
from sail import initializers
from sail import optimizers
from sail import modules
from sail.feature import FeatureSlot
from sail.feature import FeatureColumnV1
from sail.feature import FeatureColumnDense


def get_pred_and_loss(logit, label, sample_rate, channel_id_label, channel_id, loss_type):
    if loss_type == 'logloss':
        logit = logit - tf.log(sample_rate)
        pred = tf.nn.sigmoid(logit)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit)
        loss = tf.where(tf.equal(channel_id_label, channel_id), loss, tf.zeros_like(loss))
        loss = tf.reduce_sum(loss)
    elif loss_type == 'mse':
        pred = logit
        loss = tf.where(tf.equal(channel_id_label, channel_id), tf.square(label - logit), tf.zeros_like(label))
        loss = 0.5 * tf.reduce_sum(loss)
    else:
        raise ValueError('Unknown loss_type: {}'.format(loss_type))
    return pred, loss


def compress_embedding(input, output_dims, name):
    _compressed_uid = modules.DenseTower(name='{}_compress_tower'.format(name),
                            output_dims=output_dims,
                            initializers=initializers.HeNormal(),
                            )(input)
    tf.summary.histogram('{}_compressed'.format(name), _compressed_uid)
    return _compressed_uid

def get_init_funcs(dims, mean=0.0, stddev=1.0):
    init_funcs = []
    for dim in dims[:-1]:
        init_funcs.append(
            lambda s, d=dim: tf.truncated_normal(shape=s,
                                                 mean=mean,
                                                 stddev=stddev) / math.sqrt(d))

# GCN implementation
def graph_convolution_tower(neighbor_emb_vec, emb_vec, tower_dims, name, stop_grad=False, drop_out=False):
    assert len(neighbor_emb_vec.shape) == 3
    assert len(emb_vec.shape) == 2

    if stop_grad:
        neighbor_emb_vec = tf.stop_gradient(neighbor_emb_vec)
        emb_vec = tf.stop_gradient(emb_vec)

    neighbor_emb_vec = tf.reduce_mean(neighbor_emb_vec, axis=1)
    input_emb = tf.concat([neighbor_emb_vec, emb_vec], axis=1)

    if drop_out:
        input_emb = drop_feature(input_emb, 1555545600, 1577836800, 0.0)

    compress_tower = modules.DenseTower(name=name,
                                        optimizers=optimizers.ps.AdaGrad(alpha=0.01, beta=1000, init_factor=1.0),
                                        output_dims=tower_dims,
                                        activations=layers.Relu(),
                                        initializers=initializers.GlorotUniform())
    output_emb = compress_tower(input_emb)

    return output_emb


def drop_feature(feature, s, e, keep_rate):
    is_training = True
    if S.compilation() != S.AllowedCompilations.get('training'):
        is_training = False
    print("is_training: ", is_training)

    if (not is_training):
        return feature

    fc_req_time = FeatureColumnDense('fc_line_id_req_time', 1, tf.int64)
    req_time = fc_req_time.get_tensor()

    tf.summary.histogram('req_time', req_time)

    no_filter = tf.less(req_time, s)
    no_filter1 = tf.greater(req_time, e)
    no_filter = tf.logical_or(no_filter, no_filter1)

    batch_size = tf.shape(req_time)[0]
    rand = tf.random_uniform(shape=[batch_size], minval=0, maxval=1, dtype=M.get_dtype())

    to_keep = tf.less(rand, tf.fill([batch_size], tf.constant(keep_rate, dtype=M.get_dtype())))
    to_keep = tf.logical_or(to_keep, tf.reshape(no_filter, [-1]))

    multiplier = tf.where(
        to_keep,
        tf.ones_like(req_time, dtype=M.get_dtype()),
        tf.zeros_like(req_time, dtype=M.get_dtype()))

    multiplier = tf.reshape(multiplier, [-1, 1])
    tf.summary.histogram('ad_emb_multiplier', multiplier)
    return feature * multiplier

