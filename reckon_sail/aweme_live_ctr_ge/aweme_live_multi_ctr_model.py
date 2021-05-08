#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import os

os.environ["ENABLE_HALO"] = "true"

from sail.layers.halo.feature_extractor import FIDExtractor, UIDExtractor
from sail.layers.halo.feature_fetcher import FeatureFetcher
from sail.layers.halo.set_model import FeatureSetter
from tf_euler.python import euler_ops

sys.path.append('../../common/api')

import math
import tensorflow as tf
from collections import OrderedDict

import sail.common as S
import sail.model as M
from sail import layers
from sail import losses
from sail import initializers
from sail import optimizers
from sail import modules
from sail.feature import FeatureSlot, FeatureColumnV2
from sail.feature import FeatureColumnV1
from sail.feature import FeatureColumnDense

from live_aweme_feature_v2 import *
from utils import *

BUCKET_NUM = 500
FINISH_CUTOFF = 100

COMBINE_DIM = 16
COMBINE_METHOD = 'concat'

COMBINE_TOWER_DIMS = [16, 1]

BIAS_NN_DIMS = [256, COMBINE_DIM]

DEEP_FM_DIMS = [256, 64, COMBINE_DIM]

ROOM_SLOT_DIM = 64
DEEP_ROOM_NN_DIMS = [256, 128, ROOM_SLOT_DIM]

DEEP_CONCAT_DIMS = []

NEW_DEEP_CONCAT_DIMS = []

SELECT_DEEP_CONCAT_DIMS = [256, 64, COMBINE_DIM]

STOP_GRAD = False

KEEP_ORIGINAL_FM = True

ADD_AUXI_LOSS = False

USE_RMSPROP = False

DENSE_ALPHA = 0.001

LOSS_TYPE = 'logloss'

CLEAR_NN = False

# Graph Embedding
USE_GE = True

is_training = S.is_compiling_training()

channel_id_2_channel_name = {
    21: "square",
    36: "head",
    10: "nearby",
    1: "follow",
    46: "feed_head",
    37: "follow_toplist",
    0: "feed",
}

# Set defaults for FeatureColumns
FeatureSlot.set_default_bias_optimizer(optimizers.ps.FTRL(alpha=.008, beta=1., lambda1=1.2, lambda2=0.0))
FeatureSlot.set_default_occurrence_threshold(2)
FeatureSlot.set_default_expire_time(0)

default_bias_slot_bias_opt = optimizers.ps.FTRL(alpha=0.01, beta=1.0, lambda1=0.0, lambda2=0.0)
default_vec_slot_bias_opt = optimizers.ps.FTRL(alpha=0.01, beta=1.0, lambda1=0.0, lambda2=0.0)
default_vec_slot_vec_opt = optimizers.ps.AdaGrad(alpha=0.05, beta=1.0, weight_decay=0.0, init_factor=0.015625)

# Create FeatureColumn
fc_dict = {}
for slot_id in VALID_SLOTS:
    if not slot_id in VEC_SLOTS:
        fs = FeatureSlot(slot_id=slot_id,
                         bias_optimizer=default_bias_slot_bias_opt)
    else:
        fs = FeatureSlot(slot_id=slot_id,
                         bias_optimizer=default_vec_slot_bias_opt,
                         vec_optimizer=default_vec_slot_vec_opt)
    fc_dict[slot_id] = FeatureColumnV1(fs)

# get ue
ue_name = 'fc_aweme_finish_61294_uid_d128'
fc_aweme_ue = FeatureColumnDense(ue_name, 128)
fc_aweme_ue_raw = fc_aweme_ue.get_tensor()
tf.summary.histogram(ue_name, fc_aweme_ue_raw)

# Start to build model
towers = []

bias_input = M.get_bias([fc_dict[sid] for sid in VALID_SLOTS])

# Sum bias
sum_bias = tf.reduce_sum(bias_input, axis=1, keepdims=True)

# Bias NN tower
if len(BIAS_NN_DIMS) > 0:
    bias_nn_input = M.get_bias([fc_dict[sid] for sid in BIAS_NN_SLOTS])
    if STOP_GRAD:
        bias_nn_input = tf.stop_gradient(bias_nn_input)
    print('bias_nn_input: {}'.format(bias_nn_input))
    bias_nn_out = modules.DenseTower(name='bias_nn_tower',
                                     output_dims=BIAS_NN_DIMS,
                                     initializers=get_init_funcs([bias_nn_input.shape[1]] + BIAS_NN_DIMS)
                                     )(bias_nn_input)
    towers.append(bias_nn_out)
    tf.summary.histogram('bias_nn_out', bias_nn_out)

# Graph Embedding
if USE_GE:
    # we now only have one ge, so...
    # ge_list = []
    # ge_total_dim = 0

    GE_UID_COMPRESS_DIMS = [256, 128, 128]

    STOP_SLOT_GRADIENT = False  # equal to UE if set to true
    DROP_FEATURE = False

    GE_USER_DIM = 64
    GE_AUTHOR_DIM = 64

    GE_UID_MAPPING_SLOT = 10000
    GE_UID_SERVING_SLOT = 9999
    GE_AUTHOR_MAPPING_SLOT = 10001
    FID_VERSION = 2

    EDGE_TYPE = [0]  # 0: user-author
    NEIGHBOR_COUNT = 6
    NEIGHBOR_COUNT_L2 = 20

    OCCURRENCE_THRESHOLD = 2
    DEFAULT_EXPIRE_TIME = 90

    if is_training:
        # get uid
        fc_uid = FeatureColumnDense('fc_line_id_uid', 1, tf.int64)
        fc_uid_tensor = fc_uid.get_tensor()
        fc_ut = FeatureColumnDense('fc_line_id_ut', 1, tf.int64)
        fc_ut_tensor = fc_ut.get_tensor()

        ### GCN-L2 based on uid and its neighbors

        # fetch user residual embedding
        fid_extractor = FIDExtractor(slot_id=GE_UID_MAPPING_SLOT, fid_version=FID_VERSION, invalid_fid=0)
        fid_tensor = fid_extractor(fc_uid_tensor)

        # fetch feature from PS
        fs = FeatureSlot(slot_id=GE_UID_MAPPING_SLOT,
                         vec_optimizer=optimizers.ps.AdaGrad(alpha=0.05, beta=1.0, init_factor=0.015625,
                                                             # alpha=0.025, init_factor=0.0625
                                                             weight_decay=0.001),
                         occurrence_threshold=OCCURRENCE_THRESHOLD,
                         expire_time=DEFAULT_EXPIRE_TIME)  # get_expire_time(GE_UID_MAPPING_SLOT))
        fc = FeatureColumnV2('fc_ge_user_vec_{}'.format(GE_UID_MAPPING_SLOT), fs)
        fs.add_slice(GE_USER_DIM)
        fetcher = FeatureFetcher(GE_UID_MAPPING_SLOT, 'ge_user_vec_fetcher', GE_USER_DIM, 0, False)
        _, user_emb_vec = fetcher(sp_ids=fid_tensor)
        user_embedding = user_emb_vec

        # sample uid's neighbors from euler graph
        neighbors = \
            euler_ops.sample_fanout(fc_uid_tensor, [EDGE_TYPE, EDGE_TYPE], [NEIGHBOR_COUNT, NEIGHBOR_COUNT_L2], 0)[0]
        neighbors_l2 = neighbors[2]

        fid_neighbors_l2 = fid_extractor(neighbors_l2)
        neighbors_l2_emb_bias, neighbors_l2_emb_vec = fetcher(sp_ids=fid_neighbors_l2)
        neighbors_l2_emb_vec = tf.reshape(neighbors_l2_emb_vec, [-1, NEIGHBOR_COUNT, NEIGHBOR_COUNT_L2, GE_USER_DIM])
        neighbors_l2_emb_vec = tf.reduce_mean(neighbors_l2_emb_vec, axis=2)

        neighbors = neighbors[1]

        # extract neighbor to author id
        fid_extractor = FIDExtractor(slot_id=GE_AUTHOR_MAPPING_SLOT, fid_version=FID_VERSION, invalid_fid=0,
                                     only_slot_mapping=True)
        fid_tensor = fid_extractor(neighbors)

        # fetch feature from PS
        fs = FeatureSlot(slot_id=GE_AUTHOR_MAPPING_SLOT,
                         vec_optimizer=optimizers.ps.AdaGrad(alpha=0.05, beta=1.0, init_factor=0.015625,
                                                             # alpha=0.025, init_factor=0.0625
                                                             weight_decay=0.001),
                         occurrence_threshold=OCCURRENCE_THRESHOLD,
                         expire_time=DEFAULT_EXPIRE_TIME)  # get_expire_time(GE_AUTHOR_MAPPING_SLOT))
        fc = FeatureColumnV2('fc_ge_author_vec_{}'.format(GE_AUTHOR_MAPPING_SLOT), fs)
        fs.add_slice(GE_AUTHOR_DIM)
        fetcher = FeatureFetcher(GE_AUTHOR_MAPPING_SLOT, 'ge_group_vec_fetcher', GE_AUTHOR_DIM, 0)
        _, author_emb_vec = fetcher(sp_ids=fid_tensor)
        author_embedding = author_emb_vec
        author_embedding = tf.reshape(author_embedding, [-1, NEIGHBOR_COUNT, GE_AUTHOR_DIM])
        author_embedding = tf.concat([author_embedding, neighbors_l2_emb_vec], axis=-1)
        uid_conv_emb = graph_convolution_tower(emb_vec=user_embedding,
                                               neighbor_emb_vec=author_embedding,
                                               tower_dims=GE_UID_COMPRESS_DIMS,
                                               name='user_conv_tower',
                                               stop_grad=STOP_SLOT_GRADIENT,
                                               drop_out=DROP_FEATURE)

        uid_ge_emb = uid_conv_emb

        # extract uid to fid and map slot to GE_UID_SERVING_SLOT
        uid_extractor = UIDExtractor(slot_id=GE_UID_SERVING_SLOT, fid_version=FID_VERSION)
        uid_fid = uid_extractor([fc_uid_tensor, fc_ut_tensor])

        # serving slot is not trainable
        fs = FeatureSlot(slot_id=GE_UID_SERVING_SLOT,
                         vec_optimizer=optimizers.ps.AdaGrad(alpha=.0, beta=10000., init_factor=0.0),
                         expire_time=DEFAULT_EXPIRE_TIME)  # get_expire_time(GE_UID_SERVING_SLOT))
        fc = FeatureColumnV2('fc_ge_finish_gcn_uid', fs)
        fc.get_tensor().trainable = False
        fs.add_slice(GE_UID_COMPRESS_DIMS[-1])

        all_true_mask = tf.fill(tf.shape(uid_fid), True)

        feature_setter = FeatureSetter(dst_fid_version=2, dst_feature_slot=GE_UID_SERVING_SLOT)
        setter_output = feature_setter([uid_fid, uid_ge_emb, all_true_mask])
    else:
        # serving slot is not trainable
        fs = FeatureSlot(slot_id=GE_UID_SERVING_SLOT,
                         vec_optimizer=optimizers.ps.AdaGrad(alpha=.0, beta=10000., init_factor=0.0),
                         expire_time=DEFAULT_EXPIRE_TIME)  # get_expire_time(GE_UID_SERVING_SLOT))
        fc = FeatureColumnV2('fc_ge_finish_gcn_uid', fs)
        fc.get_tensor().trainable = False
        slice1 = fs.add_slice(GE_UID_COMPRESS_DIMS[-1])
        uid_ge_emb = fc.get_vector(slice1)

# FFM tower
ffm_inputs = []
for x, y, dim in FM_SLOTS:
    vec_x = [fc_dict[sid].add_vector(dim) for sid in S.generic_utils.to_list(x)]
    vec_y = [fc_dict[sid].add_vector(dim) for sid in S.generic_utils.to_list(y)]
    ffm_inputs.append((vec_x, vec_y))

# UE FFM
for y, dim in UE_FM_SLOTS:
    ue = compress_embedding(fc_aweme_ue_raw, [128, 128], 'ue_in_fm_{}'.format(y))
    vec_y = [fc_dict[sid].add_vector(dim) for sid in S.generic_utils.to_list(y)]
    ffm_inputs.append((ue, vec_y))

# uid GE FFM, consider adding a separate GE_FM_SLOTS including more slots
if USE_GE:
    for y, dim in UE_FM_SLOTS:
        ge = compress_embedding(uid_ge_emb, [128, 128], 'ge_in_fm_{}'.format(y))
        vec_y = [fc_dict[sid].add_vector(dim) for sid in S.generic_utils.to_list(y)]
        ffm_inputs.append((ge, vec_y))

# room slot embedding NN
all_room_slot_vec_list = []
const_room_slot_dim = 32
for slot_id in ROOM_TOWER_SLOTS:
    vec_list = [fc_dict[sid].add_vector(const_room_slot_dim) for sid in S.generic_utils.to_list(slot_id)]
    all_room_slot_vec_list.extend(vec_list)
room_slot_prod_concat = tf.concat(all_room_slot_vec_list, axis=1)
if len(ROOM_TOWER_SLOTS) > 0:
    if STOP_GRAD:
        room_slot_prod_concat = tf.stop_gradient(room_slot_prod_concat)
    print('room_slot_prod_concat: {}'.format(room_slot_prod_concat))
    room_slot_prod_concat_dim = int(room_slot_prod_concat.shape[1])
    room_nn_out = modules.DenseTower(name='room_nn_tower',
                                     output_dims=DEEP_ROOM_NN_DIMS,
                                     initializers=get_init_funcs([room_slot_prod_concat.shape[1]] + DEEP_ROOM_NN_DIMS)
                                     )(room_slot_prod_concat)
    tf.summary.histogram('room_nn_out', room_nn_out)

for x in ROOM_FM_SLOTS:
    vec_x = [fc_dict[sid].add_vector(ROOM_SLOT_DIM) for sid in S.generic_utils.to_list(x)]
    ffm_inputs.append((vec_x, room_nn_out))

_, dots, ffm_prods = modules.FFM()(ffm_inputs)

ffm_prods_concat = tf.concat(ffm_prods, axis=1)
dfm_out = modules.DenseTower(name='dfm_tower',
                             output_dims=DEEP_FM_DIMS,
                             initializers=get_init_funcs([ffm_prods_concat.shape[1]] + DEEP_FM_DIMS)
                             )(ffm_prods_concat)
towers.append(dfm_out)

# Original FM part
if KEEP_ORIGINAL_FM:
    towers.extend(dots)

# select concat nn， alloc_slot_vec   -> UE and GE included
if len(SELECT_CONCAT_SLOTS) > 0:

    fc_aweme_ue_tensor = compress_embedding(fc_aweme_ue_raw, [128, 128], ue_name)  # UE
    if USE_GE:
        ge_emb_tensor = compress_embedding(uid_ge_emb, [128, 128], 'ge_in_concat')  # GE

    concat_embeddings = []
    for slot_id, dim in SELECT_CONCAT_SLOTS:
        vec = fc_dict[slot_id].add_vector(dim)
        concat_embeddings.append(vec)

    concat_embeddings.append(fc_aweme_ue_tensor)  # UE
    if USE_GE:
        concat_embeddings.append(ge_emb_tensor)  # GE

    embed_concat = tf.concat(concat_embeddings, axis=1)
    deep_concat_out = modules.DenseTower(name='select_deepconcat_tower',
                                         output_dims=SELECT_DEEP_CONCAT_DIMS,
                                         initializers=get_init_funcs([embed_concat.shape[1]] + SELECT_DEEP_CONCAT_DIMS)
                                         )(embed_concat)
    towers.append(deep_concat_out)
    tf.summary.histogram('select_deep_concat_out', deep_concat_out)

combine_out = tf.concat(towers + [sum_bias], axis=1)

# computing lables and losses
staytime_label = M.get_label('staytime', 0)
fc_chnid = FeatureColumnDense('fc_line_id_chnid', 1, tf.int64)
fc_chnid_tensor = tf.cast(fc_chnid.get_tensor(), staytime_label.dtype)
channel_id_label = tf.reshape(fc_chnid_tensor, [-1])

ctr_label = tf.where(tf.greater_equal(staytime_label, 1), tf.ones_like(staytime_label), tf.zeros_like(staytime_label))
ctr10_sub_label = tf.where(tf.greater_equal(staytime_label, 10000), tf.ones_like(staytime_label),
                           tf.zeros_like(staytime_label))
ctr30_sub_label = tf.where(tf.greater_equal(staytime_label, 30000), tf.ones_like(staytime_label),
                           tf.zeros_like(staytime_label))
label_transformed_staytime = tf.log(staytime_label * 0.001 + 1)

target_name = "ctr"
loss_type = "logloss"
channel_id_2_label = {
    21: ctr10_sub_label,
    36: ctr10_sub_label,
    10: ctr10_sub_label,
    1: ctr10_sub_label,
    46: ctr10_sub_label,
    37: ctr10_sub_label,
    0: ctr_label,
}
INVALID_LABEL = tf.ones_like(channel_id_label) * tf.float32.min

run_predict = M.RunStep(name='predict_online', run_type='INFERENCE')
run_predict.add_feeds(M.get_all_input_tensors())

# Assign RunSteps for training
run_train = M.RunStep(name='oracle_train',
                      run_type='TRAIN')
run_train.add_feeds(M.get_all_input_tensors()). \
    add_gradients()

losses = []
for id, name in channel_id_2_channel_name.items():
    out = modules.DenseTower(name='combine_tower_{}'.format(name),
                             output_dims=COMBINE_TOWER_DIMS,
                             initializers=get_init_funcs([combine_out.shape[1]] + COMBINE_TOWER_DIMS),
                             )(combine_out)

    logit = tf.reduce_sum(out, axis=1)
    real_single_label = channel_id_2_label[id]
    pred, loss = get_pred_and_loss(logit, real_single_label, M.get_sample_rate(), channel_id_label, id, loss_type)

    run_predict.add_head(name='{}_{}'.format(name, target_name), prediction=pred)

    label2metric = tf.where(tf.equal(channel_id_label, id), real_single_label, INVALID_LABEL)
    run_train.add_head(name='{}_{}'.format(name, target_name),
                       prediction=pred,
                       label=label2metric,
                       loss=loss,
                       sample_rate=M.get_sample_rate())
    losses.append(loss)

M.set_global_gradient_clip_norm(250.0)

if USE_GE and is_training and not euler_ops.initialize_graph({'mode': 'Remote',
                                                              'zk_server': '10.6.15.38:2181',
                                                              'zk_path': '/tf_euler_application_aweme_live_u2a_1d_v1'}):
    raise RuntimeError('Failed to initialize graph in worker.')

# compile the whole model for training
M.compile(default_hidden_layer_optimizer=optimizers.ps.AdaGrad(alpha=0.1, beta=262144, init_factor=1),
          run_steps=[run_train, run_predict],
          losses=losses, loss_weights=[1., ] * len(channel_id_2_channel_name),
          num_estimated_bias_features=500000000,
          num_estimated_vec_features=500000,
          cold_feature_filter_capacity=20000000)
