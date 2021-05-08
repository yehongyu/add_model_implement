# -*- encoding=utf-8 -*-
# aweme followfeed nn model
from feature import *
import sys
import math
import datetime
import tensorflow as tf

import sail.common as S
import sail.model as M

from sail import layers
from sail import losses
from sail import initializers
from sail import optimizers
from sail import modules
from sail.feature import FeatureSlot
from sail.layers import *
from sail.feature import FeatureColumnV1

BIAS_DIM = 4
COMBINE_DIM = 64
COMBINE_TOWER_DIMS = [16, 1]
BIAS_NN_DIMS = [1024, 256, COMBINE_DIM]
DEEP_FM_DIMS = [256, COMBINE_DIM]
GRAD_NORM = True


class Target(object):
    def __init__(self, name, idx, loss_type, scale=1.0, filter_target=str()):
        self._name = name
        self._idx = idx
        self._loss_type = loss_type
        self._scale = scale
        self._label = M.get_label(name, idx)
        self._filter_target = filter_target


# target顺序对应selected_multi_labels，不能直接更改 skip,like,share,comment,click_comment,finish,headslideleft,download
targets = [
    Target('skip', 0, losses.binary_cross_entropy),
    Target('like', 1, losses.binary_cross_entropy),
    Target('share', 2, losses.binary_cross_entropy),
    Target('comment', 3, losses.binary_cross_entropy),
    Target('click_comment', 4, losses.binary_cross_entropy),
    Target('finish', 5, losses.binary_cross_entropy),
    Target('headslideleft', 6, losses.binary_cross_entropy),
    Target('download', 7, losses.binary_cross_entropy),

]

slots = set(VALID_SLOTS)

# nn init & opt
nn_init = initializers.GlorotNormal(mode="fan_in")
nn_opt = optimizers.ps.RMSPropV2(lr=0.001, momentum=0.999999, weight_decay=0.0)

FeatureSlot.set_default_vec_optimizer(optimizers.ps.AdaGrad(alpha=0.1, beta=1.0, init_factor=0.00015625))
FeatureSlot.set_default_bias_optimizer(optimizers.ps.FTRL(alpha=0.05, beta=1.0))
FeatureSlot.set_default_occurrence_threshold(7)
FeatureSlot.set_default_expire_time(0)

vec_optimizer_setting1 = optimizers.ps.AdaGrad(alpha=0.05, beta=1.0, init_factor=0.01)
vec_optimizer_setting2 = optimizers.ps.AdaGrad(alpha=0.1, beta=1.0, init_factor=0.00015625)
vec_optimizer_setting3 = optimizers.ps.AdaGrad(alpha=0.01, beta=1.0, init_factor=0.01)
bias_optimizer_setting = optimizers.ps.FTRL(alpha=0.05, beta=1.0)

# 设置训练参数
fc_dict = {}
for slot in slots:
    if slot in SLOT_CONFIG_GROUPS1:
        fs = FeatureSlot(
            slot_id=slot,
            vec_optimizer=vec_optimizer_setting1,
            bias_optimizer=bias_optimizer_setting,
            occurrence_threshold=7
        )
    elif slot in SLOT_CONFIG_GROUPS2:
        fs = FeatureSlot(
            slot_id=slot,
            vec_optimizer=vec_optimizer_setting2,
            bias_optimizer=bias_optimizer_setting,
            occurrence_threshold=7
        )
    elif slot in SLOT_CONFIG_GROUPS3:
        fs = FeatureSlot(
            slot_id=slot,
            vec_optimizer=vec_optimizer_setting3,
            bias_optimizer=bias_optimizer_setting,
            occurrence_threshold=7
        )
    else:
        print("left slot", slot)
        fs = FeatureSlot(slot_id=slot)
    fc_dict[slot] = FeatureColumnV1(fs)

bias_in_ = []

spvip_slots = set([1, 2, 201])
vip_slots = set([202, 204, 205])
for slot in BIAS_NN_SLOTS:
    if slot in spvip_slots:
        bias_in_.append(fc_dict[slot].add_vector(128))
    elif slot in vip_slots:
        bias_in_.append(fc_dict[slot].add_vector(64))
    else:
        bias_in_.append(fc_dict[slot].add_vector(BIAS_DIM))
bias_input = tf.concat(bias_in_, axis=1)
bias_nn_out = modules.DenseTower(name="bias_nn_tower", output_dims=BIAS_NN_DIMS, initializers=nn_init)(bias_input)

tf.summary.histogram("bias_nn_out", bias_nn_out)

towers_out = bias_nn_out  # only bias_nn_out

run_predict = M.RunStep(name="predict", run_type="INFERENCE")
run_predict.add_feeds(M.get_all_input_tensors())

run_train = M.RunStep(name="train", run_type="TRAIN")
run_train.add_feeds(M.get_all_input_tensors()).add_gradients()

all_losses = []
all_loss_weights = []
all_loss_names = []
sample_rate = M.get_sample_rate()
all_target_names = [t._name for t in targets]
all_label = [t._label for t in targets]
all_label_dict = dict(zip(all_target_names, all_label))

for idx, target in enumerate(targets):
    name = target._name
    label = target._label
    logit = modules.DenseTower(
        name="{}_tower".format(name),
        output_dims=COMBINE_TOWER_DIMS,
        initializers=nn_init
    )(towers_out)
    logit = tf.reduce_sum(logit, axis=1)
    target_pred = tf.nn.sigmoid(logit)
    loss_name = "{}_loss".format(name)
    mask = label > -10000
    blabel = tf.boolean_mask(label, mask)
    blogit = tf.boolean_mask(logit, mask)
    target_loss = target._loss_type(
        y_label=blabel, y_pred=blogit, name=loss_name
    )
    all_losses.append(target_loss)
    all_loss_weights.append(target._scale)
    all_loss_names.append(loss_name)
    run_predict.add_head(name=name, prediction=target_pred)
    run_train.add_head(
        name=name,
        prediction=target_pred,
        label=label,
        loss=None,  # deprecated
        sample_rate=sample_rate
    )
    tf.summary.scalar("{}/label".format(name), tf.reduce_mean(blabel))
    tf.summary.scalar("{}/predict".format(name), tf.reduce_mean(target_pred))
    tf.summary.scalar("loss/{}".format(name), tf.reduce_sum(target_loss))

if GRAD_NORM:
    grad_norm = modules.GradNorm(loss_names=all_loss_names, scale=1e3)
    gnorm_loss, weighted_loss = grad_norm(all_losses, shared_inputs=towers_out)
    loss = tf.add_n([gnorm_loss, weighted_loss * len(targets)])
    tf.summary.scalar("loss/grad", gnorm_loss)
    tf.summary.scalar("loss/wloss", weighted_loss)
else:
    loss = tf.reduce_sum(tf.multiply(all_losses, all_loss_weights))
tf.summary.scalar("loss/total", loss)

M.compile(
    default_hidden_layer_optimizer=nn_opt,
    run_steps=[run_train, run_predict],
    losses=[loss, ],
    loss_weights=[1., ],
    num_estimated_bias_features=12000000,
    num_estimated_vec_features=120000,
    cold_feature_filter_capacity=20000000
)