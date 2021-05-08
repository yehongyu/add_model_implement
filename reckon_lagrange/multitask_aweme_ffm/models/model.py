# -*- encoding=utf-8 -*-

"""Aweme Multitask model v6 (MMOE/SNR/SB)
"""

from common import *

# expert parameter
EXPERT_SHARE_INPUTS = True
TASK_BIAS = False  # task bias = 1

# gate parameter
'''
dynamic_mmoe, dynamic_snr, static_mmoe, static_snr, sb
'''
GATE_METHOD = 'sb'
BIAS_DIM = 4  # for experts
EXPERTS_NUM = 1

# grad norm
GRAD_NORM = True

# snr l0_regularization
L0REG = 0.0001
BETA = 0.5
GAMMA = -0.1
ETA = 1.1

# others
BUCKET_NUM = 500
COMBINE_DIM = 64
COMBINE_METHOD = 'concat'
COMBINE_TOWER_DIMS = [16, 1]
SOFTMAX_COMBINE_TOWER_DIMS = [64, BUCKET_NUM]
BIAS_NN_DIMS = [256, COMBINE_DIM]
DEEP_FM_DIMS = [256, COMBINE_DIM]
GATE_DIMS = [128, EXPERTS_NUM]
USE_RMSPROP = True
DENSE_ALPHA = 0.001
GLOBAL_GRAD_NORM_CUTOFF = 5000

# parameter checking
# if GATE_METHOD == 'sb':
#    assert(EXPERTS_NUM == 1)

# equally divide the parameters into each expert
if not EXPERT_SHARE_INPUTS:
    for (x, y, dim) in FM_SLOTS:
        if dim % EXPERTS_NUM != 0:
            raise ValueError('invalid vec dim: (%s,%s,%s)' % (x, y, dim))
    if BIAS_DIM % EXPERTS_NUM != 0:
        raise ValueError('invalid bias dim: %s' % BIAS_DIM)
    FM_SLOTS = [(x, y, dim / EXPERTS_NUM) for (x, y, dim) in FM_SLOTS]
    BIAS_DIM = BIAS_DIM / EXPERTS_NUM


class Target(object):
    def __init__(self, name, loss_type, transform_type=None, scale=1.0, label_name=None, dims=COMBINE_TOWER_DIMS):
        self._name = name
        self._loss_type = loss_type
        self._transform_type = transform_type
        self._scale = scale
        self._label_name = label_name or name
        self._dims = dims

    def _transform(self, label):
        if self._transform_type is None:
            return label
        elif self._transform_type[:4] == 'pow-':
            pow_value = float(self._transform_type[4:])
            return tf.pow(label, pow_value)
        else:
            raise ValueError('Unsupportted transform: {}'.format(self._transform_type))

    def _get_label(self, labels_dict):
        if isinstance(self._label_name, str):
            return labels_dict[self._label_name]
        elif isinstance(self._label_name, list):
            # Take max one as label from multiple sources
            label = tf.reduce_max(tf.stack([labels_dict[name] for name in self._label_name], axis=1), axis=1)
            print('label from multiple indexes: {}'.format(label))
            return label
        else:
            raise ValueError('Unsupportted label_name: {}'.format(self._label_name))

    @property
    def name(self):
        return self._name

    @property
    def loss_type(self):
        return self._loss_type

    @property
    def scale(self):
        return self._scale

    @property
    def dims(self):
        return self._dims

    @property
    def label_name(self):
        return self._label_name

    def get_label(self, labels_dict):
        label = tf.identity(self._transform(self._get_label(labels_dict)), name='label_{}'.format(self._name))
        print('label for {}: {}'.format(self._name, label))
        return label


def get_label_dict(parser, targets):
    unique_names = []
    for target in targets:
        label_names = target.label_name
        if not isinstance(label_names, list):
            label_names = [label_names]
        for label_name in label_names:
            if label_name not in unique_names:
                unique_names.append(label_name)
    labels_dict = dict(zip(unique_names, parser.get_label(unique_names)))
    return labels_dict


'''
feedbacks:
1. 8xbias, share input, mmoe
  download:-0.1
  dislike:-0.7
  cover:-0.2
  headslideleft:-0.1
  shoot:-0.1
2.
'''
TARGETS = [
    Target('finish', 'logloss'),
    # Target('pc', 'mse', 'pow-0.3'),
    # Target('pc_softmax', 'softmax', label_name='pc', dims=SOFTMAX_COMBINE_TOWER_DIMS),
    # Target('staytime', 'mse', 'pow-0.3', 0.1),
    Target('skip', 'logloss'),
    Target('click_comment', 'logloss'),

    # stage1
    Target('like', 'logloss'),
    Target('comment', 'logloss'),
    # Target('click_comment', 'logloss'),#, scale = 2),

    # stage2
    Target('share', 'logloss'),
    Target('download', 'logloss'),  # , scale = 2),
    # Target('shoot', 'logloss'), #, scale = 2),
    # Target('cover', 'logloss'),#, scale = 2),

    # stage3
    Target('headslideleft', 'logloss'),  # , scale = 2),

]


def get_bias_input(parser, slots, n):
    return tf.concat([parser.alloc_slot_vec(slot, n) for slot in slots], axis=1)


def get_tf_net(parser):
    parser.set_valid_slots(VALID_SLOTS)
    sample_rate = parser.get_sample_rate()

    # build inputs
    def generate_input():

        # bias emb
        bias_emb = get_bias_input(parser, BIAS_NN_SLOTS, BIAS_DIM)

        # fm emb
        all_vec_list = []
        all_prod_list = []
        all_dot_list = []

        for x, y, dim in FM_SLOTS:
            def get_vec(slot, dim):
                if isinstance(slot, list):
                    vec_list = []
                    for slot_id in slot:
                        vec_list.append(parser.alloc_slot_vec(slot_id, dim))
                    vec = tf.add_n(vec_list)
                else:
                    vec = parser.alloc_slot_vec(slot, dim)
                    vec_list = [vec]
                return vec, vec_list

            x_vec, x_list = get_vec(x, dim)
            y_vec, y_list = get_vec(y, dim)

            all_vec_list.extend(x_list)
            all_vec_list.extend(y_list)

            prod = tf.multiply(x_vec, y_vec)
            all_prod_list.append(prod)

            # dot = tf.reduce_sum(prod, axis=1)
            # all_dot_list.append(dot)

        prod_emb = tf.concat(all_prod_list, axis=1)
        vec_emb = tf.concat(all_vec_list, axis=1)
        # dot_emb = tf.concat(all_dot_list, axis = 1)
        # all_emb = tf.concat([prod_emb, vec_emb, dot_emb, bias_emb], axis = 1)
        all_emb = tf.concat([prod_emb, vec_emb, bias_emb], axis=1)

        print('vec_emb: {}'.format(vec_emb))

        return {
            'bias': bias_emb,
            'prod': prod_emb,
            'vec': vec_emb,
            # 'dot' : dot_emb,
            'all': all_emb
        }

    # Expert network
    def expert_net(input_dict, name=''):
        """Build two towers (BiasNN+DFM) for each expert
        """
        bias_nn_input = input_dict['bias']
        prod_concat = input_dict['prod']

        towers = []

        # Bias NN tower
        if len(BIAS_NN_DIMS) > 0:
            bias_nn_input_dim = int(bias_nn_input.shape[1])
            bias_nn_out = lgm.deep_tower(name='{}_nn_tower'.format(name),
                                         input_t=bias_nn_input,
                                         input_dim=bias_nn_input_dim,
                                         output_dims=BIAS_NN_DIMS,
                                         init_funcs=get_init_funcs([bias_nn_input_dim] + BIAS_NN_DIMS))
            towers += [bias_nn_out]
            tf.summary.histogram('bias_nn_out', bias_nn_out)

        # Deep FM tower
        if len(DEEP_FM_DIMS) > 0:
            prod_concat_dim = int(prod_concat.shape[1])
            dfm_out = lgm.deep_tower(name='{}_dfm_tower'.format(name),
                                     input_t=prod_concat,
                                     input_dim=prod_concat_dim,
                                     output_dims=DEEP_FM_DIMS,
                                     init_funcs=get_init_funcs([prod_concat_dim] + DEEP_FM_DIMS))
            towers += [dfm_out]
            tf.summary.histogram('dfm_out', dfm_out)

        return tf.concat(towers, axis=1)

    # Gate network
    def static_mmoe_gate_net(exp_out, gate_cond, name):
        '''
        this gate is not conditioned on `gate_cond`
        '''
        exp_out = tf.stack(exp_out, axis=2)
        print('exp_out: {}'.format(exp_out))

        gate_out = lg.Variable(name='mmoe_gate_{}'.format(name), initial_value=tf.zeros([1, EXPERTS_NUM]))
        gate_out = tf.tile(gate_out, [tf.shape(exp_out)[0], 1])
        gate_out = tf.nn.softmax(gate_out, axis=1)

        exp_out = tf.matmul(exp_out, tf.expand_dims(gate_out, -1))
        exp_out = tf.squeeze(exp_out, axis=2)
        return exp_out, []

    # Gate network
    def sb_gate_net(exp_out, gate_cond, name):
        '''
        this gate is not conditioned on `gate_cond`
        '''
        exp_out = tf.stack(exp_out, axis=2)
        print('exp_out: {}'.format(exp_out))

        gate_out = tf.ones([1, EXPERTS_NUM])
        gate_out = tf.tile(gate_out, [tf.shape(exp_out)[0], 1])

        exp_out = tf.matmul(exp_out, tf.expand_dims(gate_out, -1))
        exp_out = tf.squeeze(exp_out, axis=2)

        return exp_out, []

    def static_snr_gate_net(input_list, gate_cond, name):
        '''
        this gate is not conditioned on `gate_cond`
        '''
        input_list = [lgm.deep_tower(name='snr_trans_{}_{}'.format(name, i),
                                     input_t=x,
                                     input_dim=int(x.shape[1]),
                                     output_dims=[int(x.shape[1])],
                                     use_weight_norm=False,
                                     use_learnable_weight_norm=False,
                                     use_bias=False)
                      for i, x in enumerate(input_list)]

        snr_input = tf.stack(input_list, axis=2)
        bs = tf.shape(snr_input)[0]
        input_dim = int(snr_input.shape[1])

        n = len(input_list)
        u = tf.random_uniform(shape=[1, n])
        log_alpha = lg.Variable(tf.zeros(shape=[1, n]), name='snr_log_alpha_{}'.format(name))
        s = tf.sigmoid((tf.log(u) - tf.log(1 - u) + log_alpha) / BETA)
        s_ = s * (ETA - GAMMA) + GAMMA
        z = tf.minimum(1.0, tf.maximum(s_, 0.0))
        tf.summary.histogram('snr_z_{}'.format(name), z)

        z = tf.tile(z, [bs, 1])

        # apply gate
        nn_input = tf.matmul(snr_input, tf.expand_dims(z, -1))
        exp_out = tf.squeeze(nn_input, axis=2)

        # l0 regularize
        l0_reg = tf.reduce_sum(tf.sigmoid(log_alpha - BETA * math.log(- GAMMA / ETA)))
        tf.summary.scalar('snr_l0_reg_{}'.format(name), l0_reg)

        return exp_out, [l0_reg * L0REG]

    def dynamic_mmoe_gate_net(exp_out, gate_cond, name):
        exp_out = tf.stack(exp_out, axis=2)
        all_concat_dim = int(gate_cond.shape[1])
        gate_out = lgm.deep_tower(name='{}_gate'.format(name),
                                  input_t=gate_cond,
                                  input_dim=int(gate_cond.shape[1]),
                                  output_dims=GATE_DIMS,
                                  init_funcs=get_init_funcs([all_concat_dim] + GATE_DIMS))
        gate_out = tf.nn.softmax(gate_out, axis=1)

        for i in range(EXPERTS_NUM):
            tf.summary.histogram('mmoe_gate_{}_{}'.format(name, i), gate_out[:, i])

        exp_out = tf.matmul(exp_out, tf.expand_dims(gate_out, -1))
        exp_out = tf.squeeze(exp_out, axis=2)
        return exp_out, []

    def dynamic_snr_gate_net(input_list, gate_cond, name):
        # create log_alpha
        all_concat_dim = int(gate_cond.shape[1])
        log_alpha = lgm.deep_tower(name='{}_gate'.format(name),
                                   input_t=gate_cond,
                                   input_dim=int(gate_cond.shape[1]),
                                   output_dims=GATE_DIMS,
                                   init_funcs=get_init_funcs([all_concat_dim] + GATE_DIMS))
        log_alpha = tf.reshape(log_alpha, [1, len(input_list)])

        # generate gate
        input_list = [lgm.deep_tower(name='snr_trans_{}_{}'.format(name, i),
                                     input_t=x,
                                     input_dim=int(x.shape[1]),
                                     output_dims=[int(x.shape[1])],
                                     use_weight_norm=False,
                                     use_learnable_weight_norm=False,
                                     use_bias=False)
                      for i, x in enumerate(input_list)]
        snr_input = tf.stack(input_list, axis=2)
        bs = tf.shape(snr_input)[0]
        input_dim = int(snr_input.shape[1])

        n = len(input_list)
        u = tf.random_uniform(shape=[1, n])
        s = tf.sigmoid((tf.log(u) - tf.log(1 - u) + log_alpha) / BETA)
        s_ = s * (ETA - GAMMA) + GAMMA
        z = tf.minimum(1.0, tf.maximum(s_, 0.0))
        tf.summary.histogram('snr_z_{}'.format(name), z)
        z = tf.tile(z, [bs, 1])

        # apply gate
        nn_input = tf.matmul(snr_input, tf.expand_dims(z, -1))
        exp_out = tf.squeeze(nn_input, axis=2)
        print('nn_input: {}'.format(nn_input))

        # l0 regularize
        l0_reg = tf.reduce_sum(tf.sigmoid(log_alpha - BETA * math.log(- GAMMA / ETA)))
        tf.summary.scalar('snr_l0_reg_{}'.format(name), l0_reg)

        return exp_out, [l0_reg * L0REG]

    # Task specified network
    def task_net(exp_out, dims, name, gate_cond, gate_method='dynamic_mmoe'):
        """Build task specified network based experts output
        """
        # gate exp_out
        assert (len(exp_out) == EXPERTS_NUM)
        if EXPERTS_NUM == 1:
            exp_out, reg_loss = exp_out[0], []
        elif gate_method == 'dynamic_mmoe':
            exp_out, reg_loss = dynamic_mmoe_gate_net(exp_out, gate_cond, name)
        elif gate_method == 'static_mmoe':
            exp_out, reg_loss = static_mmoe_gate_net(exp_out, gate_cond, name)
        elif gate_method == 'static_snr':
            exp_out, reg_loss = static_snr_gate_net(exp_out, gate_cond, name)
        elif gate_method == 'dynamic_snr':
            exp_out, reg_loss = dynamic_snr_gate_net(exp_out, gate_cond, name)
        elif gate_method == 'sb':
            exp_out, reg_loss = sb_gate_net(exp_out, gate_cond, name)
        else:
            raise ValueError('unknown gate method: {}'.format(gate_method))

        # allocate bias for each task
        if TASK_BIAS:
            bias_input = get_bias_input(parser, BIAS_NN_SLOTS, 1)
            exp_out = tf.concat([exp_out, bias_input], axis=1)

        # task net output
        exp_out_dim = int(exp_out.shape[1])
        task_out = lgm.deep_tower(name='',
                                  input_t=exp_out,
                                  input_dim=exp_out_dim,
                                  output_dims=dims,
                                  init_funcs=get_init_funcs([exp_out_dim] + dims))

        return task_out, reg_loss

    # create inputs
    if EXPERT_SHARE_INPUTS:
        expert_inputs = [generate_input()] * EXPERTS_NUM
        gate_input = expert_inputs[0]['all']
    else:
        expert_inputs = [generate_input() for i in range(EXPERTS_NUM)]
        gate_input = tf.concat([i['all'] for i in expert_inputs], axis=1)

    # create experts
    experts = [
        expert_net(expert_inputs[i], 'expert_{}'.format(i))
        for i in range(EXPERTS_NUM)]
    experts_concat = tf.concat(experts, axis=1)

    labels_dict = get_label_dict(parser, TARGETS)
    reg_losses, obj_losses, preds, gnorms = [], [], [], []

    # create target network
    for i, target in enumerate(TARGETS):
        name = target.name
        loss_type = target.loss_type
        scale = target.scale
        dims = target.dims

        label = target.get_label(labels_dict)
        bs = tf.to_float(tf.shape(label)[0])

        task_out, reg_loss = task_net(experts, dims, name, gate_input, GATE_METHOD)

        pred, loss = get_pred_and_loss(task_out, label, sample_rate, loss_type)
        pred = tf.identity(pred, name='oracle_pred_{}'.format(name))

        if not GRAD_NORM:
            loss = loss * scale

        obj_losses += [loss]
        reg_losses += reg_loss

        # grad norm
        if GRAD_NORM:
            # batch size
            bs = tf.to_float(tf.shape(experts[0])[0])

            # caculate share part's gradient wrt task loss
            grad = tf.concat(tf.gradients(loss, experts), axis=1)
            grad_norm = (tf.reduce_sum(tf.multiply(grad, grad)) / bs) ** 0.5
            gnorms += [grad_norm]

        preds += [pred]
        tf.summary.scalar('loss_{}'.format(name), loss / bs)

    if GRAD_NORM:
        # normalize
        # loss_weights = lg.Variable(name = 'loss_weghts', initial_value = tf.ones([len(TARGETS)]))
        loss_weights = lg.Variable(tf.ones([len(TARGETS)]), name='loss_weghts')
        loss_weights_normalized = tf.nn.softmax(loss_weights)

        # weighted obj_losses
        loss_sum = tf.reduce_sum(tf.stack(obj_losses) * tf.stop_gradient(loss_weights_normalized)) * len(TARGETS)
        obj_losses = [loss_sum]

        # regularize weights to average
        gnorm_vec_ori = tf.stack(gnorms, axis=0)
        gnorm_vec = tf.stop_gradient(gnorm_vec_ori)
        avgnorm = tf.reduce_sum(gnorm_vec * loss_weights_normalized) / len(TARGETS)

        wgnorm_vec = gnorm_vec * loss_weights_normalized
        weights_loss = tf.reduce_sum((wgnorm_vec - avgnorm) ** 2) * 1e6  # to average value

        reg_losses += [weights_loss]

        for i, target in enumerate(TARGETS):
            name = target.name
            tf.summary.scalar('grad_norm_{}'.format(name), gnorm_vec[i])
            tf.summary.scalar('loss_weight_{}'.format(name), loss_weights[i])
            tf.summary.scalar('loss_weight_norm_{}'.format(name), loss_weights_normalized[i])
            tf.summary.scalar('grad_norm_x_loss_weight_{}'.format(name), wgnorm_vec[i])

        tf.summary.scalar('weights_loss', weights_loss)

    oracle_pred = tf.identity(preds[0], name='oracle_pred')  # Dummy
    loss = tf.add_n(obj_losses + reg_losses)
    loss = tf.identity(loss, name='oracle_loss')

    tf.summary.scalar('oracle_loss', loss)


def get_pred_and_loss(out, label, sample_rate, loss_type='logloss'):
    # drop nan sample(for negative sampling)
    # print(dir(tf.math))
    mask = label > -10000  # tf.is_finite(label) # tf.not_equal(tf.math.is_nan(label), False)
    # out = tf.boolean_mask(out, mask)
    # label = tf.boolean_mask(label, mask)
    # bs = tf.shape(label)[0]

    if loss_type == 'logloss':
        logit = tf.reduce_sum(out, axis=1) - tf.log(sample_rate)
        pred = tf.sigmoid(logit)
        print('logit: {}'.format(logit))
        print('label: {}'.format(label))
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit)
        loss = tf.boolean_mask(loss, mask)
        loss = tf.reduce_sum(loss)
    elif loss_type == 'mse':
        pred = tf.reduce_sum(out, axis=1)
        loss = tf.square(label - pred)
        loss = tf.boolean_mask(loss, mask)
        loss = 0.5 * tf.reduce_sum(loss)
    elif loss_type == 'softmax':
        label = tf.to_int32(label)
        label = tf.maximum(0, label)
        label = tf.minimum(BUCKET_NUM - 1, label)
        prob = tf.nn.softmax(out, axis=1)
        tf.summary.tensor_summary('softmax-prob', prob)
        classes = tf.to_float(tf.tile(tf.reshape(tf.range(BUCKET_NUM), [1, BUCKET_NUM]), [tf.shape(out)[0], 1]))
        mean = tf.multiply(prob, classes)
        pred = tf.reduce_sum(mean, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=out)
        loss = tf.boolean_mask(loss, mask)
        loss = tf.reduce_sum(loss)
    else:
        raise ValueError('Unknown loss_type: {}'.format(loss_type))
    return pred, loss


def get_init_funcs(dims, mean=0.0, stddev=1.0):
    init_funcs = []
    for dim in dims[:-1]:
        init_funcs.append(
            lambda s, d=dim: tf.truncated_normal(shape=s,
                                                 mean=mean,
                                                 stddev=stddev) / math.sqrt(d))


def gen_config():
    with lg.default_parser() as parser:
        get_tf_net(parser)

        meta = parser.get_meta()
        meta.model_size = 5000000
        meta.vec_model_size = 50000
        meta.filter_capacity = 20000000
        parser.set_global_grad_clip_norm(GLOBAL_GRAD_NORM_CUTOFF)

        for config, slots in SLOT_CONFIG_GROUPS:
            for slot in slots:
                parser.set_slot_config(slot, config)

        if USE_RMSPROP:
            parser.set_default_variable_config({
                'alpha': DENSE_ALPHA,
                'beta': 0.999999,
                'lambda1': 0,
                'lambda2': 0,
                'init_factor': 1.0,
                'opt_type': 4,
            })
        else:
            parser.set_default_variable_config({
                'alpha': DENSE_ALPHA,
                'beta': 262144,
                'lambda1': 0,
                'lambda2': 0,
                'init_factor': 1.0
            })

        model_name = 'aweme_deep_model_multitask_v5'

        parser.set_extra_parameters(
            '-label_names=like,share,comment,follow,head,click_comment,dislike,cover,challenge,shoot,finish,read,staytime,pc -multi_targets_setting=finish,pc -kafka_dump=0 -has_sort_id=0 -file_type=proto')
        parser.set_test_instance_path('../data/sample_finish_data_100')
        parser.gen_config(model_name, check_config=False)


if __name__ == '__main__':
    gen_config()
