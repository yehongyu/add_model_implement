import tensorflow as tf
import lagrange as lg
import lagrange_model as lgm
import math


def toutiao_init_func(shape):
    """Xavier initialization. See http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
    print
    shape[0], shape[1], math.sqrt(1.0 / shape[0])
    return tf.truncated_normal(shape=shape, mean=0, stddev=math.sqrt(1.0 / shape[0]))


FFM_TRIPPLES = [
    ([591, 4, 11, 15, 16, 18, 113, 500, 501, 502, 511, 512, 513],
     [2, 549, 551, 553, 554, 555, 556, 557, 558, 559, 560, 561, 202, 203, 204, 205, 206, 207, 208, 209, 210, 213, 218,
      219, 220, 245], 32),  # user,push
    ([591, 4, 11, 15, 16, 18, 113, 500, 501, 502, 511, 512, 513], [201, 217], 32),  # user,author
    ([591, 4, 11, 15, 16, 18, 113, 500, 501, 502, 511, 512, 513], [8, 10], 32),  # user,context
    ([2, 549, 551, 553, 554, 555, 556, 557, 558, 559, 560, 561, 202, 203, 204, 205, 206, 207, 208, 209, 210, 213, 218,
      219, 220, 245], [8, 10], 32),  # push,context
    ([201, 217], [8, 10], 32),  # author,context
]

BIAS_NN_SLOTS = [434]

VALID_SLOTS = BIAS_NN_SLOTS + [0]

for slots1, slots2, dim in FFM_TRIPPLES:
    for slot in slots1:
        VALID_SLOTS.append(slot)
    for slot in slots2:
        VALID_SLOTS.append(slot)
VALID_SLOTS = list(set(VALID_SLOTS))
# print(BIAS_NN_SLOTS)
print(VALID_SLOTS)

NN_DIMS = [1024, 64, 32]
FFM_DIMS = [1024, 256, 32]
C_DIMS = [512, 256, 32]


def get_embedding(slots, dim):
    with lg.default_parser() as parser:
        if type(slots) == type(()):
            embs = []
            for slot in slots:
                embs.append(parser.alloc_slot_vec(slot, dim))
            return tf.add_n(embs)
        else:
            return parser.alloc_slot_vec(slots, dim)


def get_cross_prods(inputs):
    with lg.default_parser() as parser:
        all_prods = []
        all_concats = []
        for slots_1, slots_2, dim in inputs:
            for slot1 in slots_1:
                for slot2 in slots_2:
                    input1 = get_embedding(slot1, dim)
                    input2 = get_embedding(slot2, dim)
                    m = tf.multiply(input1, input2)
                    all_concats.append(input1)
                    all_concats.append(input2)
                    all_prods.append(m)
        return all_prods, all_concats


def generate_model(model_name):
    parser = lg.LagrangeModelParser()
    parser.set_default_parser(parser)
    with lg.default_parser() as parser:
        # NN tower
        bias_input = parser.get_bias_input()
        bias_selected = tf.gather(bias_input, BIAS_NN_SLOTS, axis=1)
        bias_selected_dim = int(bias_selected.shape[1])

        # nn_out
        init_func = lambda shape: tf.truncated_normal(shape=shape, mean=0, stddev=0.1)
        init_func_last_layer = lambda shape: tf.truncated_normal(shape=shape, mean=0, stddev=0.01)
        init_funcs = [init_func, init_func, init_func, init_func_last_layer]
        nn_out = lgm.deep_tower(input_t=bias_selected, input_dim=bias_selected_dim, output_dims=NN_DIMS,
                                name='nn_tower', init_funcs=init_funcs, use_weight_norm=True)
        nn_out = tf.reduce_sum(nn_out, axis=1)
        # sum_bias
        sum_bias = tf.reduce_sum(bias_input, axis=1)

        #
        # emb_prod, emb_concat = get_cross_prods(FM_TRIPPLES)
        # prod_dfm_out
        # prod_emb = tf.concat(emb_prod, axis=1)
        # prod_emb_dim = int(prod_emb.shape[1])
        # prod_dfm_init_funcs = [toutiao_init_func] * len(FFM_DIMS)
        # prod_dfm_out = lgm.deep_tower(input_t = prod_emb, input_dim = prod_emb_dim, output_dims = FFM_DIMS, name = 'prod_dfm_tower', init_funcs=prod_dfm_init_funcs, use_weight_norm=True)
        # prod_dfm_out = tf.reduce_sum(prod_dfm_out, axis=1)
        # ffm_out
        # ffm_out = tf.reduce_sum(prod_emb, axis=1)

        emb_prod, _ = get_cross_prods(FFM_TRIPPLES)
        prod_emb = tf.concat(emb_prod, axis=1)
        prod_emb_dim = int(prod_emb.shape[1])
        prod_dfm_init_funcs = [toutiao_init_func] * len(FFM_DIMS)
        prod_dfm_out = lgm.deep_tower(input_t=prod_emb, input_dim=prod_emb_dim, output_dims=FFM_DIMS,
                                      name='prod_dfm_tower', init_funcs=prod_dfm_init_funcs, use_weight_norm=True)
        prod_dfm_out = tf.reduce_sum(prod_dfm_out, axis=1)
        # ffm_out = tf.reduce_sum(prod_emb2, axis=1)
        # concat_dfm_out
        # concat_emb = tf.concat(emb_concat, axis=1)
        # concat_emb_dim = int(concat_emb.shape[1])
        # concat_dfm_init_funcs = [toutiao_init_func] * len(C_DIMS)
        # concat_dfm_out = lgm.deep_tower(input_t = concat_emb, input_dim = concat_emb_dim, output_dims = C_DIMS, name = 'concat_dfm_tower', init_funcs=concat_dfm_init_funcs, use_weight_norm=False)

        # towers = [nn_out, prod_dfm_out]

        # init_values = tf.truncated_normal(shape=[32, 1], mean=0, stddev=0.05)
        # rank_w = lg.weight_norm_variable(init_values, name='rank_w', norm_axis=0, norm_trainable=True)
        # hidden_out = tf.add_n(towers)
        # tower_out = tf.reduce_sum(tf.matmul(hidden_out, rank_w), axis=1)

        y_out = sum_bias + prod_dfm_out + nn_out

        label = parser.get_label()
        sample_rate = parser.get_sample_rate()
        sample_y_out = tf.add_n([y_out, tf.negative(tf.log(sample_rate))])
        pred = tf.sigmoid(sample_y_out, name="oracle_pred")
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=sample_y_out),
                             name='oracle_loss')

        # Add summary
        tf.summary.scalar('oracle_loss', loss)
        tf.summary.histogram('oracle_pred', pred)
        # tf.summary.histogram('ffm_out', ffm_out)
        tf.summary.histogram('nn_out', nn_out)
        tf.summary.histogram('sum_bias', sum_bias)
        tf.summary.histogram('prod_dfm_out', prod_dfm_out)
        # tf.summary.histogram('concat_dfm_out', concat_dfm_out)

        # Set meta related config
        set_meta_config(parser)

        parser.gen_config(model_name, check_config=False)


def set_meta_config(parser):
    """Set meta and slot config
    Args:
        parser: lagrange model parser
    """
    parser.set_valid_slots(VALID_SLOTS)
    meta = parser.get_meta()
    meta.model_size = 3000000
    meta.vec_model_size = 1200000
    meta.filter_capacity = 40000000

    for slot in VALID_SLOTS:
        if slot in [1, 2]:
            parser.set_slot_config(slot, {
                'alpha': 0.01,
                'beta': 1.0,
                'lambda1': 0.1,
                'lambda2': 0.0,
                'vec_alpha': 0.1,
                'vec_beta': 1.0,
                'vec_lambda1': 0.0,
                'vec_lambda2': 0.0,
                'vec_init_factor': -0.015625,
                'occurrence_threshold': 0,
            })
        else:
            parser.set_slot_config(slot, {
                'alpha': 0.01,
                'beta': 1.0,
                'lambda1': 0.1,
                'lambda2': 0.0,
                'vec_alpha': 0.04,
                'vec_beta': 1.0,
                'vec_lambda1': 0.0,
                'vec_lambda2': 0.0,
                'vec_init_factor': -0.015625,
                'occurrence_threshold': 0,
            })
    parser.set_default_variable_config({
        'alpha': 0.01,
        'beta': 262144,
        'lambda1': 0,
        'lambda2': 0,
        'init_factor': 1})


if __name__ == '__main__':
    """please set an appropriate model name"""
    model_name = 'tiktok_tf_nn'
    generate_model(model_name)

