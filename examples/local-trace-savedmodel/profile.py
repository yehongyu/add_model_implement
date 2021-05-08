#coding:utf-8
import logging
import numpy as np
import tensorflow as tf
import time

from tensorflow.python.client import timeline


tf.app.flags.DEFINE_integer('batch_size', 200, 'Instance count per request.')
tf.app.flags.DEFINE_integer('warmup_rounds', 2, 'Run several rounds before generate trace.')
tf.app.flags.DEFINE_string('trace_path', 'timeline.json', 'Chrome trace file path.')

FLAGS = tf.app.flags.FLAGS
SM_DIR = '/path/to/search_lagrange_raw_rank_training_10999_v11_r176167'


def get_output(graph):
    return graph.get_tensor_by_name('sigmoid_merge_score/Sigmoid:0')


def fake_input(graph):
    float_inputs = {
        'pos_rel:0': 1,
        'pos_feature:0': 115,
    }
    int32_inputs = {
        'pos_anchor:0': 10,
        'pos_cq:0': 10,
        'pos_host:0': 1,
        'pos_tagtitle:0': 25
    }
    scalar_inputs = {
        'batch_size:0': FLAGS.batch_size,
        'batch_switch:0': FLAGS.batch_size
    }
    feed_dict = {}
    for name, width in float_inputs.items():
        tensor = graph.get_tensor_by_name(name)
        dim_list = [FLAGS.batch_size, width]
        feed_dict[tensor] = np.random.uniform(size=dim_list).astype('f')
    for name, width in int32_inputs.items():
        tensor = graph.get_tensor_by_name(name)
        dim_list = [FLAGS.batch_size, width]
        feed_dict[tensor] = np.random.randint(0, 10, size=dim_list, dtype=np.int32)
    for name, value in scalar_inputs.items():
        tensor = graph.get_tensor_by_name(name)
        vtype = np.int32 if isinstance(value, int) else np.float32
        feed_dict[tensor] = np.array(value, dtype=vtype)
    tensor = graph.get_tensor_by_name('query:0')
    feed_dict[tensor] = np.random.randint(0, 10, size=[1, 20], dtype=np.int32)
    return feed_dict


def write_trace(run_metadata):
    stats = timeline.Timeline(run_metadata.step_stats)
    trace = stats.generate_chrome_trace_format()
    with open('timeline.json', 'w') as fout:
        fout.write(trace)


def main(_):
    with tf.Session() as sess:
        _ = tf.compat.v1.saved_model.loader.load(sess, ['serve'], SM_DIR)
        graph = tf.compat.v1.get_default_graph()
        to_fetch = get_output(graph)
        to_feed = fake_input(graph)

        # Warm up session
        for idx in range(FLAGS.warmup_rounds):
            start_time = time.time()
            _ = sess.run(to_fetch, feed_dict=to_feed)
            time_cost = time.time() - start_time
            logging.info('[%d] %.4f seconds', idx + 1, time_cost)

        # Run one step
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        start_time = time.time()
        sess.run(to_fetch, feed_dict=to_feed, options=options, run_metadata=run_metadata)
        time_cost = time.time() - start_time
        logging.info('[FINAL] %.4f seconds', time_cost)
        write_trace(run_metadata)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s [%(filename)s:%(lineno)d] %(levelname)s %(message)s'
    )
    tf.app.run()
