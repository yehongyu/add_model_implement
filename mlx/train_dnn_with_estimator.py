import os
import sys
import time
import logging
logging.basicConfig(
    level=logging.INFO
)

import tensorflow as tf
print("tf version:", tf.__version__)

cur_ts = str(int(time.time()))
tf.app.flags.DEFINE_string("train_paths", "/mnt/comment_new/aodandan/train_data/tfrecord/20200604/part-*", "HDFS paths to input files.")
tf.app.flags.DEFINE_string("eval_paths", "/mnt/comment_new/aodandan/train_data/tfrecord/20200605/part-*", "eval data path")
tf.app.flags.DEFINE_string("model_path", "/mnt/comment_new/aodandan/model/dnn/"+cur_ts, "Where to write output files.")
tf.app.flags.DEFINE_string("last_model_path", "", "Model path for the previous run.")
tf.app.flags.DEFINE_integer("train_epochs", 10, "train epochs")
tf.app.flags.DEFINE_integer("batch_size", 512, "batch size")
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "train learning rate")
tf.app.flags.DEFINE_float("dropout", 0.1, "dropout")
tf.app.flags.DEFINE_float("clip_norm", 10.0, "clip norm")
tf.app.flags.DEFINE_integer("num_cols", 264, "num cols")

FLAGS = tf.app.flags.FLAGS

def build_feature_columns():
    columns = []
    for i in range(FLAGS.num_cols):
        num_column = tf.feature_column.numeric_column("slot_%s"%i)
        columns.append(num_column)
    return columns

def build_model(FLAGS):
    logging.info("model_path: %s" % FLAGS.model_path)
    logging.info("last model path: %s" % FLAGS.last_model_path)
    logging.info("learning rate: %s" % FLAGS.learning_rate)
    logging.info("clip norm: %s" % FLAGS.clip_norm)
    logging.info("num_cols: %s" % FLAGS.num_cols)
    logging.info("dropout: %s" % FLAGS.dropout)
    checkpoint_dir = FLAGS.model_path
    if FLAGS.last_model_path and not tf.train.latest_checkpoint(checkpoint_dir):
        warmup_dir = FLAGS.last_model_path
    else:
        warmup_dir = None

    my_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, FLAGS.clip_norm)

    #DNNClassifier: Loss is calculated by using softmax cross entropy.

    model = tf.estimator.DNNClassifier(
        feature_columns=build_feature_columns(),
        hidden_units=[256, 64],
        optimizer = my_optimizer,
        n_classes = 2,
        dropout=FLAGS.dropout,
        config=tf.estimator.RunConfig(model_dir=checkpoint_dir),
        warm_start_from=warmup_dir)
    return model

def serving_input_receiver_fn():
    features = {}
    for i in range(FLAGS.num_cols):
        fname = "slot_%s"%(i)
        features[fname] = tf.placeholder(tf.float32, shape=[None], name=fname)
    return tf.estimator.export.ServingInputReceiver(features, features)

def read_data(paths, batch_size=512, num_epochs=1, shuffle=False, buffer_size=50000, num_cols=242, num_parallels=1, num_workers=1, worker_index=0):
    def parse(value):
        desc = {
                'slot_%s'%i: tf.FixedLenFeature([1], tf.float32, default_value=0.0) for i in range(0, num_cols)
            }
        desc["label"] = tf.FixedLenFeature([1], tf.int64, default_value=0)
        example = tf.parse_single_example(value, desc)
        label = example["label"]
        label = tf.cast(label,tf.int32)
        del example["label"]
        return example, label

    logging.info('paths: %s' % paths)
    data_files = tf.data.Dataset.list_files(paths)

    dataset = tf.data.TFRecordDataset(data_files, num_parallel_reads=num_parallels)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    # dataset = dataset.shard(num_workers, worker_index)

    return dataset.map(parse, num_parallel_calls=num_parallels) \
                  .repeat(num_epochs).batch(batch_size) \
                  .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def train_input_fn():
    return read_data(FLAGS.train_paths,
                     batch_size=FLAGS.batch_size,
                     num_epochs=FLAGS.train_epochs,
                     shuffle=True,
                     num_cols=FLAGS.num_cols,
                     num_parallels=1)

def eval_input_fn():
    return read_data(FLAGS.eval_paths,
                     batch_size=FLAGS.batch_size,
                     num_cols=FLAGS.num_cols,
                     num_parallels=1)

def load_model_and_print_variable():
    path = FLAGS.model_path
    init_vars = tf.train.list_variables(path)
    for name, shape in init_vars:
        array = tf.train.load_variable(path, name)
        print(name, shape)

def find_variable():
    # TODO: find the two variable:
    model = build_model(FLAGS)
    weight_name = 'dnn/logits/kernel:0' # dnn/logits/bias, dnn/head/beta1_power, dnn/hiddenlayer_0/bias
    score_tensor_name='dnn/head/predictions/logistic:0'
    label_tensor_name='IteratorGetNext:%s' % (FLAGS.num_cols)

    graph = tf.compat.v1.get_default_graph()
    score_tensor = graph.get_tensor_by_name(weight_name)
    sess = tf.compat.v1.Session(graph=graph)
    sess.run(model)
    print("score_tensor:", sess.run(score_tensor))


def start_train():
    model = build_model(FLAGS)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=6000000)

    feature_spec = tf.feature_column.make_parse_example_spec(build_feature_columns())
    export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    exporter = tf.estimator.FinalExporter('gandalf', export_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=100, throttle_secs=10, exporters=[exporter])

    print("start to train")
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    print("start to save model")
    model.export_saved_model(FLAGS.model_path + '/saved_model',
                serving_input_receiver_fn=serving_input_receiver_fn)
    print("finish save model")

def test():
    sess=tf.Session()

    inc_dataset = tf.data.Dataset.range(100)
    dec_dataset = tf.data.Dataset.range(0, -100, -1)
    batched_dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
    batched_dataset = batched_dataset.batch(4)

    iterator = batched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    print(next_element)

    print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
    print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
    print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])

def test_parse(num_cols=242):
    def _parse_function(value):
        #desc = {
        #        'slot_%s'%i: tf.FixedLenFeature([1], tf.float32, default_value=0.0) for i in range(0, num_cols)
        #    }
        #desc["label"] = tf.FixedLenFeature([1], tf.int64, default_value=0)
        desc = {}
        desc["label"] = tf.FixedLenFeature([1], tf.int64, default_value=0)
        for i in range(0, num_cols):
            desc['slot_%s'%i] = tf.FixedLenFeature([1], tf.float32, default_value=0.0)
        example = tf.parse_single_example(value, desc)
        label = example["label"]
        label = tf.cast(label,tf.int32)
        del example["label"]
        return example, label
    filenames = ["/Users/aodandan/data/tfrecord/train/part-00000"]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    batched_dataset = dataset.batch(4)

    iterator = batched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    print(next_element)

if __name__ == '__main__':

    '''
    from optparse import OptionParser
    usage = "Usage: %s -s <date> -a <num_days>" % __file__
    parser = OptionParser(usage=usage)
    parser.add_option('--learning_rate', dest='learning_rate', default=0.001, help="lr")
    parser.add_option('--dropout', dest='dropout', default=0.1, help="dropout")
    parser.add_option('--train_epochs', dest='train_epochs', default=30, help="train epochs")
    parser.add_option('--train_paths', dest='train_paths', help="train paths")
    parser.add_option('--eval_paths', dest='eval_paths', help="eval paths")
    opts, args = parser.parse_args()
    print(opts)
    '''

    print(sys.argv)
    FLAGS.train_paths = '{}/part*'.format(sys.argv[1])
    FLAGS.eval_paths = '{}/part*'.format(sys.argv[2])
    logging.info("lr: %s" % FLAGS.learning_rate)
    logging.info("dropout: %s" % FLAGS.dropout)
    logging.info("train_epochs: %s" % FLAGS.train_epochs)
    logging.info("train_paths: %s" % FLAGS.train_paths)
    logging.info("eval_paths: %s" % FLAGS.eval_paths)
    logging.info("model_path: %s" % FLAGS.model_path)
    start_train()
