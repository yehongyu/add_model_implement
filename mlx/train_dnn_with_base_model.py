#coding=utf-8
import sys
import time
import numpy as np
import datetime
import logging
import traceback

from sklearn import metrics
import tensorflow as tf

# test tensorflow lib
tf_version = tf.__version__
print("tf version:", tf_version)
print("gpu avaibable:", tf.test.is_gpu_available())
#print(tf.test.gpu_device_name())
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
#print(tf.config.experimental.list_physical_devices('GPU'))
#print(tf.config.experimental.list_physical_devices('CPU'))

logging.basicConfig(level=logging.INFO)

cur_ts = str(int(time.time()))
tf.app.flags.DEFINE_string("train_paths", "/mnt/comment_new/aodandan/train_data/tfrecord/20200604/part-*", "HDFS paths to input files.")
tf.app.flags.DEFINE_string("eval_paths", "/mnt/comment_new/aodandan/train_data/tfrecord/20200605/part-*", "eval data path")
tf.app.flags.DEFINE_string("model_path", "/mnt/comment_new/aodandan/model/dnn_simple/"+cur_ts, "Where to write output files.")
tf.app.flags.DEFINE_string("last_model_path", "", "Model path for the previous run.")
tf.app.flags.DEFINE_integer("train_epochs", 10, "train epochs")
tf.app.flags.DEFINE_integer("batch_size", 512, "batch size")
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "train learning rate")
tf.app.flags.DEFINE_float("dropout", 0.0, "dropout")
tf.app.flags.DEFINE_float("clip_norm", 10.0, "clip norm")
tf.app.flags.DEFINE_integer("num_cols", 264, "num cols")
tf.app.flags.DEFINE_string("device", "/gpu:2", "device of tf")
tf.app.flags.DEFINE_string("f", "", "kernel")

FLAGS = tf.app.flags.FLAGS


def now_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def cal_sklearn_auc(all_y, all_pred, batch_labels, batch_probs):
    all_y.extend(batch_labels)
    all_pred.extend(batch_probs)
    auc = metrics.roc_auc_score(all_y, all_pred)
    acc = metrics.accuracy_score(all_y, np.around(all_pred).astype(int))
    return auc, acc

class DataIterator(object):
    def __init__(self, paths, shuffle, num_cols, batch_size):
        self.iterator = self.build_iterator(paths, shuffle=shuffle, num_cols=num_cols, batch_size=batch_size)
        self.initializer = self.iterator.initializer
        self.next_element = self.iterator.get_next()

    def build_iterator(self, paths, shuffle=True, num_cols=264, batch_size=2, buffer_size = 8 * 1024 * 1024, num_parallels=1):
        def parse(value):
            desc = {
                "slot_%s"%i: tf.io.FixedLenFeature([1], tf.float32, default_value=0.0) for i in range(0, num_cols)
                }
            desc["label"] = tf.io.FixedLenFeature([1], tf.int64, default_value=0)
            example = tf.io.parse_single_example(value, desc)
            label = example["label"]
            label = tf.cast(label,tf.int32)
            del example["label"]
            instance = []
            for i in range(num_cols):
                instance.append(example["slot_%s"%i])
            return instance, label

        logging.info("{} Build iterator from file: {}".format(now_time(), paths))
        data_files = tf.data.Dataset.list_files(paths, shuffle=shuffle)
        dataset = tf.data.TFRecordDataset(data_files, buffer_size=buffer_size,
                                          num_parallel_reads=num_parallels)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.map(parse, num_parallel_calls=num_parallels).batch(batch_size)
        dataset = dataset.prefetch(buffer_size=2 * batch_size)
        if tf_version.startswith('1.12'):
            return dataset.make_initializable_iterator()
            #return tf.data.make_initializable_iterator(dataset)
        else:
            return tf.compat.v1.data.make_initializable_iterator(dataset)

class VerifyDNNModel(object):
    def __init__(self, parameters):
        self.lr = parameters.get("lr", 0.001)
        self.dropout = parameters.get("dropout", 0.1)
        self.n_feature = parameters.get("n_feature", 264)
        self.n_classes = parameters.get("n_classes", 2)
        self.device = parameters.get("device", "/cpu:0")

        self.width_1th_layer = 256
        self.width_2th_layer = 64
        self.width_3th_layer = self.n_classes

        self.create_placeholders()

        self.build_graph()

        self.create_summary()

    def create_summary(self):
        with tf.name_scope("dnn_metric") as scope:
            if tf_version.startswith('1.12'):
                loss_val, self.loss_op = tf.metrics.mean(self.loss, name="loss_metric")
                auc_val, self.auc_op = tf.metrics.auc(self.labels, self.prediction, name="auc_metric")
                pred_class = tf.cast(tf.round(self.prediction), tf.int32)
                acc_val, self.acc_op = tf.metrics.accuracy(self.labels, pred_class, name="acc_metric")
            else:
                loss_val, self.loss_op = tf.compat.v1.metrics.mean(self.loss, name="loss_metric")
                auc_val, self.auc_op = tf.compat.v1.metrics.auc(self.labels, self.prediction, name="auc_metric")
                pred_class = tf.cast(tf.round(self.prediction), tf.int32)
                acc_val, self.acc_op = tf.compat.v1.metrics.accuracy(self.labels, pred_class, name="acc_metric")
            tf.summary.scalar('loss', loss_val)
            tf.summary.scalar('auc', auc_val)
            tf.summary.scalar('acc', acc_val)
            self.summary_merged = tf.summary.merge_all()

    def create_placeholders(self):
        with tf.name_scope("dnn_input") as scope, tf.device(self.device):
            self.X = tf.placeholder(tf.float32, shape=(None, self.n_feature, 1), name='X')
            self.Y = tf.placeholder(tf.int32, shape=(None, 1), name='Y')

    def build_graph(self):
        with tf.name_scope("dnn_core") as scope, tf.device(self.device):
            if tf_version.startswith('1.12'):
                input_X = tf.reshape(tf.squeeze(self.X), [-1, self.n_feature])
                onehot_Y = tf.reshape(tf.one_hot(tf.squeeze(self.Y), depth=self.n_classes), [-1, self.n_classes])
            else:
                input_X = tf.reshape(tf.compat.v1.squeeze(self.X), [-1, self.n_feature])
                onehot_Y = tf.reshape(tf.one_hot(tf.compat.v1.squeeze(self.Y), depth=self.n_classes), [-1, self.n_classes])
            self.labels = onehot_Y[:,1]

            w1 = tf.get_variable(name="w1", shape=[self.n_feature, self.width_1th_layer],
                             initializer=tf.contrib.layers.xavier_initializer())
            if tf_version.startswith('1.12'):
                b1 = tf.get_variable(name="b1", shape=[1, self.width_1th_layer], initializer=tf.zeros_initializer)
            else:
                b1 = tf.get_variable(name="b1", shape=[1, self.width_1th_layer], initializer=tf.compat.v1.zeros_initializer)

            w2 = tf.get_variable(name="w2", shape=[self.width_1th_layer, self.width_2th_layer],
                             initializer=tf.contrib.layers.xavier_initializer())
            if tf_version.startswith('1.12'):
                b2 = tf.get_variable(name="b2", shape=[1, self.width_2th_layer], initializer=tf.zeros_initializer)
            else:
                b2 = tf.get_variable(name="b2", shape=[1, self.width_2th_layer], initializer=tf.compat.v1.zeros_initializer)

            w3 = tf.get_variable(name="w3", shape=[self.width_2th_layer, self.width_3th_layer],
                             initializer=tf.contrib.layers.xavier_initializer())
            if tf_version.startswith('1.12'):
                b3 = tf.get_variable(name="b3", shape=[1, self.width_3th_layer], initializer=tf.zeros_initializer)
            else:
                b3 = tf.get_variable(name="b3", shape=[1, self.width_3th_layer], initializer=tf.compat.v1.zeros_initializer)

            z1 = tf.add(tf.matmul(input_X, w1), b1)
            a1 = tf.nn.relu(z1)
            a1 = tf.nn.dropout(a1, keep_prob=1.0-self.dropout)
            z2 = tf.add(tf.matmul(a1, w2), b2)
            a2 = tf.nn.relu(z2)
            a2 = tf.nn.dropout(a2, keep_prob=1.0-self.dropout)
            logits = tf.add(tf.matmul(a2, w3), b3)
            probabilities = tf.nn.softmax(logits)
            self.prediction = probabilities[:,1]

            self.loss = tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits,
                        labels=onehot_Y
                    )
            self.loss_rmean = tf.reduce_mean(self.loss)
            if tf_version.startswith('1.12'):
                train_vars = tf.trainable_variables()
            else:
                train_vars = tf.compat.v1.trainable_variables()
            for v in train_vars: logging.info("model train_var: {}, {}".format(v.name, v))
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_rmean, train_vars), clip_norm=5)
            optimizer = tf.train.AdamOptimizer(self.lr)
            #optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, train_vars))

    def get_train_task(self):
        task_ops = {
            "train_op": self.train_op,
            "loss_op": self.loss_op,
            "auc_op": self.auc_op,
            "acc_op": self.acc_op,
            "summary": self.summary_merged,
            "loss_rmean": self.loss_rmean,
        }
        return task_ops

    def get_eval_task(self):
        task_ops = {
            "loss_op": self.loss_op,
            "auc_op": self.auc_op,
            "acc_op": self.acc_op,
            "loss_rmean": self.loss_rmean,
            "prediction": self.prediction,
        }
        return task_ops

class DNNTrainer(object):
    def __init__(self):
        self.log_path = FLAGS.model_path + "/log/" # tensorboard -–logdir
        self.checkpoint_path = FLAGS.model_path + "/ckpt/dnn"
        self.last_checkpoint_path = FLAGS.last_model_path
        self.train_epochs = FLAGS.train_epochs

        parameters = {}
        parameters["lr"] = FLAGS.learning_rate
        parameters["dropout"] = FLAGS.dropout
        parameters["n_feature"] = FLAGS.num_cols
        #parameters["device"] = "/gpu:{}".format(FLAGS.device)
        parameters["device"] = FLAGS.device
        self.dnn_model = VerifyDNNModel(parameters)

        self.train_iterator = DataIterator(paths=FLAGS.train_paths, num_cols=FLAGS.num_cols,
                                                        batch_size=FLAGS.batch_size, shuffle=True)
        self.eval_iterator = DataIterator(paths=FLAGS.eval_paths, num_cols=FLAGS.num_cols,
                                                       batch_size=FLAGS.batch_size, shuffle=False)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.train_epochs)

    def get_sess_config(self):
        sess_config = tf.ConfigProto()
        sess_config.log_device_placement = True # log device placement
        #sess_config.gpu_options.allow_growth = True # dynamic allocate mem
        #sess_config.allow_soft_placement = True # auto select device
        logging.info("tf session config: {}".format(sess_config))
        return sess_config

    def reset_running_variables(self, sess, scope):
        if tf_version.startswith('1.12'):
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope)
        else:
            running_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOCAL_VARIABLES, scope=scope)
        for v in running_vars: logging.info("run var: {}, {}".format(v.name, v))
        if tf_version.startswith('1.12'):
            running_vars_initializer = tf.variables_initializer(var_list=running_vars)
        else:
            running_vars_initializer = tf.compat.v1.variables_initializer(var_list=running_vars)
        sess.run(running_vars_initializer)

    def train_one_epoch(self, sess, log_writer, epoch, step):
        sess.run(self.train_iterator.initializer)
        self.reset_running_variables(sess, "dnn_metric") # accumulative, reset for each epoch
        next_element = self.train_iterator.next_element
        task_ops = self.dnn_model.get_train_task()
        while True:
            try:
                batch_instances, batch_labels = sess.run(next_element)
                feed = {self.dnn_model.X:batch_instances, self.dnn_model.Y:batch_labels}

                results = sess.run(task_ops, feed_dict=feed)

                step += 1
                log_writer.add_summary(results['summary'], step)

                if step % 100 == 0:
                    logging.info("{} Epoch-{}, step-{}: batch_loss={}, loss={}, auc={}, acc={}".format(
                        now_time(), epoch, step, results["loss_rmean"], results["loss_op"], results["auc_op"], results["acc_op"])
                     )
            except tf.errors.OutOfRangeError:
                logging.info("{} Epoch-{}: consumed all examples.".format(now_time(), epoch))
                break
            except Exception as e:
                err_msg = traceback.format_exc()
                looging.error("err: {}".format(err_msg))
                break
        return step

    def eval_one(self, sess, step=0):
        sess.run(self.eval_iterator.initializer)
        self.reset_running_variables(sess, "dnn_metric")
        next_element = self.eval_iterator.next_element
        task_ops = self.dnn_model.get_eval_task()
        all_y, all_pred = [], []
        results = None
        while True:
            try:
                batch_instances, batch_labels = sess.run(next_element)
                feed = {self.dnn_model.X:batch_instances, self.dnn_model.Y:batch_labels}

                results = sess.run(task_ops, feed_dict=feed)
                auc, acc = cal_sklearn_auc(all_y, all_pred, batch_labels, results["prediction"])

                step += 1
                if step % 2 == 0:
                    logging.info("{} Eval, step-{}: batch_loss={}, loss={}, auc={}, acc={}, skauc={}, skacc={}".format(
                        now_time(), step, results["loss_rmean"], results["loss_op"], results["auc_op"], results["acc_op"], auc, acc)
                     )
            except tf.errors.OutOfRangeError:
                logging.info("{} Eval, step-{}: loss={}, auc={}, acc={}, skauc={}, skacc={}".format(
                    now_time(), step, results["loss_op"], results["auc_op"], results["acc_op"], auc, acc)
                    )
                break
            except Exception as e:
                err_msg = traceback.format_exc()
                looging.error("err: {}".format(err_msg))
                break
        return step


    def train(self):
        with tf.Session(config=self.get_sess_config()) as sess:
        #with tf.Session() as sess:
            if self.last_checkpoint_path:
                logging.info("Restore variable from file: {}".format(self.last_checkpoint_path))
                self.saver.restore(sess, self.last_checkpoint_path)
            else:
                if tf_version.startswith('1.12'):
                    logging.info("Init global variable: {}".format([v.name for v in tf.global_variables()]))
                else:
                    logging.info("Init global variable: {}".format([v.name for v in tf.compat.v1.global_variables()]))
                if tf_version.startswith('1.12'):
                    sess.run(tf.global_variables_initializer())
                else:
                    sess.run(tf.compat.v1.global_variables_initializer())
            if tf_version.startswith('1.12'):
                logging.info("Init local variable: {}".format([v.name for v in tf.local_variables()]))
            else:
                logging.info("Init local variable: {}".format([v.name for v in tf.compat.v1.local_variables()]))
            if tf_version.startswith('1.12'):
                sess.run(tf.local_variables_initializer())
            else:
                sess.run(tf.compat.v1.local_variables_initializer())

            if tf_version.startswith('1.12'):
                log_writer = tf.summary.FileWriter(self.log_path, sess.graph)
            else:
                log_writer = tf.compat.v1.summary.FileWriter(self.log_path, sess.graph)
            step = 0
            for epoch in range(self.train_epochs):
                # train model
                step = self.train_one_epoch(sess, log_writer, epoch, step)

                # save model
                last_store_path = self.saver.save(sess, self.checkpoint_path, global_step=step)
                logging.info("{} Store model to {}".format(now_time(), last_store_path))

                # eval model
                self.eval_one(sess)

            log_writer.close()

if __name__ == '__main__':
    print("main sys.argv", sys.argv)
    if len(sys.argv) >= 3:
        FLAGS.train_paths = '{}/part*'.format(sys.argv[1])
        FLAGS.eval_paths = '{}/part*'.format(sys.argv[2])
    logging.info("lr: %s" % FLAGS.learning_rate)
    logging.info("batch_size: %s" % FLAGS.batch_size)
    logging.info("dropout: %s" % FLAGS.dropout)
    logging.info("device: %s" % FLAGS.device)
    logging.info("train_epochs: %s" % FLAGS.train_epochs)
    logging.info("train_paths: %s" % FLAGS.train_paths)
    logging.info("eval_paths: %s" % FLAGS.eval_paths)
    logging.info("model_path: %s" % FLAGS.model_path)

    trainer = DNNTrainer()
    trainer.train()

