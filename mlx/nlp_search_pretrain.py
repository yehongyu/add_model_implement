#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Authors: Shenjian Zhao
#   Date: 2019/12/05 5:59 PM
#
import json
import os

import tensorflow as tf
from absl import logging
from absl import flags
from absl import app

from archer import models
from archer.models.model_config import parse_json_model_config
from archer.utils.info_log import logging_dict
from archer.utils.ckpt_manipulate import get_assignment_map_from_checkpoint
from archer.data.tfrecord_loader import tpu_input_fn_builder
from archer.losses.mlm_output import get_masked_lm_output, \
  get_masked_lm_output_split_batch
from archer.losses.sentence_level_output import get_next_sentence_output
from archer.optimization import optimizers

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_config", None,
    "The model config file which define the configuration of model, "
    "dimension etc")

flags.DEFINE_string(
    "train_config", None,
    "The train params file which define the params of training, optimizer, "
    "lr etc")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "input_dir", None,
    "The input directory where the training will read from.")

flags.DEFINE_bool(
    "use_tpu", True,
    "If we use TPU or not, default it is true.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

def model_fn_builder(model_config,
                     train_params):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""
    logging.info("*** Features ***")
    for name in sorted(features.keys()):
      logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = getattr(models, model_config.model_name)(config=model_config,
                                                     is_training=is_training)
    _ = model(input_ids, input_mask=input_mask, token_type_ids=segment_ids)

    # TODO (@zhaoshenjian.01): check conditional_jit_scope
    # split loss calculation across batch
    batch_splits = train_params.get("batch_splits", 1)
    if batch_splits == 1:
      # sparse_softmax_cross_entropy_with_logits
      masked_lm_output_dict = get_masked_lm_output(model_config,
                                                   model.get_sequence_output(),
                                                   model.get_embedding_table(),
                                                   masked_lm_positions,
                                                   masked_lm_ids,
                                                   masked_lm_weights)
    else:
      # use for large vocab
      masked_lm_output_dict = get_masked_lm_output_split_batch(
          model_config,
          model.get_sequence_output(),
          model.get_embedding_table(),
          masked_lm_positions,
          masked_lm_ids,
          masked_lm_weights,
          batch_splits=batch_splits)

    masked_lm_loss = masked_lm_output_dict["loss"]

    use_nsp = train_params.get("use_nsp", True)
    if use_nsp:
      next_sentence_labels = features["next_sentence_labels"]
      next_sentence_output_dict = get_next_sentence_output(
          model_config, model.get_pooled_output(), next_sentence_labels)
      next_sentence_loss = next_sentence_output_dict["loss"]
    else:
      next_sentence_loss = 0

    total_loss = masked_lm_loss + next_sentence_loss

    tvars = tf.compat.v1.trainable_variables()
    # run init
    init_checkpoint = train_params.get("init_checkpoint")
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map,
       initialized_variable_names) = get_assignment_map_from_checkpoint(
           tvars, init_checkpoint)
      tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint,
                                      assignment_map)
        return tf.train.Scaffold()
      scaffold_fn = tpu_scaffold
    logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      logging.info("  name = {}, shape = {} {}".format(var.name, var.shape,
                                                       init_string))

    # default `bert_decay` lr_scheduler
    lr_params = train_params.get(
        'lr_scheduler', {
            'name': 'bert_decay',
            'learning_rate': 1e-4,
            'warmup_steps': 10000,
            'num_train_steps': 1000000
        })
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op, _ = optimizers.create_optimizer(
          loss=total_loss,
          init_lr=lr_params['learning_rate'],
          num_train_steps=lr_params['num_train_steps'],
          num_warmup_steps=lr_params['warmup_steps'])

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                    loss=total_loss,
                                                    train_op=train_op,
                                                    scaffold_fn=scaffold_fn)
      return output_spec
    raise NotImplementedError

  return model_fn


def main(_):
  # load model and train config
  model_config = parse_json_model_config(FLAGS.model_config)

  logging.info(" %s" % FLAGS.output_dir)

  # Automatically load the tensorboard in the system.
  tensorboard_cmd = "tensorboard  --port=0 --logdir=" + FLAGS.output_dir + " &"
  os.system(tensorboard_cmd)

  with tf.io.gfile.GFile(FLAGS.train_config, "r") as reader:
    text = reader.read()
    train_params = json.loads(text)
  logging_dict("Train params: {}.", train_params)
  logging_dict("Model config: {}.", model_config.to_dict())

  # input files
  input_files = []
  input_dir = FLAGS.input_dir.split(",")  if FLAGS.input_dir != None else train_params['input_files']
  for input_pattern in  input_dir:
    input_files.extend(tf.io.gfile.glob(input_pattern))
  tf.compat.v1.logging.info("*** Input Files are ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)
  
  model_fn = model_fn_builder(model_config=model_config,
                              train_params=train_params)
  save_checkpoints_steps = train_params.get('save_checkpoints_steps', 100)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu:
    # Resolve TPU cluster and runconfig for this.
    from oauth2client.client import GoogleCredentials
    credentials = GoogleCredentials.get_application_default()

    tpu_name = os.environ['ARNOLD_TPU_NAME'].split(";")[0]
    tpu_zone = os.environ['ARNOLD_TPU_ZONE']
    tpu_project = os.environ['ARNOLD_TPU_PROJECT']

    tpu_type = os.environ['ARNOLD_TPU_TYPE'].split("-")[0]
    tpu_count = int(os.environ['ARNOLD_TPU_TYPE'].split("-")[1])
    
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      tpu=tpu_name,
      zone=tpu_zone,
      project=tpu_project,
      credentials=credentials)


  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    master=None,
    model_dir=FLAGS.output_dir,
    save_checkpoints_steps=save_checkpoints_steps,
    keep_checkpoint_max=100,
    tpu_config=tf.contrib.tpu.TPUConfig(
      iterations_per_loop=FLAGS.iterations_per_loop,
      num_shards=tpu_count,
      per_host_input_for_training=is_per_host))

  if tpu_type == 'v2':
    training_batch_size = int((train_params['batch_size'] * tpu_count) / 2)
  elif tpu_type == 'v3':
    training_batch_size = (train_params['batch_size'] * tpu_count)
  else:
    training_batch_size = train_params['batch_size']

  estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=FLAGS.use_tpu,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=training_batch_size,
    eval_batch_size=8)

  # train
  logging.info("***** Running training *****")
  logging.info("  train_params: {} ".format(train_params))
  logging.info("  Batch size = %d", training_batch_size)

  train_input_fn = tpu_input_fn_builder(
      input_files=input_files,
      max_seq_length=train_params['max_seq_length'],
      max_predictions_per_seq=train_params['max_predictions_per_seq'],
      is_training=True)

  estimator.train(input_fn=train_input_fn,
                  max_steps=train_params['num_train_steps'])


if __name__ == "__main__":
  flags.mark_flag_as_required("model_config")
  flags.mark_flag_as_required("train_config")
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("input_dir")

  app.run(main)

