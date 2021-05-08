#!/bin/bash
#coding=utf-8
BIN_DIR=xgboost_runner
bvc clone data/reckon/xgboost_runner $BIN_DIR

# IO arguments
INPUT_PATHS=hdfs:///recommend/tmp/ad_creative_pruning_model_train_9048_v2_r99599.data/part_*
MODEL_PATH=hdfs:///recommend/tmp/ad_creative_pruning_model_train_9048_v2_r99599.xgb
OUTPUT_DIR=hdfs:///recommend/tmp/ad_creative_pruning_model_train_9048_v2_r99599.evaluated
/opt/tiger/yarn_deploy/hadoop/bin/hadoop fs -rm -r $OUTPUT_DIR

# Submit!
NUM_WORKERS=40
XGBOOST_VERSION=0.83-SNAPSHOT
YARN_CLUSTER_NAME=default /opt/tiger/spark_deploy/spark-stable/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --name ad_creative_pruning_model_train_9048_v2_r99599_eval_$USER \
    --num-executors $NUM_WORKERS \
    --executor-cores 1 \
    --executor-memory 4g \
    --conf spark.driver.memory=4g \
    --conf spark.driver.cores=4 \
    --conf spark.dynamicAllocation.enabled=false \
    --conf spark.task.cpus=1 \
    --conf spark.yarn.maxAppAttempts=1 \
    --queue root.reckon.gbdt \
    --class com.bytedance.aml.xgboost.prediction.App \
    --jars ${BIN_DIR}/xgboost4j-${XGBOOST_VERSION}.jar,${BIN_DIR}/xgboost4j-spark-${XGBOOST_VERSION}.jar \
    ${BIN_DIR}/xgboost-runner-0.3-SNAPSHOT.jar \
    --feature-start-column 1 \
    --input-paths $INPUT_PATHS \
    --model-path $MODEL_PATH \
    --num-features 643 \
    --num-workers $NUM_WORKERS \
    --objective binary:logistic \
    --output-path $OUTPUT_DIR
