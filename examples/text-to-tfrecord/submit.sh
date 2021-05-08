#!/bin/bash
export HADOOP_HOME=${HADOOP_HOME:-/opt/tiger/yarn_deploy/hadoop}

# Prepare output
OUTPUT_DIR=/recommend/tmp/tfrecord/search_raw_rank
${HADOOP_HOME}/bin/hadoop fs -rm -r ${OUTPUT_DIR}

# Call the MR client
CUR_DIR=`cd $(dirname $0); pwd`
DOCKER_ENV="YARN_CONTAINER_RUNTIME_TYPE=docker,YARN_CONTAINER_RUNTIME_DOCKER_IMAGE=hub.byted.org/instance_to_tfrecord:4e1ec290737250e72e3e92976eb17dba"
${HADOOP_HOME}/bin/hadoop jar ${HADOOP_HOME}/share/hadoop/tools/lib/hadoop-streaming-2.6.0-cdh5.4.4.jar \
-libjars ${HADOOP_HOME}/bytedance-data-1.0.1.jar,${CUR_DIR}/tensorflow-hadoop-1.13.2-SNAPSHOT.jar \
-Dmapred.job.name=text.to.tfrecord_$USER \
-Dmapred.job.map.memory.mb=5120 \
-Dmapred.job.reduce.memory.mb=5120 \
-Dmapred.job.queue.name=root.lagrange_lite \
-Dmapred.job.priority=VERY_HIGH \
-Dmapred.reduce.tasks=0 \
-Dmapred.output.compress=true \
-Dmapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec \
-Dmapreduce.input.combinefileformat.tasks=100 \
-Dmapreduce.map.env=${DOCKER_ENV} \
-Dstream.io.identifier.resolver.class=com.bytedance.hadoop.mapred.PBIdResolver \
-Dstream.map.input=text \
-Dstream.map.input.ignoreKey=true \
-Dstream.map.output=pb \
-inputformat com.bytedance.data.CustomCombineFileInputFormat \
-outputformat org.tensorflow.hadoop.io.TFRecordFileOutputFormatV1 \
-input '/data/kafka_dump/raw_rank_text_feature/2019081[6-7]/*' \
-output ${OUTPUT_DIR} \
-mapper "python map.py $*" \
-file ${CUR_DIR}/map.py
