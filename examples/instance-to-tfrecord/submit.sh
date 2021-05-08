#!/bin/bash
export HADOOP_HOME=${HADOOP_HOME:-/opt/tiger/yarn_deploy/hadoop}

OUTPUT_DIR=/recommend/tmp/tfrecord/output
${HADOOP_HOME}/bin/hadoop fs -rm -r ${OUTPUT_DIR}

CUR_DIR=`dirname $0`
DOCKER_ENV="YARN_CONTAINER_RUNTIME_TYPE=docker,YARN_CONTAINER_RUNTIME_DOCKER_IMAGE=hub.byted.org/instance_to_tfrecord:4e1ec290737250e72e3e92976eb17dba"
${HADOOP_HOME}/bin/hadoop jar ${HADOOP_HOME}/share/hadoop/tools/lib/hadoop-streaming-2.6.0-cdh5.4.4.jar \
-libjars ${HADOOP_HOME}/bytedance-data-1.0.1.jar,${CUR_DIR}/tensorflow-hadoop-1.13.2-SNAPSHOT.jar \
-archives ${CUR_DIR}/idl.tar.gz#idl \
-Dmapreduce.map.env=${DOCKER_ENV} \
-Dstream.io.identifier.resolver.class=com.bytedance.hadoop.mapred.PBIdResolver \
-Dstream.map.input=pb \
-Dstream.map.output=pb \
-Dmapred.job.name=instance.to.tfrecord_$USER \
-Dmapred.job.map.memory.mb=1024 \
-Dmapred.job.reduce.memory.mb=1024 \
-Dmapred.job.queue.name=root.lagrange_lite \
-Dmapred.job.priority=VERY_HIGH \
-Dmapred.reduce.tasks=0 \
-Dmapred.output.compress=true \
-Dmapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec \
-inputformat com.bytedance.hadoop.mapred.PBInputFormat \
-outputformat org.tensorflow.hadoop.io.TFRecordFileOutputFormatV1 \
-input '/data/kafka_dump/ad_online_joiner_output_test_gq/20190729/22_*' \
-output ${OUTPUT_DIR} \
-mapper "python map.py $*" \
-file ${CUR_DIR}/map.py
 
