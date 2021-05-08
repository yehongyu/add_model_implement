# 说明
本项目提供了一个样例 MapReduce 程序，用于将 Instance PB 格式的推荐/广告样本转化成 TFRecord 格式，方便被 [tf.data.TFRecordDataset](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset) 消费。

# 使用
以下各步骤均是可选的，请根据您的训练需求进行配置。

## 1. 修正 Instance PB 的读取逻辑
请检查 [map.py](map.py) 中 read_xx_instance() 函数的实现，是否匹配了您的 Instance PB 的格式。一般来说，数据流处理器的 kafka_dump 和 has_sort_id 两个参数会影响二进制流的组成。

## 2. 自定义 tf.Example 的填充内容
请修改 [map.py](map.py) 中 dump_tf_example() 函数的实现，将您需要的 [Instance PB](https://code.byted.org/data/idl/blob/master/matrix/proto/proto_parser.proto) 的字段填入 [tf.Example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) 中。您可以尽可能多地抽取感兴趣的数据，因为 TensorFlow 读取数据时，可以按需反序列化字段。

## 3. 配置 MapReduce 的作业参数
请修改 [submit.sh](submit.sh) 中一下参数：
- `YARN_CLUSTER_NAME` 是有权限的 Yarn 集群。
- `mapred.job.queue.name` 是有权限的 Yarn 队列。
- `OUTPUT_DIR` 是期望的 HDFS 输出目录。
- `input` 是输入文件的路径描述，支持通配符。
- `mapred.output.compress` 是 Gzip 压缩的开关。

此时，可以执行 Bash 命令测试一下：
```bash
bash -x submit.sh
```

## 4. 注册成天级或小时级的定时任务
推荐在 [Unicron](https://bytedance.feishu.cn/space/doc/doccnkUyf1hg9MBobzhDmD) 或 [Dorado](https://data.bytedance.net/dorado) 两个平台上托管您的 MapReduce 作业，从而激活在 [Reckon-Forge](http://reckon.bytedance.net/forge) 的小时级、天级的追新训练功能。