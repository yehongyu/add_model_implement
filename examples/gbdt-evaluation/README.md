# 说明
如果您有以下需求，那么 XGBoost 的评估模式将是不二选择。
- 对 XGBoost on Reckon 的效果指标存疑。
- 想用预训练好的 XGBoost 模型来对大规模的 LibSVM 数据进行算分。

本项目提供了一些示例代码，方便您使用或二次开发。

## 本地算分且评估
请将 XGBoost 模型和训练数据都在本地准备好，接着执行工具脚本。
```bash
python -u predict_and_eval.py \
    --model-path=xgboost.mdl \
    --input-path=test_data.libsvm \
    --num-features=128 \
    --slot-masks=3,4,5 \
    --threshold=0.5
```

特别提醒， `predict_and_eval.py` 只支持二分类模型，如有其他模型请二次开发。

## 分布式算分再本地评估
首先开启算分模式，在 `predict.sh` 中设置好 HDFS 上模型和测试数据的路径，提交一个 XGBoost on Spark 作业。
```bash
bash -x predict.sh # 用 XGBoost on Spark 算分
```

接着把算分的结果和模型下载到本地，执行工具脚本。
```bash
python -u eval.py \
    --output-path=predicted.tsv \
    --threshold=0.5
```

特别提醒，虽然 XGBoost on Spark 支持二分类、多分类、回归等模型的算分，但 `eval.py` 只支持二分类模型，如有其他模型请二次开发。
