# 说明

本实例用于加载您训练好的 SavedModel 文件，用本地的 TensorFlow 引擎执行一次推理，生成可视化的性能计量文件。

## 使用

将您的 SavedModel 目录准备在本地，该目录下有一个 `saved_model.pb` 文件。

请执行 Bash 命令，校验模型的合法性：
```bash
saved_model_cli show --dir /path/to/saved_model --tag_set serve --signature_def serving_default
```

修改 `profile.py` 的代码，填对模型目录名，对齐输入和输出的 Tensor 名称和形状。


在合适的 Python 环境下，执行 Bash 命令：
```bash
python -u profile.py --batch_size=64 --warmup_rounds=2 --trace_path=timeline.json
```

## 查看

打开 Chrome 浏览器，在地址栏输入 `chrome://tracing`，点击 `Load` 按钮，选中生成的性能计量文件（上一步名为 `timeline.json`）。