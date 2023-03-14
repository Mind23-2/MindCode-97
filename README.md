
# 目录

<!-- TOC -->

- [目录](#目录)
- [概述](#概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
        - [训练](#训练)
    - [参数](#参数)
        - [参数](#参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
            - [Ascend处理器上运行](#ascend处理器上运行)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# 概述

Multi task learning (MTL) has been used in many NLP tasks to obtain better language representations. Hence, we experiment with several auxiliary tasks to improve the generalization capability of a MRC model. The auxiliary tasks that we use include

 - Unsupervised Task: masked Language Model
 - Supervised Tasks:
   -  natural language inference
   -  paragraph ranking

# 模型架构

D-NET的主干结构为BERT。下面接了3个下游任务。

# 数据集

To download the MRQA training and development data, as well as other auxiliary data for MTL, run

```
bash wget_data.sh
```
The downloaded data will be saved into `data/mrqa` (combined MRQA training and development data), `data/mrqa_dev` (seperated MRQA in-domain and out-of-domain data, for model evaluation), `mlm4mrqa` (training data for masked language model task) and `data/am4mrqa` (training data for paragraph matching task).

# 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend或GPU处理器搭建硬件环境。如需试用昇腾处理器，请发送[申请表](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx)至ascend@huawei.com，申请通过后，即可获得资源。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

从官网下载安装MindSpore之后，您可以按照如下步骤进行训练和评估：

- 在Ascend上运行

```bash

  bash run.sh

```

在Ascend设备上做分布式训练时，请提前创建JSON格式的HCCL配置文件。

在Ascend设备上做单机分布式训练时，请参考[here](https://gitee.com/mindspore/mindspore/tree/master/config/hccl_single_machine_multi_rank.json)创建HCCL配置文件。

在Ascend设备上做多机分布式训练时，训练命令需要在很短的时间间隔内在各台设备上执行。因此，每台设备上都需要准备HCCL配置文件。请参考[here](https://gitee.com/mindspore/mindspore/tree/master/config/hccl_multi_machine_multi_rank.json)创建多机的HCCL配置文件。

如需设置数据集格式和参数，请创建JSON格式的模式配置文件，详见[TFRecord](https://www.mindspore.cn/doc/programming_guide/zh-CN/master/dataset_loading.html#tfrecord)格式。

```text
Schema file contains ['task_id', 'src_ids', 'pos_ids', 'sent_ids', 'input_mask', 'start_positions', 'end_positions', 'mask_label', 'mask_pos', 'labels']

`numRows` is the only option which could be set by user, other values must be set according to the dataset.

For example, the schema file shows as follows:
{
    "datasetType": "TF",
    "numRows": 7680,
    "columns": {
        "task_id": {
            "type": "int64",
            "rank": 1,
            "shape": [1]
        },
        "src_ids": {
            "type": "int64",
            "rank": 1,
            "shape": [512]
        },
        "pos_ids": {
            "type": "int64",
            "rank": 1,
            "shape": [512]
        },
        "sent_ids": {
            "type": "int64",
            "rank": 1,
            "shape": [512]
        },
        "input_mask": {
            "type": "int64",
            "rank": 1,
            "shape": [512]
        },
        "start_positions": {
            "type": "int64",
            "rank": 1,
            "shape": [1]
        },
        "end_positions": {
            "type": "float32",
            "rank": 1,
            "shape": [1]
        }
        "mask_label": {
            "type": "float32",
            "rank": 1,
            "shape": [76]
        }
        "mask_pos": {
            "type": "float32",
            "rank": 1,
            "shape": [76]
        }
        "labels": {
            "type": "float32",
            "rank": 1,
            "shape": [1]
        }
    }
}
```

## 脚本说明

## 脚本和样例代码

```shell
.
└─D-NET
  ├─README_CN.md
  ├─backbone
    └─bert_model.py         # 骨干网络 bert
  ├─config
    ├─answer_matching.yaml                              # answer_matching 配置文件
    ├─mask_language_model.yaml                      # mask_language_model 配置文件
    └─reading_comprehension.yaml                           # reading_comprehension 配置文件
  ├─paradigm
    ├─joint_model.py                                     # 合并后的模型
    ├─answer_matching.py                              # answer_matching 模型文件
    ├─mask_language_model.py                      # mask_language_model 模型文件
    └─reading_comprehension.py                           # reading_comprehension 模型文件
  ├─reader
    ├─joint_reader.py                                     # 合并后的数据加载模块
    ├─answer_matching_reader.py                              # answer_matching 数据加载
    ├─mask_language_model_reader.py                      # mask_language_model 数据加载
    └─reading_comprehension_reader.py                           # reading_comprehension 数据加载
  ├─utils
    ├─batching.py                                     # 用于生成batch
    ├─configure.py                              # 处理参数
    └─tokenization.py                           # 将文本token化
  ├─mtl_config.yaml                              # 多任务训练参数
  ├─mtl_run.py                              # 训练脚本
  ├─run.sh                             # 训练脚本
  └─wget_data.sh                              # 数据集下载脚本
```

## 脚本参数

### 训练

```shell
用法：mtl_run.py  
```

## 参数

可以在`mtl_config.yaml`文件中分别配置参数。


### 参数

```text


Parameters for optimizer:
    learning_rate                   学习率
    weight_decay                    权重衰减

```

## 训练过程

### 用法

#### Ascend处理器上运行

```bash
bash run.sh
```


# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。

hello