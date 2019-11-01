# 医学图像目标检测

## 实验背景

本实验基于两千多张私有数据集，通过 pretrained_model 和 weights 结合 faster-rcnn 对甲状腺结节进行目标检测，从而辅助医学图像的分类。

## 实验环境

实验所用的服务器部署在国外，服务器共32块 CPU、8块 GPU（8G/块）。框架版本如下：Python 3.6、Keras 2.2.4、Pytorch 0.4.1。

先通过 Xftp 6 连接服务器，将代码和数据集上传，再通过 Xshell 连接服务器进行实验，连接成功之后，根据不同模型的运行环境需要激活对应的虚拟环境，有 tensorflow、pytorch、keras、caffe 可以选择，通过 source activate (keras、tensorflow等) 激活，之后就可以运行程序。

## 数据集的标注

## 代码说明

- pascal_voc_parser.py: 数据预处理，读取图片和其 VOC 格式(xml)的标注，转换成"path/image.jpg,  x1, y1, x2, y2, class_name"这种格式
  - 输入为数据路径，图片存储在 JPEGImages 中，标注存储在 Annotations 中
  - 目前文件中将路径中的所有图片都设定为训练集，如需分割测试集，则自行调整代码

- train_frcnn.py: 模型训练。
  - 通过 num_epochs 和 epoch_length 设置训练次数和每次的训练长度
  - 代码里面没有设置 validation set，只针对训练集

- test_frcnn.py: 测试模型
  - 将图片放到 test 文件夹中
  - 需要读取训练时保存的 config.pickle 文件和训练好的模型 model.hdf5

## 训练过程

## 实验结果与展示
