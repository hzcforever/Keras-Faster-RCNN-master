# 医学图像目标检测

## 实验背景

本实验基于两千多张医学图像的私有数据集，拟通过预训练好的 model 和 weights，结合 faster-rcnn 目标检测模型对预处理后的甲状腺超声图像数据集进行再训练，从而达到对甲状腺结节进行定位的目的，辅助甲状腺结节超声图像的分类。

**不包含目标检测的迁移学习分类**

直接通过 vgg19 对预处理后的数据集直接进行图像分类，分类准确率为62.0%，TP = 54，FP = 46，TN = 70，FN = 30，敏感性 = 64.3%，特异性 = 60.3%。由此可见，将迁移学习直接用于医学图像分类，效果并不理想，因此以下实验想通过先进行目标(病变结节)检测再分类的思想，进一步提高模型性能。

## 实验环境及配置

实验所用的服务器共32块 CPU、8块 GPU（8G/块）。实验的运行环境配置如下：Ubuntu 18.04、Python 3.6、Keras 2.1.6、Pytorch 0.4.1、Tensorflow 1.10.0(supports AVX and AVX2)。

先通过 Xftp 连接服务器，将代码和数据集上传，再通过 Xshell 连接服务器，连接成功之后，根据需要可以激活不同的虚拟环境，有 tensorflow、pytorch、keras、caffe 可以选择，通过 source activate (keras、tensorflow 等) 激活后，便可运行模型。

## 数据集的标注

通过 labelImg 标记软件对医学图像数据集进行标注，格式为 PascalVOC(一张图像对应一个 XML 文件)。

<div align="center"><img src="/project_img//thyroid_label.jpg" width="800px"/></div>
<div align="center"><b>lableImg</b></div>

<div align="center"><img src="/project_img//thyroid_xml.jpg" width="700px"/></div>
<div align="center"><b>XML 文件</b></div>

训练集与测试集以5：1的比例分割，其中存在数据集的不均衡现象，即 negative(1100) 比 positive(520) 的数据量多出将近一半，暂时将 negative 的数据截取与 positive 数据相同数据量的图像，之后再考虑通过数据增强等方式增加 positive 的数据量。

## 代码的简单说明

- pascal_voc_parser.py: 数据预处理，读取图片和对应 XML 文件的内容并解析 XML 文件结构，将其转换成"path/image.jpg,  x1, y1, x2, y2, class_name"这种格式
  - 输入为数据路径，图片存储在 JPEGImages 中，标注存储在 Annotations 中
  - 目前文件中将路径中的所有图片都设定为训练集，如果要分割测试集，则需自行调整代码

- train_frcnn.py: 模型训练。
  - 通过 num_epochs 和 epoch_length 分别设置训练次数和每次的训练长度
  - 代码里面没有设置 validation set，只针对训练集

- test_frcnn.py: 测试模型
  - 将图片放到 test 文件夹中
  - 需要读取训练时保存的 config.pickle 文件和训练好的 model.hdf5

## 训练过程及结果展示

### 模型的预训练

<div align="center"><img src="/project_img//train.jpg" width="800px"/></div>

利用四百多张细胞显微图进行模型的预训练，Epoch=12，epoch_length=200，每个 epoch 耗时约 30 min。

对细胞测试集进行目标检测：

<div align="center"><img src="/project_img//test.jpg" width="800px"/></div>

部分结果如下：

<div align="center"><img src="/project_img//1.png" width="500px"/></div>
<div align="center"><b>cell detection</b></div>

<div align="center"><img src="/project_img//2.png" width="500px"/></div>
<div align="center"><b>cell detection</b></div>

### 私有数据集训练

保存预训练好的模型为 dataset/model.h5，并通过标注好的甲状腺超声图像私有数据集上，继续训练。

由于数据集标注工作比较繁琐，现用201张已标注的甲状腺图像在 pre_model 的基础上训练，等后续所有数据集标注完成后再加入新的数据。训练的 Epoch=12，epoch_length=100，每个 epoch 耗时约 7 min。

训练过程如下：

<div align="center"><img src="/project_img//thyroid_train.jpg" width="800px"/></div>

对若干甲状腺测试图片进行预测：

<div align="center"><img src="/project_img//thyroid_test.jpg" width="700px"/></div>

可以看出，其中有的图像出现了多个目标检测框，总的分类效果已经比只用迁移分类模型好很多了，具体的性能参数值后续再通过计算列出来。

部分结果如下：

<div align="center"><img src="/project_img//negative1.png" width="500px"/></div>
<div align="center"><b>negative</b></div>

<div align="center"><img src="/project_img//negative2.png" width="500px"/></div>
<div align="center"><b>negative</b></div>

<div align="center"><img src="/project_img//positive1.png" width="500px"/></div>
<div align="center"><b>positive</b></div>

<div align="center"><img src="/project_img//positive2.png" width="500px"/></div>
<div align="center"><b>positive</b></div>

## 实验结果分析

性能指标：总体分类准确率、TP、FP、TN、FN、准确率、召回率、AP、mAP、mmAP...
