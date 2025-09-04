| 第五届计图挑战赛

#  第五届Jittor算法挑战赛--赛道一

## 简介
1、本项目围绕着（Jittor算法挑战赛赛道一：超声图像的智能筛查与分级）进行，核心任务为：基于收集的乳腺癌超声影像数据集，运用深度学习技术构建分类模型，克服数据类别不平衡问题，实现对乳腺癌超声影像的精准分类。

2、模型基座：由于赛题要求全部算法流程包括模型构建，要基于jittor框架而非pytorch框架，我们调研了jittor框架下可以加载imagenet的模型库和代码，最后选用了优秀的开源模型代码库jimm作为模型backbone的基座库，jimm是由早期版本的timm改动而来，虽然版本较老但可以使用较多经典模型及imagenet预训练权重。

3、 最终我们使用effv2s作为backbone构建的多尺度特征融合的分类模型，并且训练过程中模型使用multi dropout层，解决模型快速过拟合的问题。

## 安装 

#### 运行环境、安装依赖
- 显卡：RTX 4090D(24GB)
- 系统：ubuntu 18.04
- Cuda: 12.1
- Python: 3.12.2

```
pip install -r requirements.txt
# torch需要单独安装对应cuda的gpu版本
```

## 训练

```
python train.py
```

## 推理

```
bash test.sh
```
