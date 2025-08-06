[English](README_en.md) | 中文

# MindCos

- [MindCos介绍](#mindcos介绍)
- [应用场景](#应用场景)
- [安装教程](#安装教程)

## MindCos介绍

宇宙学是研究宇宙本性、起源和演化的科学领域，依赖于望远镜、粒子加速器等先进设备产生的复杂实验数据。这些实验生成的数据量庞大且复杂，传统数据处理方法难以应对这一挑战。MindCos利用先进的深度学习技术，提升了宇宙学数据分析的效率和准确性。

MindCos基于昇思MindSpore框架开发，是一个专为宇宙学研究设计的AI工具包。它支持多种应用场景，包括喷注鉴别、磁场预测以及Grad–Shafranov方程、亥姆霍兹方程和Lane–Emden方程的求解。这些应用在宇宙学领域具有重要意义：
- **喷注鉴别**：分析高能碰撞产生的粒子喷注，用于研究夸克-胶子等离子体和宇宙早期事件，助力理解宇宙初期状态。
- **磁场预测**：求解麦克斯韦方程，建模宇宙结构中的磁场或加速器、对撞机中的磁场
- **Grad–Shafranov方程**：求解天体物理环境中等离子体的平衡问题，如恒星磁场或围绕黑洞等致密天体的吸积盘，适用于研究宇宙学系统中的磁流体力学过程。
- **亥姆霍兹方程**：建模宇宙学模拟中的波传播，适用于研究引力波或宇宙微波背景波动，使用矩形域内的合成边界和配点数据。
- **Lane–Emden方程**：描述多方程星体的结构，对研究恒星演化、恒星形成和银河系动态等宇宙学问题至关重要。

与传统方法相比，MindCos显著降低了计算成本，缩短了数据处理时间，同时提高了结果的准确性。它为科研人员、高校教师和学生提供了一个高效、易用的工具，推动宇宙学研究的进一步发展。

## 应用场景

下表总结了支持的应用场景、数据集、模型架构及硬件兼容性：

| 应用场景 | 数据集 | 模型架构 | CPU | 昇腾 |
|----------|--------|----------|-----|------|
| Grad–Shafranov方程（恒星磁场） | D形轮廓上的合成边界点和配点数据，可用于恒星磁场等离子体平衡模拟 | PINN | ✔️ | ✔️ |
| Grad–Shafranov方程（吸积盘） | 滴状轮廓上的合成边界点和配点数据，可用于吸积盘等离子体平衡模拟 | PINN | ✔️ | ✔️ |
| 亥姆霍兹方程 | 矩形域内的合成边界和配点数据，用于宇宙波传播模拟 | PINN | ✔️ | ✔️ |
| Lane–Emden方程 | 笛卡尔坐标系中的合成边界和配点数据，用于多方程星体结构模拟 | PINN | ✔️ | ✔️ |
| 喷注鉴别 | 夸克-胶子喷注数据集 | LorentzNet | ✔️ | ✔️ |
| 磁场预测 | 三维空间点及基于麦克斯韦方程的磁场数据，用于宇宙磁场或加速器环境磁场模拟 | PINN | ✔️ | ✔️ |

## 安装教程

按照以下步骤安装MindCos及其依赖项。

### 安装MindSpore
```bash
# 参考MindSpore官方安装指南
https://www.mindspore.cn/install
```

示例安装命令：
```bash
conda create -n MindCos python==3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.14/MindSpore/unified/x86_64/mindspore-2.2.14-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 安装依赖库
```bash
pip install -r requirements.txt
```

### 安装GPU版本的MindSpore
有关昇腾或GPU支持的安装，请参考[安装说明](gpu_version_install.txt)。