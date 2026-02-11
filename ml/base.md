# 机器学习基础

三要素

- 模型：根据具体问题，确定假设空间。
- 策略：根据评价标准，确定选取最优模型的策略（损失函数）。
- 算法：求解损失函数，确定最优模型。


## 基本术语

算法：从数据学得“模型”的具体方法，线性回归、对数几率
决策树等。

模型：算法产出的结果，通常是具体的函数或者可抽象看作为函数。

样本：关于一个事件或对象的描述。一般数据是向量的形式，向量的各个维度称为“特征”或者“属性”

样本空间：也称为“输入空间”或“属性空间”。表示样本的特征向量所在的空间为样本空间。

标记空间：标记所在空间称为“标记空间”或“输出空间”。

学习任务：标记取值为离散，称此类任务为“分类”。标记取值为连续时，称此类任务为“回归”。

监督学习：模型训练阶段有用到标记信息。（线性模型等）

无监督学习：模型训练阶段没用到标记信息。（聚类等）

数据集：数据集通常用集合来表示，令集合 $D = \{x_1,x_2,\cdots,x_m \}$ 表示包含 $m$ 个样本的数据集，一般同一份数据集的每个样本都含有相同个数的特征，假设此数据集中的每个样本都含有 $d$ 个特征，则第 $i$ 个样本的数据表示为 $d$ 维向量：$x_i = \{ x_{i1}; \cdots; x_{id} \}$，其中 $x_{ij}$ 表示样本 $x_i$ 在第 $j$ 个属性上的取值。

泛化：对未知物判断的准确与否才是衡量一个模型的关键，我们称此为“泛化”能力。



## 数学

### 极大似然估计（MLE）

使得观测样本出现概率最大的分布

设样本 $X_1, ..., X_n$ 独立同分布，概率密度（或质量）函数为 $f(x; \theta)$ ，其中 $\theta$ 是待估参数。

似然函数：

$$
L(\theta) = L(\theta; x_1, ..., x_n) = \prod_{i=1}^{n} f(x_i; \theta)
$$

对数似然：

方便求导，单调递增

$$
\ell(\theta) = \ln L(\theta) = \sum_{i=1}^{n} \ln f(x_i; \theta)
$$

### 凸集

集合 $C \subseteq \mathbb{R}^n$ 是凸集，当且仅当：

$$
\forall x, y \in C, \quad \forall \theta \in [0, 1], \quad \theta x + (1-\theta)y \in C
$$

$\theta x + (1-\theta)y$ 就是 $x$ 和 $y$  之间的凸组合（线段上的任意点）。

| 凸集                             | 非凸集      |
| ------------------------------ | -------- |
| 整个空间 $\mathbb{R}^n$            | 两个分离的圆盘  |
| 半空间 $\{x \mid a^Tx \leq b\}$   | 圆环（中间有洞） |
| 球体 $\{x \mid \|x\|_2 \leq r\}$ | 月牙形区域    |
| 单纯形（概率单纯形）                     | 任意非连通集合  |

### 凸函数

函数 $f: \mathbb{R}^n \to \mathbb{R}$ 是凸函数，当且仅当（最优化）：

$$
\forall x, y \in \text{dom}(f), \quad \forall \theta \in [0, 1]:
$$

$$
f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)
$$

### 梯度

对于函数 $f: \mathbb{R}^n \to \mathbb{R}$ （输入向量，输出标量），其梯度为：

$$
\nabla f(x) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}
$$

所有偏导数组成的列向量。

### Hessian 矩阵

对于 $f: \mathbb{R}^n \to \mathbb{R}$，Hessian 矩阵 $\mathbf{H} \in \mathbb{R}^{n \times n}$：

$$
\mathbf{H} = \nabla^2 f(x) = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\[6pt]
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\[6pt]
\vdots & \vdots & \ddots & \vdots \\[6pt]
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
$$

所有二阶偏导数组成的对称矩阵（若 $f$ 二阶连续可微，则$\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$）

设 $D \in \mathbb{R}^n$ 是非空开凸集，$f(x)$ 是定义在$D$上的实值函数，且 $f(x)$ 在 $D$ 上二阶连续可微，如果 $f(x)$ 的 Hessian 矩阵 $\nabla^2f(x)$ 在 $D$ 上是半正定的，则 $f(x)$ 是 $D$ 上的凸函数；如果 $\nabla^2f(x)$ 在
$D$ 上是正定的，则 $f(x)$ 是 $D$ 上的严格凸函数。


证正定：顺序主子式全 $> 0$（西尔维斯特）

证正半定：所有主子式 $\ge 0$，或直接用二次型 $z^\top H z \geq 0$（更常用）


### 矩阵微分

#### 标量求导

$$
(uv)' = u'v + uv'
$$

#### 迹

是方阵对角线元素之和，记作 $\text{tr}(A)$ 。在矩阵求导中，它是把标量包装成矩阵形式的利器，核心作用是让乘法顺序可以轮换。

$$
\text{tr}(A) = \sum_{i=1}^n A_{ii} = A_{11} + A_{22} + \dots + A_{nn}
$$

性质

$$
\text{tr}(ABC) = \text{tr}(BCA) = \text{tr}(CAB)
$$

套路：

1. 标量 $a = \text{tr}(a)$ （包装）
2. $\text{tr}(ABC) = \text{tr}(CAB)$ （轮换，把 $dx$ 转到合适位置）
3. 或直接利用 $u^\top v = v^\top u$ （标量转置等于自身）


#### 矩阵求导 

| 函数形式                        | 导数                                    | 备注                      |
| --------------------------- | ----------------------------------------------- | ----------------------- |
| $f(x) = x^\top A x$         | $\frac{\partial f}{\partial x} = (A + A^\top)x$ | **一般情况**                |
| $f(x) = x^\top A x$（$A$ 对称） | $\frac{\partial f}{\partial x} = 2Ax$           | **最常见**（如 $A=X^\top X$） |
| $f(x) = a^\top x$           | $\frac{\partial f}{\partial x} = a$             | 线性项                     |


### 拉格朗日乘法

对于仅含等式约束的优化问题：

$$
min \; f(x) \\
s.t. \; h_i(x) = 0 \; i = 1,2,3,\cdots,n
$$

其中自变量 $x \in \mathbb{R}^n$，$f(x)$ 和 $h_i(x)$ 均有连续的一阶偏导数，其拉格朗日函数：

$$
L(x, \lambda) = f(x) + \sum_{i=1}^n \lambda_i h_i(x)
$$

其中 $\lambda = (\lambda_1,\lambda_2,\cdots,\lambda_n)^T$ 为拉格朗日乘子。然后对拉格朗日函数关于 $x$ 求偏导，并令导数等于 $0$ 再搭配约束条件 $h_i(x) = 0$解出 $x$，求解出的所有 $x$ 即为上述优化问题的所有可能的极值点。

### 广义特征值

设 $A$、$B$ 为 $n$ 阶方阵，若存在 $\lambda$，使得方程 $Ax = \lambda Bx$ 存在非零解，则称 $\lambda$ 为 $A$ 相对与 $B$ 的广义特征向值，$x$ 为 $A$ 相对与 $B$ 的广义特征向值 $\lambda$ 的特征向量。