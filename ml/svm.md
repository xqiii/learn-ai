# 支持向量机

## 算法原理

从几何角度，对于线性可分数据集，支持向量机就是找距离正负样本都远的超平面，相比于感知机，其解是唯一的，且不偏不倚，泛化性能更好。


### 超平面

$n$ 维空间的超平面 $(w^Tx + b = 0)$，其中 $w,x \in \mathbb{R^n}$：

- 超平面方程不唯一
- 法向量 $w$ 和位移项 $b$ 确定一个唯一超平面
- 法向量 $w$ 垂直于超平面
- 法向量 $w$ 指向的那一半空间为空间，另一半为负空间
- 任意点 $x$ 到超平面的距离公式为：
$$
r = \frac{w^Tx + b}{ \left \| w \right \| }
$$

### 几何间隔

对于给定的数据集 $X$ 和超平面 $w^Tx+b = 0$，定义数据集 $X$ 中的任意一个样本点 $(x_i,y_i)$，$y_i \in {-1,1}$，$i = 1,2,3,\cdots,m$ 关于超平面的几何间隔为：

$$
\gamma_i = \frac{y_i(w^Tx_i+b)}{\left \| w  \right \| }
$$

正确分类时：$\gamma_i > 0$，几何间隔等价于点到超平面的距离，没有正确分类时：$\gamma_i < 0$

对于给定的数据集 $X$ 和超平面 $w^Tx+b=0$，定义数据集 $X$ 关于超平面的几何间隔为：数据集 $X$ 中所有样本点的几何间隔最小值

$$
\gamma = \min_{i=1,2,\cdots,m} \; \gamma_i
$$

## 算法

模型：给定线性可分数据集 $X$，支持向量机模型希望求得数据集 $X$ 关于超平面的几何间隔 $\gamma$ 达到最大的那个超平面，然后套上一个 $sign$ 函数实现分类功能

$$
y = sign(w^Tx+b)
$$

## 策略

给定线性可分数据集 $X$，设 $X$ 中几何间隔最小的样本为 $(x_min, y_min)$，那么支持向量机找超平面的过程可以转换为带约束条件的优化问题：

$$
max \; \gamma \\
s.t. \gamma_i \ge \gamma，\; i = 1,2,\cdots,m
$$

展开如下：

$$
\max_{w,b} \frac{y_{min}(w^Tx_min + b)}{\left \| w  \right \|} \\

s.t. \; \frac{y_{i}(w^Tx_i + b)}{\left \| w  \right \|} \ge \frac{y_{min}(w^Tx_min + b)}{\left \| w  \right \|} , \; i = 1,2,\cdots,m
$$

$$
\max_{w,b} \frac{y_{min}(w^Tx_min + b)}{\left \| w  \right \|} \\

s.t. \; y_{i}(w^Tx_i + b) \ge y_{min}(w^Tx_min + b) , \; i = 1,2,\cdots,m
$$

假设该问题的最优解为 $(w^*,b^*)$，那么 $(\alpha w^*, \alpha b^*)$，$\alpha \in \mathbb{R}^+$ 也是最优解（尺度自由），且超平面不变，因此还需要对 $w, b$ 做一定限制才能使得上述优化问题有可解的唯一解。不妨零 $y_min(w^Tx_min+b) = 1$，因为对于特定的 $(x_min,y_min)$ 来说，能使得 $y_min(w^Tx_min+b) = 1$ 的 $\alpha$ 有且仅有一个。因此上述优化问题进一步转换为：

$$
\max_{w,b} \frac{1}{\left \| w  \right \|} \\
s.t. y_i(w^Tx_i + b) \ge 1, \; i = 1,2,\cdots,m
$$

进一步进行恒等变换，最大化问题转换为最小化问题：

$$
\min_{w,b} \frac{1}{2}\left \| w  \right \|^2 \\
s.t. 1 - y_i(w^Tx_i + b) \le 0, \; i = 1,2,\cdots,m
$$

此优化问题为含不等式约束的优化问题，且为凸优化问题，因此可以直接用很多专门求解凸优化问题的方法求解该问题。通常使用拉格朗日对偶求解。



### 一般约束优化问题
$$
min \; f(x) \\
s.t. \; g_i(x) \le 0, \; i = 1,2,\cdots,m \\
\; \; h_j(x) = 0, \; j = 1,2,\cdots,m
$$

若目标函数 $f(x)$ 是凸函数，约束集合是凸集，则称上述优化问题为凸优化问题，特别地，$g_i(x)$ 是凸函数，$h_j(x)$ 是线性函数时，约束集合为凸集，该优化问题为凸优化问题。显然支持向量机的目标函数 $\frac{1}{2}\left \| w  \right \|^2$ 是关于 $w$ 的凸函数，不等式约束 $1 - y_i(w^Tx_i + b) \le 0$ 也是关于 $w$ 的凸函数，因此支持向量机是一个凸优化问题。

### 拉格朗日函数

$$
min \; f(x) \\
s.t. \; g_i(x) \le 0, \; i = 1,2,\cdots,m \\
\; \; h_j(x) = 0, \; j = 1,2,\cdots,m
$$

设上述优化问题的定义域为 $D = dom \; f \cap \cap_{i=1}^m \; dom \; g_i \cap \cap_{j=1}^n \; dom \; h_j$，可行集为 $\tilde{D} = \{ x | x \in D, g_i(x) \le 0, h_j(x) = 0 \}$，显然 $\tilde{D}$ 是 $D$ 的子集，最优值为 $p^* = min\{ f(\tilde{x}) \}$。由拉格朗日的定义，可得：

$$
L(x,\mu,\lambda) = f(x) + \sum_{i=1}^m \mu_i g_i(x) + \sum_{j=1}^n \lambda_j h_j(x)
$$

其中 $\mu = (\mu_1,\mu_2,\cdots,\mu_m)^T$，$\lambda = (\lambda_1,\lambda_2,\cdots,\lambda_m)^T$ 是拉格朗日乘子向量