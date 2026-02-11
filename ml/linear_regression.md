# 线性回归

## 一元线性回归

给定数据 $\{(x_i, y_i)\}_{i=1}^n$ ，其中 $x_i \in \mathbb{R}^d$ （特征），$y_i \in \mathbb{R}$ （目标，寻找最佳线性映射：

$$
y \approx f(x) = w^T x + b
$$

或写成矩阵形式（把 $b$ 吸进 $w$ ，增广一个常数1特征）：

$$
\mathbf{y} \approx \mathbf{X}\mathbf{w}
$$

### 最小二乘法

最小化预测值与真实值的欧氏距离平方和

$$
\min_{\mathbf{w}} \quad J(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 = \sum_{i=1}^n (y_i - \mathbf{w}^T\mathbf{x}_i)^2
$$

### MLE 视角

假设偏差 $\varepsilon \sim N(0, \sigma^2)$，则：

$$
p(\mathbf{\varepsilon}) =  \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\varepsilon^2}{2\sigma^2}\right)
$$

等价替换 $\varepsilon = y - (w\mathbf{x}+b)$ 得：

$$
p(\mathbf{y}) =  \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y-(w\mathbf{x}+b))^2}{2\sigma^2}\right)
$$

所以：
$$
y = w\mathbf{x} + b + \varepsilon \sim N(w\mathbf{x}+b, \sigma^2)
$$

使用最大似然估计 $w$ 和 $b$ 的值：

$$
L(\mathbf{w},\mathbf{b}) = \prod_{i=1}^m \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y_i-(w\mathbf{x}_i+b))^2}{2\sigma^2}\right)
$$

取对数：

$$
lnL(\mathbf{w},\mathbf{b}) = \sum_{i=1}^m ln \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y_i-(w\mathbf{x}_i+b))^2}{2\sigma^2}\right) = 
$$
$$
\sum_{i=1}^m ln \frac{1}{\sqrt{2\pi}\sigma} + 
\sum_{i=1}^m ln \exp\left(-\frac{(y_i-(w\mathbf{x}_i+b))^2}{2\sigma^2}\right)
$$

化简：

$$
lnL(\mathbf{w},\mathbf{b}) = mln\frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{2\sigma^2} \sum_{i=1}^m (y_i-w\mathbf{x}_i-b)^2
$$

其中 $w$，$\sigma$ 均为常数，所以最大化 $lnL(\mathbf{w},\mathbf{b})$ 等价最小化 $\sum_{i=1}^m (y_i-w\mathbf{x}_i-b)^2$，即：

$$
(\mathbf{w}^*,\mathbf{b}^*) = arg max lnL(\mathbf{w},\mathbf{b}) = argmin \sum_{i=1}^m (y_i-w\mathbf{x}_i-b)^2
$$

可看出等价最小二乘估计。

### 求解

求解 $w$ 和 $b$ 本质是多元函数求最值：

1. 证明 $E_{(w,b)} = \sum_{i=1}^m (y_i-w\mathbf{x}_i-b)^2$ 是关于 $w$ 和 $b$ 的凸函数。
2. 用凸函数求最值思路求解出 $w$ 和 $b$。


求1：

证明 $E_{(w,b)}$ 的 Hessian 矩阵是半正定的：

求一阶偏导数：

求解 $\frac{\partial E_{(w,b)}}{\partial w}$：
$$
\begin{aligned}
\frac{\partial E_{(w,b)}}{\partial w} &= \frac{\partial }{\partial w}[\sum_{i=1}^m (y_i-w\mathbf{x}_i-b)^2]  \\
&= \sum_{i=1}^m \frac{\partial }{\partial w} (y_i-w\mathbf{x}_i-b)^2 \\
&= \sum_{i=1}^m 2 \cdot (y_i-w\mathbf{x}_i-b) \cdot (-\mathbf{x}_i) \\
&= 2 (w\sum_{i=1}^m x_i^2 - \sum_{i=1}^m(y_i-b)x_i)
\end{aligned}
$$

求解 $\frac{\partial E_{(w,b)}}{\partial b}$ ：
$$
\begin{aligned}
\frac{\partial E_{(w,b)}}{\partial b} &= \frac{\partial }{\partial b}[\sum_{i=1}^m (y_i-w\mathbf{x}_i-b)^2]  \\
&= \sum_{i=1}^m \frac{\partial }{\partial b} (y_i-w\mathbf{x}_i-b)^2 \\
&= \sum_{i=1}^m 2 \cdot (y_i-w\mathbf{x}_i-b) \cdot (-1) \\
&= 2(mb - \sum_{i=1}^m 2 \cdot (y_i-w\mathbf{x}_i))
\end{aligned}
$$

求二阶偏导：

求解 $\frac{\partial E_{(w,b)}}{\partial w^2}$ ：
$$
\begin{aligned}
\frac{\partial E_{(w,b)}}{\partial w^2} &= \frac{\partial }{\partial w} (\frac{\partial E_{(w,b)}}{\partial w}) \\
&=  \frac{\partial }{\partial w}(2w\sum_{i=1}^m x_i^2) \\
&=  2\sum_{i=1}^mx_i^2 
\end{aligned}
$$

求解 $\frac{\partial E_{(w,b)}}{\partial w \partial b}$ ：

$$
\begin{aligned}
\frac{\partial E_{(w,b)}}{\partial w \partial b} &= \frac{\partial }{\partial b} (\frac{\partial E_{(w,b)}}{\partial w}) \\
&= \frac{\partial }{\partial b}[2 (w\sum_{i=1}^m x_i^2 - \sum_{i=1}^m(y_i-b)x_i)]\\
&= \frac{\partial }{\partial b}[- 2\sum_{i=1}^m(y_i-b)x_i] \\
&= \frac{\partial }{\partial b}(-2\sum_{i=1}^my_ix_i + 2 \sum_{i=1}^m bx_i) \\
&= 2\sum_{i=1}^m x_i
\end{aligned}
$$

求解 $\frac{\partial E_{(w,b)}}{\partial b^2}$ ：

$$
\begin{aligned}
\frac{\partial E_{(w,b)}}{\partial b^2} &= \frac{\partial }{\partial b} (\frac{\partial E_{(w,b)}}{\partial b}) \\
&= \frac{\partial }{\partial b}[2(mb - \sum_{i=1}^m 2 \cdot (y_i-w\mathbf{x}_i))] \\
&= 2m

\end{aligned}
$$

根据上式求得：

$$
\nabla^2E_(w,b) = 
\begin{bmatrix}
\frac{\partial E_{(w,b)}}{\partial w^2} & \frac{\partial E_{(w,b)}}{\partial w \partial b} \\
\frac{\partial E_{(w,b)}}{\partial b \partial w}  & \frac{\partial E_{(w,b)}}{\partial b^2}
\end{bmatrix} = 
\begin{bmatrix}
2\sum_{i=1}^mx_i^2 & 2\sum_{i=1}^m x_i \\
2\sum_{i=1}^m x_i & 2m
\end{bmatrix}
$$

根据半正定判断定理：所有主子式非负，则该矩阵为半正定矩阵。

$$
|2\sum_{i=1}^mx_i^2| > 0
$$

$$
\begin{aligned}
\begin{vmatrix}
2\sum_{i=1}^mx_i^2 & 2\sum_{i=1}^m x_i \\
2\sum_{i=1}^m x_i & 2m
\end{vmatrix} &= 2\sum_{i=1}^m x_i^2 \cdot 2m - 
2\sum_{i=1}^m x_i \cdot 2\sum_{i=1}^m x_i \\
&= 4m \sum_{i=1}^m x_i^2 - 4 (\sum_{i=1}^m x_i)^2 
\end{aligned}
$$

$$
\begin{aligned}
4m \sum_{i=1}^m x_i^2 - 4 (\sum_{i=1}^m x_i)^2 &= 
4m \sum_{i=1}^m x_i^2 - 4 \cdot m \cdot \frac{1}{m} \cdot (\sum_{i=1}^m x_i)^2 \\
&= 4m (\sum_{i=1}^m x_i^2 - \sum_{i=1}^m x_i \bar{x}) \\
&= 4m \sum_{i=1}^m(x_i^2 - x_i \bar{x}) \\
\end{aligned}
$$

由于 $\sum_{i=1}^m x_i \bar{x} = \bar{x}\sum_{i=1}^m x_i = \bar{x} \cdot m \cdot \frac{1}{m} \cdot \sum_{i=1}^m x_i = m \bar{x}^2 = \sum_{i=1}^m \bar{x}^2$ 

$4m \sum_{i=1}^m(x_i^2 - x_i \bar{x}) = 4m \sum_{i=1}^m(x_i^2 - x_i \bar{x} - x_i \bar{x} + x_i \bar{x}) = 4m \sum_{i=1}^m(x_i^2 - x_i\bar{x} - x_i\bar{x} + \bar{x}^2) = 4m \sum_{i=1}^m (x_i - \bar{x})^2 \ge 0$

进而关于 $w$ 和 $b$ 的凸函数得证。所以，$\nabla E_{(w,b)} = 0$ 的点即为最小值点，也即：

$$
\nabla E_{(w,b)} = 
\begin{bmatrix}
\frac{\partial E_{(w,b)}}{\partial w} \\
\frac{\partial E_{(w,b)}}{\partial b} 
\end{bmatrix} =

\begin{bmatrix}
0 \\
0
\end{bmatrix}
$$

求解：

$$
\frac{\partial E_{(w,b)}}{\partial b} = 
2(mb - \sum_{i=1}^m 2 \cdot (y_i-w\mathbf{x}_i)) = 0
$$

$$
mb - \sum_{i=1}^m 2 \cdot (y_i-w\mathbf{x}_i) = 0
$$

$$
b = \frac{1}{m} \sum_{i=1}^m 2 \cdot (y_i-w\mathbf{x}_i) = \frac{1}{m} \sum_{i=1}^m y_i - w \cdot \frac{1}{m} \sum_{i=1}^m x_i = \bar{y} - w \bar{x}
$$

$$
\frac{\partial E_{(w,b)}}{\partial w} = 
2 (w\sum_{i=1}^m x_i^2 - \sum_{i=1}^m(y_i-b)x_i) = 0
$$

$$
w\sum_{i=1}^m x_i^2 - \sum_{i=1}^m(y_i-b)x_i = 0
$$

$$
w\sum_{i=1}^m x_i^2 = \sum_{i=1}^m y_i x_i - \sum_{i=1}^m b x_i
$$

将 $b = \bar{y} - w \bar{x}$ 回代：

$$
\begin{aligned}
w\sum_{i=1}^m x_i^2 &= \sum_{i=1}^m y_i x_i - \sum_{i=1}^m (\bar{y} - w \bar{x}) x_i \\
&= \sum_{i=1}^m y_i x_i - \bar{y} \sum_{i=1}^m x_i - w \bar{x} \sum_{i=1}^m x_i
\end{aligned}
$$

$$
w\sum_{i=1}^m x_i^2 - w \bar{x} \sum_{i=1}^m x_i = 
\sum_{i=1}^m y_i x_i - \bar{y} \sum_{i=1}^m x_i
$$

$$
w(\sum_{i=1}^m x_i^2 -  \bar{x} \sum_{i=1}^m x_i) = 
\sum_{i=1}^m y_i x_i - \bar{y} \sum_{i=1}^m x_i
$$


利用：$\sum (x_i - \bar{x}) = 0$



$$
\begin{aligned}
w &= \frac{\sum_{i=1}^m y_i x_i - \bar{y} \sum_{i=1}^m x_i}{\sum_{i=1}^m x_i^2 -  \bar{x} \sum_{i=1}^m x_i} \\
&= \frac{\sum_{i=1}^m y_i (x_i - \bar{x})}{\sum_{i=1}^m (x_i^2 -  x_i\bar{x} )} \\
&= \frac{\sum_{i=1}^m y_i (x_i - \bar{x}) - \sum_{i=1}^m \bar{y}(x_i - \bar{x})}{\sum_{i=1}^m x_i(x_i - \bar{x} ) - \sum_{i=1}^m \bar{x}(x_i - \bar{x}) } \\
&= \frac{\sum_{i=1}^m (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^m (x_i - \bar{x})^2} 
\end{aligned}
$$

转化为矩阵：若令 $x = (x_1;x_2;\cdots;x_m)$，$x_d = (x_1 - \bar{x};\cdots;x_m - \bar{x})$为去均值的 $x$，同理 $y$ 可以表示成 $y_d$，($m \times 1$ 的列向量) 代入上式：

$$
w = \frac{x_d^Ty_d}{x_d^Tx_d}
$$


## 多元线性回归

数据集 $D$，样本由 $d$ 个属性描述，此时：

$$
f(x_i) = w^Tx_i + b_i，使得 f(x_i) \simeq  y_i
$$

导出 $E_w$：

$$
f(x_i) = 
\begin{pmatrix}
w_1 & w_2 & \cdots & w_d & w_{d+1}
\end{pmatrix}

\begin{pmatrix}
x_{i1} \\
x_{i2} \\
\vdots \\
x_{id} \\
1
\end{pmatrix}
$$

$$
f(\hat{x_i}) = \hat{w}^T\hat{x_i}
$$

由最小二乘法得：

$$
\begin{aligned}
E_{\hat{w}} &= \sum_{i=1}^m (y_i - \hat{w}^T \hat{x_i})^2 \\
&= (y_1 - \hat{w}^T \hat{x_1})^2 + \cdots + (y_m - \hat{w}^T \hat{x_m})^2 \\
&= 
\begin{pmatrix}
y_1 - \hat{w}^T \hat{x_1} &
\cdots &
y_m - \hat{w}^T \hat{x_m}
\end{pmatrix}

\begin{pmatrix}
y_1 - \hat{w}^T \hat{x_1} \\
\vdots \\
y_m - \hat{w}^T \hat{x_m}
\end{pmatrix}
\end{aligned}
$$

其中：

$$
\begin{pmatrix}
y_1 - \hat{w}^T \hat{x_1} \\
\vdots \\
y_m - \hat{w}^T \hat{x_m}
\end{pmatrix} = 

\begin{pmatrix}
y_1 \\
\vdots \\
y_m 
\end{pmatrix} -

\begin{pmatrix}
\hat{w}^T \hat{x_1} \\
\vdots \\
\hat{w}^T \hat{x_m}
\end{pmatrix} = 

\begin{pmatrix}
y_1 \\
\vdots \\
y_m 
\end{pmatrix} -

\begin{pmatrix}
\hat{x_1}^T \hat{w} \\
\vdots \\
\hat{x_m}^T \hat{w} 
\end{pmatrix}

$$

$$
y = \begin{pmatrix}
y_1 \\
\vdots \\
y_m 
\end{pmatrix}, 

\begin{pmatrix}
\hat{x_1}^T \hat{w} \\
\vdots \\
\hat{x_m}^T \hat{w} 
\end{pmatrix} = 

\begin{pmatrix}
\hat{x_1}^T \\
\vdots \\
\hat{x_m}^T
\end{pmatrix} \cdot \hat{w} = 

\begin{pmatrix}
\hat{x_1}^T & 1 \\
\vdots & \vdots \\  
\hat{x_m}^T & 1
\end{pmatrix} \cdot \hat{w} = X \cdot \hat{w}

$$

由上式可得：

$$
E_{\hat{w}} = (y - X\hat{w})^T(y - X\hat{w})
$$

求解的问题依然是一个多元函数求最值的问题：
1. 证明 $E_{\hat{w}}$ 是关于 $\hat{w}$ 的凸函数。
2. 用凸函数求最值的思路求解出 $\hat{w}$。

证明1：

$$
\begin{aligned}
E_{\hat{w}} &= (y - X\hat{w})^T(y - X\hat{w}) \\
&= (y^T - \hat{w}^TX^T)(y - X\hat{w}) \\
&= y^Ty - y^TX\hat{w}^T - \hat{w}^TX^Ty + w^TX^TX\hat{w} \\
\end{aligned}
$$

由矩阵微分公式：$\frac{\partial a^Tx}{\partial x}$ = 
$\frac{\partial xa^T}{\partial x} = a$  （标量相同），$\frac{\partial x^TAx}{\partial x} = (A+A^T)x$ （二次型求导）。 
$$
\frac{\partial E_{\hat{w}}}{\partial w} = 
0 - X^Ty - X^Ty + (X^TX + X^TX) \hat{w} =
2 X^T (X \hat{w} - y)
$$

$$
\begin{aligned}
\nabla^2 E_{\hat{w}} &= \frac{\partial}{\partial \hat{w}}(\frac{\partial E_{\hat{w}}}{\partial \hat{w}}) \\
&= \frac{\partial}{\partial \hat{w}}[2 X^T (X \hat{w} - y)] \\
&= 2 X^TX
\end{aligned}
$$


假定 $X^TX$ 为正定矩阵，因此得证。

求解2：

$$
\frac{\partial E_{\hat{w}}}{\partial w} = 2 X^T (X \hat{w} - y) = 0
$$

$$
2X^TX\hat{w} - 2X^Ty = 0 \\
2X^TX\hat{w} = 2X^Ty \\
\hat{w} = (X^TX)^{-1}X^Ty
$$


## 对数几率回归

在线性模型的基础上套一个映射函数来实现分类功能

### 极大似然估计

一：确定概率质量函数。

已知 $y \in \{0,1\}$，则：

$$
\begin{aligned}
p(y = 1 | x) &= \frac{1}{1 + e^{-(w^Tx+b)}} = \frac{e^{w^Tx+b}}{1 + e^{w^Tx+b}} \\
p(y = 0 | x) &= 1 - p(y = 1 | x) = \frac{1}{1 + e^{w^Tx+b}}
\end{aligned}
$$

令 $\beta = (w;b)$，$\hat{x} = (x;1)$，上式简写：
$$
\begin{aligned}
p(y = 1 | \hat{x};\beta) &= \frac{e^{\beta^T\hat{x}}}{1 + e^{\beta^T\hat{x}}} = p_1(\hat{x};\beta) \\
p(y = 0 | \hat{x};\beta) &= \frac{1}{1 + e^{\beta^T\hat{x}}} = p_0(\hat{x};\beta)
\end{aligned}
$$

以上可以得出概率质量函数：

$$
p(y|\hat{x};\beta) = y \cdot p_1(\hat{x};\beta) + (1-y) \cdot p_0(\hat{x};\beta)
$$

或者为：

$$
p(y|\hat{x};\beta) = [p_1(\hat{x};\beta)]^y + [p_0(\hat{x};\beta)]^{1-y}
$$

二：写出似然函数

$$
L(\beta) = \prod_{i=1}^m p(y_i | \hat{x}; \beta)
$$

对数似然函数为：

$$
\ell(\beta) = lnL(\beta) = \sum_{i=1}^m p(y_i | \hat{x}; \beta) = \sum_{i=1}^m ln(y_ip_1(\hat{x};\beta) + (1-y_i)p_0(\hat{x};\beta))
$$

将 $p_1(\hat{x};\beta) = \frac{e^{\beta^T\hat{x}}}{1 + e^{\beta^T\hat{x}}}$ ，$p_0 = \frac{1}{1 + e^{\beta^T\hat{x}}} = p_0(\hat{x};\beta)$ 带入上式：

$$
\begin{aligned}
\ell(\beta) &= \sum_{i=1}^m ln (\frac{y_i e^{\beta^T\hat{x}}}{1 + e^{\beta^T\hat{x}}} + \frac{1 - y_i}{1 + e^{\beta^T\hat{x}}}) \\
&= \sum_{i=1}^m ln (\frac{y_i e^{\beta^T\hat{x}} + 1 - y_i}{1 + e^{\beta^T\hat{x}}}) \\
&= \sum_{i=1}^m (ln(y_i e^{\beta^T\hat{x}} + 1 - y_i) - ln(1 + e^{\beta^T\hat{x}}))
\end{aligned}
$$

由于 $y \in \{0, 1\}$：

$y_i = 0$ 时：

$$
\ell(\beta) = \sum_{i=1}^m(-ln(1 + e^{\beta^T\hat{x}}))
$$

$y_i = 1$ 时：
$$
\ell(\beta) = \sum_{i=1}^m(\beta^T\hat{x}-ln(1 + e^{\beta^T\hat{x}}))
$$

综合：

$$
\ell(\beta) = \sum_{i=1}^m(y_i\beta^T\hat{x}-ln(1 + e^{\beta^T\hat{x}}))
$$

损失函数是最小化目标，因此将最大化 $\ell(\beta)$ 转化为最小化 $-\ell(\beta)$，即：

$$
\ell(\beta) = \sum_{i=1}^m(-y_i\beta^T\hat{x}+ln(1 + e^{\beta^T\hat{x}}))
$$

## 二分类线性判别分析

从几何角度出发，让全体训练样本经过投影后：
- 异类样本的中心尽可能远
- 同类样本的方差尽可能小

经过投影后，异类样本的中心尽可能远（非严格投影）：

$$
max \left \| w^T\mu_0 - w^T\mu_1 \right \|_2^2 
$$

$$
max \left \| \left | w \right | \cdot \left |  \mu_0 \right | \cdot cos\theta_0 - \left | w \right | \cdot \left |  \mu_1 \right | \cdot cos\theta_1 \right \|_2^2
$$

经过投影后，同类样本的方差尽可能小：

$$
min w^T \Sigma w
$$

$\Sigma_0$ 为协方差矩阵
$$
\Sigma_0 = \begin{bmatrix}
\sigma_{11}^2 & \sigma_{12} & \cdots & \sigma_{1d} \\
\sigma_{21} & \sigma_{22}^2 & \cdots & \sigma_{2d} \\
\vdots & \vdots & \ddots & \vdots \\
\sigma_{d1} & \sigma_{d2} & \cdots & \sigma_{dd}^2
\end{bmatrix}
$$

$$
(x_i - \mu)(x_i - \mu)^T = 
\begin{bmatrix}
(x_i^{(1)}-\mu^{(1)})^2 & (x_i^{(1)}-\mu^{(1)})(x_i^{(2)}-\mu^{(2)}) & \cdots \\
(x_i^{(2)}-\mu^{(2)})(x_i^{(1)}-\mu^{(1)}) & (x_i^{(2)}-\mu^{(2)})^2 & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}
$$

$$
\begin{aligned}
\text{Var}_{\text{proj}} &= \frac{1}{m_0} \sum_{i=1}^{m_0} w^T(x_i - \mu_0)(x_i - \mu_0)^T w \\
&= w^T \left( \frac{1}{m_0} \sum_{i=1}^{m_0} (x_i - \mu_0)(x_i - \mu_0)^T \right) w \\
&= w^T \Sigma_0 w
\end{aligned}
$$

所以损失函数：

$$
\begin{aligned}
max J &= \frac{\left \| w^T\mu_0 - w^T\mu_1 \right \|_2^2}{w^T\Sigma_0w + w^T\Sigma_1w} \\
&= \frac{\left \| (w^T\mu_0 - w^T\mu_1)^T \right \|_2^2}{w^T(\Sigma_0 + \Sigma_1)w} \\
&= \frac{\left \| (\mu_0 - \mu_1)^Tw \right \|_2^2}{w^T(\Sigma_0 + \Sigma_1)w} \\
&= \frac{[(\mu_0 - \mu_1)^Tw]^T(\mu_0 - \mu_1)^Tw}{w^T(\Sigma_0 + \Sigma_1)w} \\
&= \frac{w^T(\mu_0 - \mu_1)(\mu_0 - \mu_1)^Tw}{w^T(\Sigma_0 + \Sigma_1)w}
\end{aligned}
$$

将 $(\mu_0 - \mu_1)(\mu_0 - \mu_1)^T$ 定义为 $S_b$，(\Sigma_0 + \Sigma_1) 定义为 $S_w$，式子可化为：

$$
max J = \frac{w^TS_bw}{w^TS_ww}
$$

分子分母都是关于 $w$ 的二次项，因此与 $w$ 长度无关，只与其方向有关。令 $w^TS_ww = 1$（固定分母），则等价：

$$
min \; -w^TS_bw
$$
$$
s.t. \; w^TS_ww = 1
$$

而 $ w^TS_ww - 1= 0$，可使用拉格朗日乘子法：

$$
L(w,\lambda) = -w^TS_bw + \lambda(w^TS_ww - 1)
$$

对 $w$ 求偏导：

$$
\frac{\partial L(w, \lambda)}{\partial w} = 
-(S_b + S_b^T)w + \lambda(S_w + S_w^T)w
$$

由于 $S_b = S_b^T$，$S_w = S_w^T$，所以：

$$
\frac{\partial L(w, \lambda)}{\partial w} = 
-2S_bw + 2\lambda S_w w
$$

令偏导数等于0，得：

$$-2S_bw + 2\lambda S_w w = 0$$
$$S_bw = \lambda S_ww$$
$$(\mu_0 - \mu_1)(\mu_0 - \mu_1)^Tw = \lambda S_ww$$

若令 $(\mu_0 - \mu_1)^Tw = \gamma$，则：

$$
\gamma (\mu_0 - \mu_1) = \lambda S_ww
$$

$$
w = \frac{\gamma}{\lambda}S_w^{-1} (\mu_0 - \mu_1)
$$

最终求解的 $w$ 不关心其大小，只关心其方向，所以 $\frac{\gamma}{\lambda}$ 这个常数项可以任意取值，令 $\gamma = \lambda$，进而使得 $\frac{\gamma}{\lambda} = -1$，此时求解出的 $w = S_w^{-1} (\mu_0 - \mu_1)$