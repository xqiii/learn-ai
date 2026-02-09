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

$$
w = \frac{\sum_{i=1}^m y_i x_i - \bar{y} \sum_{i=1}^m x_i}{\sum_{i=1}^m x_i^2 -  \bar{x} \sum_{i=1}^m x_i}
$$


## 多元线性回归