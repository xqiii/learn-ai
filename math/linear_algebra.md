# 线性代数

## 向量

### 定义

$n$ 个有序的数 $a_1,a_2 \cdots,a_n$ 所组成的数组称为 $n$ 维向量。

- $n$ 维列向量  $\begin{pmatrix}
a_1\\
a_2\\
\vdots\\
a_n
\end{pmatrix}$

- $n$ 维行向量 $\begin{pmatrix}
a_1,
a_2,
\cdots,
a_n
\end{pmatrix}$

### 运算

加法
$$
\begin{pmatrix}
a_1\\
a_2\\
\vdots\\
a_n
\end{pmatrix} + 

\begin{pmatrix}
b_1\\
b_2\\
\vdots\\
b_n
\end{pmatrix} = 

\begin{pmatrix}
a_1 + b_1\\
a_2 + b_2\\
\vdots\\
a_n + b_n
\end{pmatrix}
$$

数乘
$$
k \cdot \begin{pmatrix}
a_1\\
a_2\\
\vdots\\
a_n
\end{pmatrix} =

\begin{pmatrix}
k \cdot a_1\\
k \cdot a_2 \\
\vdots\\
k \cdot a_n
\end{pmatrix}
$$

交换律：
$\vec{u} + \vec{v} = \vec{v} + \vec{u}$

分配律：
$(\vec{u} + \vec{v}) + \vec{w} = \vec{u} + (\vec{v}+ \vec{w}) $

模：
$ \vec{u} = (u_1,u_2,\cdots,u_n)^T,
\left \| \vec{u} \right \| = \sqrt{u_1^2+u_2^2+\cdots+u_n^2} $

点乘：
$$\vec{u} \cdot \vec{v} = 
\begin{pmatrix}
u_1\\
u_2\\
\vdots\\
u_n
\end{pmatrix} \cdot

\begin{pmatrix}
v_1\\
v_2\\
\vdots\\
v_n
\end{pmatrix} = 

u_1v_1 + u_2v_2 + \cdots + u_nv_n$$
从向量相似的角度来看，点乘结果越大，这两个向量越相似，越小越背离，等于0则无关。

余弦相似度：$cos\theta = \frac{a \cdot b}{\left \| \vec{a} \right \|\left \| \vec{b} \right \|} $

### 线性相关和线性无关

#### 定义
给定向量组 $v = {v_1,v_2,\cdots,v_m}$，如果存在不全为零的实数 $k1,k2,\cdots,k_m$ 使：
$$
k_1v_1 + k_2v_2 + \cdots + k_mv_m = 0
$$
则称向量组 $v$ 是线性相关的，否则就线性无关（只存在全为零的实数 $k1,k2,\cdots,k_m$ 使得式子成立）。

### 向量空间
设 $v$ 为一向量组，如果 $v$ 非空，且 $v$ 对向量的加法和数乘两种运算封闭，那么就称 $v$ 为向量空间。
- 若 $a \in v, b \in v$，则 $a + b \in v$
- 若 $a \in v, k \in \mathbb{R}$，则 $ka \in v$

### 张成空间
某向量组 $v = {v_1,v_2,\cdots,v_p}$，其所有线性组合构成的集合为向量空间，也称为向量组 $v$ 的张成空间，记为 $span(v_1,v_2,\cdots,v_p)$，即：

$$
span(v_1,v_2,\cdots,v_p) = \{k_1v_1 + k_2v_2 + \cdots + k_pv_p，k_{1,2,3,\cdots,p} \in \mathbb{R} \}
$$

### 最大无关组
设有向量组 $v$，如果在 $v$ 中能选出 $r$ 个向量 $v_1,v_2,\cdots,v_r$，满足：

- 向量组 $v_0 = \{v_1,v_2,\cdots,v_n\}$ 线性无关
- 向量组 $v_0$ 中任意 $r+1$ 个向量都线性相关，那么称向量组 $v_0$ 是向量组 $v$ 的一个最大线性无关组，简称最大无关组。

### 向量的秩
假设向量组 $A$ 的最大无关组为：
$$A_0 = \{a_1,a_2,\cdots,a_n\}$$
$A_0$ 的向量个数 $r$ 称为向量组 $A$ 的秩，记作 $rank(A)$。

### 空间的基
已只 $v$ 为向量空间，如果其中的某组向量组
$$A = \{a_1,a_2,\cdots,a_n\}$$
是 $v$ 的最大无关组，那么 $A$ 被称作向量空间 $v$ 的一组基。

### 坐标
假设 $A = \{a_1,a_2,\cdots,a_n\}$ 是向量空间 $v$ 的一个基，所以 $v$ 中的任意向量 $x$ 可表示为：
$$x = k_1a_1 + k_2a_2 + \cdots + k_na_n$$
上式的系数可以组成向量：
$$\left [ x \right ]_A = (k_1,k_2,\cdots,k_n) $$
我们将其称为 $x$ 在基 $A$ 下的坐标向量。

选择不同的基，就是在向量空间中建立了不同的坐标系


## 矩阵

### 定义
由 $m$ x $n$ 个数 $a_{ij}(i = 1,2,\cdots,m;j= 1,2,\cdots,n)$ 排成的 $m$ 行 $n$ 列的数表称为 $m$ 行 $n$ 列矩阵，简称 $m$ x $n$ 矩阵：
$$
A = \begin{pmatrix}  
  a_{11} & \cdots & a_{1n} \\  
  \vdots & \ddots & \vdots \\  
  a_{m1} & \cdots & a_{mn}  
\end{pmatrix} 
$$

第 $i$ 行可以看作行向量，第 $j$ 列可以看成列向量。

行数和列数相等且等于 $n$ 的矩阵称为 $n$ 阶方阵：
$$\begin{pmatrix}  
  1 & 0 \\  
  0 & 1  
\end{pmatrix} $$
元素都是 $0$ 的矩阵称为零矩阵：
$$
\begin{pmatrix}  
  0 & 0 \\  
  0 & 0  
\end{pmatrix} 
$$

### 线性方程组

#### 定义

未知数最高次数为一的方程组，称为线性方程组。
$$
\left\{\begin{matrix} 
  x + 2y = 3\\  
  3x + 4y = 5 
\end{matrix}\right.
$$

把系数提取出来为系数矩阵，等号右边提取出来合在一起叫增广矩阵：
$$
\begin{pmatrix}  
  1 & 2 & 3 \\  
  3 & 4 & 5  
\end{pmatrix} 
$$

#### 高斯消元法
消元：
1. 一个方程左右两边同时乘一个常数
2. 一个方程加（减）一个方程
3. 交换位置

高斯-约旦消元法：

从上到下：
1. 选择最上的主元，化为1
2. 主元下面的所有行减去主元所在行的某个倍数，使得主元下面的所有元素为0

从下到上：
1. 选择最下面的主元
2. 主元上面的所有行减去所在行的某个倍数，使得主元上面所有元素为0

整个过程可以看作是若干个单位矩阵的乘积：
$$
E_p \cdot E_{p-1} \cdots E_2 \cdot E_1 = mef(A)
$$

#### 特殊矩阵

 行阶梯型矩阵：非零行在零行（存在的话）的上面；非零行的最左边的首非零元素在上一行（存在的话）的首非零元素右面。
 $$
 \begin{pmatrix}
  a  & *  & *  & * & * \\
  0  & b  & *  & * & * \\
  0  & 0  & 0  & c & * 
\end{pmatrix}

\begin{pmatrix}
  a  & *  & *  & * & * \\
  0  & b  & *  & * & * \\
  0  & 0  & 0  & 0 & * 
\end{pmatrix}
 $$
对角阵：$n$ 阶方阵除了对角元素以外都为 $0$ ，也记作 $A_n = diag(\lambda_1,\lambda_2,\cdots,\lambda_n)$：
$$
A_n =
 \begin{bmatrix}  
  \lambda_1 & 0 & \cdots & 0 \\  
  0 & \lambda_2 & \cdots & 0 \\  
  \vdots & \vdots & \ddots & \vdots \\  
  0 & 0 & \cdots & \lambda_n
\end{bmatrix} 
$$

行最简型矩阵

主元为1；除主元外，其所在列的其他元素均为0。
$$
\begin{pmatrix}
  1  & *  & *  & * & * \\
  0  & 1  & *  & * & * \\
  0  & 0  & 0  & 0 & * 
\end{pmatrix}
$$

初等行变换与初等行矩阵

1. 倍加变换。行加（减）某行的若干倍。$r_1' = r_1 + kr_2$
$$
\begin{pmatrix}
  1 & k & 0  \\
  0 & 1 & 0  \\
  0 & 0 & 1  
\end{pmatrix}
$$

2. 倍乘变换。某一行乘以一个常数。$r_1' = kr1(k \neq 0)$
$$
\begin{pmatrix}
  k & 0 & 0  \\
  0 & 1 & 0  \\
  0 & 0 & 1  
\end{pmatrix}
$$

3. 对换变换。交换两行。$r_1 \leftrightarrow r_2$
$$
\begin{pmatrix}
  0 & 1 & 0  \\
  1 & 0 & 0  \\
  0 & 0 & 1  
\end{pmatrix}
$$

### 运算

## 矩阵的秩

## 线性方程组

## 行列式

## 相似矩阵

## 特征向量