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
左乘矩阵 $E$ 是在对 $A$ 的行做操作。
右乘矩阵 $E$ 是在对 $A$ 的列做操作。

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

#### 加法
两个同维度的矩阵相加，对应元素相加：
$$
A + B = [a_{ij} + b_{ij}]_{m \times n}
$$

满足：

交换律：$A + B = B + A$

结合律：$(A + B) + C = A + (B + C)$

#### 数乘
数 $k$ 与矩阵 $A$ 的乘积：
$$
kA = Ak = [ka_{i,j}]_{m \times n}
$$

#### 乘法
$A$、$B$ 相乘满足如下条件：

- $m \times n$ 的矩阵只能和 $n \times p$ 的矩阵相乘。
- 相乘后大小为 $m \times p$。

$A_{m \times n} \cdot B_{n \times p} = C_{m \times p}$

计算方法：

$C = AB \quad \text{其中} \quad c_{ij} = \sum_{k=1}^{n} a_{ik} \cdot b_{kj}$

行视角：

$$
xA = \begin{bmatrix} x_1^T \\ x_2^T \\ \vdots \end{bmatrix} A = \begin{bmatrix} x_1^T A \\ x_2^T A \\ \vdots \end{bmatrix}
$$

$$
xA = \begin{pmatrix}
  x_1 & x_2 & \cdots & x_n
\end{pmatrix}
\begin{pmatrix}  
  a_{11} & \cdots & a_{1n} \\  
  \vdots & \ddots & \vdots \\  
  a_{m1} & \cdots & a_{mn}  
\end{pmatrix} 
= x_1(a_{11}, a_{12}, \cdots , a_{1n}) + \cdots + x_m(a_{m1}, a_{m2}, \cdots , a_{mn})
$$

列视角：
$$
Ax = A \cdot [x_1, x_2, ..., x_p] = [Ax_1, Ax_2, ..., Ax_p]
$$

$$
Ax = \begin{pmatrix}  
  a_{11} & \cdots & a_{1n} \\  
  \vdots & \ddots & \vdots \\  
  a_{m1} & \cdots & a_{mn}  
\end{pmatrix} 
\begin{pmatrix}
  x_1 \\
  x_2 \\
  \vdots \\
  x_n
\end{pmatrix}

= x_1\begin{pmatrix}
  a_{11} \\
  a_{21} \\
  \vdots \\
  a_{m1}
\end{pmatrix} 
+ \cdots + x_n\begin{pmatrix}
  a_{1n} \\
  a_{2n} \\
  \vdots \\
  a_{mn}
\end{pmatrix}
$$

点积视角：
$$
AB = \sum_{k=1}^{n} (\text{A的第k列}) \cdot (\text{B的第k行})
$$
$$
c_{ij} = a_{i*} \cdot b_{*j} = a_{i1}b_{1j} + \cdots + a_{is}b_{sj} = \sum_{k=1}^s a_{ik}b_{kj}  (i = 1, \cdots, m; j = 1, \cdots, n)
$$

性质：

交换律：不一定不满足

数乘交换律：$\lambda (AB) = (\lambda A)B = A(\lambda B)$

结合律：$(AB)C = A(BC)$

分配律：$A(B+C) = AB + AC$

幂运算：
设 $A$ 是方阵，定义：
$A^1 = A, A^2 = A^1A^1, \cdots, A^{k+1} = A^kA^1$

转置：

把矩阵 $A$ 的行换成同序的列，该操作称为转置。转置后得到一个新的矩阵，记为 $A^T$：
$$
A = (a_{ij})，A^T = (a_{ji})
$$

性质：

$(A^T)^T = A$

$(AB)^T = B^TA^T$

$(A^T)^n = (A^n)^T$

$(A+B)^T = A^T + B^T$

$x^Ty = x \cdot y$

#### 矩阵函数
$Ax = y$

旋转矩阵：

$Aa = b$ 使得 $b$ 相对 $a$ 逆时针旋转 $\theta$

$$
{A = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}}
$$

单位阵：

在单位阵作用下，向量保持不变
$$
\begin{pmatrix}  
  1 & 0 \\  
  0 & 1  
\end{pmatrix} 
$$

镜像矩阵：

$Ax = y$，在 $A$ 作用下，相对于 $y = x$ 对称

$$\begin{pmatrix}  
  0 & 1 \\  
  1 & 0  
\end{pmatrix}$$


## 矩阵的秩

### 列空间

$A$ 的所有列向量的向量组，即：

$$\{c_1, c_2, \cdots, c_n\}$$

列向量组的张成空间称为列空间，记作 $colsp(A)$，即：

$$
colsp(A) = span(\{c_1, c_2, \cdots, c_n\}) = x_1c_1 + x_2c_2 + \cdots + x_nc_n
$$

列向量组的秩，也就是列空间的维度，称为列秩，即：

$$
列秩 = rank(colsp(A))
$$


### 行空间

$A$ 的所有行向量的向量组，即：

$$\{r_1^T, r_2^T, \cdots, r_m^T\}$$

行向量组的张成空间称为行空间，记作 $colsp(A)$，即：

$$
rowsp(A) = span(\{r_1^T, r_2^T, \cdots, r_m^T\}) = x_1r_1^T + x_2r_2^T + \cdots + x_mc_m^T
$$

行向量组的秩，也就是行空间的维度，称为行秩，即：

$$
行秩 = rank(rowsp(A))
$$

### 矩阵的秩
对于任意矩阵：

$$
矩阵的秩 = 行秩 = 列秩
$$

记作：$rank(A)$

### 矩阵函数

四要素:

| 要素 | 名称                            | 符号                              | 含义                                  |
| :- | :---------------------------- | :------------------------------ | :---------------------------------- |
| 1  | **定义域**            | $\mathbb{R}^n$ 或 $\mathbb{C}^n$ | 输入空间，所有可能的 $\mathbf{x}$             |
| 2  | **到达域**         | $\mathbb{R}^m$ 或 $\mathbb{C}^m$ | 输出空间，$f(\mathbf{x})$ 所在的 ambient 空间 |
| 3  | **值域/像**       | $\mathcal{colsp}(A)$                | 实际能输出的所有向量（列空间）                     |
| 4  | **映射法则** | $A$                | 映映射矩阵             |

#### 秩的性质

如果某矩阵即行满秩又列满秩，那么就称该矩阵为满秩矩阵，满秩矩阵必为方阵。

$rank(A) = rank(A^T)$

$rank(AB) <= min(rank(A), rank(B))$

#### 矩阵函数的映射

当定义域为向量空间时：

$定义域的维度 \ge 值域的维度$


单射：

$矩阵函数是单射 \Leftrightarrow 定义域的维度 = 值域维度$

等价判定：

-  $rank(A) = n$ (列满秩)
- 零空间只有零向量：$Ax = 0$ 只有 x = 0 唯一解
- 列向量线性无关

必要条件：$m \ge n$

满射：

$矩阵函数是非单射 \Leftrightarrow 定义域的维度 > 值域维度$

等价判定：

-  $rank(A) = m$ (行满秩)
- 列空间 = 整个 $\mathbb{R}^m$
- 行向量线性无关

必要条件：$n \ge m$


双射:

$矩阵函数是双射 \Leftrightarrow 定义域的维度 = 值域维度 = 到达域的维度$

等价判定：

-  $m = n，rank(A) = n$ (方阵+满秩=可逆)
- 行列式 $det(A) \neq 0$
- 存在逆矩阵 $A^{-1}$

#### 逆矩阵

当 $A$ 为满秩矩阵时，对应的矩阵函数为双射，此时 $A$ 存在反函数，称为 $A$ 可逆。其反函数记作 $A^(-1)$，称 $A$ 的逆矩阵。

若存在两个 $n$ 阶方阵 $A，C$，两者的乘积为 $n$ 阶单位阵 $I$：

$$
AC = I 且 CA = I
$$

那么 $C$ 就是 $A$ 的逆矩阵，即 $A^{-1} = C$，且 $A^{-1}$
 是唯一的。

求逆矩阵

初等行矩阵求逆矩阵

$E_1E_2\cdots E_nA = I$




## 线性方程组

## 行列式

## 相似矩阵

## 特征向量