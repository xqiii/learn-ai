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

$E_1E_2\cdots E_nA = I7`$

性质：

$(A^{-1})^{-1} = A$

$(A\cdot B)^{-1} = B^{-1}A^{-1}$

$(A^{-1})^T = (A^T)^{-1}$


## 线性方程组的解

### 解的存在

在 $Ax = y$ 中，是否有 $x$ 与 $b$ 相对应。

线性方程组 $Ax = b$ 有解的充分条件是，系数矩阵 $A$ 的秩等于增广矩阵 $B$ 的秩，即：

$$rank(A) = rank(A|B) \Longleftrightarrow 方程有解$$

### 解的个数

线性方程组 $Ax = b$，它的增广矩阵为 $B = (A|b)$，如果 $A$ 为 $m x n$ 的矩阵，那么：

- 有唯一解，当且仅当 $rank(A) = rank(B) = n$
- 有无数解，当且仅当 $rank(A) = rank(B) < n$

从映射的角度来看，满秩矩阵满秩有解，单射表示有唯一解。将增广矩阵化为行最简形式可以看出，系数矩阵非零行个数和未知数个数相等，代表唯一解。

### 解集

对于线性方程组 $Ax = b$ ：

- $b = 0$ 时，即 $Ax = 0$ 时，称为齐次方程组。

- $b \neq 0$ 时，称为非齐次方程组。

零空间（Null Space）

矩阵 $A$ 满足 $Ax = 0$ 的 $x$ 组成的集合。
$$Null(A) = \{x | Ax = 0\}$$

在 $A$ 的变换下，映射到原点。集合的向量，和 $A$ 的行向量点乘为 $0$ ，和 $A$ 的行空间向量正交。

已知矩阵 $A$ 以及零空间 $Null(A)$，那么非齐次线性方程组
$Ax = b$ 的解集为：

$$x + p + Null(A)， 其中 p 为 Ax = b 的特解$$

### 秩-零化度定理

对于 $m x n$ 的矩阵 $A$：
$$
rank(A) + rank(Null(A)) = n
$$

$n$ 为定义域的维度：

$$
rank(定义域) - rank(Null(A)) = rank(A) = rank(值域)
$$

也可以这样等价：

零空间的维度 = 自由变量的个数 = 解空间的维度 = $Ax = 0$ 独立参数的个数


## 行列式

行列式是方阵的一个属性。

### 几何意义

$$
det\begin{vmatrix}
 a & b\\
 c & d
\end{vmatrix} = ad - bc
$$

行列式可以代表向量在空间形式的有向体积。对于某方阵 $A$，它的行列式是矩阵函数 $Ax = y$ 的伸缩比例。

### 子式和余子式

### 性质

对于 $n$ 阶方阵 $A = (a_{ij})$，有：$|A| = |A^T|$

对于方阵 $A$，有 $|A| \neq 0$ $\Longleftrightarrow$ $A$ 满秩 $\Longleftrightarrow$ $A$ 可逆

### 运算

数乘：行列式乘以 $k$ 倍，是某一行（列）乘以 $k$。
$$
\begin{vmatrix}
 ka & kb\\
 c & d
\end{vmatrix} = 
k\begin{vmatrix}
 a & b\\
 c & d
\end{vmatrix}
$$

行（列）互换：行列式的行（列）互换后，行列式正负号发生改变。

$$
\begin{vmatrix}
 a & b\\
 c & d
\end{vmatrix} = 
-\begin{vmatrix}
 c & d\\
 a & b
\end{vmatrix}
$$

倍加：将一行（列）的 $k$ 倍加进另一行（列）里，行列式的值不变。

加法：某一行（列）每个元素是两数之和，则此行列式可以拆分为两个相加的行列式。

乘法：$|AB| = |A||B|$

从乘法证明矩阵可逆的条件：
$det(A\cdot A^{-1}) = det(A) \cdot det(A^{-1}) = det(I) \Longrightarrow det(A^{-1}) = \frac{1}{det(A)}$ 

如果行列式的一行是另一行的 $k$ 倍，则行列式的值为 $0$。（可以看作向量共线）
$$
\begin{vmatrix}
 ka & kb\\
 a & b
\end{vmatrix} = 
k\begin{vmatrix}
 a & b\\
 a & b
\end{vmatrix} = 0
$$
行列式为 $0$：
 - 一行为 $0$
 - 两行相同
 - 一行是另外一行的 $k$ 倍
 - 一行是其他的线性组合

对角行列式：

$$
\begin{vmatrix}  
  d_1 & 0 & \cdots & 0 \\  
  0 & d_2 & \cdots & 0 \\  
  \vdots & \vdots & \ddots & \vdots \\  
  0 & 0 & \cdots & d_n  
\end{vmatrix} = 
d_1d_2\cdots d_n
\begin{vmatrix}  
  1 & 0 & \cdots & 0 \\  
  0 & 1 & \cdots & 0 \\  
  \vdots & \vdots & \ddots & \vdots \\  
  0 & 0 & \cdots & 1  
\end{vmatrix} =
d_1d_2\cdots d_n
$$


## 相似矩阵

### 基变换公式

已知两个基 $m_1,m_2,\cdots,m_s$ 和 $n_1,n_2,\cdots,n_s$，当且仅当它们是同一个向量空间时，才存在唯一的矩阵 $P$，使得下式成立：

$$
(n_1,n_2,\cdots,n_s) = (m_1,m_2,\cdots,m_s)P
$$

该矩阵 $P$ 称为由基 $m_1,m_2,\cdots,m_s$ 到基 $n_1,n_2,\cdots,n_s$ 的过渡矩阵，上式称为基转换公式。

### 坐标变换
坐标是向量在一组基下的线性组合系数

已知 两个基 $m_1,m_2,\cdots,m_s$ 和 $n_1,n_2,\cdots,n_s$ 的过渡矩阵：

$$
(n_1,n_2,\cdots,n_s) = (m_1,m_2,\cdots,m_s)P
$$

又已知 $x$ 在基 $M$下的坐标为 $\left [ x \right ]_M$ 及在基 $N$ 下的坐标为 $\left [ x \right ]_N$，则有坐标变换公式为：

$$
\left [ x \right ]_N = P^{-1}\left [ x \right ]_M，
\left [ x \right ]_M = P\left [ x \right ]_N
$$


| 类型            | 对象   | 矩阵位置   | 矩阵含义     |
| ------------- | ---- | ------ | -------- |
| **基变换**（换坐标系） | 基向量  | **右乘** | 过渡矩阵 $P$ |
| **坐标变换**（换坐标） | 坐标向量 | **左乘** | $P^{-1}$ |
基向量是行排列，坐标向量是列排列。


### 相似矩阵
设 $A$、$B$ 都是 $n$ 阶方阵，若有可逆矩阵 $P$，使得 $B = P^{-1}AP$，则称 $P$ 为相似变换矩阵，称 $B$ 是 $A$ 的相似矩阵。

几何直观：

$[x]_p  ---B---> [y]_P$

$[x]_p ---P---> [x]_\epsilon$

$[x]_\epsilon ---A---> [y]_\epsilon$

$[y]_\epsilon ---P^{-1}---> [y]_p$

$P$：把新基下的坐标翻译成标准基下的坐标

$A$：在标准基下执行线性变换

$P^{-1}$：把结果翻译回新基下的坐标

#### 性质

$A^k \sim B^k, k \in \mathbb{Z}$

$A^T \sim B^T$

$A^{-1} \sim B^{-1}$

$A \sim B, B \sim C, A \sim C$

## 特征向量

### 定义

设 $A$ 是 $n$ 阶方阵，$x$ 为非零向量，若存在 $\lambda$ 使得下式成立：

$$
Ax = \lambda x
$$

那么 $\lambda$ 称为 $A$ 的特征值，非零向量 $x$ 称为 $A$ 的对应 $\lambda$ 的特征向量。

### 特征方程和特征空间

假设：
$$
A = \begin{pmatrix}  
  a_{11} & \cdots & a_{1n} \\  
  \vdots & \ddots & \vdots \\  
  a_{m1} & \cdots & a_{nn}  
\end{pmatrix} 
$$

那么 $|A-\lambda I| = 0$ 可以写作：

$$
|A-\lambda I| = 
\begin{vmatrix}  
  a_{11} - \lambda & \cdots & a_{1n} \\  
  \vdots & \ddots & \vdots \\  
  a_{n1} & \cdots & a_{nn} - \lambda  
\end{vmatrix} 
$$

其中 $|A-\lambda I|$ 展开后就是关于特征值 $\lambda$ 的多项式，称为特征多项式。进而 $|A-\lambda I| = 0$ 被称为特征方程。

已知 $\lambda_1,\lambda_2,\cdots,\lambda_m$ 是 $n$ 阶方阵的特征向量，则向量组 $\{ v_1,v_2,\cdots,v_m \}$ 线性无关。

### 对角化

如果 $n$ 阶方阵 $A$ 有 $n$ 个线性无关的特征向量 $p_1,p_2,\cdots,p_n$ ，那么构造矩阵 $P = (p_1,p_2,\cdots,p_n)$ ，使得：

$$
A = P \Lambda P^{-1}
$$

其中 $\Lambda $ 为如下对角矩阵：

$$
\begin{pmatrix}  
  \lambda_1 & & \\  
   & \ddots &  \\  
   &  & \lambda_n
\end{pmatrix} 
$$

其中 $\lambda_1,\lambda_2,\cdots,\lambda_n$ 为特征向量 $ v_1,v_2,\cdots,v_n $ 对应的特征值，该过程称为对角化。

### 正交矩阵

#### 正交基

已知 $p_1,p_2,\cdots,p_r$ 是向量空间 $V$ 的一个基，如果两两正交，即满足：

$$
p_i \cdot p_j = 0, i \neq j
$$

那么称为正交基，如果长度都为1，就称为标准正交基。

#### 正交矩阵

假设 $p_1,p_2,\cdots,p_r$ 是向量空间 $\mathbb{R}^n$ 的一个标准正交基，那么由它们构造的 $n$ 阶方阵 $P$ 也称为正交矩阵。

$$
P = (p_1,p_2,\cdots,p_n)
$$

该方阵必然满足：

$$
P^TP = P^{-1}P = I
$$

$P^T$ 就是 $P$ 的逆矩阵。


#### 施密特正交化

给定线性无关组 $\{\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_n\}$，构造正交组 $\{\mathbf{u}_1, \mathbf{u}_2, ..., \mathbf{u}_n\}$ ：

$$
\begin{aligned}
\mathbf{u}_1 &= \mathbf{v}_1 \\
\mathbf{u}_2 &= \mathbf{v}_2 - \frac{\langle \mathbf{v}_2, \mathbf{u}_1 \rangle}{\langle \mathbf{u}_1, \mathbf{u}_1 \rangle} \mathbf{u}_1 \\
\mathbf{u}_3 &= \mathbf{v}_3 - \frac{\langle \mathbf{v}_3, \mathbf{u}_1 \rangle}{\langle \mathbf{u}_1, \mathbf{u}_1 \rangle} \mathbf{u}_1 - \frac{\langle \mathbf{v}_3, \mathbf{u}_2 \rangle}{\langle \mathbf{u}_2, \mathbf{u}_2 \rangle} \mathbf{u}_2 \\
&\vdots \\
\mathbf{u}_k &= \mathbf{v}_k - \sum_{j=1}^{k-1} \frac{\langle \mathbf{v}_k, \mathbf{u}_j \rangle}{\langle \mathbf{u}_j, \mathbf{u}_j \rangle} \mathbf{u}_j
\end{aligned}
$$

#### 正交对角化

对称矩阵所有不同的特征值对应的特征向量互相垂直。实对称矩阵 $A = A^T \in \mathbb{R}^{n \times n}$ 必然可以正交对角化。

$$
A = Q\Lambda Q^T
$$

### 对称矩阵

若 $A$ 是 $m \times n $ 的矩阵，则 $A^TA$ 是 $n \times n$ 的方阵且对称。所以：

- $A^TA 可以被正交对角化$

- 拥有 $n$ 个实数特征值

- $n$ 个相互垂直的特征向量

#### 奇异值
对矩阵 $A \in \mathbb{R}^{m \times n}$，其奇异值是矩阵 $A^TA$ 或 $AA^T$ 特征值的非负平方根：

$$
\sigma_i = \sqrt{\lambda_i(A^TA)} = \sqrt{\lambda_i(AA^T)}
$$

其中 $\lambda_i$ 表示第 $i$ 大的特征值。奇异值也可以代表向量 $Av_i$ 的长度（$\left \| A \vec{v_i} \right \| $）。

如果 $A$ 有 $r$ 个不为零的奇异值，则 $\{ A\vec{v_1}, A\vec{v_2}, \cdots, A\vec{v_r} \}$ 是 $A$ 的列空间的一组正交基，$\lambda_i \neq 0$。所以有 $rank(A) = r$。$\{ \frac{A\vec{v_1}}{\sigma_1}, \frac{A\vec{v_2}}{\sigma_2}, \cdots, \frac{A\vec{v_r}}{\sigma_r} \}$ 是标准正交基。 



#### SVD分解

如果有一个 $m \times n$ 的矩阵 $A$，它可以被分解成：

$$
A = U \Sigma V^T
$$


| 矩阵       | 角色    | 几何操作       | 性质                                           |
| :------- | :---- | :--------- | :------------------------------------------- |
| $V^T$    | 右奇异矩阵 | **旋转**输入空间 | 正交矩阵，$V^TV = I$                              |
| $\Sigma$ | 奇异值矩阵 | **伸缩**各坐标轴 | 对角阵，$\sigma_1 \geq \sigma_2 \geq ... \geq 0$ |
| $U$      | 左奇异矩阵 | **旋转**输出空间 | 正交矩阵，$U^TU = I$ |
