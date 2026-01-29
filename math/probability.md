# 概率与统计

## 计数

### 阶乘
如果 $n$ 是一个正整数，那么 $n! = n \cdot (n-1) \cdots 1 $，并且 $0! = 1$，我们把 $n!$ 解释为 $n$ 种排序方法数。

### 二项式系数
从 $n$ 中选出 $k$ 种一共有 $\frac{n!}{k!(n-k)!}$ 种方法，$n$ 和 $k$ 都是非负整数，并且 $k <= n$，称这个式子为二项式系数，记为 $\binom{n}{k}$ 。
常见等式：
$$\binom{n}{k} + \binom{n}{k+1} = \binom{n+1}{k+1}$$
$$\binom{n}{k} = \binom{n}{n-k}$$
$$\sum_{k=0}^{n} \binom{n}{k} = 2^n$$

### 二项式定理
任意两个数 a  和 b  之和的 n  次幂的展开公式：
$$(a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k}b^k$$

## 概率论

### 概率公理
$\Omega$ 是一个结果空间, $P$ 是一个 $\sigma$ 代数. 如果概率函数满足下列件, 那么 $(\Omega,
P ,Prob) $就是一个概率空间：

1. 如果 $A \in P$ , 那么 $Pr(A)$ 是有定义的, 并且 $0 <= Pr(A) <= 1$。
2. $Pr(∅)$ = 0 且 $Pr(Ω) = 1$。
3. 设 ${A_i}$ 是由有限个或可数个两两互不相交的集合构成的集族, 并且每一个集合都是 $P$ 中的元素. 那么 $Pr(\cup_i A_i) = \sum_{i}Pr(A_i)$。

### 条件概率
设 $B$ 是满足条件 $P(B)>0$ 的事件，那么已知 $B$ 时 $A$ 的条件概率为：
$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$
可以推出：
$$P(A \cap B) = P(A|B)P(B)$$

### 贝叶斯定理
根据 $B \cap A = A \cap B$ 得到：
$$P(B \cap A) = P(B|A)P(A)$$
所以可以推出：
$$P(B|A)P(A) = P(A|B)P(B)$$
因此，只要 $P(B) \neq 0$，则：
$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

### 划分和全概率
样本空间 $S$ 的一个划分就是满足下列条件的可数个集合 $\{A_1,A_2 \cdots\}$。
1. 如果 $i \neq j$，那么 $A_i$ 和 $A_j$ 不相交，通常用 $A_i \cap A_j = $ 来表示这个两个集合的交集是空集。
2. 全体 $A_i$ 的并就是整个样本空间：$\cup_i A_i = S$。

全概率法则：
如果 $\{B_1,B_2 \cdots \}$ 构成了样本空间 $S$ 的一个划分，那么对任意 $A \subset S$，我们有：
$$P(A) = \sum_{n}P(A|B_n) \cdot P(B)$$

设 $\{ A_1,A_2,\cdots,A_n\}$ 是样本空间的一个划分，那么贝叶斯定理可以表示成：
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{\sum_{i=1}^n P(B|A_i) \cdot P(A_i)}$$

### 离散随机变量

#### 定义
离散型随机变量 X 就是定义在一个离散的结果空间 $\Omega$(这意味
着 $\Omega$ 是有限的或至多可数的) 上的实值函数. 具体地说, 我们为每个元素 $\omega \in \Omega$
指定了一个实数 $X(\omega)$。

#### 密度函数
设 $X$ 是一个随机变量, 它定义在离散的结果
空间 $\Omega$ 上 ($\Omega$ 是有限的或至多可数的)。 那么 $X$ 的概率密度函数 (常记作 $f_X$) 就是 $X$ 取某个特定值的概率：
$$f_X(x) = P(\omega \in \Omega : X(\omega) = x)$$
也有些地方称作“概率质量函数”。


### 连续随机变量

## 定义
连续型随机变量是可以在某一区间内取任意实数值的随机现象的量。如果一个随机变量的取值不可数（无法一一列举），且可以取某个区间内的所有实数值，那么它就是连续型的。

### 概率密度函数和概率分布函数
设 $X$ 是一个随机变量. 如果
存在一个实值函数 $f_X$ 满足：
1. $f_X$ 是一个分段连续函数
2. $f_X(x) \ge 0 $
3. $\int_{-\infty}^{+\infty}f_X(t)dt = 1$

那么 $X$ 是一个连续型随机变量, $f_X$ 是 $X$ 的概率密度函数. 

$X$ 的累积分布函数$F_X(x)$ 就是 X 不大于 $x$ 的概率：
$$F_X(x) = P(X \le x) = \int_{-\infty}^{x}f_X(t)dt$$

### 期望

#### 期望值
期望值, 矩：设 $X$ 是定义在 $\mathbb{R}$ 上的随机变量, 它的概率密度函数是 $f_X$. 函数$g(X)$ 的期望值是
$$\mathbb{E}[g(X)]=\left\{\begin{array}{ll}
\int_{-\infty}^{\infty} g(x) \cdot f_{X}(x) \mathrm{d} x & \text { 若 } X \text { 是连续的 } \\
\sum_{n} g\left(x_{n}\right) \cdot f_{X}\left(x_{n}\right) & \text { 若 } X \text { 是离散的. }
\end{array}\right.$$

最重要的情形是 $g(x) = x^r$
。我们把 $E[Xr]$ 称为 $X$ 的 $r$ 阶矩, 把 $E[(X −E[X])^r]$ 称为 $X$ 的 $r$ 阶中心矩。

#### 均值
$X$ 的均值 (即平均值或期望值) 是一阶矩. 我们把它表示为 $E[X]$ 或 $µ_X$
(当随机变量很明确时, 通常不给出下标 $X$, 而只写 $µ$)。具体地说：
$$\mu =\left\{\begin{array}{ll}
\int_{-\infty}^{\infty} x \cdot f_{X}(x) \mathrm{d} x & \text { 若 } X \text { 是连续的 } \\
\sum_{n} x_n  \cdot f_{X}(x_{n}) & \text { 若 } X \text { 是离散的 }
\end{array}\right.$$

#### 方差
方差 (记作 $\sigma_X^2$ 或 $Var(X)$) 是二阶中心距, 也可以说是 $g(X) =
(X − µX)^2$ 的期望值。同样, 当随机变量很明确时, 通常不给出下标 $X$,而只写 $\sigma^2$。 把它完整地写出来, 就是：
$$\sigma_X^2=\left\{\begin{array}{ll}
\int_{-\infty}^{\infty} (x-\mu x)^2 f_X(x) \mathrm{d} x & \text { 若 } X \text { 是连续的 } \\
\sum_{n} (x-\mu x)^2 f_{X}\left(x_{n}\right) & \text { 若 } X \text { 是离散的 }
\end{array}\right.
$$
因为 $\mu_X = \mathbb{E}[X]$，所以在一系列代数运算后，有：
$$\sigma^2 = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$
标准差是方差的平方根，即：$\sigma = \sqrt{\sigma_X^2}$

### 协方差
对于任意两个 (离散型或连续型的) 随机变量 $X$ 和 $Y$ , 如果它们的均值分别是
$\mu X$ 和 $\mu Y$ , 那么 $X$ 和 $Y$ 的协方差可以写成：
$$Cov(X,Y) = E[XY] - \mu_X \mu_Y$$

### 概率分布

#### 伯努利分布

#### 二项分布

#### 松柏分布

#### 正态分布

### 极限定理

#### 大数定理

#### 中心极限定理

## 统计