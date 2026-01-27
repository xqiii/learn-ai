# 微积分

## 数列

对于每个 $n \in \mathbb{R}$，对应一个确定的实数$a_n$，这些实数按照下标大小排列得到一个序列，这个序列就叫做数列
$$\{a_n\} = \{a_1,a_2,a_3,...a_n\}$$


## 极限

### 数列极限
若某数列无限趋于某实数，与实数的差可以任意小，则该确定的实数称为数列的极限

对于数列$\{a_n\}$，如果$\forall\epsilon>0$，$\exists N\in \mathbb{R}$，$\forall n>N$，有
$$|a_n-L|<\epsilon，L\in \mathbb{R}$$
那么就称$L$是数列$\{a_n\}$的极限，或者称数列$\{a_n\}$收敛于$L$，记作
$$\lim_{n \to \infty}a_n = L 或 a_n \to L(n \to \infty)$$
若不存在这样的常数$L$，就说数列$\{a_n\}$没有极限，或者说数列$\{a_n\}$是发散的。 

### 函数极限
设函数$f(x)$当$|x|$大于某正整数时有定义，如果$\forall\epsilon>0$，$\exists X > 0$，$\forall|x|>X$，有：
$$|f(x)-L|<\epsilon，L\in \mathbb{R}$$
那么就称L是函数$f(x)$当$x\to \infty$的极限，或者称当$当$x\to \infty$时函数$f(x)$收敛于$L$，记作：
$$lim_{x \to \infty}f(x) = L 或 f(x) \to L(x \to \infty)$$



## 无穷小与无穷大

### 无穷小
对于函数$f(x)$，如果满足$\lim f(x)=0$，则称函数$f(x)$为此自变量变化过程（指$x \to x_0$，$x \to \infty$等）的无穷小

### 无穷大
1. 设函数$f(x)$在$\mathring{U}(x_0)$上有定义。如果$\forall M>0$，$\exists \delta > 0$，$\forall x \in \mathring{U}(x_0,\delta)$，有$f(x)>M$，那么称函数$f(x)$是$x \to x_0$ 时的正无穷大，可以记作$\lim_{x \to x_0}f(x) = +\infty$ 。
1. 设函数$f(x)$在$\mathring{U}(x_0)$上有定义。如果$\forall M>0$，$\exists \delta > 0$，$\forall x \in \mathring{U}(x_0,\delta)$，有$f(x)<-M$，那么称函数$f(x)$是$x \to x_0$ 时的负无穷大，可以记作$\lim_{x \to x_0}f(x) = -\infty$ 。
1. 设函数$f(x)$在$\mathring{U}(x_0)$上有定义。如果$\forall M>0$，$\exists \delta > 0$，$\forall x \in \mathring{U}(x_0,\delta)$，有$|f(x)|>M$，那么称函数$f(x)$是$x \to x_0$ 时的无穷大，可以记作$\lim_{x \to x_0}f(x) = \infty$ 。



### 海涅定理

对函数$f(x)$在定义域内的任意满足$\lim_{n \to \infty}x_n=x_0$，$x_n \neq x_0$，$n \in \mathbb{Z^+}$的数列$\{x_n\}$，有$\lim_{n \to \infty}f(x_n) = L \iff \lim_{x \to x_0}f(x) = L$。 

该定理把“连续变量趋近”的极限问题，归结成了“离散点列趋近”的数列极限问题。通常用来证明极限不存在。

例：证明$\lim_{x \to 0}sin{\frac{1}{x}}$不存在

取两条数列：

$x_n = \frac{1}{n \pi} \to 0$，则$sin\frac{1}{x_n} = sin(n\pi) = 0 \to 0$

$y_n = \frac{1}{2n\pi+\frac{\pi}{2}}$，则$sin\frac{1}{x_n} = sin(2n\pi+\frac{\pi}{2}) = 1\to 1$

$lim_{n \to \infty}f(x_n) \neq lim_{n \to \infty}f(y_n)$故得证



### 极限运算法则

如果$\lim f(x)=A$，$\lim g(x) = B$，那么：

1. $\lim[f(x) \pm g(x)]$ = $\lim f(x) \pm \lim g(x)$ = $A \pm B$
2. $\lim[f(x) \times g(x)]$ = $\lim f(x) \times lim g(x)$ = $A \times B$
3. 若又有 $B \neq 0$，则$\lim \frac{f(x)}{g(x)}$ = $\frac{\lim f(x)}{\lim g(x)}$ = $\frac{A}{B}$



## 连续性与导数

### 连续

如果$\lim_{\Delta x \to 0}\Delta y = \lim_{\Delta x \to 0}[f(x_0 + \Delta x) - f(x_0)] = 0$ ，那么就称函数$f(x)$在$x_0$点连续。也可以用以下定义：

设函数$f(x)$在$x_0$点的某一邻域内有定义，若$\lim_{x \to x_0}f(x) = f(x_0)$就称函数在$x_0$处连续。

## 导数



## 偏导数



## 方向导数



## 梯度



## 微积分



## 定积分



## 牛顿-莱布尼茨



## 泰勒公式





## 阶数与阶乘



## 拉格朗日乘子



