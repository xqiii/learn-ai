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

## 微分导数

### 微分

设函数$y = f(x)$在某区间内有定义，$x_0$ 及$x_0+\Delta x$在此区间内，如果函数值增量：

$$\Delta y = f(x_0+\Delta x) - f(x_0)$$

可表示为：

$$\Delta y = A \Delta x + o(\Delta x)$$ 

其中 $A$ 是不依赖与 $\Delta x$ 的常数，那么称函数 $y = f(x)$ 在 $x_0$ 处是可微的，而$A \Delta x$叫做函数$y = f(x)$ 在 $x_0$ 点相应于自变量增量 $\Delta x$的微分，记作$dy$，即：

$dy = A\Delta x$

通常令 $dx = \Delta x$，所以微分又表示为 $dy = Adx$。

其实是在说，在$x_0$附近，微分和曲线非常接近：

若 曲线 - 直线 = $o(\Delta x)  \implies$ 该直线就是曲线的微分。

$\Delta y = f(x_0 + \Delta x) - f(x_0)$，其实是曲线的表达式。

$dy = A \Delta x$，其实是直线的表达式，也就是$x_0$点微分的表达式。

$o(\Delta x) = \Delta y - A \Delta x $，该式是曲线和直线的相差。

### 导数

由微分定义可知，如果我们需要求$A$，其中$o(\Delta x)$是$x \to x_0$ 时 $\Delta x$ 的高阶无穷小，所以可以推出：

$$\Delta y = A \Delta x + o(\Delta x) \implies \frac{\Delta y}{\Delta x} = A + \frac{o(\Delta x)}{\Delta x} 
\implies $$

$$\lim_{x \to x_0} \frac{\Delta y}{\Delta x} = \lim_{x \to x_0} (A + \frac{o(\Delta x)}{\Delta x}) \implies$$ 

$$\lim_{x \to x_0} \frac{\Delta y}{\Delta x} = \lim_{x \to x_0} A + \lim_{x \to x_0} \frac{o(\Delta x)}{\Delta x} = A$$ 

因为$\Delta x = x - x_0$，所以$x \to x_0$ 就是$\Delta x \to 0$；又因为$\Delta y = f(x_0 + \Delta x) - f(x_0)$ ，所以：

$$\lim_{x \to x_0} \frac{\Delta y}{\Delta x} = \lim_{\Delta x \to 0} \frac{\Delta y}{\Delta x} = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x} = A$$

A的算式在数学中被称作导数，定义如下: 
设函数$y = f(x)$在$x_0$点的某个邻域内有定义，当自变量$x$ 在 $x_0$处取得增量$\Delta y = f(x_0+\Delta x) - f(x_0)$。如果$\Delta y$ 与 $\Delta x$之比当$\Delta x \to 0$时的极限存在，那么称函数在$x_0$处可导，并称这个极限为函数$y = f(x)$在$x_0$处的导数，记为$f\prime (x_0)$，即：
$$f\prime (x_0) = \lim_{\Delta x \to 0} \frac{\Delta y}{\Delta x} = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}$$

可微即可导，可导即可微。

## 偏导数
设n元函数 $f(x_1,x_2,...x_n)$，其中在点 $(a_1,a_2,...,a_n)$ 处关于变量$x_i$的偏导数定义为：
$$\frac{\partial f}{\partial x} = f_x = \lim_{h \to 0} \frac{f(x+h,y)-f(x,y)}{h}$$
关键点：计算 $\frac{\partial f}{\partial x}$
时，将其他所有变量（如 $y,z$ ）视为常数，仅让 $x$ 发生微小变化，观察函数值的响应。

## 方向导数与梯度
设$f : \mathbb{R^n} \to \mathbb{R}$ 在点 $x_0 = (x_1,...,x_n)$ 处可微，$u = (u_1,...,u_n)$ 是单位向量 $(|u| = 1)$ ，则 $f$ 在 沿 方向 $u$ 的方向导数是：
$$ D_uf(x_0) = \lim_{h \to 0} \frac{f(x_0 + hu) - f(x_0)}{h}$$
计算公式：
$$D_uf = \nabla f \cdot u = \frac{\partial f}{\partial x_1}u_1 + \frac{\partial f}{\partial x_2}u_2 + \ldots \frac{\partial f}{\partial x_n}u_n$$
其中$\nabla f = (\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n})$ 称为梯度（Gradient）。 

**梯度方向是函数增长最快的方向**，其模长就是最大增长率。

## 泰勒公式
若$f(x)$ 在 $x_0$ 处有 $n$ 阶导数，那么存在 $x_0$ 的一个邻域，对于该邻域的任意以 $x$ ，有： 
$$f(x) = \sum_{k=0}^{n} \frac{f^{(k)}(x_0)}{k!}(x-x_0)^k + o\left((x-x_0)^n\right) \quad (x \to x_0)$$
多项式部分称为泰勒多项式，剩余称为余项。$n$ 次泰勒多项式 $p_n(x)$ 是对函数 $f(x)$ 的多项式逼近。

## 不定积分
### 原函数
如果区间 $I$ 上，可导函数 $F(x)$ 的导函数为 $f(x)$，即 $\forall x \in I$ 时有：
$$F\prime(x) = f(x) 或 dF(x) = f(x)dx$$
那么函数 $F(x)$ 就称为 $f(x)$ 在区间 $I$ 上的一个原函数。

### 不定积分定义
如果 $F(x)$ 是 $f(x)$ 在区间 $I$ 上的一个原函数，那么 $F(x) + C$ 称为 $f(x)$ 在区间 $I$ 上的不定积分，记作：
$$\int f(x)\,dx = F(x) + C, C \in \mathbb{R}$$

## 定积分
设 $f(x)$ 在 $[a,b]$ 上有界，将$[a,b]$ 分成 $n$ 个小区间，分点 $a=x_0<x_1<\dots<x_n=b$，长度 $\Delta x_i = x_i - x_{i-1}$，在每个 $[x_{i-1}, x_i]$ 任取 $\xi_i$，做函数值与区间长度的乘积并求和：
$$S_n = \sum_{i=1}^{n} f(\xi_i)\Delta x_i$$
记 $\lambda = \max\{\Delta x_i\} \to 0$，若极限存在且与分割、取点无关，则：
$$\int_a^b f(x)\,dx = \lim_{\lambda \to 0} \sum_{i=1}^{n} f(\xi_i)\Delta x_i$$
其中 $f(x)$ 叫被积函数，$f(x)dx$ 叫做被积表达式，$x$ 叫做积分变量，$a$、$b$分别叫做积分的下限和上限，$[a,b]$ 叫做积分区间。
如果函数 $f(x)$ 在区间 $[a,b]$ 上的定积分存在，那么就说函数 $f(x)$ 在区间上 $[a,b]$ 可积



