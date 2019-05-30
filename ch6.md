### 间隔与支持向量
给定训练样本集$D=\{(x_1,y_1),(x_2,y_2),\cdots,(x_m,y_m)\},y_i\in\{-1,+1\}$，分类学习最基本的想法是基于训练集$D$在样本空间中找到一个划分超平面，将不同类别的样本分开。
在样本空间中，划分超平面可通过如下线性方程描述：
$$w^Tx+b=0$$
其中$w=(w_1;w_2;\cdots;w_d)$为法向量，决定超平面的方向；$b$为位移项，决定超平面与原点之间的距离。
将超平面记为$(w,b)$，样本空间中任意点$x$到超平面的距离可写为：
$$r=\frac{|w^Tx+b|}{||w||}$$
令
$$\{\begin{array}{ll}{W^{T} x_{i}+b \geq+1} & {y_{i}=+1} \\ {W^{T} x_{i}+b \leq-1} & {y_{i}=-1}\end{array}$$
距离超平面最近的这几个训练样本点使得上式等号成立，则称为“支持向量”，两个异类支持向量到超平面的距离之和（间隔）为
$$\gamma=\frac{2}{||w||}$$
最大间隔的划分超平面，即
$${\max_{w,b} \frac{2}{\|W\|}} \\ {\text {s.t.y}_{i}\left(W^{T} x_{i}+b\right) \geq+1}$$
可重写为
$$
\begin{array}{l}{\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}} \\ {\text { s.t. } y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right) \geqslant 1, \quad i=1,2, \ldots, m}\end{array}
$$
这就是支持向量机的基本型。

### 对偶问题
- 大间隔划分的超平面模型
$$f(x)=w^Tx+b$$
- 拉格朗日函数
$$
L(\boldsymbol{w}, b, \boldsymbol{\alpha})=\frac{1}{2}\|\boldsymbol{w}\|^{2}+\sum_{i=1}^{m} \alpha_{i}\left(1-y_{i}\left(\boldsymbol{w}^{\top} \boldsymbol{x}_{i}+b\right)\right)
$$
其中$\alpha=(\alpha_1;\alpha_2;\cdots;\alpha_m)$，令$L(\lambda,b,\alpha)$对$\lambda$和$\alpha$的偏导为0可得
$$\lambda=\sum^m_{i=1}\alpha_iy_ix_i,$$
$$0=\sum^m_{i=1}\alpha_iy_i$$
得到
$$
f(x)=w^Tx+b=\sum^m_{i=1}\alpha_iy_ix^T_ix+b
$$

- KKT条件
$$
\{\begin{array}{c}{\alpha_{i} \geq 0} \\ {y_{i} f\left(x_{i}\right)-1 \geq 0} \\ {\alpha_{i}\left(y_{i} f\left(x_{i}\right)-1\right)=0}\end{array}
$$

### 核函数
- 定理（核函数）
令$\mathcal{X}$为输入空间,$k(.,.)$是定义在$\mathcal{X}\times\mathcal{X}$上的对称函数, 则$k$是核函数当且仅当对于任意数据$D=\left\{\boldsymbol{x}_{1}, \boldsymbol{x}_{2}, \ldots, \boldsymbol{x}_{m}\right\}$, 核矩阵K总是半正定的：
$$
\mathbf{K}=\left[ \begin{array}{cccc}{\kappa\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{1}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{j}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{m}\right)} \\ {\vdots} & {\ddots} & {\vdots} & {\ddots} & {\vdots} \\ {\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{1}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{m}\right)} \\ {\vdots} & {\ddots} & {\vdots} & {\ddots} & {\vdots} \\ {\kappa\left(\boldsymbol{x}_{m}, \boldsymbol{x}_{1}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{m}, \boldsymbol{x}_{j}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{m}, \boldsymbol{x}_{m}\right)}\end{array}\right]
$$

### 软间隔和正则化
- 优化目标
$$\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \ell_{0 / 1}\left(y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)-1\right)$$

- 替换损失函数
hinge损失: $\ell_{\text {hinge}}(z)=\max (0,1-z)$
指数损失: $\ell_{e x p}(z)=\exp (-z)$
对率损失: $\ell_{\log }(z)=\log (1+\exp (-z))$

引入松弛变量，有
$$\min _{\boldsymbol{w}, b, \xi_{i}} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \xi_{i}$$
$$\begin{array}{c}{\text { s.t. } y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right) \geqslant 1-\xi_{i}} \\ {\xi_{i} \geqslant 0, i=1,2, \ldots, m}\end{array}$$

得到拉格朗日函数
$$
\begin{aligned} L(\boldsymbol{w}, b, \boldsymbol{\alpha}, \boldsymbol{\xi}, \boldsymbol{\mu})=& \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \xi_{i} \\ &+\sum_{i=1}^{m} \alpha_{i}\left(1-\xi_{i}-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i} \end{aligned}
$$
令$L(\boldsymbol{w}, b, \boldsymbol{\alpha}, \boldsymbol{\xi}, \boldsymbol{\mu})$偏导为0，可得对偶问题
$$
\begin{aligned} \max _{\boldsymbol{\alpha}} & \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j} \\ \text { s.t. } & \sum_{i=1}^{m} \alpha_{i} y_{i}=0 \\ & 0 \leqslant \alpha_{i} \leqslant C, \quad i=1,2, \ldots, m \end{aligned}
$$

### 支持向量回归
SVR问题可形式化为
$$
\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \ell_{c}\left(f\left(\boldsymbol{x}_{i}\right)-y_{i}\right)
$$

拉格朗日函数
$$
\begin{array}{l}{L(\boldsymbol{w}, b, \boldsymbol{\alpha}, \hat{\boldsymbol{\alpha}}, \boldsymbol{\xi}, \hat{\boldsymbol{\xi}}, \boldsymbol{\mu}, \hat{\boldsymbol{\mu}})} \\ {=\frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m}\left(\xi_{i}+\hat{\xi}_{i}\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i}-\sum_{i=1}^{m} \hat{\mu}_{i} \hat{\xi}_{i}} \\ {+\sum_{i=1}^{m} \alpha_{i}\left(f\left(\boldsymbol{x}_{i}\right)-y_{i}-\epsilon-\xi_{i}\right)+\sum_{i=1}^{m} \hat{\alpha}_{i}\left(y_{i}-f\left(\boldsymbol{x}_{i}\right)-\epsilon-\hat{\xi}_{i}\right)}\end{array}
$$

对偶问题
$$
\begin{aligned} \max _{\boldsymbol{\alpha}, \boldsymbol{\alpha}} & \sum_{i=1}^{m} y_{i}\left(\hat{\alpha}_{i}-\alpha_{i}\right)-\epsilon\left(\hat{\alpha}_{i}+\alpha_{i}\right) \\ &-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m}\left(\hat{\alpha}_{i}-\alpha_{i}\right)\left(\hat{\alpha}_{j}-\alpha_{j}\right) \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j} \\ \text { s.t. } & \sum_{i=1}^{m}\left(\hat{\alpha}_{i}-\alpha_{i}\right)=0 \\ & 0 \leqslant \alpha_{i}, \hat{\alpha}_{i} \leqslant C \end{aligned}
$$

### 核方法
- 表示定理
令$\mathbb{H}$为核函数$\kappa$对应的再生核希尔伯特空间, $\|h\|_{ \mathrm{H}}$表示$\mathbb{H}$空间中关于$h$的范数, 对于任意单调递增函数$\Omega : [0, \infty] \mapsto \mathbb{R}$和任意非 负损失函数$\ell : \mathbb{R}^{m} \mapsto[0, \infty]$, 优化问题

$$
\min _{h \in \mathbb{H}} F(h)=\Omega\left(\|h\|_{\mathrm{H}}\right)+\ell\left(h\left(\boldsymbol{x}_{1}\right), h\left(\boldsymbol{x}_{2}\right), \ldots, h\left(\boldsymbol{x}_{m}\right)\right)
$$

解总可写为
$$
h^{*}(\boldsymbol{x})=\sum_{i=1}^{m} \alpha_{i} \kappa\left(\boldsymbol{x}, \boldsymbol{x}_{i}\right)
$$
