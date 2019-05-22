## 决策树

### 基本流程
![](https://tva1.sinaimg.cn/large/007rAy9hgy1g3aeegskyyj30pj0htq7r.jpg)

### 划分过程
决策树学习的关键是如何选择最优划分属性，希望分支结点所包含的样本尽可能属于同一类别，即结点的“纯度”越来越高。
#### 信息增益
假定当前样本集合$D$中第$k$类样本所占比例为$p_k(k=1,2,\cdots,|y|)$，则$D$的信息熵定义为
$$Ent(D)=-\sum^{|y|}_{k=1}p_k{\log_2p_k}$$
$Ent(D)$的值越小，则$D$的纯度最高。

假定离散属性$a$有$V$个可能的取值${a^1,a^2,\cdots,a^V}$,属性$a$对样本集$D$进行划分的信息增益
$$Gain(D,a)=Ent(D)-\sum^V_{v=1}\frac{|D^v|}{|D|}Ent(D^v)$$

#### 增益率
属性$a$的可能取值数目越多（即$V$越大），$a$的固有值越大，固有值表示为
$$IV(a)=-\sum^V_{v=1}\frac{|D^v|}{|D|}\log_2\frac{|D^v|}{|D|}$$
增益率定义为
$$Gain_ratio(D,a)=\frac{Gain(D,a)}{IV(a)}$$

#### 基尼系数
数据集$D$的纯度可用基尼值来度量
$$Gini(D)=\sum^{|y|}_{k=1}\sum_{k'\neq{k}}p_kp_{k'}=1-\sum_{|y|}^{k=1}p^2_k$$

基尼指数
$$Gini_index(D,a)=\sum^V_{v=1}\frac{|D^v|}{|D|}Gini(D^v)$$

### 剪枝处理
通过`剪枝`降低过拟合。
两种策略：`预剪枝`和`后剪枝`。

### 连续与缺失值
#### 连续值处理
二分法最连续属性进行处理。
假定连续属性$a$在样本集$D$上出现了$n$个不同的取值，将这些值从小到大排序，记为$\{a^1,a^2,\cdots,a^n\}$。把区间${[a^i,a^{i+1})}$的中位点$\frac{a^i+a^{i+1}}{2}$作为候选划分点，得到信息增益
$$Gain(D,a)=\max_{t\in{T_a}}Gain(D,a,t)=\max_{t\in{T_a}}Ent(D)-\sum_{\lambda\in\{-,+\}}\frac{|D^{\lambda}_t|}{|D|}Ent(D^{\lambda}_t)$$
其中$Gain(D,a,t)$是样本集$D$基于划分点$t$二分后的信息增益。

#### 缺失值处理
给定训练集$D$和属性$a$，令$\widetilde{D}$表示$D$中在属性$a$上没有缺失值的样本子集。假定属性$a$有$V$个可取值$\{a^1,a^2,\cdots,a^V\}$，令$\widetilde{D}^v$表示在$\widetilde{D}$中属性$a$上取值为$a^v$的样本子集，$\widetilde{D}_k$表示$\widetilde{D}$中属于第$k$类$k(k=1,2,\cdots,|y|)$的样本子集，则显然有$\widetilde{D}=\cup^{|y|}_{k=1}\widetilde{D}_k$，$\widetilde{D}=\cup^{|V|}_{v=1}\widetilde{D}^v$
为样本$x$赋予权重$w_x$，定义
$$\rho=\frac{\sum_{x\in\widetilde{D}}w_x}{\sum_{x\in{D}}w_x}$$
$$\widetilde{p}_k=\frac{\sum_{x\in\widetilde{D}_k}w_x}{\sum_{x\in\widetilde{D}}w_x}\qquad{(1{\leq}{k}{\leq}|y|)}$$
$$\widetilde{r}_v=\frac{\sum_{x\in\widetilde{D}^v}w_x}{\sum_{x\in\widetilde{D}}w_x}\qquad{(1{\leq}{v}{\leq}V)}$$
对属性$a$，$\rho$表示无缺失值样本所占的比例，$\widetilde{p}_k$表示无缺失值样本中第$k$类所占的比例，$\widetilde{r}_v$则表示无缺失值样本中在属性$a$上取值$a^v$的样本所占的比例。
可将信息增益推广为
$$Gain(D,a)=\rho\times{Gain(\widetilde{D},a)}=\rho\times{(Ent(\widetilde{D})-\sum^V_{v=1}\widetilde{r}_vEnt(\widetilde{D}^v))}$$
其中，
$$Ent(\widetilde{D})=-\sum^{|y|}_{k=1}\widetilde{p}_k\log_2\widetilde{p}_k$$

### 多变量决策树
在多变量决策树的学习过程中，不是为了每个非叶结点寻找一个最优划分属性，而是试图简历一个合适的线性分类器。
