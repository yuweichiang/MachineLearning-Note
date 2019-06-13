## 贝叶斯分类器
### 贝叶斯决策论
- 贝叶斯最优分类器
$$h^*(x)=\mathop{\arg\min}_{c\in{y}}R(c|x)$$
- 最小化分类错误率的贝叶斯最优分类器
$$h^*(x)=\mathop{\arg\max}_{c\in{y}}P(c|x)$$
基于贝叶斯定理，$P(c|x)$可写为
$$P(c|x)=\frac{P(c)P(x|c)}{P(x)}$$

### 极大似然估计
令$D_c$表示训练集$D$中第$c$类样本组成的集合，假设这些样本是独立同分布的，则参数$\theta_c$对于数据集$D_c$的似然是
$$P(D_c|\theta_c)=\prod_{x\in{D_c}}P(x|\theta_c)$$
使用对数似然
$$LL(\theta_c)=\log{P(D_c|\theta_c)}=\sum_{x\in{D_c}}\log{P(D_c|\theta_c)}$$
此时参数$\theta_c$的极大似然估计$\hat{\theta_c}$为
$$\hat{\theta_c}=\mathop{\arg\max}_{\theta_c}LL(\theta_c)$$

### 朴素贝叶斯分类器
$$h_{nb}(x)=\mathop{\arg\max}_{c\in{y}}P(c)\prod^d_{i=1}P(x_i|c)$$

###  半朴素贝叶斯分类器
$$P(c|x)\propto{P(c)\prod^d_{i=1}P(x_i|c,pa_{i})}$$

### 贝叶斯网

### EM算法
