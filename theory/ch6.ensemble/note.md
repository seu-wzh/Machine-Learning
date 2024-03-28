假设有 $n$ 个二分类基学习器，基学习器在样本 $\boldsymbol{x}$ 上的分类结果通过简单投票法进行集成
$$
\mathcal{H}(\boldsymbol{x}) = sgn\left( \sum_{i = 1}^{n} h_{i}(\boldsymbol{x}) \right) \quad h_{i}(\boldsymbol{x}) \in \{ +1,\ -1 \}
$$

假设每个基学习器在 $\boldsymbol{x}$ 上的误差率相等
$$
p(h_{i}(\boldsymbol{x}) \ne f(\boldsymbol{x})) = \epsilon
$$

在基学习器的误差相互独立的情况下，根据 **Hoeffding 不等式**，集成的学习器的误差率为
$$
p(\mathcal{H}(\boldsymbol{x}) \ne f(\boldsymbol{x})) = \sum_{k = 0}^{[n / 2]} \mathrm{C}_{n}^{k} \epsilon^{k} (1 - \epsilon)^{n - k} \le \exp\left( -\frac{1}{2} n(1 - 2 \epsilon)^{2} \right)
$$

可见随着基学习器数量的增加，集成的学习器的误差率将成指数级下降并趋于零。

# 集成算法

从偏差-方差角度考虑，Adaboost 通过多级分类器、权重的校准，着重于降低分类的偏差

而 Bagging 和 Random Forest 通过增加基分类器的多样性，着重于降低分类方差

## AdaBoost

Boosting 族算法中最著名的 **AdaBoost 算法** 对基学习器进行线性组合集成
$$
\mathcal{H}(\boldsymbol{x}) = \sum_{i = 1}^{n} \alpha_{i} h_{i}(\boldsymbol{x}) \quad h_{i}(\boldsymbol{x}) \in \{ +1,\ -1 \}
$$

所有的基学习器$h_{i}$基于数据集$D = \{ (\boldsymbol{x}_{1},\ y_{1}),\ \cdots,\ (\boldsymbol{x}_{m},\ y_{m}) \}$和学习算法$\mathfrak{L}$进行学习
$$
h_{i} = \mathfrak{L}(D \mid \mathcal{P}_{i})
$$

其中$\mathcal{P}_{i}$为人为设定的样本分布，实际上体现为学习器在训练时数据集样本被使用的概率

### 损失函数

为了生成数据分布$\mathcal{P}$下的最优分类器，应用指数损失函数
$$
\begin{align*}
    \ell_{\exp}(h \mid \mathcal{P}) &= \mathcal{E}_{\mathcal{P}} \mathcal{E}(e^{-y h(\boldsymbol{x})} \mid \boldsymbol{x}) \\ \\
    &= \int \mathcal{E}(e^{-y h(\boldsymbol{x})} \mid \boldsymbol{x}) \mathcal{P}(\boldsymbol{x}) d\boldsymbol{x}
\end{align*}
$$

其中
$$
\mathcal{E}(e^{-y h(\boldsymbol{x})} \mid \boldsymbol{x}) = e^{-h(\boldsymbol{x})} p(y = 1 \mid x) + e^{h(\boldsymbol{x})} p(y = -1 \mid \boldsymbol{x})
$$

最小化损失函数$\ell_{\exp}$即最小化条件损失函数$\mathcal{E}(e^{-y h(\boldsymbol{x})} \mid \boldsymbol{x})$，$h(\boldsymbol{x})$需要满足方程
$$
\frac{\partial \mathcal{E}(e^{-y h(\boldsymbol{x})} \mid \boldsymbol{x})}{\partial h(\boldsymbol{x})} = -e^{-h(\boldsymbol{x})} p(y = 1 \mid x) + e^{h(\boldsymbol{x})} p(y = -1 \mid \boldsymbol{x}) = 0
$$

解得
$$
\hat{h}(\boldsymbol{x}) = \frac{1}{2} \ln \frac{p(y = 1 \mid \boldsymbol{x})}{p(y = -1 \mid \boldsymbol{x})}
$$

该分类器的预测结果满足
$$
\hat{y} = sgn(\hat{h}(\boldsymbol{x})) = sgn\left[ \frac{1}{2} \ln \frac{p(y = 1 \mid \boldsymbol{x})}{p(y = -1 \mid \boldsymbol{x})} \right] = \argmax_{y} p(y \mid \boldsymbol{x})
$$

即$\hat{h}(\boldsymbol{x})$是一个贝叶斯分类器，拥有理论最小分类误差率

### 算法流程

- （一）确定当前生成的基学习器的最优权重

根据数据分布$\mathcal{P}_{i}$应用基学习算法$\mathfrak{L}$生成基分类器$h_{i}$
$$
h_{i} = \mathfrak{L}(D \mid \mathcal{P}_{i})
$$
**指数损失函数**
$$
\begin{align*}
    \ell_{\exp}(\alpha_{i} h_{i} \mid \mathcal{P}) &= \mathcal{E}_{\mathcal{P}} \mathcal{E}(\exp\{-y \alpha_{i} h_{i}(\boldsymbol{x})\} \mid \boldsymbol{x}) \\ \\
    &= \mathcal{E}_{\mathcal{P}} \bigg[ e^{-\alpha_{i}} p(y = h_{i}(\boldsymbol{x}) \mid \boldsymbol{x}) + e^{\alpha_{i}} p(y \ne h_{i}(\boldsymbol{x})) \bigg] \\ \\
    &= e^{-\alpha_{i}} \mathcal{E}_{\mathcal{P}} p(y = h_{i}(\boldsymbol{x}) \mid \boldsymbol{x}) + e^{\alpha_{i}} \mathcal{E}_{\mathcal{P}} p(y \ne h_{i}(\boldsymbol{x}) \mid \boldsymbol{x}) \\ \\
    &= e^{-\alpha_{i}} (1 - \epsilon_{i}) + e^{\alpha_{i}} \epsilon_{i}
\end{align*}
$$
针对分类器权重$\alpha_{i}$最小化指数损失函数
$$
\frac{\partial \ell_{\exp}}{\partial \alpha_{i}} = -e^{-\alpha_{i}} (1 - \epsilon_{i}) + e^{\alpha_{i}} \epsilon_{i} = 0
$$
解出最优的基分类器权重
$$
\hat{\alpha}_{i} = \frac{1}{2} \ln \frac{1 - \epsilon_{i}}{\epsilon_{i}}
$$
其中基分类器$h_{i}$的分类误差率
$$
\epsilon_{i} = \mathcal{E}_{\mathcal{P}}p(y \ne h_{i} \mid \boldsymbol{x}) = \sum_{k = 1}^{m} \mathcal{P}_{i}(\boldsymbol{x}_{k}) \mathbb{I}(y_{k} \ne h_{i}(\boldsymbol{x}_{k}))
$$

- （二）准备下一个基学习器训练的最优数据分布

在原始数据分布$D$上最小化集成了下一个基学习器$h_{i + 1}$的学习器的指数损失函数
$$
\hat{h}_{i + 1} = \argmin_{h} \ell_{\exp}(\mathcal{H}_{i} + h \mid D)
$$
其中
$$
\mathcal{H}_{i}(\boldsymbol{x}) = \sum_{j = 1}^{i} \alpha_{j} h_{j}(\boldsymbol{x})
$$
指数损失函数
$$
\begin{align*}
    \ell_{\exp}(\mathcal{H}_{i} + h \mid D) &= \mathcal{E}_{D} \mathcal{E}(e^{-y\mathcal{H}_{i}(\boldsymbol{x})} e^{-yh(\boldsymbol{x})} \mid \boldsymbol{x}) \\ \\
    &\approx \mathcal{E}_{D} \mathcal{E}\left[ e^{-y\mathcal{H}_{i}(\boldsymbol{x})} \left( 1 - yh(\boldsymbol{x}) + \frac{y^{2} h^{2}(\boldsymbol{x})}{2} \right) \mid \boldsymbol{x} \right] \\ \\
    &\rightarrow \frac{1}{m} \sum_{k = 1}^{m} \left[ e^{-y_{k}\mathcal{H}_{i}(\boldsymbol{x}_{k})} \left( 1 - y_{k}h(\boldsymbol{x}_{k}) + \frac{y_{k}^{2} h^{2}(\boldsymbol{x}_{k})}{2} \right) \right]
\end{align*}
$$
理想的下一个基学习器
$$
\begin{align*}
    \hat{h}_{i + 1}(\boldsymbol{x}) &= \argmax_{h} \sum_{k = 1}^{m} y_{k}h(\boldsymbol{x}_{k}) e^{-y_{k}\mathcal{H}_{i}(\boldsymbol{x}_{k})} \\ \\
    &= \argmax_{h} \sum_{k = 1}^{m} y_{k}h(\boldsymbol{x}_{k}) e^{-y_{k}\mathcal{H}_{i}(\boldsymbol{x}_{k})} \bigg/ Z_{i + 1}
\end{align*}
$$
其中归一化因子$Z_{i + 1}$为
$$
Z_{i + 1} = \sum_{k = 1}^{m} e^{-y_{k}\mathcal{H}_{i}(\boldsymbol{x}_{k})}
$$
令
$$
\mathcal{P}_{i + 1}(\boldsymbol{x}_{k}) = e^{-y_{k}\mathcal{H}_{i}(\boldsymbol{x}_{k})} \bigg/ Z_{i + 1}
$$
相当于一个分布律，改写原式为
$$
\begin{align*}
    \hat{h}_{i + 1}(\boldsymbol{x}) &= \argmax_{h} \sum_{k = 1}^{m} y_{k}h(\boldsymbol{x}_{k}) \mathcal{P}_{i + 1}(\boldsymbol{x}_{k}) \\ \\
    &\rightarrow \argmax_{h} \mathcal{E}_{\mathcal{P}_{i + 1}} \mathcal{E}(yh(\boldsymbol{x}) \mid \boldsymbol{x}) \\ \\
    &= \argmax_{h} \mathcal{E}_{\mathcal{P}_{i + 1}} \bigg[ p(y = h(\boldsymbol{x}) \mid \boldsymbol{x}) - p(y \ne h(\boldsymbol{x}) \mid \boldsymbol{x}) \bigg] \\ \\
    &= \argmax_{h} \mathcal{E}_{\mathcal{P}_{i + 1}} \bigg[ 1 - 2 p(y \ne h(\boldsymbol{x}) \mid \boldsymbol{x}) \bigg] \\ \\
    &= \argmin_{h} \mathcal{E}_{\mathcal{P}_{i + 1}} p(y \ne h(\boldsymbol{x}) \mid \boldsymbol{x})
\end{align*}
$$
可以认为最优的下一个基学习器是通过数据分布$\mathcal{P}_{i + 1}$得到的

重复以上过程，并得到最终的集成分类器
$$
\mathcal{H}(\boldsymbol{x}) = \sum_{i = 1}^{n} \alpha_{i} h_{i}(\boldsymbol{x})
$$
可以将以上过程看作是权重和学习器的多级校准过程，进而不断减小集成后的偏差

## Bagging

在原有数据集$D$上**随机自助采样**出采样集$D_{i}$，基于采样集训练出学习器$h_{i}$
$$
h_{i} = \mathfrak{L}(D \mid D_{i})
$$
通过随机采样的$n$个采样集训练$n$个基学习器，并进行简单投票集成（回归任务使用简单平均集成）
$$
\mathcal{H}(\boldsymbol{x}) = \argmax_{y\ \in\ \mathcal{Y}} \sum_{i = 1}^{n} \mathbb{I}(h_{i}(\boldsymbol{x}) = y)
$$
样本在一次随机采样中没有被抽到的概率
$$
p(\boldsymbol{x} \notin D_{i}) = (1 - \frac{1}{m})^{m} \to \frac{1}{e}
$$
剩余的样本可以用作验证集，在仅考虑未使用过样本$\boldsymbol{x}$的学习器的集成预测结果 &rArr; **包外预测**
$$
\mathcal{H}^{oob}(\boldsymbol{x}) = \argmax_{y\ \in\ \mathcal{Y}} \sum_{i = 1}^{n} \mathbb{I}(h_{i}(\boldsymbol{x}) = y) \mathbb{I}(\boldsymbol{x} \notin D_{i}) 
$$
通过包外预测来估计$\mathrm{Bagging}$算法泛化误差率
$$
\epsilon^{oob} = \frac{1}{|D|} \sum_{k = 1}^{m} \mathbb{I}(\mathcal{H}^{oob}(\boldsymbol{x}_{k}) \ne y_{k})
$$

## Random Forest

使用决策树作为基学习器，构建子树时从原来的$d$个属性中随机选取$k$个属性作为子集，在**随机属性子集**中选取最优属性来构建决策树，通过这种方式引入基分类器的随机性

# 结合策略

**平均法** $\Rightarrow$ 回归任务

- **简单平均法**
$$
\mathcal{H}(\boldsymbol{x}) = \frac{1}{n} \sum_{i = 1}^{n} h_{i}(\boldsymbol{x})
$$
- **加权平均法**
$$
\begin{gather*}
    \mathcal{H}(\boldsymbol{x}) = \sum_{i = 1}^{n} w_{i} h_{i}(\boldsymbol{x}) \\ \\
    w_{i} \ge 0 \quad \sum_{i}^{n} w_{i} = 1
\end{gather*}
$$

**投票法** $\Rightarrow$ 分类任务

- **绝对多数投票法**
$$
\mathcal{H}(\boldsymbol{x}) = \left\{ 
    \begin{matrix}
        \omega_{j} & \sum_{i = 1}^{T} h_{i}^{j}(\boldsymbol{x}) > \frac{1}{2} \sum_{i = 1}^{n} \sum_{k = 1}^{c} h_{i}^{k}(\boldsymbol{x}) \\ \\
        none & otherwise
    \end{matrix}
\right.
$$
- **相对多数投票法**
$$
\mathcal{H}(\boldsymbol{x}) = \argmax_{\omega_{j}} \sum_{i = 1}^{n} h_{i}^{j}(\boldsymbol{x})
$$
- **加权投票法**
$$
\mathcal{H}(\boldsymbol{x}) = \argmax_{\omega_{j}} \sum_{i = 1}^{n} w_{i} h_{i}^{j}(\boldsymbol{x})
$$

# 多样性

## 误差-分歧分解

假设在回归任务$f: \mathbb{R}^{d} \to \mathbb{R}$上进行加权平均集成学习，定义基学习器$h_{i}$在样本$\boldsymbol{x}$上的**分歧**
$$
A(h_{i} \mid \boldsymbol{x}) = (h_{i}(\boldsymbol{x}) - \mathcal{H}(\boldsymbol{x}))^{2}
$$
集成的分歧
$$
\bar{A}(h \mid \boldsymbol{x}) = \sum_{i = 1}^{n} w_{i} A(h_{i} \mid \boldsymbol{x})
$$
从一定程度上可以反映基学习器的多样性，考虑学习器的**误差**
$$
\begin{gather*}
    E(h_{i} \mid \boldsymbol{x}) = (h_{i}(\boldsymbol{x}) - f(\boldsymbol{x}))^{2} \\ \\
    E(\mathcal{H} \mid \boldsymbol{x}) = (\mathcal{H}(\boldsymbol{x}) - f(\boldsymbol{x}))^{2}
\end{gather*}
$$
集成的误差
$$
\bar{E}(h \mid \boldsymbol{x})  = \sum_{i = 1}^{n} w_{i} E(h_{i} \mid \boldsymbol{x})
$$
可得
$$
\begin{align*}
    \bar{A}(h \mid \boldsymbol{x}) &= \sum_{i = 1}^{n} w_{i}(h_{i}(\boldsymbol{x}) - \mathcal{H}(\boldsymbol{x}))^{2} \\ \\
    &= \sum_{i = 1}^{n} w_{i}(h_{i}(\boldsymbol{x}) - f(\boldsymbol{x}) + f(\boldsymbol{x}) - \mathcal{H}(\boldsymbol{x}))^{2} \\ \\
    &= \bar{E}(h \mid \boldsymbol{x}) - E(\mathcal{H} \mid \boldsymbol{x})
\end{align*}
$$
在泛化样本上基分类器的分歧与误差的期望
$$
\begin{gather*}
    A_{i} = \mathcal{E} A(h_{i} \mid \boldsymbol{x}) = \int A(h_{i} \mid \boldsymbol{x}) p(\boldsymbol{x}) d\boldsymbol{x} \\ \\
    E_{i} = \mathcal{E} E(h_{i} \mid \boldsymbol{x}) = \int E(h_{i} \mid \boldsymbol{x}) p(\boldsymbol{x}) d\boldsymbol{x}
\end{gather*}
$$
考虑集成学习器的泛化误差
$$
E(\mathcal{H}) = \mathcal{E} E(\mathcal{H} \mid \boldsymbol{x}) = \bar{E} - \bar{A}
$$
其中
$$
\begin{gather*}
    \bar{A} = \mathcal{E} \bar{A}(h \mid \boldsymbol{x}) = \sum_{i = 1}^{n} w_{i} A_{i} \\ \\
    \bar{E} = \mathcal{E} \bar{E}(h \mid \boldsymbol{x}) = \sum_{i = 1}^{n} w_{i} E_{i}
\end{gather*}
$$
说明了基分类器的误差越小，多样性程度越高，集成分类器的性能越好

## 多样性度量

为了度量两个分类器的相似度，考虑两个分类器在统一数据集上的分类结果
$$
\begin{matrix}
&   h_{i} = +1  &   h_{i} = -1 \\ \\
h_{j} = +1  &   a   &   c \\ \\
h_{j} = -1  &   b   &   d
\end{matrix}
$$
常见的多样性度量标准
- 不合度量
$$
dis_{ij} = \frac{b + c}{m}
$$
- 相关系数
$$
\rho_{ij} = (ad - bc) \bigg/ \sqrt{(a + b)(a + c)(c + d)(b + d)}
$$
- $Q$-统计量
$$
Q_{ij} = \frac{ad - bc}{ad + bc}
$$
- $\kappa$-统计量
$$
\kappa = \frac{p_{1} - p_{2}}{1 - p_{2}}
$$
其中
$$
\begin{gather*}
    p_{1} = \frac{a + d}{m} \\ \\
    p_{2} = \frac{(a + b)(a + c) + (c + d)(b + d)}{m^{2}}
\end{gather*}
$$