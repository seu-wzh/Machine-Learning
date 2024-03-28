# 聚类任务

聚类任务通过无标记的样本信息给出数据集的一个**划分**，划分的子集称为**簇**
$$
\begin{gather*}
    D \to \{ C_{i} \mid i = 1,\ 2,\ \cdots,\ k \} \\ \\
    D = \cup_{i = 1}^{k} C_{i} \qquad C_{i} \cap_{i \ne j} C_{j} = \emptyset
\end{gather*}
$$
样本划分的**簇标记**（类比于分类任务的类别标记）
$$
\lambda_{i} \in \{ 1,\ 2,\ \cdots,\ k \} \qquad \boldsymbol{x}_{i} \in C_{\lambda_{i}}
$$

# 性能度量

## 外部指标

将聚类算法给出的簇划分和参考模型给出的簇划分进行比较，定义以下基本指标
$$
\begin{gather*}
    a = \bigg| SS = \{ (\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) \mid \lambda_{i} = \lambda_{j},\ \lambda_{i}^{*} = \lambda_{j}^{*},\ i < j \} \bigg| \\ \\
    b = \bigg| SD = \{ (\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) \mid \lambda_{i} = \lambda_{j},\ \lambda_{i}^{*} \ne \lambda_{j}^{*},\ i < j \} \bigg| \\ \\
    c = \bigg| DS = \{ (\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) \mid \lambda_{i} \ne \lambda_{j},\ \lambda_{i}^{*} = \lambda_{j}^{*},\ i < j \} \bigg| \\ \\
    d = \bigg| DD = \{ (\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) \mid \lambda_{i} \ne \lambda_{j},\ \lambda_{i}^{*} \ne \lambda_{j}^{*},\ i < j \} \bigg| \\ \\
    a + b + c + d = \mathrm{C}_{m}^{2} = \frac{m(m - 1)}{2}
\end{gather*}
$$
根据以上基本指标定义的常用的外部度量指标
- **$\mathrm{Jaccard}$系数**
$$
\mathrm{JC} = \frac{a}{a + b + c}
$$
- **$\mathrm{FM}$指数**
$$
\mathrm{FMI} = \sqrt{\frac{a}{a + b} \times \frac{a}{a + c}}
$$
- **$\mathrm{Rand}$指数**
$$
\mathrm{RI} = \frac{2(a + d)}{m(m - 1)} = \frac{a + d}{a + b + c + d}
$$

以上度量的值域均在$[0,\ 1]$内，并且数值越大聚类效果更接近参考模型

## 内部指标

直接考察聚类算法给出的划分结果，定义以下基本指标
$$
\begin{gather*}
    \mathrm{avg}(C) = \frac{2}{|C|(|C| - 1)} \sum_{i < j} d(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) \\ \\
    \mathrm{diam}(C) = \max_{i < j} d(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) \\ \\
    d_{min}(C_{i},\ C_{j}) = \min_{\boldsymbol{x}_{i} \in C_{i},\ \boldsymbol{x}_{j} \in C_{j}} d(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) \\ \\
    d_{cen}(C_{i},\ C_{j}) = d(\boldsymbol{\mu}_{i},\ \boldsymbol{\mu}_{j})
\end{gather*}
$$
根据以上基本指标定义的常用的内部度量指标
- **$\mathrm{DB}$指数**
$$
\mathrm{DBI} = \frac{1}{k} \sum_{i = 1}^{k} \max_{j \ne i} \left( \frac{\mathrm{avg}(C_{i}) + \mathrm{avg}(C_{j})}{d_{cen}(C_{i},\ C_{j})} \right)
$$
- **$\mathrm{Dunn}$指数**
$$
\mathrm{DI} = \min_{i \le j} d_{min}(C_{i},\ C_{j}) \bigg/ \max_{k} \mathrm{diam}(C_{k})
$$

$\mathrm{DB}$指数的数值越大，$\mathrm{Dunn}$指数的数值越小，聚类的效果越好

# 距离度量

距离度量需要满足的基本性质
- 非负性
$$
d(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) \ge 0
$$
- 同一性
$$
d(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) = 0  \Leftrightarrow \boldsymbol{x}_{i} = \boldsymbol{x}_{j}
$$
- 对称性
$$
d(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}) = d(\boldsymbol{x}_{j},\ \boldsymbol{x}_{i})
$$
- 直递性
$$
d(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}) \le d(\boldsymbol{x}_{i}, \boldsymbol{x}_{k}) + d(\boldsymbol{x}_{k}, \boldsymbol{x}_{j})
$$

常用的距离度量
- 闵可夫斯基距离（$\mathcal{L}_{p}$范数）
$$
d_{mk}^{(p)}(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) = \left( \sum_{u = 1}^{d} | x_{iu} - x_{ju} |^{p} \right)^{1/p}
$$
- 曼哈顿距离（$\mathcal{L}_{1}$范数）
$$
d_{man}(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) = \sum_{u = 1}^{d} | x_{iu} - x_{ju} |
$$
- 欧式距离（$\mathcal{L}_{2}$范数）
$$
d_{ed}(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) = \left( \sum_{u = 1}^{d} | x_{iu} - x_{ju} |^{2} \right)^{1/2} = \sqrt{\boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j}}
$$
- 切比雪夫距离（$\mathcal{L}_{\infty}$范数）
$$
d_{che}(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) = \max_{u} | x_{iu} - x_{ju} |
$$
- 马氏距离
$$
d_{mah}(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) = \sqrt{\boldsymbol{x}_{i}^{\mathrm{T}} \mathbf{M} \boldsymbol{x}_{j}} = \sqrt{\boldsymbol{x}_{i}^{\mathrm{T}} \mathbf{A} \mathbf{A}^{\mathrm{T}} \boldsymbol{x}_{j}}
$$

其中度量矩阵$\mathbf{M} = \mathbf{A} \mathbf{A}^{\mathrm{T}}$为（半）正定对称矩阵，$\mathbf{A}$为一组正交基
- $\mathrm{VDM}$距离
$$
\mathrm{VDM}^{(p)}(a,\ b) = \sum_{i = 1}^{k} \left| \frac{m_{u,\ a,\ i}}{m_{u,\ a}} - \frac{m_{u,\ b,\ i}}{m_{u,\ b}} \right|^{p}
$$

代表离散无序属性$u$的取值$a$和$b$的距离，其中$m_{u,\ x}$ $m_{u,\ x,\ i}$分别为总样本和样本簇$i$中取值为$x$的样本数量

- 混合闵可夫斯基-$\mathrm{VDM}$距离
$$
\mathrm{MinkovDM}^{(p)}(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) = \left[ \sum_{u = 1}^{d_{c}} | x_{iu} - x_{ju} |^{p} + \sum_{u = d_{c} + 1}^{d} \mathrm{VDM}^{(p)}(x_{iu},\ x_{ju}) \right]^{1/p}
$$

前$d_{c}$个属性为有序属性，后$d - d_{c}$个为无序属性

- 加权闵可夫斯基距离
$$
d_{wmk}(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) = \left( \sum_{u = 1}^{d} w_{u} | x_{iu} - x_{ju} |^{p} \right)^{1/p}
$$

# 原型聚类

## $k-means$

针对聚类划分所得簇最小化平方误差
$$
error = \sum_{i = 1}^{k} \sum_{\boldsymbol{x} \in C_{i}} (\boldsymbol{x} - \boldsymbol{\mu}_{i})^{\mathrm{T}} (\boldsymbol{x} - \boldsymbol{\mu}_{i})
$$
即最大化簇内样本相似程度，采取贪心算法进行迭代优化：（1）根据当前簇均值对数据集进行最近邻分类，得到$k$个新的簇（2）通过（1）得到的新簇计算新的簇均值。重复以上算法直到簇均值不再改变

## 学习向量量化（$\mathrm{LVQ}$）

假设数据含有标签数据，在标签数据作为先验知识的基础上进行半监督聚类
$$
D = \{ (\boldsymbol{x}_{1},\ y_{1}),\ \cdots,\ (\boldsymbol{x}_{n},\ y_{n}) \}
$$
类似$k-means$算法，$\mathrm{LVQ}$采用$k$个原型向量代表聚类簇
$$
\boldsymbol{p}_{1},\ \boldsymbol{p}_{2},\ \cdots,\ \boldsymbol{p}_{k}
$$
并且每个原型向量都被预设了一个类别标记，同时原型向量的数量可以比类别数更多
$$
t_{1},\ t_{2},\ \cdots,\ t_{k}
$$
随机地从数据集中选取一个样本$(\boldsymbol{x}_{i},\ y_{i})$，并找到该样本对应的最近邻原型向量$\hat{p}$

- 样本$\boldsymbol{x}_{i}$与原型向量$\hat{p}$的类别相同时
$$
\hat{p} \gets \hat{p} + \eta (\boldsymbol{x}_{i} - \hat{p}) = \eta \boldsymbol{x}_{i} + (1 - \eta) \hat{p}
$$

使得原型向量$\hat{p}$更新后更加靠近样本$\boldsymbol{x}_{i}$

- 样本$\boldsymbol{x}_{i}$与原型向量$\hat{p}$的类别不同时
$$
\hat{p} \gets \hat{p} - \eta (\boldsymbol{x}_{i} - \hat{p}) = -\eta \boldsymbol{x}_{i} + (1 + \eta) \hat{p}
$$

使得原型向量$\hat{p}$更新后更加远离样本$\boldsymbol{x}_{i}$

重复以上步骤直至最大迭代数或原型向量更新很小，算法结束

## 高斯混合聚类（$\mathrm{GMM}$）

假设样本的分布服从高斯混合分布
$$
p_{\mathcal{M}}(\boldsymbol{x}) = \sum_{i = 1}^{k} \alpha_{i} p_{i}(\boldsymbol{x}) = \sum_{i = 1}^{k} \alpha_{i} p(\boldsymbol{x} \mid \boldsymbol{\mu}_{i},\ \mathbf{\Sigma}_{i})
$$
该分布由$k$个混合成分组成，每个成分服从高斯分布
$$
p_{i}(\boldsymbol{x}) = \frac{1}{(2\pi)^{d/2} |\mathbf{\Sigma}_{i}|^{1/2} } \exp\left[ -\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu}_{i})^{\mathrm{T}} \mathbf{\Sigma}_{i}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}_{i}) \right]
$$
为了保证混合分布的合法性，混合系数$\alpha$需要满足
$$
\alpha_{i} > 0 \qquad \sum_{i = 1}^{k} \alpha_{i} = 1
$$
可以认为高斯混合分布的样本包含一个服从分布律$\alpha_{1},\ \alpha_{2},\ \cdots,\ \alpha_{k}$的隐变量$Z \in \{ 1,\ 2,\ \cdots,\ k \}$

隐变量$Z$和样本特征$\boldsymbol{x}$的联合分布
$$
p(\boldsymbol{x},\ Z = j \mid \theta) = \alpha_{j} p_{j}(\boldsymbol{x}) = \alpha_{j} p(\boldsymbol{x} \mid \boldsymbol{\mu}_{j},\ \mathbf{\Sigma}_{j})
$$
隐变量$Z$关于样本特征$\boldsymbol{x}$的后验分布
$$
p(Z = j \mid \boldsymbol{x},\ \theta) = p(\boldsymbol{x},\ Z = j \mid \theta) \bigg/ p_{\mathcal{M}}(\boldsymbol{x})
$$
高斯混合模型的参数$\{ \theta_{i} = (\alpha_{i},\ \boldsymbol{\mu}_{i},\ \mathbf{\Sigma}_{i}) \mid 1 \le i \le k \}$可以通过$\mathrm{EM}$算法来估计

- $\mathrm{E}$步
$$
\begin{align*}
    Q(\theta,\ \theta^{(m)}) &= \sum_{i = 1}^{n} \sum_{j = 1}^{k} \ln p(\boldsymbol{x}_{i},\ Z_{i} = j \mid \theta) p(Z_{i} = j \mid \boldsymbol{x}_{i},\ \theta^{(m)}) \\ \\
    &= \sum_{i = 1}^{n} \sum_{j = 1}^{k} \gamma_{ij}^{(m)} \left[ \ln \alpha_{j} - \frac{1}{2} (\boldsymbol{x}_{i} - \boldsymbol{\mu}_{j})^{\mathrm{T}} \mathbf{\Sigma}_{j}^{-1} (\boldsymbol{x}_{i} - \boldsymbol{\mu}_{j}) - \frac{1}{2} \ln |\mathbf{\Sigma}_{j}| - \frac{d}{2} \ln(2\pi) \right] \\ \\
    &\Rightarrow \sum_{i = 1}^{n} \sum_{j = 1}^{k} \gamma_{ij}^{(m)} \left[ \ln \alpha_{j} - \frac{1}{2} (\boldsymbol{x}_{i} - \boldsymbol{\mu}_{j})^{\mathrm{T}} \mathbf{\Sigma}_{j}^{-1} (\boldsymbol{x}_{i} - \boldsymbol{\mu}_{j}) - \frac{1}{2} \ln |\mathbf{\Sigma}_{j}| \right]
\end{align*}
$$

其中
$$
\gamma_{ij}^{(m)} = p(Z_{i} = j \mid \boldsymbol{x}_{i},\ \theta^{(m)})
$$

- $\mathrm{M}$步
$$
\begin{gather*}
    \max_{\theta} Q(\theta,\ \theta^{(m)}) \\ \\
    s.t.\quad \alpha_{i} \ge 0 \quad \sum_{j = 1}^{k} \alpha_{j} = 1
\end{gather*}
$$

对于$\boldsymbol{\mu}$和$\mathbf{\Sigma}$需要满足方程
$$
\begin{gather*}
    \frac{\partial Q}{\partial \boldsymbol{\mu}_{j}} = \sum_{i = 1}^{n} \gamma_{ij}^{(m)} \mathbf{\Sigma}_{j}^{-1} (\boldsymbol{x}_{i} - \boldsymbol{\mu}_{j}) = 0 \\ \\
    \frac{\partial Q}{\partial \mathbf{\Sigma}_{j}^{-1}} = \sum_{i = 1}^{n} \gamma_{ij}^{(m)} \left[ \frac{1}{2} \mathbf{\Sigma}_{j} - \frac{1}{2} (\boldsymbol{x}_{i} - \boldsymbol{\mu}_{j})^{\mathrm{T}} (\boldsymbol{x}_{i} - \boldsymbol{\mu}_{j})  \right] = 0
\end{gather*}
$$
解得
$$
\begin{gather*}
    \boldsymbol{\mu}_{j}^{(m + 1)} = \sum_{i = 1}^{n} \gamma_{ij}^{(m)} \boldsymbol{x}_{i} \bigg/ \sum_{i = 1}^{n} \gamma_{ij}^{(m)} \\ \\
    \mathbf{\Sigma}_{j}^{(m + 1)} = \sum_{i = 1}^{n} \gamma_{ij}^{(m)} (\boldsymbol{x}_{i} - \boldsymbol{\mu}_{j}^{(m + 1)})^{\mathrm{T}} (\boldsymbol{x}_{i} - \boldsymbol{\mu}_{j}^{(m + 1)}) \bigg/ \sum_{i = 1}^{n} \gamma_{ij}^{(m)}
\end{gather*}
$$
额外考虑$\alpha$的约束条件
$$
\mathcal{L}(\alpha,\ \lambda) = Q(\theta,\ \theta^{(m)}) + \lambda \left( \sum_{j = 1}^{k} \alpha_{j} - 1 \right)
$$
$\alpha$需要满足方程
$$
\begin{gather*}
    \frac{\partial \mathcal{L}}{\partial \alpha_{j}} = \sum_{i = 1}^{n} \gamma_{ij}^{(m)} \frac{1}{\alpha_{j}} + \lambda = 0 \\ \\
    \frac{\partial \mathcal{L}}{\partial \lambda} = \sum_{j = 1}^{k} \alpha_{j} - 1 = 0
\end{gather*}
$$
解得
$$
\alpha_{j}^{(m + 1)} = \frac{1}{m} \sum_{i = 1}^{n} \gamma_{ij}^{(m)}
$$
重复以上算法直至收敛即可估计出混合模型的参数$\hat{\alpha}$、$\hat{\boldsymbol{\mu}}$以及$\hat{\mathbf{\Sigma}}$，每个混合分布可以看作一簇

# 密度聚类

通过样本密度的角度考察样本的可连接性，在此基础上形成以样本作为顶点的图，图中的每个连通分量可以看作是一簇样本

## $\mathrm{DBSCAN}$

初始化核心对象集合
$$
\Omega = \{ \boldsymbol{x} \in D \mid |N_{\epsilon}(\boldsymbol{x})| \ge MinPts \}
$$
其中，$N_{\epsilon}(\boldsymbol{x})$为处于样本$\boldsymbol{x}$的$\epsilon$-邻域内的样本集合
$$
N_{\epsilon}(\boldsymbol{x}) = \{ \boldsymbol{x}' \in D \mid d(\boldsymbol{x},\ \boldsymbol{x}') \le \epsilon \}
$$
随机从核心对象集合中选取一个对象$o$作为迭代种子，通过广度优先搜索的变种遍历邻接节点。即出队节点只有核心对象才能够将$\epsilon$-邻域内的其他未访问节点入队，单步迭代的结果为图的一个连通分量，即一簇样本

# 层次聚类

层次聚类在不同层次上对样本进行划分，形成树形的聚类结构，划分可以采取自下而上或自上而下的策略

## $\mathrm{AGNES}$

$\mathrm{AGNES}$采取自下而上的聚合策略，将数据集中的每一个样本看作是一个初始聚类簇

计算聚类簇两两之间的距离，可以选用以下几种距离的形式
$$
\begin{gather*}
    d_{min}(C_{i},\ C_{j}) = \min_{\boldsymbol{x} \in C_{i},\ \boldsymbol{y} \in C_{j}} d(\boldsymbol{x},\ \boldsymbol{y}) \\ \\
    d_{max}(C_{i},\ C_{j}) = \max_{\boldsymbol{x} \in C_{i},\ \boldsymbol{y} \in C_{j}} d(\boldsymbol{x},\ \boldsymbol{y}) \\ \\
    d_{avg}(C_{i},\ C_{j}) = \frac{1}{|C_{i}||C_{j}|} \sum_{\boldsymbol{x} \in C_{i}} \sum_{\boldsymbol{y} \in C_{j}} d(\boldsymbol{x},\ \boldsymbol{y})
\end{gather*}
$$
算法的每次迭代选取距离最近的两个聚类簇进行合并，重复该过程直至达到预设的聚类簇数目