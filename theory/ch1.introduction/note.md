# 基本术语对照表

<style>
table td, table th {
  line-height: 1.5cm;
}
</style>
|术语|数学形式|描述|
|:-:|:-:|:-:|
|样本特征向量|$\boldsymbol{x}_{i} = \begin{pmatrix} x_{i1} \\[3mm] x_{i2} \\[3mm] \vdots \\[3mm] x_{id} \end{pmatrix}$|代表样本特征的向量|
|样本特征空间|$\boldsymbol{x}_{i} \in \mathcal{X}$|特征向量张成的空间|
|样本标记|$y_{i}$|样本对应的结果信息|
|标记空间|$y_{i} \in \mathcal{Y}$|标记张成的空间|
|样例|$(\boldsymbol{x}_{i},\ y_{i})$|样本特征和标记的元组|
|数据集|$\left \{ \boldsymbol{x}_{1}, \ \boldsymbol{x}_{2},\ \dots,\ \boldsymbol{x}_{m} \right \}$|样本的集合|
|训练（验证）集|$\{ (\boldsymbol{x}_{1},\ y_{1}),\ (\boldsymbol{x}_{2},\ y_{2}),\ \dots,\ (\boldsymbol{x}_{m},\ y_{m}) \}$|样例的集合|
|分类任务|$\mathcal{Y} = \{ 0,\ 1,\ \dots,\ c \}$|预测值为离散有限集合|
|回归任务|$\mathcal{Y} = \mathbb{R}$|预测值为连续实数空间|
|聚类任务|$D \to \{ C_{i} \mid i = 1,\ 2,\ \cdots,\ k \}$|将特征相似的样本进行分组|
|假设空间|$\mathcal{H} = \{ f \mid f: \mathcal{X} \mapsto \mathcal{Y} \}$|模型所能表示的所有映射的集合|
|学习||建立从样本空间到标记空间映射的过程|
|监督学习||所有训练样本的标记信息已知|
|半监督学习||部分训练样本的标记信息已知|
|无监督学习||训练样本的标记信息未知|

# 独立于算法的准则

**奥卡姆剃刀原则**；在所有与观察一致的假设中选择最简单的假设

**没有免费午餐定理**($\mathbb{NFL}$定理)
$$
\begin{align*}
E_{ote}(\mathfrak{L}_{a}) &= \sum_{f} E_{ote}(\mathfrak{L}_{a} | X,\ Y,\ f) \\ \\
        &= \sum_{f} \sum_{\boldsymbol{x} \in \mathcal{X} - X} \sum_{h} P(\boldsymbol{x}) P(h | \mathfrak{L}_{a},\ X,\ Y) \mathbb{I}(h(\boldsymbol{x}) \ne f(\boldsymbol{x})) \\ \\
        &= \sum_{\boldsymbol{x} \in \mathcal{X} - X} P(\boldsymbol{x}) \sum_{h} P(h | \mathfrak{L}_{a},\ X,\ Y) \sum_{f} \mathbb{I}(h(\boldsymbol{x}) \ne f(\boldsymbol{x})) \\ \\
        &= \sum_{\boldsymbol{x} \in \mathcal{X} - X} P(\boldsymbol{x}) \sum_{h} P(h | \mathfrak{L}_{a},\ X,\ Y) 
           |\mathcal{Y}|^{|\mathcal{X}| - |X|} \frac{1}{|\mathcal{Y}|} \\ \\
        &= |\mathcal{Y}|^{|\mathcal{X}| - |X| - 1} \sum_{\boldsymbol{x} \in \mathcal{X} - X} P(\boldsymbol{x}) \sum_{h} P(h | \mathfrak{L}_{a},\ X,\ Y) \\ \\
        &= |\mathcal{Y}|^{|\mathcal{X}| - |X| - 1} \sum_{\boldsymbol{x} \in \mathcal{X} - X} P(\boldsymbol{x}) = constant
\end{align*}
$$
定理表明，任何算法$\mathfrak{L}$通过相同训练集$\mathcal{D}$得到的假设函数$h$在任意真实目标函数$f$下在未见样本上产生的误差期望相同