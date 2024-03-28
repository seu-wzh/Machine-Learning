|术语|描述|
|:-:|:-:|
|训练误差|模型在训练集上的误差期望|
|泛化误差|模型在未见样本上的误差期望|
|过拟合|模型将训练样本具有的特殊性质作为所有样本的一般性质，产生过度的拟合|
|欠拟合|模型对样本性质学习不到位，拟合效果欠佳|

# 评估方法

**留出法**；将数据集$D$划分为两个互斥的集合分别作为训练集$S$和验证集$T$

**交叉验证法**；将数据集$D$划分为$k$个大小相似的互斥子集，分别使每个子集作为验证集来进行训练和评估，再将验证结果进行平均

**自助法**；对数据集$D$进行随机有放回抽取得到相同数量的数据集${D}'$，将$D - {D}'$作为验证集

# 性能度量

**均方误差**$MSE$
$$
MSE(f; D) = \frac{1}{m} \sum_{i = 1}^{m} (f(\boldsymbol{x}_{i}) - y_{i})^{2}
$$
**错误率**$error$ && **精度**$acc$
$$
\begin{gather*}
error(f; D) = \frac{1}{m} \sum_{i = 1}^{m} \mathbb{I}(f(\boldsymbol{x}_{i}) \ne y_{i}) \\ \\
acc(f; D) = \frac{1}{m} \sum_{i = 1}^{m} \mathbb{I}(f(\boldsymbol{x}_{i}) = y_{i}) \\ \\
error(f; D) + acc(f; D) = 1
\end{gather*}
$$
**混淆矩阵**$\boldsymbol{C}$
$$
\boldsymbol{C} = (c_{ij})_{c \times c}
$$
其中$c_{ij}$代表将真实类别i预测为类别j的样本个数

**二分类的混淆矩阵**
$$
\boldsymbol{C} = 
\begin{pmatrix}
TP & FN \\ \\
FP & TN
\end{pmatrix}
$$
**查准率$P$**；被预测为正例的正例占所有预测正例的比例
$$
P = \frac{TP}{TP + FP}
$$
**召回率$R$**；被预测为正例的正例占所有真实正例的比例
$$
R = \frac{TP}{TP + FN}
$$
**$P-R$曲线**；将预测概率进行排序，按顺序逐个将样本预测为正例得到的$P,\ R$值连成的曲线

**$F_{\beta}$度量**；查准率$P$和召回率$R$的加权调和平均
$$
\begin{gather*}
\frac{1}{F_{\beta}} = \frac{1}{1 + \beta^2}  \frac{1}{P} + \frac{\beta^2}{1 + \beta^2}  \frac{1}{R} \\ \\
F_{\beta} = \frac{(1 + \beta^2) \times P \times R}{\beta^2 \times P + R}
\end{gather*}
$$
当$\beta^2 < 1$时查准率更重要，当$\beta^2 > 1$时召回率更重要

**$\mathrm{TPR}$**；被预测为正例的正例占所有真实正例的比例
$$
\mathrm{TPR} = \frac{TP}{TP + FN}
$$
**$\mathrm{FPR}$**；被预测为正例的负例占所有真实负例的比例
$$
\mathrm{FPR} = \frac{FP}{TN + FP}
$$
**正负例排序误差**；排在每个正例前面的负例的个数之和
$$
\ell_{rank} = \sum_{x^{+} \in D^{+}} \sum_{x^{-} \in D^{-}} \mathbb{I}(f(x^{+}) < f(x^{-}))
$$
对误差进行归一化得
$$
\ell_{rank} = \frac{1}{|D^{+}||D^{-}|} \sum_{x^{+} \in D^{+}} \sum_{x^{-} \in D^{-}} \mathbb{I}(f(x^{+}) < f(x^{-}))
$$
**$\mathrm{ROC}$曲线**；类似于$P-R$曲线，将预测概率进行排序，再按顺序逐个将样本预测为正例得到的$\mathrm{TPR},\ \mathrm{FPR}$连成的曲线

**$\mathrm{AUC}$**；$\mathrm{ROC}$曲线下的面积
$$
\begin{align*}
\mathrm{AUC} &= \sum_{x^{-} \in D^{-}} \frac{1}{|D^{-}|} \mathrm{ROC}(x^{-}) \\ \\
&= \sum_{x^{-} \in D^{-}} \frac{1}{|D^{-}|} \frac{|D^{+}| - \sum_{x^{+} \in D^{+}} \mathbb{I}(f(x^{+}) < f(x^{-}))}{|D^{+}|} \\ \\
&= 1 - \frac{1}{|D^{+}||D^{-}|} \sum_{x^{+} \in D^{+}} \sum_{x^{-} \in D^{-}} \mathbb{I}(f(x^{+}) < f(x^{-})) \\ \\
&= 1 - \ell_{rank}
\end{align*}
$$
即$\mathrm{AUC}$ &uarr; &rArr; 排序效果 &uarr;

# 偏差与方差

假设有多个从样本空间独立采样的大小相似的数据集
$$
D_{1},\ D_{2},\ \dots,\ D_{n}
$$
学习算法$\mathfrak{L}$通过数据集$D$学习得到的样本$\boldsymbol{x}$预测值为
$$
f(\boldsymbol{x}; D)
$$
学习算法$\mathfrak{L}$通过各个数据集学习得到的预测函数的均值为
$$
\bar{f}(\boldsymbol{x}) = \mathcal{E}_{D}\left[ f(\boldsymbol{x}; D) \right]
$$
学习算法$\mathfrak{L}$由于数据集而产生的的预测方差为
$$
var(\boldsymbol{x}) = \mathcal{E}_{D}\left[ (f(\boldsymbol{x}; D) - \bar{f}(\boldsymbol{x}))^2 \right]
$$
数据集$D$在采样同样的样本$\boldsymbol{x}$的标记时产生的随机噪声
$$
\varepsilon_{D} = y_{D} - y
$$
假设随机噪声$\varepsilon_{D}$的期望为0
$$
\mathcal{E}_{D} \varepsilon_{D} = 0
$$
随机噪声$\varepsilon_{D}$平方的的期望
$$
\varepsilon^2 = \mathcal{E}_{D} \varepsilon_{D}^2 = \mathcal{E}_{D}\left[ (y_{D} - y)^2 \right]
$$
预测均值和真实标记之间的偏差
$$
bias^2(\boldsymbol{x}) = (\bar{f}(\boldsymbol{x}) - y)^2
$$
学习算法$\mathfrak{L}$学到的各个预测函数在样本$\boldsymbol{x}$上的泛化误差 &rArr; 样本$\boldsymbol{x}$在所有数据集上都属于学习算法$\mathfrak{L}$学习时的未见样本
$$
\begin{align*}
error(\boldsymbol{x}; D) &= \mathcal{E}_{D}\left[ (f(\boldsymbol{x}; D) - y_{D})^2 \right] \\ \\
&= \mathcal{E}_{D}\left[ (f(\boldsymbol{x}; D) - \bar{f}(\boldsymbol{x}) + \bar{f}(\boldsymbol{x}) - y_{D})^2 \right] \\ \\
&= \mathcal{E}_{D}\left[ (f(\boldsymbol{x}; D) - \bar{f}(\boldsymbol{x}))^2 \right] +
\mathcal{E}_{D}\left[ (\bar{f}(\boldsymbol{x}) - y_{D})^2 \right] +
2\mathcal{E}_{D}\left[ (f(\boldsymbol{x}; D) - \bar{f}(\boldsymbol{x}))(\bar{f}(\boldsymbol{x}) - y_{D}) \right] \\ \\
&= \mathcal{E}_{D}\left[ (f(\boldsymbol{x}; D) - \bar{f}(\boldsymbol{x}))^2 \right] +
\mathcal{E}_{D}\left[ (\bar{f}(\boldsymbol{x}) - y_{D})^2 \right] +
2\mathcal{E}_{D}\left[ (f(\boldsymbol{x}; D) - \bar{f}(\boldsymbol{x}))(\bar{f}(\boldsymbol{x}) - y - \varepsilon_{D}) \right]
\end{align*}
$$
由于样本的未见性，所以每个数据集在采集样本$\boldsymbol{x}$的标记$y$时的随机噪声$\varepsilon$与学习算法的学习过程无关，也就与最终学到的预测函数无关
$$
\begin{align*}
error(\boldsymbol{x}; D) &= \mathcal{E}_{D}\left[ (f(\boldsymbol{x}; D) - \bar{f}(\boldsymbol{x}))^2 \right] + \mathcal{E}_{D}\left[ (\bar{f}(\boldsymbol{x}) - y_{D})^2 \right] + 2\mathcal{E}_{D} [f(\boldsymbol{x}; D) - \bar{f}(\boldsymbol{x})] \mathcal{E}_{D} [\bar{f}(\boldsymbol{x}) - y - \varepsilon_{D}] \\ \\
&= \mathcal{E}_{D}\left[ (f(\boldsymbol{x}; D) - \bar{f}(\boldsymbol{x}))^2 \right] + \mathcal{E}_{D}\left[ (\bar{f}(\boldsymbol{x}) - y_{D})^2 \right] \\ \\
&= \mathcal{E}_{D}\left[ (f(\boldsymbol{x}; D) - \bar{f}(\boldsymbol{x}))^2 \right] + \mathcal{E}_{D}\left[ (\bar{f}(\boldsymbol{x}) - y - \varepsilon_{D})^2 \right] \\ \\
&= \mathcal{E}_{D}\left[ (f(\boldsymbol{x}; D) - \bar{f}(\boldsymbol{x}))^2 \right] + \mathcal{E}_{D}\left[ (\bar{f}(\boldsymbol{x}) - y)^2 \right] + \mathcal{E}_{D}\varepsilon_{D}^2 - 2(\bar{f}(\boldsymbol{x}) - y)\mathcal{E}_{D} \varepsilon_{D} \\ \\
&= bias^2(\boldsymbol{x}) + var(\boldsymbol{x}) + \varepsilon^2
\end{align*}
$$
* 即误差 = 偏差 + 方差 + 噪声