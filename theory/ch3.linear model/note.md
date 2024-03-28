线性回归模型基本表现形式
$$
f(\boldsymbol{x}) = w \cdot \boldsymbol{x} + b
$$
广义线性回归模型
$$
f(\boldsymbol{x}) = g^{-1}(w \cdot \boldsymbol{x} + b)
$$
其中$g(\cdot)$单调递增

# 线性回归
增广权重向量 &rArr; 权重&偏置
$$
w^{*} =
\begin{pmatrix}
w \\ \\
b
\end{pmatrix}
$$
扩展参数向量的线性模型表述
$$
f(\boldsymbol{x}) = w^{*} \cdot
\begin{pmatrix}
\boldsymbol{x} \\ \\
1
\end{pmatrix} = w^{*} \cdot \boldsymbol{x}^{*}
$$
将数据集表示为
$$
\mathbf{X} =
\begin{pmatrix}
x_{11}  &   x_{12}  &   \cdots  &   x_{1d}  &   1       \\ \\
x_{21}  &   x_{22}  &   \cdots  &   x_{2d}  &   1       \\ \\
\vdots  &   \vdots  &   \ddots  &   \vdots  &   \vdots  \\ \\
x_{m1}  &   x_{m2}  &   \cdots  &   x_{md}  &   1
\end{pmatrix}
\quad \quad
\mathbf{Y} =
\begin{pmatrix}
y_{1}   \\ \\
y_{2}   \\ \\
\vdots  \\ \\
y_{m}   
\end{pmatrix}
$$
最小化平方误差损失
$$
\hat{w^{*}} =
\argmin_{w^{*}} \frac{1}{2} (\mathbf{X} w^{*} -\mathbf{Y})^{\mathrm{T}} (\mathbf{X} w^{*} - \mathbf{Y})
$$
损失函数$\ell$对参数$w^{*}$的梯度
$$
\nabla_{w} \ell = \mathbf{X}^{\mathrm{T}} (\mathbf{X} w^{*} - \mathbf{Y})
$$
损失函数$\ell$极小值点
$$
\begin{gather*}
\nabla_{w} \ell = 0 \\ \\
\hat{w^{*}} = (\mathbf{X}^{\mathrm{T}} \mathbf{X})^{-1} \mathbf{X}^{\mathrm{T}} \mathbf{Y}
\end{gather*}
$$
通过上式即可求得线性回归的闭式解，但考虑到$\mathbf{X}^{\mathrm{T}} \mathbf{X}$不满秩产生无穷解的情况，在损失函数$\ell$的基础上引入正则化项
$$
\Delta \ell = \frac{1}{2} \lambda || w^{*} ||^2
$$
岭回归
$$
\begin{gather*}
\nabla_{w} (\ell + \Delta \ell) =
\mathbf{X}^{\mathrm{T}} (\mathbf{X} w^{*} - \mathbf{Y}) + \lambda \boldsymbol{I} w^{*} = 0 \\ \\
\hat{w^{*}} = (\mathbf{X}^{\mathrm{T}} \mathbf{X} + \lambda \boldsymbol{I})^{-1}
\mathbf{X}^{\mathrm{T}} \mathbf{Y}
\end{gather*}
$$
正则化项的意义在于通过对模型权重的衰减筛选出无关特征，达到对模型特征的降维效果

数值优化算法
* 对于高阶连续可微凸函数
$$
\begin{gather*}
\ell(\theta) \\ \\
\hat{\theta} = \argmin_{\theta} \ell(\theta)
\end{gather*}
$$
* **梯度下降法** &rArr; 沿着梯度负方向移动
$$
\theta^{n + 1} = \theta^{n} - \alpha \nabla_{\theta} \ell
$$
* **牛顿法** &rArr; 将函数局部看作二次函数，直接移动至该二次函数的最低点
$$
\theta^{n + 1} = \theta^{n} - H_{\theta}^{-1}(\ell) \nabla_{\theta} \ell
$$

# Logistic 回归

sigmoid 函数
$$
\begin{gather*}
y = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^{z}}{1 + e^{z}} \\ \\
{y}' = {\sigma}'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \frac{e^{-z}}{1 + e^{-z}}
= \sigma(z) (1 - \sigma(z)) = y(1 - y)
\end{gather*}
$$
根据广义线性回归的定义
$$
\begin{gather*}
z = w \cdot \boldsymbol{x} + b \\ \\
y = \sigma(z) = \sigma(w \cdot \boldsymbol{x} + b) \\ \\
\sigma^{-1}(y) = \ln{\frac{y}{1 - y}} = w \cdot \boldsymbol{x} + b
\end{gather*}
$$
将$y$看作样本类别后验概率，达到近似概率预测的效果
$$
\begin{gather*}
\ln{\frac{p(y = 1\ |\ \boldsymbol{x})}{p(y = 0\ |\ \boldsymbol{x})}} = w \cdot \boldsymbol{x} + b \\ \\
p(y = 1\ |\ \boldsymbol{x}) = \frac{\exp(w \cdot \boldsymbol{x} + b)}{1 + \exp(w \cdot \boldsymbol{x} + b)} \\ \\
p(y = 0\ |\ \boldsymbol{x}) = \frac{1}{1 + \exp(w \cdot \boldsymbol{x} + b)}
\end{gather*}
$$
通过最大似然法估计参数$w,\ b$
$$
L(w,\ b) = \sum_{i = 1}^{m} \ln{p(y_{i}\ |\ \boldsymbol{x}_{i};\ w,\ b)}
$$
重写式中的似然项
$$
\begin{gather*}
p(y_{i}\ |\ \boldsymbol{x}_{i};\ w,\ b) = y_{i}p_{1}(\boldsymbol{x}^{*};\ w^{*}) + (1 - y_{i})p_{0}(\boldsymbol{x}^{*};\ w^{*}) \\ \\
or \\ \\
p(y_{i}\ |\ \boldsymbol{x}_{i};\ w,\ b) = p_{1}(\boldsymbol{x}^{*};\ w^{*})^{y_{i}} p_{0}(\boldsymbol{x}^{*};\ w^{*})^{(1 - y_{i})}
\end{gather*}
$$
对数似然函数
$$
\begin{align*}
L(w^{*}) &= \sum_{i = 1}^{m} \ln{p(y_{i}\ |\ \boldsymbol{x}_{i};\ w^{*})} \\ \\
&= \sum_{i = 1}^{m} \ln\left[ y_{i}p_{1}(\boldsymbol{x}_{i}^{*};\ w^{*}) + (1 - y_{i})p_{0}(\boldsymbol{x}_{i}^{*};\ w^{*}) \right] \\ \\
&= \sum_{i = 1}^{m} \ln\frac{{y_{i} \exp(w^{*} \cdot \boldsymbol{x}_{i}^{*}) + (1 - y_{i})}}{1 + \exp(w^{*} \cdot \boldsymbol{x}_{i}^{*})} \\ \\
&= \sum_{i = 1}^{m}\left[\ln\left[ (\exp(w^{*} \cdot \boldsymbol{x}_{i}^{*}) - 1)y_{i} + 1 \right] -
\ln(1 + \exp(w^{*} \cdot \boldsymbol{x}_{i}^{*})) \right] \\ \\
&= \sum_{i = 1}^{m} \left[\left\{
\begin{matrix}
w^{*} \cdot \boldsymbol{x}_{i}^{*},\quad y_{i} = 1 \\ \\
0,\quad y_{i} = 0
\end{matrix} \right.  - \ln(1 + \exp(w^{*} \cdot \boldsymbol{x}_{i}^{*})) \right] \\ \\
&= \sum_{i = 1}^{m} \left[y_{i} w^{*} \cdot \boldsymbol{x}_{i}^{*} - \ln(1 + \exp(w^{*} \cdot \boldsymbol{x}_{i}^{*})) \right]
\end{align*}
$$
损失函数
$$
\ell(w^{*}) = -L(w^{*}) = \sum_{i = 1}^{m} \left[-y_{i} w^{*} \cdot \boldsymbol{x}_{i}^{*} +
\ln(1 + \exp(w^{*} \cdot \boldsymbol{x}_{i}^{*})) \right]
$$
对负对数似然损失函数进行变形
$$
\begin{align*}
    \ell(w^{*}) &= \sum_{i = 1}^{m} \left[ -y_{i} \ln \frac{\hat{y}_{i}}{1 - \hat{y}_{i}} + \ln(1 + \frac{\hat{y}_{i}}{1 - \hat{y}_{i}}) \right] \\ \\
    &= \sum_{i = 1}^{m} \bigg[ -y_{i} \ln \hat{y}_{i} + y_{i} \ln(1 - \hat{y}_{i}) - \ln(1 - \hat{y}_{i}) \bigg] \\ \\
    &= \sum_{i = 1}^{m} \bigg[ -y_{i} \ln \hat{y}_{i} - (1 - y_{i}) \ln(1 - \hat{y}_{i}) \bigg]
\end{align*}
$$
可以发现与交叉熵损失函数的形式一致。最小化损失函数$\ell$
$$
\hat{w^{*}} = \argmin_{w^{*}} \ell(w^{*})
$$
损失函数$\ell$对参数的梯度以及$Hessian$矩阵
$$
\begin{gather*}
\begin{align*}
\nabla_{w} \ell &= \sum_{i = 1}^{m} \left[-y_{i} \boldsymbol{x}_{i}^{*} +
\frac{\exp(w^{*} \cdot \boldsymbol{x}_{i}^{*})}{1 + \exp(w^{*} \cdot \boldsymbol{x}_{i}^{*})} \boldsymbol{x}^{*}_{i} \right] \\ \\
&= \sum_{i = 1}^{m} \left[-y_{i} +
\frac{\exp(w^{*} \cdot \boldsymbol{x}_{i}^{*})}{1 + \exp(w^{*} \cdot \boldsymbol{x}_{i}^{*})} \right] \boldsymbol{x}^{*}_{i} \\ \\
&= \sum_{i = 1}^{m} (\hat{y}_{i} - y_{i}) \boldsymbol{x}^{*}_{i}
\end{align*} \\ \\
H_{w}(\ell) = J_{w}(\nabla_{w} \ell) = \sum_{i = 1}^{m} \hat{y}_{i} (1 - \hat{y}_{i}) \boldsymbol{x}^{*}_{i} \boldsymbol{x}^{* \mathrm{T}}_{i}
\end{gather*}
$$
通过数值优化方法求解参数迭代最优解