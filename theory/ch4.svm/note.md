# 支持向量机（原始版本）
在线性可分问题中，存在一个能够将两类样本完全分开的判决边界（超平面）
$$
w^{\mathrm{T}} \boldsymbol{x} + b = 0
$$
任一样本点$\boldsymbol{x}$到判决边界的距离
$$
r(\boldsymbol{x}) = \frac{ |w^{\mathrm{T}} \boldsymbol{x} + b| }{||w||}
$$
当判决边界的法向量方向取定时，可以通过调节偏置项使得判决边界到两类样本点的最短距离相等，不妨设
$$
\begin{gather*}
\forall \boldsymbol{x} \in \mathcal{D}^{+} \quad w^{\mathrm{T}} \boldsymbol{x} + b \ge 1 \\ \\
\forall \boldsymbol{x} \in \mathcal{D}^{-} \quad w^{\mathrm{T}} \boldsymbol{x} + b \le 1
\end{gather*}
$$
其中到判决边界距离最小的样本称为**支持向量**，并满足
$$
y(w^{\mathrm{T}} \tilde{\boldsymbol{x}} + b) = 1
$$
为了尽可能提高分类器的鲁棒性，需要最大化支持向量的间隔
$$
\gamma = 2r(\tilde{\boldsymbol{x}}) = \frac{2}{||w||}
$$
该优化问题可以表述为
$$
\begin{gather*}
\max_{w,\ b} \frac{2}{||w||} \\ \\
s.t.\quad y_{i}(w^{\mathrm{T}} \boldsymbol{x}_{i} + b) \ge 1
\end{gather*}
$$
为方便计算，将以上优化问题转换为
$$
\begin{gather*}
\min_{w,\ b} \frac{||w||^2}{2} \\ \\
s.t.\quad y_{i}(w^{\mathrm{T}} \boldsymbol{x}_{i} + b) \ge 1
\end{gather*}
$$
定义拉格朗日函数
$$
\mathcal{L}(w,\ b,\ \alpha) = \frac{||w||^2}{2} + \sum_{i} \alpha_{i} 
\left[ 1 - y_{i}(w^{\mathrm{T}} \boldsymbol{x}_{i} + b) \right] \\ \\
$$
最优解满足方程
$$
\begin{gather*}
\frac{\partial \mathcal{L}}{\partial w} = w - \sum_{i} \alpha_{i} y_{i} \boldsymbol{x}_{i} = 0 \\ \\
\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i} \alpha_{i} y_{i} = 0
\end{gather*}
$$
在不等式约束下还需满足 **$\mathrm{KKT}$条件**
$$
\left \{
\begin{matrix}
\alpha_{i} \ge 0 \\ \\
y_{i}(w^{\mathrm{T}} \boldsymbol{x}_{i} + b) \ge 1 \\ \\
\alpha_{i} \left[ 1 - y_{i}(w^{\mathrm{T}} \boldsymbol{x}_{i} + b) \right] = 0
\end{matrix}
\right.
$$
将原优化问题分解为两步

* 在只考虑$\mathrm{KKT}$第一条件和第二条件下针对$\alpha$最大化拉格朗日函数
$$
\begin{gather*}
\max_{\alpha} \mathcal{L}(w,\ b,\ \alpha) \\ \\
s.t.\quad \alpha_{i} \ge 0 \quad y_{i}(w^{\mathrm{T}} \boldsymbol{x}_{i} + b) \ge 1
\end{gather*}
$$
当某个参数取值$w$和$b$能够满足所有第二条件时
$$
\max_{\alpha} \mathcal{L}(w,\ b,\ \alpha) = \frac{||w||^2}{2}
$$
当某个参数取值$w$和$b$不能够满足所有第二条件时
$$
\max_{\alpha} \mathcal{L}(w,\ b,\ \alpha) = \infty
$$
值得注意的是，在第一种情况下，优化后的$\alpha$同时满足第三条件
$$
\alpha_{i} \left[ 1 - y_{i}(w^{\mathrm{T}} \boldsymbol{x}_{i} + b) \right] = 0
$$
* 进而针对参数$w$和$b$最小化拉格朗日函数即可得到约束下原函数的条件极小值
$$
\min_{w,\ b} \max_{\alpha} \mathcal{L}(w,\ b,\ \alpha)
$$

优化后的极小值点属于上述的第一情况，即同时满足第二条件，因而优化问题只需要在第一条件下进行便可同时满足其他两个
$$
\begin{gather*}
\min_{w,\ b} \max_{\alpha} \mathcal{L}(w,\ b,\ \alpha) \\ \\
s.t.\quad \alpha_{i} \ge 0
\end{gather*}
$$
原优化问题转换为了$\min \max \mathcal{L}$的形式，相应的**对偶问题**
$$
\begin{gather*}
\max_{\alpha} \min_{w,\ b} \mathcal{L}(w,\ b,\ \alpha) \\ \\
s.t.\quad \alpha_{i} \ge 0
\end{gather*}
$$
内层的$\min_{w,\ b} \mathcal{L}$需要满足等式
$$
\begin{gather*}
\frac{\partial \mathcal{L}}{\partial w} = w - \sum_{i} \alpha_{i} y_{i} \boldsymbol{x}_{i} = 0 \\ \\
\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i} \alpha_{i} y_{i} = 0
\end{gather*}
$$
代入原拉格朗日函数
$$
\begin{align*}
\mathcal{L}(\alpha) &= \frac{1}{2} ||\sum_{i} \alpha_{i} y_{i} \boldsymbol{x}_{i}||^2 + \sum_{i} \alpha_{i} 
\left[ 1 - y_{i}((\sum_{j} \alpha_{j} y_{j} \boldsymbol{x}_{j}) \boldsymbol{x}_{i} + b) \right] \\ \\
&= \sum_{i} \alpha_{i} - \frac{1}{2} \sum_{i,\ j} (\alpha_{i} y_{i} \boldsymbol{x}_{i})^{\mathrm{T}} (\alpha_{j} y_{j} \boldsymbol{x}_{j}) \\ \\
&= \sum_{i} \alpha_{i} - \frac{1}{2} \sum_{i,\ j} \alpha_{i} \alpha_{j} y_{i} y_{j} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j}
\end{align*}
$$
对偶问题转化为
$$
\begin{gather*}
\max_{\alpha} \mathcal{L}(\alpha) = \sum_{i} \alpha_{i} - \frac{1}{2} \sum_{i,\ j} 
\alpha_{i} \alpha_{j} y_{i} y_{j} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j} \\ \\
s.t. \quad \sum_{i} \alpha_{i} y_{i} = 0 \quad \alpha_{i} \ge 0
\end{gather*}
$$
通过优化该对偶问题得到最优解$\hat{\alpha}$，即可解得
$$
\begin{gather*}
\hat{w} = \sum_{i} \hat{\alpha}_{i} y_{i} \boldsymbol{x}_{i} \\ \\
\hat{b} = y_{s} - \hat{w}^{\mathrm{T}} \boldsymbol{x}_{s}
\end{gather*}
$$
其中$\boldsymbol{x}_{s}$是任一支持向量（支持向量对应的$\alpha_{s} \ne 0$），为了使计算结果更加鲁棒，通常采取
$$
\hat{b} = \frac{1}{|\mathcal{D}_{s}|} \sum_{\boldsymbol{x} \in \mathcal{D}_{s}} (y_{s} - \hat{w}^{\mathrm{T}} \boldsymbol{x}_{s})
$$
**$\mathrm{SMO}$算法**

固定其他变量，单独优化两个变量$\alpha_{i}$和$\alpha_{j}$
$$
\begin{gather*}
\max_{\alpha_{i},\ \alpha_{j}} \mathcal{L}(\alpha_{i},\ \alpha_{j}) \\ \\
s.t. \quad \alpha_{i} y_{i} + \alpha_{j} y_{j} = -\sum_{k \ne i,\ j} \alpha_{k} y_{k};\ \alpha_{i},\ \alpha_{j} \ge 0 \\ \\
\mathcal{L}(\alpha_{i},\ \alpha_{j}) =
\begin{pmatrix}
1 - y_{i} \mathrm{R}^{\mathrm{T}} \boldsymbol{x}_{i} \\ \\
1 - y_{j} \mathrm{R}^{\mathrm{T}} \boldsymbol{x}_{j}
\end{pmatrix} \cdot
\begin{pmatrix}
\alpha_{i} \\ \\
\alpha_{j}
\end{pmatrix} -
\begin{pmatrix}
\alpha_{i} & \alpha_{j}
\end{pmatrix}
\begin{pmatrix}
\frac{1}{2} (y_{i} \boldsymbol{x}_{i})^2    &   y_{i} y_{j} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j} \\ \\
y_{i} y_{j} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j}  &   \frac{1}{2} (y_{j} \boldsymbol{x}_{j})^2
\end{pmatrix}
\begin{pmatrix}
\alpha_{i} \\ \\
\alpha_{j}
\end{pmatrix} \\ \\
\mathrm{R} = \sum_{k \ne i,\ j} \alpha_{k} y_{k} \boldsymbol{x}_{k}
\end{gather*}
$$
通过优化条件可以通过$\alpha_{i}$来表示$\alpha_{j}$，单步优化转化为单变量二次优化，可以求得解析解。在不断的单步优化下可以得到最终的数值解

# 支持向量机（核方法）
对于线性不可分问题，可以将样本映射到更高维的空间使得样本在新的特征空间线性可分
$$
\boldsymbol{x} \rightarrow \phi(\boldsymbol{x}),\quad \phi:\ \mathbb{R}^{\beta} \rightarrow \mathbb{R}^{\gamma}
$$
在新的特征空间判决边界可以表示为
$$
w^{\mathrm{T}} \phi(\boldsymbol{x}) + b = 0
$$
相应的对偶问题可以表示为
$$
\begin{gather*}
\max_{\alpha} \mathcal{L}(\alpha) = \sum_{i} \alpha_{i} - \frac{1}{2} \sum_{i,\ j} 
\alpha_{i} \alpha_{j} y_{i} y_{j} \phi(\boldsymbol{x}_{i})^{\mathrm{T}} \phi(\boldsymbol{x}_{j}) \\ \\
s.t.\quad \sum_{i} \alpha_{i} y_{i} = 0 \quad \alpha_{i} \ge 0
\end{gather*}
$$
利用**核函数**来代替映射后的特征向量的点积
$$
\kappa(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) = \phi(\boldsymbol{x}_{i})^{\mathrm{T}} \phi(\boldsymbol{x}_{j})
$$
问题转化为
$$
\begin{gather*}
\max_{\alpha} \mathcal{L}(\alpha) = \sum_{i} \alpha_{i} - \frac{1}{2} \sum_{i,\ j} 
\alpha_{i} \alpha_{j} y_{i} y_{j} \kappa(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) \\ \\
s.t. \quad \sum_{i} \alpha_{i} y_{i} = 0 \quad \alpha_{i} \ge 0
\end{gather*}
$$
求解后的判决边界
$$
f(\boldsymbol{x}) = w^{\mathrm{T}} \phi(\boldsymbol{x}) + b = (\sum_{i} \alpha_{i} y_{i} \boldsymbol{x}_{i})^{\mathrm{T}} \boldsymbol{x} + b
= \sum_{i} \alpha_{i} y_{i} \kappa(\boldsymbol{x}_{i},\ \boldsymbol{x}) + b
$$

## 核函数定理
为了判断核函数是否合法（是否满足点积的性质），考虑原空间任意向量组映射得到的新空间向量张成的线性空间
$$
\begin{gather*}
\boldsymbol{y} = a_{1} \phi(\boldsymbol{x}_{1}) + a_{2} \phi(\boldsymbol{x}_{2}) + \cdots + a_{m} \phi(\boldsymbol{x}_{m}) \\ \\
\forall\ m,\ a_{i} \in \mathbb{R},\ \boldsymbol{x}_{i} \in \mathbb{R}^{\beta}
\end{gather*}
$$
在该线性空间中的向量与自身进行点积
$$
\begin{align*}
\boldsymbol{y}^{2} &= (\phi(\boldsymbol{x}_{1}) + a_{2} \phi(\boldsymbol{x}_{2}) + \cdots + a_{m} \phi(\boldsymbol{x}_{m}))^{2} \\ \\
&= \sum_{i,\ j} a_{i} a_{j} \phi(\boldsymbol{x}_{i})^{\mathrm{T}} \phi(\boldsymbol{x}_{j}) = \sum_{i,\ j} a_{i} a_{j} \kappa(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) \\ \\
&=
\begin{pmatrix}
a_{1} & a_{2} & \cdots & a_{m}
\end{pmatrix}
\begin{pmatrix}
\kappa(\boldsymbol{x}_{1},\ \boldsymbol{x}_{1})     &   \kappa(\boldsymbol{x}_{1},\ \boldsymbol{x}_{2})  & \cdots   &   \kappa(\boldsymbol{x}_{1},\ \boldsymbol{x}_{m}) \\ \\
\kappa(\boldsymbol{x}_{2},\ \boldsymbol{x}_{1})     &   \kappa(\boldsymbol{x}_{2},\ \boldsymbol{x}_{2})  & \cdots   &   \kappa(\boldsymbol{x}_{2},\ \boldsymbol{x}_{m}) \\ \\
\vdots  &   \vdots  &   \ddots  &   \vdots \\ \\
\kappa(\boldsymbol{x}_{m},\ \boldsymbol{x}_{1})     &   \kappa(\boldsymbol{x}_{m},\ \boldsymbol{x}_{2})  & \cdots   &   \kappa(\boldsymbol{x}_{m},\ \boldsymbol{x}_{m})
\end{pmatrix}
\begin{pmatrix}
a_{1}  \\ \\
a_{2}  \\ \\
\vdots \\ \\
a_{m}
\end{pmatrix}
\end{align*}
$$
点积的性质要求上式非负，即要求式中的核矩阵$\mathbf{K}$正定
$$
\mathbf{K} =
\begin{pmatrix}
\kappa(\boldsymbol{x}_{1},\ \boldsymbol{x}_{1})     &   \kappa(\boldsymbol{x}_{1},\ \boldsymbol{x}_{2})  & \cdots   &   \kappa(\boldsymbol{x}_{1},\ \boldsymbol{x}_{m}) \\ \\
\kappa(\boldsymbol{x}_{2},\ \boldsymbol{x}_{1})     &   \kappa(\boldsymbol{x}_{2},\ \boldsymbol{x}_{2})  & \cdots   &   \kappa(\boldsymbol{x}_{2},\ \boldsymbol{x}_{m}) \\ \\
\vdots  &   \vdots  &   \ddots  &   \vdots \\ \\
\kappa(\boldsymbol{x}_{m},\ \boldsymbol{x}_{1})     &   \kappa(\boldsymbol{x}_{m},\ \boldsymbol{x}_{2})  & \cdots   &   \kappa(\boldsymbol{x}_{m},\ \boldsymbol{x}_{m})
\end{pmatrix}
$$
同时核函数也应该具有对称的特性，满足这些条件的核函数合法

常见的核函数

* 线性核
$$
\kappa(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) = \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j}
$$
* 多项式核
$$
\kappa(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) = \left( \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j} \right)^{d}
$$
* 高斯核/径向基核
$$
\kappa(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) = \exp \left( -\frac{||\boldsymbol{x}_{i} - \boldsymbol{x}_{j}||^2}{2\sigma^2} \right)
$$
* 拉普拉斯核
$$
\kappa(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) = \exp \left( -\frac{||\boldsymbol{x}_{i} - \boldsymbol{x}_{j}||}{\sigma} \right)
$$
* $sigmoid$核
$$
\kappa(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) = \tanh(\beta \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j} + \theta)
$$

也可以在已有的核函数的基础上通过函数组合得到新的核函数
$$
\tilde{\kappa}(\star,\ \ast) = 
\left\{ 
\begin{matrix}
\gamma_{1} \kappa_{1}(\star,\ \ast) + \gamma_{2} \kappa_{2}(\star,\ \ast) \\ \\
\kappa_{1}(\star,\ \ast) \kappa_{2}(\star,\ \ast) \\ \\
g(\star) \kappa(\star,\ \ast) g(\ast)
\end{matrix}
\right.
$$

# 支持向量机（软间隔）
线性不可分问题或存在某些异常点情况下，允许部分样本点不符合约束
$$
y_{i}(w^{\mathrm{T}} \boldsymbol{x}_{i} + b) \ge 1 \overset{\xi_{i}}{\longrightarrow} y_{i}(w^{\mathrm{T}} \boldsymbol{x}_{i} + b) \ge 1 - \xi_{i}
$$
优化问题转换为
$$
\begin{gather*}
\min_{w,\ b,\ \xi} \frac{||w||^2}{2} + C\sum_{i = 1}^{n} \xi_{i} \\ \\
s.t.\quad y_{i}(w^{\mathrm{T}} \boldsymbol{x}_{i} + b) \ge 1 - \xi_{i} \quad \xi_{i} \ge 0
\end{gather*} 
$$
构造拉格朗日函数
$$
\mathcal{L}(w,\ b,\ \xi,\ \alpha,\ \beta) = \frac{||w||^2}{2} + C\sum_{i = 1}^{n} \xi_{i} +
\sum_{i = 1}^{n} \alpha_{i} (1 - \xi_{i} - y_{i}(w^{\mathrm{T}} \boldsymbol{x}_{i} + b)) +
\sum_{i = 1}^{n} \beta_{i} (-\xi_{i})
$$
类似地，将优化问题重写为
$$
\begin{gather*}
\min_{w,\ b,\ \xi} \max_{\alpha,\ \beta} \mathcal{L}(w,\ b,\ \xi,\ \alpha,\ \beta) \\ \\
s.t.\quad \alpha_{i} \ge 0 \quad \beta_{i} \ge 0
\end{gather*}
$$
对应的对偶问题
$$
\begin{gather*}
\max_{\alpha,\ \beta} \min_{w,\ b,\ \xi} \mathcal{L}(w,\ b,\ \xi,\ \alpha,\ \beta) \\ \\
s.t.\quad \alpha_{i} \ge 0 \quad \beta_{i} \ge 0
\end{gather*}
$$
内层的$\min_{w,\ b,\ \xi} \mathcal{L}$需要满足方程
$$
\begin{gather*}
\frac{\partial \mathcal{L}}{\partial w} = w - \sum_{i = 1}^{n} \alpha_{i} y_{i} \boldsymbol{x}_{i} = 0 \\ \\
\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i = 1}^{n} \alpha_{i} y_{i} = 0 \\ \\
\frac{\partial \mathcal{L}}{\partial \xi_{i}} = C - \alpha_{i} - \beta_{i} = 0
\end{gather*}
$$
代入原拉格朗日函数
$$
\mathcal{L}(\alpha) = \sum_{i} \alpha_{i} - \frac{1}{2} \sum_{i,\ j} \alpha_{i} \alpha_{j} y_{i} y_{j} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j}
$$
考虑上述方程带来的约束，优化问题转换为
$$
\begin{gather*}
\max_{\alpha} \mathcal{L}(\alpha)= \sum_{i} \alpha_{i} - \frac{1}{2} \sum_{i,\ j} \alpha_{i} \alpha_{j} y_{i} y_{j} 
\boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j} \\ \\
s.t.\quad 0 \le \alpha_{i} \le C \quad \sum_{i = 1}^{n} \alpha_{i} y_{i} = 0
\end{gather*}
$$
同样地，采取$\mathrm{SMO}$算法对该对偶问题进行优化，即可得到最优解$\hat{w}$和$\hat{b}$

# 支持向量回归
对于任意有界的样本集合，总可以通过两个平行的超平面**边带**将样本集合完全包裹
$$
w^{\mathrm{T}} \boldsymbol{x} + b = \pm 1
$$
这里的样本向量$\boldsymbol{x}$包含了自变量与函数值。这两个边带的宽度为
$$
\gamma = \frac{2}{||w||}
$$
将回归任务看作寻找边带宽度最小的一对超平面边带，相应的回归方程
$$
\hat{w}^{\mathrm{T}} \boldsymbol{x} + \hat{b} = 0
$$
优化问题与支持向量机类似，只不过变为了最大化边带间隔
$$
\begin{gather*}
\max_{w,\ b} \frac{2}{||w||} \\ \\
s.t.\quad -1 \le w^{\mathrm{T}} \boldsymbol{x}_{i} + b \le 1
\end{gather*}
$$
将优化问题重写为
$$
\begin{gather*}
\max_{w,\ b} \frac{||w||^2}{2} \\ \\
s.t.\quad -1 \le w^{\mathrm{T}} \boldsymbol{x}_{i} + b \le 1
\end{gather*}
$$
类似地，为了使结果更具鲁棒性，在约束条件上添加松弛变量$\xi$和$\zeta$，并重写优化问题
$$
\begin{gather*}
\max_{w,\ b,\ \xi,\ \zeta} \frac{||w||^2}{2} - C \sum_{i = 1}^{n} (\xi_{i} + \zeta_{i}) \\ \\
s.t.\quad -1 - \zeta_{i} \le w^{\mathrm{T}} \boldsymbol{x}_{i} + b \le 1 + \xi_{i} \quad \xi_{i} \ge 0 \quad \zeta_{i} \ge 0
\end{gather*}
$$
构造拉格朗日函数
$$
\mathcal{L}(w,\ b,\ \xi,\ \zeta,\ \alpha,\ \beta,\ \mu,\ \nu) = 
\frac{||w||^2}{2} - C \sum_{i = 1}^{n} (\xi_{i} + \zeta_{i}) + \tilde{\mathcal{L}}
$$
其中拉格朗日乘子项$\tilde{\mathcal{L}}$为
$$
\tilde{\mathcal{L}} = \sum_{i = 1}^{n} \alpha_{i} (1 + \xi_{i} - (w^{\mathrm{T}} \boldsymbol{x}_{i} + b)) + 
\sum_{i = 1}^{n} \beta_{i} ((w^{\mathrm{T}} \boldsymbol{x}_{i} + b) + 1 + \zeta_{i}) + 
\sum_{i = 1}^{n} \mu_{i} \xi_{i} + \sum_{i = 1}^{n} \nu_{i} \zeta_{i}
$$
优化问题重写为
$$
\begin{gather*}
\max_{w,\ b,\ \xi,\ \zeta} \min_{\alpha,\ \beta,\ \mu,\ \nu} \mathcal{L}(w,\ b,\ \xi,\ \zeta,\ \alpha,\ \beta,\ \mu,\ \nu) \\ \\
s.t.\quad \alpha_{i} \ge 0 \quad \beta_{i} \ge 0 \quad \mu_{i} \ge 0 \quad \nu_{i} \ge 0
\end{gather*}
$$
对应的对偶问题
$$
\begin{gather*}
\min_{\alpha,\ \beta,\ \mu,\ \nu} \max_{w,\ b,\ \xi,\ \zeta} \mathcal{L}(w,\ b,\ \xi,\ \zeta,\ \alpha,\ \beta,\ \mu,\ \nu) \\ \\
s.t.\quad \alpha_{i} \ge 0 \quad \beta_{i} \ge 0 \quad \mu_{i} \ge 0 \quad \nu_{i} \ge 0
\end{gather*}
$$
内层的$\max_{w,\ b,\ \xi,\ \zeta} \mathcal{L}$需要满足方程
$$
\begin{gather*}
\frac{\partial \mathcal{L}}{\partial w} = w - \sum_{i = 1}^{n} (\alpha_{i} - \beta_{i}) \boldsymbol{x}_{i} = 0 \\ \\
\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i = 1}^{n} (\alpha_{i} - \beta_{i}) = 0 \\ \\
\frac{\partial \mathcal{L}}{\partial \xi_{i}} = \alpha_{i} + \mu_{i} - C = 0 \quad
\frac{\partial \mathcal{L}}{\partial \zeta_{i}} = \beta_{i} + \nu_{i} - C = 0
\end{gather*}
$$
代入原拉格朗日函数
$$
\mathcal{L}(\alpha,\ \beta) = \sum_{i = 1}^{n} (\alpha_{i} + \beta_{i}) - \frac{1}{2} \sum_{i,\ j}^{n} (\alpha_{i} - \beta_{i})
(\alpha_{j} - \beta_{j}) \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j}
$$
考虑到上述方程带来的约束，优化问题转换为
$$
\begin{gather*}
\min_{\alpha,\ \beta} \mathcal{L}(\alpha,\ \beta) = \sum_{i = 1}^{n} (\alpha_{i} + \beta_{i}) - \frac{1}{2} \sum_{i,\ j}^{n} (\alpha_{i} - \beta_{i}) (\alpha_{j} - \beta_{j}) \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j}\\ \\
s.t.\quad 0 \le \alpha_{i} \le C \quad 0 \le \beta_{i} \le C \quad \sum_{i = 1}^{n} (\alpha_{i} - \beta_{i}) = 0
\end{gather*}
$$
在非线性回归问题中也可以引入核函数或直接给出映射形式，采取$\mathrm{SMO}$算法对该对偶问题进行优化，即可得到最优解$\hat{w}$和$\hat{b}$