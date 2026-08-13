# 小批次多次训练等价于大批次训练：完整数学推导（修订定稿）

> 本文是 `es_batch_equivalence.md` 与 `es_batch_equivalence_proof.md` 的**修正整合定稿**，已吸收两份
> 审查报告的全部有效结论，并修正以下缺陷：① 符号统一为"最大化 fitness → 梯度上升"；② 定理 1 区分
> 无偏基线与同批均值基线的 $1/N$ 偏差；③ 方差严格区分**迹（总方差）**与**单分量方差**；④ 定理 5 显式
> 声明无偏前提，并纳入局部中心化的 $O(\eta'/N_s)$ 漂移项；⑤ 定理 6 重写极限设定；⑥ 定理 4 改为"主阶相等"。
>
> 代码事实锚点（`src/hyperscalees/noiser/alteggroll.py`）：z-score `convert_fitnesses` 执行
> `(raw - mean)/sqrt(var + 1e-5)`（同批均值 + 除以样本标准差）；对偶采样 `sigma = ±base_sigma`
> （`thread_id % 2`）、`true_thread_idx = thread_id // 2`。

---

## 0. 摘要与定理地图

设小批量 $N_s$、大批量 $N_L=KN_s$（$K$ 为累积次数）、微观学习率 $\eta$、宏观学习率 $\eta'=\eta K$。

**核心结论**：在"每步使用全新独立噪声 + 无偏基线 + 学习率按宏观步长折算"的前提下，小批次多次训练与
大批次训练是**同一梯度流的两种一阶离散化**——漂移一致到 $O(\eta'^2)$、噪声协方差主阶精确相等
（误差 $O(\eta'^3)$），因此在 $\eta'\to0$ 的连续极限下严格弱收敛到同一条梯度流（及同一扩散近似 SDE）。

| 编号 | 结论 | 等价类型 | 误差阶 |
|---|---|---|---|
| 定理 2 | 参数**冻结**时，$K$ 段累积 = 单个大批次 | 代数精确 | $0$ |
| 定理 3 | 冻结点上两者**渐近同分布**（CLT） | 渐近分布 | $O(1/\sqrt N)$ |
| 定理 4 | 动态下累积噪声**协方差主阶精确相等** | 二阶矩主阶 | $O(\eta'^3)$ |
| 定理 5 | 动态下漂移一阶相容 | 一阶相容 | $O(\eta'^2)$ |
| 定理 6 | $\eta'\to0$ 时弱收敛到同一梯度流 / 同一 SDE | 极限等价 | $O(\eta')$ |

---

## 1. 设定、记号与符号约定

### 1.1 基本量

- $x\in\mathcal M=\mathbb R^d$：参数向量；
- $f:\mathcal M\to\mathbb R$：**fitness（最大化目标）**；
- $\varepsilon\sim\mathcal N(0,I_d)$：扰动方向；
- $\sigma>0$：扰动尺度；
- $N$：批量（并行样本数）。

**符号约定（重要）**：ES 最大化 fitness，因此更新为**梯度上升** $x\leftarrow x+\eta\hat g$；流形上的
梯度流为 $\dot x=+\operatorname{grad}F_\sigma$。

### 1.2 ES 估计器与两种基线

$$
\hat g_N(x;b)=\frac1N\sum_{i=1}^N\frac{f(x+\sigma\varepsilon_i)-b}{\sigma}\,\varepsilon_i. \tag{1}
$$

基线 $b$ 有两种选择，二者对无偏性的影响截然不同：

- **独立基线**：$b$ 与 $\{\varepsilon_i\}$ 独立（如真实总体均值 $F_\sigma(x)$，或另一批样本的均值）→ 严格无偏；
- **同批样本均值**：$b=\bar f=\frac1N\sum_j f_j$（EggRoll `convert_fitnesses` 的实际做法）→ 有 $1/N$ 偏差。

> EggRoll 的 z-score 实际是 $\dfrac{f_i-\bar f}{\sqrt{\operatorname{var}+10^{-5}}}$，还包含**除以样本标准差**
> 的非线性操作；下文严格分析中先讨论纯均值中心化，再在 §8 说明标准差归一化带来的额外偏差/尺度。

---

## 2. 第一性原理：ES = 热核平滑目标的梯度

### 2.1 引理 1（Stein 引理 / 高斯分部积分）

> **引理 1。** 对任意光滑 $\varphi:\mathbb R^d\to\mathbb R$ 满足适当增长条件，
>
> $$
> \mathbb E_{\varepsilon\sim\mathcal N(0,I)}\big[\varepsilon_j\,\varphi(\varepsilon)\big]
> =\mathbb E\!\left[\frac{\partial\varphi}{\partial\varepsilon_j}(\varepsilon)\right]. \tag{2}
> $$

**证明。** 记密度 $p(\varepsilon)=(2\pi)^{-d/2}e^{-\|\varepsilon\|^2/2}$，则 $\partial_j p=-\varepsilon_j p$。分部积分：

$$
\mathbb E[\varepsilon_j\varphi]=\int\varepsilon_j\varphi\,p\,d\varepsilon
=-\int\varphi\,\partial_j p\,d\varepsilon
=\int\partial_j\varphi\,p\,d\varepsilon=\mathbb E[\partial_j\varphi]. \qquad\blacksquare
$$

### 2.2 定义 1（高斯平滑目标 / 热核磨光）

$$
F_\sigma(x):=\mathbb E_{\varepsilon\sim\mathcal N(0,I)}[f(x+\sigma\varepsilon)]
=\int f(y)\,\mathcal N(y;x,\sigma^2 I)\,dy. \tag{3}
$$

**热核解释**：$F_\sigma=e^{\frac{\sigma^2}{2}\Delta}f$ 是热方程 $\partial_\tau u=\tfrac12\Delta u$（初值 $u(\cdot,0)=f$）
在时刻 $\tau=\sigma^2$ 的解。ES 本质上是**热核磨光景观 $F_\sigma$ 上的梯度上升**——把噪声尺度 $\sigma$ 与几何正则化
（热方程）统一起来，这是全文的"第一性原理锚点"。

### 2.3 定理 1（无偏情形）与推论 1（同批均值 $1/N$ 偏差）

> **定理 1（独立基线无偏）。** 若基线 $b$ 与 $\{\varepsilon_i\}$ 独立，则对任意 $x$，
>
> $$
> \mathbb E[\hat g_N(x;b)]=\nabla F_\sigma(x). \tag{4}
> $$

**证明。** 由独立性，$\mathbb E[(b)\varepsilon_i]=\mathbb E[b]\mathbb E[\varepsilon_i]=0$；由引理 1 取
$\varphi(\varepsilon)=f(x+\sigma\varepsilon)$，且 $\partial_{\varepsilon_j}f(x+\sigma\varepsilon)=\sigma\partial_{x_j}f(x+\sigma\varepsilon)$：

$$
\mathbb E[\hat g_N]_j=\frac1{\sigma N}\sum_i\mathbb E[f(x+\sigma\varepsilon_i)\varepsilon_{i,j}]
=\frac1{\sigma N}\sum_i\mathbb E[\sigma\partial_{x_j}f(x+\sigma\varepsilon_i)]
=\partial_{x_j}F_\sigma(x). \qquad\blacksquare
$$

> **推论 1（同批样本均值有 $1/N$ 偏差）。** 若 $b=\bar f=\frac1N\sum_j f_j$，则
>
> $$
> \mathbb E[\hat g_N(x;\bar f)]=\Big(1-\frac1N\Big)\nabla F_\sigma(x). \tag{5}
> $$

**证明。** 展开 $\mathbb E[\bar f\,\varepsilon_i]=\frac1N\sum_j\mathbb E[f_j\varepsilon_i]$。当 $j\ne i$，
$\varepsilon_j\perp\varepsilon_i$，故 $\mathbb E[f_j\varepsilon_i]=\mathbb E[f_j]\mathbb E[\varepsilon_i]=0$；
当 $j=i$，由 Stein 得 $\sigma\nabla F_\sigma$。故 $\mathbb E[\bar f\varepsilon_i]=\frac{\sigma}{N}\nabla F_\sigma$，于是

$$
\mathbb E[\hat g_N]=\frac1{\sigma N}\sum_i\Big(\sigma\nabla F_\sigma-\frac{\sigma}{N}\nabla F_\sigma\Big)
=\Big(1-\frac1N\Big)\nabla F_\sigma. \qquad\blacksquare
$$

> **严格无偏的三种修正**：(i) 独立基线；(ii) leave-one-out 基线 $\bar f_{-i}=\frac1{N-1}\sum_{j\ne i}f_j$；
> (iii) Bessel 类校正 $\frac{N}{N-1}\hat g_N$。EggRoll 未做这些修正，故工程上存在 $O(1/N)$ 偏差（$N=200$ 时约 $0.5\%$）。

---

## 3. 方差结构：迹与单分量的精确区分（Isserlis/Wick）

单样本估计器（已中心化）的一阶展开：

$$
\hat g_1\approx \underbrace{\frac{f(x)-b}{\sigma}\varepsilon}_{\text{基线项}}
+\underbrace{(\varepsilon^\top g)\varepsilon}_{\text{梯度项}},\qquad g=\nabla f(x). \tag{6}
$$

### 3.1 基线项（无中心化时）

$\operatorname{Cov}\!\big(\frac{f}{\sigma}\varepsilon\big)=\frac{f(x)^2}{\sigma^2}I_d$，故

- 单分量方差 $=\dfrac{f(x)^2}{\sigma^2}$；
- **迹（总方差）** $=d\,\dfrac{f(x)^2}{\sigma^2}$。

### 3.2 梯度项（Isserlis 精确）

令 $G=(\varepsilon^\top g)\varepsilon$。由 $\mathbb E[G]=g$（Stein），且由 Isserlis/Wick：

$$
\mathbb E[\varepsilon_k\varepsilon_l\varepsilon_i\varepsilon_j]=\delta_{kl}\delta_{ij}+\delta_{ki}\delta_{lj}+\delta_{kj}\delta_{li}.
$$

于是

$$
\mathbb E[(\varepsilon^\top g)^2\varepsilon_i\varepsilon_j]
=\sum_{k,l}g_kg_l\,\mathbb E[\varepsilon_k\varepsilon_l\varepsilon_i\varepsilon_j]
=\|g\|^2\delta_{ij}+2g_ig_j,
$$

故 $\mathbb E[GG^\top]=\|g\|^2I+2gg^\top$，从而

$$
\operatorname{Cov}(G)=\|g\|^2 I + gg^\top. \tag{7}
$$

取迹与单分量平均：

$$
\operatorname{tr}\operatorname{Cov}(G)=(d+1)\|g\|^2,
\qquad
\tfrac1d\operatorname{tr}\operatorname{Cov}(G)=\big(1+\tfrac1d\big)\|g\|^2. \tag{8}
$$

> **注**：$(d+1)\|g\|^2$ 是**总方差（迹）**，不是"每分量"；单分量平均方差是 $(1+1/d)\|g\|^2$，第 $j$ 分量是
> $(\partial_j f)^2+\|g\|^2$。

### 3.3 总方差（单样本，$N$ 样本再除 $N$）

$$
\operatorname{tr}\operatorname{Var}(\hat g_1)\approx d\frac{f(x)^2}{\sigma^2}+(d+1)\|g\|^2,\qquad
\text{单分量均值}\approx\frac{f(x)^2}{\sigma^2}+\big(1+\tfrac1d\big)\|g\|^2. \tag{9}
$$

中心化（独立基线）后基线项被消除，仅剩梯度项。

### 3.4 对偶采样：梯度项方差 = 独立采样的 2 倍

对偶单对 $\pm\varepsilon$ 的梯度项为

$$
\frac{f(x+\sigma\varepsilon)-f(x-\sigma\varepsilon)}{2\sigma}\varepsilon
\approx\frac{2\sigma\varepsilon^\top g}{2\sigma}\varepsilon=(\varepsilon^\top g)\varepsilon=G,
$$

即**每个对偶对贡献与一个独立样本完全相同的梯度项随机量**。总样本 $N$ → $N/2$ 对，故

$$
\operatorname{Cov}(\hat g_{\text{anti}})\big|_{\text{梯度项}}=\frac{\operatorname{Cov}(G)}{N/2}=2\cdot\frac{\operatorname{Cov}(G)}{N}.
$$

**比值精确为 $2$**。这是对偶采样"方向数减半"的代价；它靠消除通常更大的基线项与偶次项而总体占优。

---

## 4. 流形结构：梯度流与 retraction

### 4.1 Riemannian 设定

设 $(\mathcal M,G)$ 为 $d$ 维 Riemannian 流形，度量张量 $G(x)$ 正定。两种度量：

- **Euclidean** $G=I_d$：普通梯度上升；
- **Fisher 信息度量** $G(x)=\mathbb E[\nabla_x\log p(\cdot|x)\nabla_x\log p(\cdot|x)^\top]$：自然梯度 / 信息几何。

流形上的**最速上升方向**为

$$
\operatorname{grad}F_\sigma(x)=G^{-1}(x)\nabla F_\sigma(x). \tag{10}
$$

**梯度流（上升）**：

$$
\dot x(t)=+\operatorname{grad}F_\sigma(x(t)). \tag{11}
$$

### 4.2 retraction 与一阶相容性

**定义 2（retraction）。** $R_x:T_x\mathcal M\to\mathcal M$，$R_x(0)=x$，$DR_x(0)=\mathrm{id}$。一阶 retraction：

$$
R_x(v)=x+v+O(\|v\|^2). \tag{12}
$$

Euclidean 取 $R_x(v)=x+v$（精确、零曲率）。任何一阶优化步 $x_{t+1}=R_{x_t}(+\eta\,v(x_t))$ 都是向量场
$v=\operatorname{grad}F_\sigma$ 的**一阶离散化**，局部截断误差 $O(\eta^2)$，曲率只进入二阶项。

---

## 5. 三种"等价"的精确定义与区分

把 ES 估计器写成"真梯度 + 零均值噪声"（**无偏基线**下）：

$$
\hat g_N(x)=\nabla F_\sigma(x)+\xi_N(x),\qquad \mathbb E[\xi_N]=0,\quad
\Sigma_N(x):=\operatorname{Cov}(\xi_N)=\frac{\Sigma_1(x)}{N}. \tag{13}
$$

其中 $\Sigma_1(x)=\operatorname{Cov}(\hat g_1(x))$ 为单样本协方差。

| 情形 | 噪声是否复用 | 参数是否移动 | 等价性 |
|---|---|---|---|
| (a) 梯度累积 | 否（各 chunk 新噪声） | **冻结** | **代数精确**（定理 2） |
| (b) 小批次多次 | **否**（每步新噪声） | **移动** | 一阶相容 + 二阶矩主阶一致 + 极限同一（定理 4–6） |
| (c) 多 epoch 复用旧噪声 | **是** | 移动 | **不等价** |

> **重要约定**：本文定理 4–6 针对情形 (b)，且**每步使用全新独立噪声**。(c) 复用旧方向不增加单步信息量，
> 不构成等价（见 §8）。

---

## 6. 主要定理与证明

### 6.1 定理 2：参数冻结时梯度累积**代数精确**等于大批次

**设定。** 参数冻结在 $x$，把 $N_L$ 个样本分成 $K$ 个不相交 chunk $\{C_k\}$（各 $N_s$ 个），中心化使用
**同一基线** $b$（独立基线，或跨所有 chunk 的全局均值 $\bar f$）。

> **定理 2。** 对同一参数点 $x$，
>
> $$
> \hat g_{N_L}(x;b)=\frac1K\sum_{k=1}^K\hat g^{(k)}_{N_s}(x;b)
> =\frac1K\sum_{k=1}^K\frac1{N_s}\sum_{i\in C_k}\frac{f_i-b}{\sigma}\varepsilon_i. \tag{14}
> $$

**证明。** 求和可交换（线性）：

$$
\frac1K\sum_k\frac1{N_s}\sum_{i\in C_k}\frac{f_i-b}{\sigma}\varepsilon_i
=\frac1{KN_s}\sum_{i=1}^{N_L}\frac{f_i-b}{\sigma}\varepsilon_i=\hat g_{N_L}(x;b).\qquad\blacksquare
$$

> **精确性条件**：必须全局线性归一、一次 optimizer step、无 per-chunk 局部 z-score、无 optimizer state。
> 若每 chunk 单独做 $\frac{f-\operatorname{mean}_k}{\sqrt{\operatorname{var}_k+10^{-5}}}$（局部 mean+std），
> 则 (14) 不再成立（局部 std 非线性破坏了线性性）。

### 6.2 定理 3：冻结点上两者渐近同分布（CLT）

> **定理 3。** 设 $\hat g_1$ 有有限二阶矩，样本独立，则当 $N_s,N_L\to\infty$，
>
> $$
> \hat g_{N_L}(x)\Rightarrow\mathcal N\!\Big(\nabla F_\sigma(x),\ \frac{\Sigma_1(x)}{N_L}\Big),
> $$
>
> 且 $\frac1K\sum_k\hat g^{(k)}_{N_s}(x)$ 收敛到**相同**正态分布。

**证明。** CLT 给出 $\hat g_{N_s}\Rightarrow\mathcal N(\nabla F_\sigma,\Sigma_1/N_s)$；$K$ 个 iid 正态之和仍正态，
协方差 $\frac1{K^2}\cdot K\cdot\frac{\Sigma_1}{N_s}=\frac{\Sigma_1}{KN_s}=\frac{\Sigma_1}{N_L}$。$\qquad\blacksquare$

> **注**：这是**渐近**结论，高斯近似误差 $O(1/\sqrt N)$；有限样本下仅近似。定理 2 已给出**逐点精确相等**，
> 比"同分布"更强。

### 6.3 定理 4：动态下累积噪声协方差**主阶精确相等**

> **定理 4。** 设每微步噪声 $\xi_{N_s}(x_{t,k})$ 构成鞅差序列（每步用全新独立噪声，且条件均值零），
> $\Sigma_1$ 沿路径 Lipschitz。则
>
> $$
> \operatorname{Cov}\!\Big(\eta\sum_k\xi_{N_s}(x_{t,k})\Big)
> =\eta'^2\frac{\Sigma_1(x_t)}{N_L}+O(\eta'^3)
> =\operatorname{Cov}\big(\eta'\xi_{N_L}(x_t)\big)+O(\eta'^3). \tag{15}
> $$

**证明。** 鞅差 ⇒ 交叉协方差为零（$\mathbb E[\xi_k\xi_{k-1}^\top]=0$，见 §6.5 引理 2），故

$$
\operatorname{Cov}\!\Big(\eta\sum_k\xi_{N_s}(x_{t,k})\Big)
=\eta^2\sum_k\frac{\Sigma_1(x_{t,k})}{N_s}
=\eta^2 K\frac{\Sigma_1(x_t)}{N_s}+O(\eta^2 K\,\eta')
=\eta'^2\frac{\Sigma_1(x_t)}{N_L}+O(\eta'^3).
$$

（末步用 $N_L=KN_s$，$\eta'=\eta K$，$\Sigma_1(x_{t,k})=\Sigma_1(x_t)+O(\eta')$。）$\qquad\blacksquare$

> 标题严格说应为"主阶精确相等、误差 $O(\eta'^3)$"，而非"精确相等"。

### 6.4 定理 5：漂移一阶相容，误差 $O(\eta'^2)$

> **定理 5。** 设梯度估计**无偏**（独立基线）且 $\nabla F_\sigma$ 为 $C^2$、$\|\nabla^2F_\sigma\|$ 有界，则
> 两者确定性漂移之差为
>
> $$
> \Big\|\eta'\nabla F_\sigma(x_t)-\eta\sum_{k=0}^{K-1}\nabla F_\sigma(x_{t,k})\Big\|
> \le C\,L\,\|\nabla F_\sigma(x_t)\|\,\eta'^2+O(\eta'^3). \tag{16}
> $$

**证明。** 记 $\delta_k=x_{t,k}-x_t$（$\delta_k=O(k\eta)$），Taylor：

$$
\nabla F_\sigma(x_{t,k})=\nabla F_\sigma(x_t)+\nabla^2F_\sigma(x_t)\delta_k+O(\|\delta_k\|^2).
$$

求和乘 $\eta$：

$$
\eta\sum_k\nabla F_\sigma(x_{t,k})
=\eta K\nabla F_\sigma(x_t)+\eta\nabla^2F_\sigma(x_t)\sum_k\delta_k+\eta\sum_kO(\|\delta_k\|^2).
$$

第二项 $\le L\cdot\eta\cdot O(\eta K^2)=O(L\eta'^2)$；第三项 $O(\eta^3K^3)=O(\eta'^3)$。$\qquad\blacksquare$

> **重要前提**：(16) 的 $O(\eta'^2)$ 建立在**无偏基线**上。若用同批均值基线，漂移实为
> $(1-\frac1N)\nabla F_\sigma$，则附加一项（推论 2）：

> **推论 2（局部中心化使漂移差达 $O(\eta')$）。** 同批均值基线下，小批次 $K$ 步与大批次单步的漂移差为
>
> $$
> \eta'\Big(\frac1{N_L}-\frac1{N_s}\Big)\nabla F_\sigma
> =-\eta'\,\frac{K-1}{KN_s}\nabla F_\sigma
> \xrightarrow{K\gg1}-\frac{\eta'}{N_s}\nabla F_\sigma. \tag{17}
> $$
>
> 对固定 $N_s$，量级为 $O(\eta')$，**严格大于** $O(\eta'^2)$。因此定理 5 必须显式假设无偏基线（或 $N_s$ 足够大、
> 或全局归一）。

### 6.5 引理 2（噪声的鞅差结构）

> **引理 2。** 无偏基线下，$\{\xi_{N_s}(x_{t,k})\}_k$ 是鞅差序列：$\mathbb E[\xi_k\mid\mathcal F_{k-1}]=0$，故
> 交叉协方差 $\operatorname{Cov}(\xi_k,\xi_{k-1})=0$。

**证明。** 每步噪声 $\varepsilon^{(k)}$ 独立于历史 $\mathcal F_{k-1}$；给定 $x_{t,k}$，$\xi_k=\hat g(x_{t,k})-\nabla F_\sigma(x_{t,k})$
仅依赖 $\varepsilon^{(k)}$ 且条件均值为零，故 $\mathbb E[\xi_k\mid\mathcal F_{k-1}]=0$，进而
$\mathbb E[\xi_k\xi_{k-1}^\top]=\mathbb E[\mathbb E[\xi_k\mid\mathcal F_{k-1}]\xi_{k-1}^\top]=0$。$\qquad\blacksquare$

### 6.6 定理 6：极限等价（重写极限设定）

**定义 3（极限过程）。** 固定 $K$ 与 $N_s$（故 $N_L=KN_s$ 固定），令微观学习率 $\eta\to0$，则 $\eta'=\eta K\to0$。
定义宏观时间插值 $X^\eta(t)=x_{\lfloor t/\eta'\rfloor}$。

> **定理 6（极限等价）。**
>
> **(a) ODE 极限（噪声消失）。** 上述极限下，大批次与小批次多次的插值过程都在紧时间区间上**一致收敛**到
> **同一梯度流（上升）**
>
> $$
> \dot X(t)=+\nabla F_\sigma(X(t)),\qquad X(0)=x_0, \tag{18}
> $$
>
> 收敛阶 $O(\eta')$。
>
> **(b) 扩散极限（SDE，纯理论标度）。** 在形式标度 $\eta'\to0$ 且 $\eta'/N_L=c$（$c$ 固定，即 $N_L$ 随 $\eta'$
> 同步缩小）下，两者**弱收敛到同一 Itô SDE**
>
> $$
> dX_t=+\nabla F_\sigma(X_t)\,dt+\sqrt{c\,\Sigma_1(X_t)}\,dW_t. \tag{19}
> $$

**证明（骨架）。**

**(a)** 每宏观步漂移误差 $O(\eta'^2)$（定理 5），累计 $T/\eta'$ 步后全局误差 $O(T\eta')$；噪声 $O(\eta'^2\Sigma_1/N_L)\to0$。
由 Kushner–Clark / Gronwall：一阶相容 ⇒ 一致收敛到 (18)。

**(b)** 单位宏观时间的扩散系数，两方案**逐阶相等**：

$$
D_L=\frac{\operatorname{Cov}(\eta'\xi_{N_L})}{\eta'}=\eta'\frac{\Sigma_1}{N_L},
\qquad
D_s=\frac{\eta'^2\Sigma_1/N_L+O(\eta'^3)}{\eta'}=\eta'\frac{\Sigma_1}{N_L}+O(\eta'^2).
$$

漂移一阶相容、扩散一致，Euler–Maruyama 弱收敛 ⇒ 同一 SDE；标度 $\eta'/N_L=c$ 下扩散系数 $=c\Sigma_1$ 不消失。$\qquad\blacksquare$

> **注**：(b) 是**纯理论存在性结果**（用于证明"同一 SDE"），$N_L\to0$ 不可作为实际训练策略（审查报告二 §三.5
> 正确指出）。

### 6.7 定理 7：误差阶与流形曲率

> **定理 7（局部 / 分布意义误差）。** 在紧集上，两者宏观步轨迹之差满足
>
> $$
> \big\|x_{t+K}^{\text{small}}-x_{t+1}^{\text{large}}\big\|
> \le C_1\eta'^2+C_2\eta'\sqrt{\frac{\|\Sigma_1\|}{N_L}}, \tag{20}
> $$
>
> 其中 $C_1$ 由流形 Hessian $\|\nabla^2F_\sigma\|$（Christoffel/曲率贡献）决定，$C_2$ 由噪声决定。**均值意义**下
> 随机项相消，仅剩 $O(\eta'^2)$。

> **注**：(20) 是**局部或分布意义**的刻画，不表示任意两条实际轨迹逐点接近——即使同分布，单条轨迹的随机实现
> 也可能差异明显。

---

## 7. 完整推导链条（串联）

$$
\underbrace{\hat g_N}_{\text{ES 估计器}}
\xrightarrow[\text{引理1（Stein）}]{\text{定理1}}
\underbrace{\nabla F_\sigma}_{\text{热核磨光 fitness 的梯度}}
\xrightarrow[\text{定义2（retraction）}]{\text{式(11)}}
\underbrace{\text{梯度流 }\dot x=+\operatorname{grad}F_\sigma}_{\text{流形最速上升}}
$$

两条离散化路径：

$$
\begin{aligned}
\text{大批次} &: x_{t+1}=R_{x_t}\big(+\eta'\hat g_{N_L}(x_t)\big),\\
\text{小批次多次} &: x_{t,K}=\underbrace{R_{\cdot}(+\eta\hat g_{N_s}(\cdot))\circ\cdots\circ R_{\cdot}(+\eta\hat g_{N_s}(\cdot))}_{K\text{ 次}}(x_t).
\end{aligned}
$$

由定理 2–7：

- **冻结**：二者**代数相等**（定理 2）、渐近同分布（定理 3）；
- **移动**：漂移差 $O(\eta'^2)$（定理 5，无偏前提下）、噪声协方差主阶精确相等（定理 4）；
- **极限**：弱收敛同一梯度流 (18) / 同一 SDE (19)（定理 6）；
- **误差阶**：$O(\eta'^2)$（均值），曲率进入 $C_1$（定理 7）。

---

## 8. 假设、局限与 SNN 现实的差距

1. **光滑性**：所有定理假设 $f\in C^2$、$\nabla^2F_\sigma$ Lipschitz、Stein 引理的增长条件。真实 SNN 的
   非凸 landscape、泊松编码噪声、稀疏硬奖励、离散输出可能破坏这些条件；此时 $\Sigma_1$ 的 Lipschitz 假设与
   定理 5 中 $L$（Hessian 范数）都可能失效或极大。
2. **无偏前提**：EggRoll 用同批均值 + `/std`，引入 $O(1/N)$ 偏差与非线性尺度，严格分析需作为近似处理。
3. **带状态优化器**：定理仅适用于无状态一阶光滑更新（vanilla ascent / retraction）；Adam/momentum/weight
   decay/clipping/warmup 下等价性不自动成立，需另行分析。
4. **多 epoch 复用旧噪声不等价**：复用旧方向不增加独立方向覆盖，也不按 $1/N$ 降低新鲜蒙特卡洛方差，可能导致
   对旧方向过拟合、探索不足、平台期提前。
5. **方差等效 ≠ 训练结果等效**：等效批次倍数只是梯度噪声（标量）指标，不能直接外推为准确率/fitness 收益；
   且"等效批次 $F\times$"是迹标量化，忽略了协方差矩阵的各向异性（回归式 ES 尤甚）。
6. **扩散极限是形式化标度**：定理 6(b) 的 $\eta'/N_L=c$ 意味着 $N_L\to0$，仅为证明"同一 SDE"的理论存在性。

---

## 附录 A：符号表

| 符号 | 含义 |
|---|---|
| $x\in\mathcal M$ | 参数 |
| $f$ | fitness（最大化） |
| $\varepsilon\sim\mathcal N(0,I_d)$ | 扰动方向 |
| $\sigma$ | 扰动尺度 |
| $N,N_s,N_L$ | 批量 / 小批量 / 大批量，$N_L=KN_s$ |
| $b$ | 基线（独立基线 or 同批均值 $\bar f$） |
| $\hat g_N$ | ES 梯度估计器 (1) |
| $F_\sigma$ | 高斯平滑 fitness，$F_\sigma=e^{\frac{\sigma^2}{2}\Delta}f$ |
| $g=\nabla F_\sigma$ | 真梯度 |
| $\Sigma_1$ | 单样本协方差 $\operatorname{Cov}(\hat g_1)$ |
| $\eta,\eta'$ | 微观 / 宏观学习率，$\eta'=\eta K$ |
| $G,\operatorname{grad},\Gamma$ | 度量 / 流形梯度 / 联络（曲率） |
| $R_x$ | retraction，$R_x(v)=x+v+O(\|v\|^2)$ |

## 附录 B：精确结果的数值复核

数值复核见 `pythonScript/verify_review_claims.py`（$1/N$ 偏差、Isserlis 协方差、对偶 2×、局部中心化漂移差四项，
均精确吻合）；等价性定理见 `tests/test_batch_equivalence.py`（7 项定理全过）。
