"""Checked notation and first-appearance reference ordering for the paper."""
import re


METHOD = r'''## 3 方法

### 3.1 符号与总体框架

网络由共享特征提取、逐源匹配、可靠性估计、多视图融合和三级深度回归组成。Stage 1、Stage 2 和 Stage 3 分别在输入分辨率的 $1/8$、$1/4$ 和 $1/2$ 尺度运行。Stage 1 使用全局深度候选，B 根据前一级深度分布生成 Stage 2/3 的候选。每个阶段先对参考—源视图匹配体进行正则化，再由成对深度分布估计不确定性；C 在逐深度候选处融合逐源隐变量，融合体经第二次正则化产生阶段深度。A 在训练时提供几何遮挡监督，真值深度不作为推理输入。

| 符号 | 定义及形状（省略 batch 维） |
| --- | --- |
| $t\in\{1,2,3\},s\in\{1,\ldots,S\},k\in\{0,\ldots,N_t-1\}$ | 阶段、源视图和深度候选索引；$S$ 不含参考视图 |
| $I_0,I_s;K_i,R_i,\mathbf t_i$ | 参考/源图像，内参与世界到相机的外参 |
| $\phi_i^t\in\mathbb R^{C_t\times H_t\times W_t}$ | 阶段图像特征 |
| $\mathbf p,\widetilde{\mathbf p}$ | 参考像素二维及齐次坐标 |
| $d_k^t(\mathbf p)$ | 光轴深度候选，单位 mm |
| $F_s^t,V_s^t\in\mathbb R^{8\times N_t\times H_t\times W_t}$ | 组相关匹配体与逐源正则化隐变量 |
| $a_s^t,P_s^t\in\mathbb R^{N_t\times H_t\times W_t}$ | 成对候选得分及其深度维概率 |
| $E_s^t,u_s^t,z_s^t,q_s^t$ | 成对熵、预测对数不确定性尺度、可见性 logit 和概率；均为 $H_t\times W_t$ |
| $r_s^t,w_s^t\in\mathbb R^{N_t\times H_t\times W_t}$ | 候选相关融合残差和归一化源视图权重 |
| $P^t,\hat d^t,\sigma^t$ | 融合深度概率、期望深度和标准差 |
| $\mu^t,s^t,h^t,d_-^t,d_+^t$ | 前级均值/标准差对齐值、搜索半宽及区间两端；深度量单位均为 mm |
| $G_i,g_s,m_s,v_s,o_s$ | 真值深度、几何有效掩码、监督掩码、可见与遮挡标签 |
| $\tau_a,\tau_r,\kappa_t$ | 几何绝对容差（mm）、相对容差和半宽倍率（后两者无量纲） |
| $c_+,c_-,\gamma,\eta_t$ | Focal 类别权重、难度指数和残差幅度 |
| $\beta_t,\lambda_p,\lambda_u,\lambda_A,\lambda_C$ | 阶段、成对深度、不确定性、A、C 监督权重 |

所有内参与像素坐标使用对应特征尺度。设 $R_{s0}=R_sR_0^\top$、$\mathbf t_{s0}=\mathbf t_s-R_sR_0^\top\mathbf t_0$，参考像素和光轴深度对应的源相机坐标为：

$$
\mathbf X_s(\mathbf p,d)=R_{s0}(dK_0^{-1}\widetilde{\mathbf p})+\mathbf t_{s0},
\qquad \mathbf p_s(d)=\pi(K_s\mathbf X_s(\mathbf p,d)),
\quad \pi([x,y,z]^\top)=[x/z,y/z]^\top.
\tag{1}
$$

将源特征双线性重采样到参考视锥，与参考特征做八组组相关，构成逐源匹配体：

$$
F_s^t(\mathbf p,k)=\operatorname{GWC}_{8}\!\left(
\phi_0^t(\mathbf p),\operatorname{Sample}(\phi_s^t,\mathbf p_s(d_k^t(\mathbf p)))\right),
\quad V_s^t=\mathcal R_{\mathrm{pair}}^t(F_s^t).
\tag{2}
$$

成对得分 $a_s^t=\mathcal H_{\mathrm{depth}}^t(V_s^t)$ 经深度维 softmax 产生 $P_s^t$，成对深度为 $\hat d_s^t=\sum_kP_s^td_k^t$。概率熵 $E_s^t=-\sum_kP_s^t\log(P_s^t+\epsilon)$ 经共享二维编码器产生 $u_s^t$ 和 $z_s^t$。这里 $u_s^t$ 是可正可负的对数尺度，基础可靠性 logit 为 $-u_s^t$，对应正权重 $\exp(-u_s^t)$；它与融合概率的深度标准差 $\sigma^t$ 含义不同。该成对不确定性加权机制继承自 Vis-MVSNet [1]。

![图1 三级 Vis-MVSNet 详细总体架构](paper_complete_assets/framework.png)

**图1. 三级总体结构及单阶段内部数据流。** 每个阶段先构建并正则化逐源匹配体，回归成对深度和对数不确定性，再融合隐变量并回归阶段深度。蓝色虚线为 A 的训练监督；橙色为 B 的候选生成；青色为 C 的逐候选融合。B 作用于 Stage 2/3，C 在三个阶段使用。

![图2 三个改进模块的内部计算流程](paper_complete_assets/modules_abc_detailed.png)

**图2. A、B、C 的内部计算。** A 由真值几何生成可见性标签并监督熵编码分支；B 根据均值与标准差调整候选范围；C 将对数不确定性基础分数与候选相关残差相加，在源视图维度归一化。GT 只参与训练监督和实验诊断。

### 3.2 A：逐源视图遮挡感知监督

将参考真值 $G_0(\mathbf p)$ 代入式(1)，得到源投影深度 $Z_s(\mathbf p)=[\mathbf X_s]_3$ 和像素 $\mathbf p_s^\star$，采样源深度 $\widetilde G_s(\mathbf p)=G_s(\mathbf p_s^\star)$。深度一致性容差定义为：

$$
\tau_s(\mathbf p)=\max(\tau_a,\tau_r\widetilde G_s(\mathbf p)).
\tag{3}
$$

几何有效掩码 $g_s=1$ 要求参考深度有效、源投影在图像内、投影深度为正、采样源深度及其掩码有效。定义：

$$
v_s=g_s\mathbb1[|Z_s-\widetilde G_s|\leq\tau_s],\qquad
o_s=g_s\mathbb1[Z_s>\widetilde G_s+\tau_s],\qquad m_s=v_s+o_s.
\tag{4}
$$

位于源表面前方的不一致点、越界投影和缺失深度不计为遮挡负样本。标签在相应尺度下由几何生成；只有 $m_s=1$ 的位置进入监督。

在共享熵编码特征 $H_s^t=\mathcal E^t(E_s^t)$ 上预测逐源可见性：

$$
z_s^t=f_A^t(H_s^t),\qquad q_s^t=\operatorname{sigmoid}(z_s^t).
\tag{5}
$$

为兼顾类别比例与难样本，定义 $p_y=qv+(1-q)(1-v)$、$c(v)=c_+v+c_-(1-v)$，采用加权 Focal BCE [8]：

$$
\ell_{\mathrm{vis}}(q,v)=c(v)(1-p_y)^\gamma[-v\log q-(1-v)\log(1-q)].
\tag{6}
$$

对每个阶段与源视图，在有效监督集合计算可见比例 $f_+=\sum m_sv_s/\max(1,\sum m_s)$；类别权重为 $c_+=\min(10,0.5/\max(f_+,10^{-3}))$ 和 $c_-=\min(10,0.5/\max(1-f_+,10^{-3}))$。实际计算使用带 logits 的稳定 BCE。

$$
\mathcal L_A^t=\frac1S\sum_s
\frac{\sum_{\mathbf p}m_s^t\ell_{\mathrm{vis}}(q_s^t,v_s^t)}
{\max(\epsilon,\sum_{\mathbf p}m_s^tc(v_s^t))}.
\tag{7}
$$

无有效样本时该源损失为零。$q_s^t$ 用于训练监督和诊断，不直接乘入融合权重。A 同时限制成对深度监督的有效位置，并使被遮挡样本的不确定性损失只更新不确定性分支，见式(20)。

### 3.3 B：自适应深度搜索范围

对 $t>1$，将前级深度与标准差双线性对齐，得到 $\mu^t$、$s^t$；当前阶段名义间距为 $\delta_t$，固定半宽记为 $h_0^t=\lfloor N_t/2\rfloor\delta_t$。搜索半宽为：

$$
h^t(\mathbf p)=\operatorname{clip}(\kappa_ts^t(\mathbf p),h_{\min}^t,h_{\max}^t),
\quad h_{\min}^t=\rho_{\min}h_0^t,\quad h_{\max}^t=\rho_{\max}h_0^t.
\tag{8}
$$

$\kappa_t\geq0$、$0<\rho_{\min}\leq\rho_{\max}$ 是无量纲参数。半宽上下界分别防止候选塌缩和过度扩大。不确定性引导薄体积的研究基础见 UCS-Net [4]；本文将其用于大视差条件下的级联纠错，并与几何监督、候选相关融合共同分析。

前级深度是全局有效候选的加权均值，因此 $\mu^t\in[d_{\min},d_{\max}]$。区间与全局边界相交：

$$
d_-^t=\max(d_{\min},\mu^t-h^t),\qquad
d_+^t=\min(d_{\max},\mu^t+h^t).
\tag{9}
$$

对 $N_t\geq2$，候选深度为：

$$
d_k^t=d_-^t+\frac{k}{N_t-1}(d_+^t-d_-^t),\qquad k=0,\ldots,N_t-1.
\tag{10}
$$

实际采样间距为：

$$
\Delta^t(\mathbf p)=\frac{d_+^t(\mathbf p)-d_-^t(\mathbf p)}{N_t-1}.
\tag{11}
$$

同样的候选预算下，扩大范围也会增大间距。因此范围覆盖和采样精度必须联合评价；标准差反映当前候选分布的离散程度，不能直接当作真实误差。实验同时报告候选覆盖率、归一化范围宽度和最终误差。

### 3.4 C：深度假设感知源视图融合

候选深度坐标归一化为：

$$
\xi_k^t=2\frac{d_k^t-d_0^t}{\max(d_{N_t-1}^t-d_0^t,\epsilon_d)}-1.
\tag{12}
$$

其中 $\epsilon_d$ 与深度具有相同单位。将八通道隐变量、成对得分的 $\tanh$、成对概率和候选坐标连接为十一通道输入，预测残差：

$$
J_s^t=\operatorname{Concat}(V_s^t,\tanh a_s^t,P_s^t,\xi^t),\quad
\zeta_s^t=g_C^t(J_s^t),\quad r_s^t=\eta_t\tanh\zeta_s^t.
\tag{13}
$$

$g_C^t$ 依次包含 $3\times3\times3$ 卷积（11→8 通道）、四组 GroupNorm、ReLU 与 $1\times1\times1$ 卷积（8→1 通道）。输出层以零初始化，使训练开始时残差为零。$\eta_t\geq0$ 约束残差幅度。

候选相关分数 $\ell_{s,k}^t=-u_s^t+r_{s,k}^t$ 在源视图维度归一化：

$$
w_{s,k}^t(\mathbf p)=\frac{\exp(-u_s^t(\mathbf p)+r_{s,k}^t(\mathbf p))}
{\sum_{j=1}^{S}\exp(-u_j^t(\mathbf p)+r_{j,k}^t(\mathbf p))}.
\tag{14}
$$

融合对象为逐源正则化隐变量：

$$
V^t(\mathbf p,k)=\sum_{s=1}^{S}w_{s,k}^t(\mathbf p)V_s^t(\mathbf p,k).
\tag{15}
$$

权重归一化满足 $w_{s,k}^t\geq0$、$\sum_sw_{s,k}^t=1$。$r_s^t=0$ 时恢复对数不确定性的基础归一化权重。此式与对正指数权重求和后归一化数学等价；softmax 的归一化维度为源视图，不是深度候选。

融合体经第二次三维正则化产生得分 $a^t$，深度维 softmax 得到：

$$
P^t(\mathbf p,k)=\frac{\exp a^t(\mathbf p,k)}{\sum_{j=0}^{N_t-1}\exp a^t(\mathbf p,j)}.
\tag{16}
$$

预测深度和标准差分别为：

$$
\hat d^t(\mathbf p)=\sum_kP^t(\mathbf p,k)d_k^t(\mathbf p).
\tag{17}
$$

$$
\sigma^t(\mathbf p)=\sqrt{\max\left(\sum_kP^t(\mathbf p,k)(d_k^t(\mathbf p)-\hat d^t(\mathbf p))^2,\epsilon_\sigma^2\right)}.
\tag{18}
$$

这里 $\epsilon_\sigma$ 为深度尺度的数值稳定下界。$\hat d^t$ 和 $\sigma^t$ 传递到下一阶段的候选生成。

### 3.5 损失与训练流程

设 $\Omega_t$ 为阶段有效真值集合，$\delta_0$ 为数据输入的基础深度间距。定义归一化误差 $e^t=|\hat d^t-G_0^t|/\delta_0$ 与 $e_s^t=|\hat d_s^t-G_0^t|/\delta_0$。阶段融合深度损失为：

$$
\mathcal L_d^t=\frac{\sum_{\mathbf p\in\Omega_t}e^t(\mathbf p)}{\max(1,|\Omega_t|)}.
\tag{19}
$$

定义 $\operatorname{Mean}_{M}(f)=\sum Mf/\max(1,\sum M)$。A 启用时，成对深度掩码 $M_{p,s}=v_s$，不确定性掩码 $M_{u,s}=m_s$，且 $\bar e_s=v_se_s+(1-v_s)\operatorname{sg}(e_s)$；A 关闭时，两掩码均为阶段真值有效掩码，$\bar e_s=e_s$。$\operatorname{sg}$ 表示停止梯度。成对深度和不确定性项为：

$$
\mathcal L_p^t=\frac1S\sum_s\operatorname{Mean}_{M_{p,s}}(e_s^t),\qquad
\mathcal L_u^t=\frac1S\sum_s\operatorname{Mean}_{M_{u,s}}(\bar e_s^t\exp(-u_s^t)+u_s^t).
\tag{20}
$$

不确定性项采用归一化误差对应的 Laplace 负对数似然形式（忽略常数）。若对 C 增加候选可见性监督，则以最近真值候选 $k^*=\arg\min_k|d_k^t-G_0^t|$ 处的 $\operatorname{sigmoid}(\zeta_{s,k^*}^t)$ 代替式(7)中的 $q_s^t$，得到 $\mathcal L_C^t$。联合目标为：

$$
\mathcal L=\sum_{t=1}^{3}\beta_t\left(\mathcal L_d^t+
\lambda_p\mathcal L_p^t+\lambda_u\mathcal L_u^t+
\lambda_A\mathcal L_A^t+\lambda_C\mathcal L_C^t\right).
\tag{21}
$$

关闭 A 时 $\lambda_A=0$；不使用候选可见性监督时 $\lambda_C=0$。损失权重、阶段候选数量和半宽倍率属于训练配置，应随所用检查点记录。监督标签和推理输入保持分离。

**算法1：三级前向过程。**

1. 提取三尺度参考和源特征。
2. Stage 1 构建全局候选，Stage 2/3 由前级深度及标准差生成候选。
3. 对每个源视图按式(1)—(2)生成逐源匹配隐变量，回归成对概率、深度、熵、对数不确定性及可见性 logit。
4. C 启用时按式(12)—(15)逐候选融合，否则使用基础不确定性权重。
5. 按式(16)—(18)回归阶段深度和标准差，将前级结果双线性对齐到下一尺度；阶段间候选生成保留计算图。
6. 推理输出最终深度；训练时按式(3)—(7)构造遮挡监督并计算式(19)—(21)。

'''


def renumber_references(text):
    body, bibliography = text.split('## 参考文献', 1)
    # Numeric labels only; never replace equation tags or table values.
    matches = list(re.finditer(r'\[(?:文献)?(\d+)\]', body))
    order = list(dict.fromkeys(int(m.group(1)) for m in matches))
    entries = {int(m.group(1)): m.group(2).strip()
               for m in re.finditer(r'(?m)^\[(\d+)\]\s+(.*)$', bibliography)}
    if set(order) != set(entries):
        raise ValueError(f'Citation population differs: {set(order)^set(entries)}')
    mapping = {old: new for new, old in enumerate(order, 1)}
    body = re.sub(r'\[(?:文献)?(\d+)\]', lambda m: f'[{mapping[int(m.group(1))]}]', body)
    bibliography = '\n\n'.join(f'[{mapping[old]}] {entries[old]}' for old in order)
    return body + '## 参考文献\n\n' + bibliography + '\n', mapping


def revise_manuscript(text):
    a, b = text.index('## 3 方法'), text.index('## 4 实验设置')
    text = text[:a] + METHOD + text[b:]
    text = text.replace('S1–S3 width 表示各阶段平均候选区间宽度。',
                        'S1–S3 width 表示候选区间宽度除以数据基础深度间距后的均值，单位为基础间距倍数；它不是 mm。')
    text = text.replace('Ours (A+B+C)', 'Base+A+B+C')
    return renumber_references(text)[0]
