# 航天遥测数据降采样预处理流水线 — 设计文档

> 版本:v1.0(2026-09-02)
> 用途:本文档是完整的设计与需求基线,新开发会话应仅凭本文档即可开工,无需补充上下文。
> 背景:面向航天遥测参数的**未来趋势预测**任务,需要对不同类型的遥测参数采用不同的降采样(数据缩减)策略,而不是全类型统一处理。核心思想:**按参数类型路由预处理方法**。

---

## 1. 已确认的需求决策(基线,不再讨论)

| 决策点 | 结论 |
|---|---|
| 类型判别方式 | **无任何标注样本**,采用"信号特征 + 规则阈值"判别器;输出类型 + 置信度 + 特征快照 |
| 处理模式 | **离线批处理**(可自由使用 LTTB/DP 等批式全局算法,无因果性约束) |
| 职责范围 | **完整预处理链 + 评价闭环**:载入 → 时间规整 → 清洗 → 类型判别 → 路由降采样 → 质量评价与工作点选择 |
| 工程形态 | **Python 库**,文件输入(CSV / HDF5 / Parquet),单参数序列独立处理 |
| 降采样强度控制 | **双模式**:固定压缩比 / 误差有界;默认模式为率失真曲线扫描 + 拐点自动选工作点 |
| 评价基准 | 代理指标闭环(压缩比 / RMSE / MaxAE / 极值保留率 / 趋势方向一致率),**预留端到端预测模型接口**(当前无基线模型) |
| 多参数 | 单参数独立处理,架构上预留"多参数时间对齐"扩展位(本期不实现) |

非目标(本期明确不做):在线/流式处理、多参数对齐、端到端模型训练、有损压缩(SZ/ZFP 类)。

---

## 2. 总体架构

```
输入文件 (CSV/HDF5/Parquet, 单参数序列)
   │
 ① 载入与规整 ── 时间戳解析排序、重复时间戳合并、码值→工程量纲转换(可选)、
   │              上游已降采样体检(采样均匀性检查)
 ② 时间轴处理 ── 丢帧/空洞检测与标记、按时间戳(非点索引)分桶
   │
 ③ 清洗 ───────── 野值剔除(Hampel)、野值 vs 真阶跃持续性检验、
   │              缺失段标记(禁止静默插值)
 ④ 类型判别器 ── 特征提取 → 规则阈值(经特征画像定标)
   │              → TypeVerdict{type, confidence, features, rule_version}
   │              低置信度 → 保守路径
 ⑤ 路由降采样 ── 缓变→SDT/LTTB;含噪缓变→PAA降噪+LTTB;阶跃→LTTB/DP(保沿);
   │              速变→抗混叠低通+抽取;复合→两段式;常值→首尾点
 ⑥ 评价与选点 ── 代理指标报告、率失真曲线、拐点检测(Kneedle)自动选工作点、
   │              硬约束越限自动回退保守模式;[预留] E2EEvaluator 接口
   ▼
输出: 降采样序列 + 元数据(决策记录) + 质量报告(JSON/Markdown/图表)
```

设计原则:
- 每一步的**决策与参数全部落盘**(可复现、可回溯);
- 判别器输出只依赖统一接口,规则引擎将来可平滑替换为弱监督分类器;
- 清洗与降采样的**顺序不可颠倒**:野值必须在降采样前剔除,否则 LTTB 会把野值钉为"重要点"、SDT 容差带被拉飞。

---

## 3. 数据模型(核心类)

```python
@dataclass
class TelemetrySeries:
    param_id: str                # 参数标识
    t: np.ndarray                # float64, 秒, 严格递增
    y: np.ndarray                # float64, 工程量纲
    flags: np.ndarray            # 每点位掩码: OK/OUTLIER_REMOVED/GAP/INTERPOLATED/STEP_EDGE

@dataclass
class TypeVerdict:
    ptype: str                   # CONSTANT/SLOW/NOISY_SLOW/STEP/FAST/COMPOSITE/UNCERTAIN
    confidence: float            # [0,1]
    features: dict[str, float]   # 特征快照(落盘)
    hit_rule: str                # 命中的规则 id(含 rule_version)

@dataclass
class DownsampleSpec:
    mode: str                    # "fixed_cr" | "error_bounded" | "auto" (默认 auto)
    target_points: int | None    # fixed_cr 模式
    max_ae_limit: float | None   # error_bounded 模式(绝对值或 %量程)
    cr_grid: list[int]           # auto 模式扫描档位, 默认 [2,4,8,16,32,64,128,256]

@dataclass
class QualityReport:
    compression_ratio: float     # N_out / N_in
    rmse: float; mae: float; max_ae: float
    extrema_retention: float     # 极值保留率 [0,1]
    trend_consistency: float     # 趋势方向一致率 [0,1]
    passed: bool                 # 是否满足硬约束
    details: dict

@dataclass
class PipelineResult:
    series_down: TelemetrySeries # 降采样结果(分段线性可重建)
    verdict: TypeVerdict
    report: QualityReport
    meta: dict                   # 输入sha256、config sha256、rule_version、算法+参数+版本、各阶段时间戳
```

---

## 4. 模块设计

### 4.1 载入与规整(`io/`)

- 支持 CSV / HDF5 / Parquet;时间列、数值列名可配置;统一输出 `TelemetrySeries`。
- 排序、重复时间戳合并(取均值,可配)。
- 码值→工程量纲转换:可选(线性 `y = a*raw + b`,系数来自配置),本期多数输入已是工程量纲。
- **上游已降采样体检**(关键防线):估计采样率 `f̂ = 1/median(diff(t))`;检查时间间隔方差异常与配置中的原始采样率字段。若发现输入疑似已被隔点抽取(间隔均匀但远大于声明的原始采样率),在 meta 中标记 `pre_decimated_suspect: true` 并告警——因为上游混叠会使速变参数被误判为缓变,整条路由从源头出错。

### 4.2 时间轴处理(`timeline/`)

- 丢帧/空洞检测:`gap = diff(t) > gap_factor × median(diff(t))`,gap_factor 默认 3。空洞段打 GAP 标记。
- **分桶按时间等宽**(桶宽 = 总时长/桶数),严禁按点索引切分——遥测数据存在丢帧与变采样率。
- 空桶输出 NaN 并计数,聚合时忽略;**桶不得跨越 GAP**。

### 4.3 清洗(`cleaning/`)

**Hampel 野值检测**:滑窗 W(默认 2×round(0.5s×f̂)+1),判据 `|y - median| > k × 1.4826 × MAD`,k 默认 4(可配,偏保守以免误杀真实瞬态)。

**野值 vs 真阶跃持续性检验**(清洗模块最关键逻辑,判错两个方向都致命):
1. Hampel 标记的异常点中,若出现 **≥ m 个连续点**(m 默认 max(2, round(0.5s×f̂)))构成连续异常段,触发检验;
2. 检验段后窗口(默认 2s)内的值是否保持在"段内中位数 ± ρ×MAD"(ρ 默认 3):
   - 保持 → 判为**真阶跃**:恢复数据,打 STEP_EDGE 标记,并把 step 事件反馈给类型判别器;
   - 回到原电平 → 维持**野值**判定,剔除,打 OUTLIER_REMOVED 标记。
3. 被剔除点不做静默插值;如配置开启插值,必须打 INTERPOLATED 标记并计数进报告。

**常值/近常值段**:不算异常,交给分类器(CONSTANT 类)。

### 4.4 类型判别器(`classify/`)

**特征清单**(全部落盘为特征快照):

| 特征 | 定义 | 用途 |
|---|---|---|
| `hf_ratio` | Welch PSD 中 cutoff(默认 f̂/16)以上能量 / 总能量 | 速变判别 |
| `step_density` | 持续性检验通过的跳变数 / 时长(次/分钟) | 阶跃判别 |
| `constant_ratio` | \|diff(y)\| < ε·range 的点占比(ε 默认 1e-6) | 常值判别 |
| `rel_derivative` | median(\|diff(y)\|) / (P95−P05) | 变化强度 |
| `detrend_var_ratio` | var(y − 线性拟合) / var(y) | 趋势性 |
| `slope_flip_rate` | diff 符号变号频率 | 噪声强度 |
| `spectral_flatness` | 谱平坦度(Wiener 熵) | 谐波 vs 噪声 |

**决策规则(有序,命中即出)**:

1. `constant_ratio > 0.95` 且 range≈0 → **CONSTANT**(输出首尾点即可,天然极限压缩)
2. `step_density ≥ θ_s` 且 `hf_ratio` 低 → **STEP**(阶跃/开关/模式字类)
3. `hf_ratio > θ_f` → **FAST**(振动/噪声等速变参数)
4. `slope_flip_rate > θ_n` 且 `hf_ratio` 中等 → **NOISY_SLOW**(含噪缓变)
5. 复合检测:先鲁棒趋势提取(中值滤波或 LOWESS),**残差**的 `hf_ratio > θ_f` → **COMPOSITE**(缓变趋势叠加振动)
6. 其余 → **SLOW**(典型缓变趋势:温度、压力、电压、姿态角)
7. 任一步命中时特征值与阈值差 < margin → **UNCERTAIN**,走保守路径

**置信度**:`conf = clip(|feature − θ| / (θ·κ), 0, 1)`,κ 默认 0.25(实现可调);多条件规则取最小。

**阈值定标流程(强制,不许拍脑袋)**:
1. 先对全部参数批量跑**特征画像**(每个特征输出分布直方图/分位数,存 PNG/CSV);
2. 遥测参数族群在 hf_ratio、step_density 等特征上常见天然双峰,阈值取**分布谷底**;
3. 无天然分界的特征才人工定值,并在配置注释中记录依据;
4. 阈值全部外置 YAML、版本化(`rule_version`),判别记录必须带版本号。

**保守路径(UNCERTAIN / 判别失败)**:SDT 误差有界模式 + MaxAE 上限收紧(默认 0.5% 量程),宁可压缩比低也不冒丢失极值/跳变的险。

**升级路径**:人工复核 verdict 后的 corrections 回流落盘,即未来弱监督分类器的免费标注集;路由只依赖 `TypeVerdict` 接口,不依赖规则本身。

### 4.5 降采样算法与路由(`downsample/`)

| 类型 | 方法 | 说明 |
|---|---|---|
| CONSTANT | 首尾点 | — |
| SLOW | **SDT**(要分段线性特征)或 **LTTB**(要保留原始点值),配置选择,默认 LTTB | 保趋势/保极值 |
| NOISY_SLOW | **PAA 桶均值降噪 → LTTB** | 均值等效低通,抑制噪声后再保形状 |
| STEP | **LTTB 或 DP**(阈值取跳变幅度的比例) | **保沿**,严禁桶均值(会把跳变沿抹成斜坡) |
| FAST | **抗混叠低通 + 整数抽取** | FIR(Kaiser/窗函数),截止 ≤ 0.45×(目标率/2);`scipy.signal` 实现;**禁止直接等距抽点** |
| COMPOSITE | 两段式:鲁棒趋势提取(SDT/LTTB)→ 残差按 FAST 处理 | 或至少标记进人工复核队列 |
| UNCERTAIN | SDT 误差有界,MaxAE 收紧 | 保守路径 |

实现要求:
- 每个算法统一接口 `downsample(t, y, spec) -> (t_out, y_out, meta)`;
- SDT 需实现容差 E 参数(误差有界模式的核心);
- LTTB 分桶按时间戳(复用 timeline 模块),支持非均匀采样;
- 所有算法版本号写入输出 meta。

**双模式控制**:
- `fixed_cr`:给定 target_points;
- `error_bounded`:趋势类用 SDT,tolerance ≈ 上限/2(SDT 重建误差 ≤ tolerance,留余量);LTTB 无误差保证 → 对 target_points 二分搜索直到 MaxAE 达标;FAST 类对抽取率二分;
- `auto`(默认):扫 `cr_grid` 档位,每档算代理指标 → 率失真曲线 → **Kneedle/最大曲率**找拐点 → 硬约束检查(MaxAE ≤ 上限、极值保留率 ≥ 下限、趋势一致率 ≥ 下限),越限回退更低保真档。

### 4.6 评价闭环(`evaluate/`)

**代理指标定义**(重建 = 降采样点分段线性插值回原时间戳得 ŷ):

- `RMSE / MAE / MaxAE`:max|ŷ − y| 必报,趋势预测任务平均误差小但削峰不可接受;
- `compression_ratio = N_out / N_in`;
- **极值保留率**:`scipy.signal.find_peaks(prominence=ρ_p·range)` 取原序列极值集 E;某极值 e 命中 ⇔ 半桶宽内存在保留点且 |y_kept − y_e| ≤ δ_e·range;retention = 命中数/|E|;
- **趋势方向一致率**:以保留点切区间,区间内原序列净变化与输出序列同号的比例。

**报告产物**(每参数):`report.json` + 汇总 `summary.csv/Markdown` + 率失真曲线 PNG;越限项显式告警字段。

**端到端接口(预留,本期只定义不实现)**:

```python
class E2EEvaluator(Protocol):
    def evaluate(self, original: TelemetrySeries,
                 reconstructed: TelemetrySeries) -> dict[str, float]: ...
    # 未来实现:同一预测模型分别在 original / reconstructed 上滚动预测 holdout,
    # 返回 MSE/MAE 衰减;接入后自动并入 QualityReport
```

### 4.7 元数据与可复现(贯穿)

每条 `PipelineResult.meta` 必含:输入文件 sha256、配置 sha256、`rule_version`、各算法名+参数+版本、判别 verdict(特征值+命中规则)、六阶段耗时与时间戳。

**原始数据不删除**:降采样产物与原始数据分级存储,保证可回退重算(算法/阈值升级后历史数据可重跑对比)。

**输出目录布局**:

```
out/<param_id>/
  ├── series.parquet        # 降采样序列
  ├── meta.json             # 决策记录
  ├── report.json           # 质量报告
  └── ratedistortion.csv/png # auto 模式的扫描结果与拐点
```

---

## 5. 关键注意事项清单(踩坑记录,实现时逐条对照)

1. **清洗 → 降采样顺序不可反**(LTTB 保野值 / SDT 容差带被拉飞);
2. **时间轴不假设均匀**:分桶按时间戳;桶不跨 GAP;多弧段拼接场景同理;
3. **类型不止缓变/速变**:阶跃类(开关/指令响应/模式字)是缓变采样率但含硬跳变,桶均值会抹沿;常值段 SDT 极优而 LTTB 退化为均匀抽取;
4. **判别器必须有兜底**:UNCERTAIN → 保守路径(误差有界+保极值),禁止默认静默选某方法;
5. **上游已降采样体检**:防混叠假象导致速变被系统性误判为缓变(4.1);
6. **复合类型显式处理**:缓变叠振动不可静默归入缓变(4.4 规则 5);
7. **野值/真阶跃用持续性检验区分**:把真状态切换当野值剔掉,预测模型永远学不到切换事件;
8. **阈值经特征画像定标 + 外置 YAML + 版本化**:否则调一次阈值,历史判别全部不可比;
9. **决策全落盘**:方法/参数/版本/特征快照随数据走,换算法后可回溯;
10. **评价至少四维**:压缩比、RMSE+MaxAE、极值保留率、趋势方向一致率(+ 预留端到端);只报压缩比或只报 RMSE 站不住。

---

## 6. 项目结构

```
telemetry_preproc/
├── pyproject.toml              # 依赖: numpy pandas scipy h5py pyarrow pyyaml matplotlib
├── configs/
│   └── default.yaml            # 全部阈值/参数外置(含 rule_version)
├── src/telemetry_preproc/
│   ├── io/                     # loader(csv/hdf5/parquet)、体检
│   ├── timeline/               # 去重、丢帧检测、时间分桶
│   ├── cleaning/               # hampel、野值/阶跃持续性检验、GAP 标记
│   ├── classify/               # features / rules / profile(画像) / verdict
│   ├── downsample/             # base, lttb, sdt, dp, paa, minmax, antialias, router
│   ├── evaluate/               # metrics, ratedistortion(含Kneedle), report, e2e接口
│   ├── pipeline.py             # 编排: run(series, config) -> PipelineResult
│   └── cli.py                  # 命令行批处理入口(可选)
├── tests/                      # 每模块单测 + 端到端集成测试
├── examples/
│   └── demo_synthetic.py       # 合成数据演示
└── docs/DESIGN.md              # 本文档
```

`default.yaml` 骨架(节选,所有第 4 节出现的参数都必须出现在此文件并带注释):

```yaml
rule_version: "rules-v1"
io:        {time_col: t, value_col: y, raw_convert: null}
timeline:  {gap_factor: 3.0}
cleaning:  {hampel_window_s: 0.5, hampel_k: 4.0, persist_window_s: 2.0, persist_m: null, persist_rho: 3.0, interpolate: false}
classify:  {hf_cutoff_div: 16, theta_constant: 0.95, theta_step: 0.5, theta_fast: 0.5,
            theta_noise: 0.3, conf_kappa: 0.25, conf_margin: 0.1}   # 阈值须画像定标后修订
downsample:{slow_method: lttb, spec: {mode: auto, cr_grid: [2,4,8,16,32,64,128,256], max_ae_limit_pct: 1.0}}
evaluate:  {peak_prominence_pct: 2.0, extrema_tol_pct: 2.0,
            hard_limits: {max_ae_pct: 1.0, extrema_retention_min: 0.9, trend_consistency_min: 0.9}}
```

---

## 7. 开发里程碑

| 阶段 | 内容 | 出口标准 |
|---|---|---|
| M1 | 数据模型 + io + 合成数据生成器 | 6 类合成序列(常值/缓变/含噪缓变/阶跃/速变/复合)+ 注入野值/丢帧,可载入 |
| M2 | timeline + cleaning + classify(含特征画像工具) | 画像输出各特征分布;6 类全判对(复合允许进复核队列);野值/阶跃区分正确 |
| M3 | downsample 各算法 + router | SDT/LTTB/PAA/抗混叠各有单测;各类型路由正确;双模式可用 |
| M4 | evaluate + 率失真 + Kneedle + report | 指标公式与第 4.6 节一致;auto 模式拐点合理;越限回退可触发 |
| M5 | pipeline 集成 + demo + 文档 | 端到端 demo 通过第 8 节全部验收项 |

## 8. 验收标准(demo)

1. 合成数据集 6 类各 ≥3 条,含注入的野值与丢帧,全部走完流水线;
2. 类型路由全部正确(复合类型至少被标记);
3. 每条输出 series + meta + report 齐全,率失真曲线生成且拐点选中;
4. error_bounded 模式 MaxAE 达标率 100%;
5. 硬约束回退路径可被测试用例触发(构造越限场景);
6. 把任一阈值改错 → UNCERTAIN 保守路径被触发(规则鲁棒性测试)。

---

## 9. 算法与指标背景(供评审答辩参考)

- **LTTB**(Largest-Triangle-Three-Buckets):分桶选最大三角形面积点,保形状/极值,Grafana、Uber M3、TimescaleDB 均内置,适合保原始点值;
- **SDT**(旋转门/摆动门):容差带内一段直线代替多点,输出分段线性、天然在线可扩展,工业时序库(Apache IoTDB、OSIsoft PI)标配;
- **DP / VW**:经典折线简化,保形状但 DP 对噪点敏感;
- **PAA**:分桶均值,等效低通,降噪优先;
- 抗混叠降采样 = FIR 低通 + 整数抽取,速变参数硬性要求;
- 评价指标现状:**无单一公认指标**,工业界事实标准是"压缩比 × 重建误差(RMSE/MaxAE)"率失真评价;趋势/形状维度补极值保留率、DTW/SED 等;预测任务的最诚实评价是端到端精度衰减(本设计预留接口,即此逻辑)。
