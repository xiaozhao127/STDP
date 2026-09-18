# telemetry_preproc — 航天遥测数据降采样预处理流水线

面向航天遥测参数**未来趋势预测**任务的离线批处理预处理库:按参数类型路由
降采样方法(常值/缓变/含噪缓变/阶跃/速变/复合),带类型判别、质量评价闭环与
可复现决策落盘。设计与需求基线见 [docs/DESIGN.md](docs/DESIGN.md)。

## 快速开始

```bash
pip install -e ".[dev]"

# 合成数据端到端演示(6 类 × 3 条,含野值/丢帧注入)
python examples/demo_synthetic.py out/demo

# 命令行批处理真实文件(CSV/HDF5/Parquet)
telemetry-preproc run data/ -c configs/default.yaml -o out/batch

# 特征画像(阈值定标:分布直方图 + 分位数)
telemetry-preproc profile data/ -o profile_out
```

库内使用:

```python
from telemetry_preproc import load_config, run, write_artifacts
from telemetry_preproc.io import load_series, check_sampling_health

cfg = load_config("configs/default.yaml")
series, io_meta = load_series("param_001.csv", cfg)
io_meta["health"] = check_sampling_health(series.t, cfg["io"])
result = run(series, cfg, input_meta=io_meta)     # PipelineResult
write_artifacts(result, "out")                    # out/<param_id>/…
```

## 流水线阶段

```
① 载入与规整(排序/去重/量纲转换/上游已降采样体检)
② 时间轴处理(丢帧检测、按时间戳分桶,桶不跨 GAP)
③ 清洗(Hampel → 野值/真阶跃持续性检验 → 受控插值)
④ 类型判别(7 类特征 → 有序规则阈值 → TypeVerdict,贴阈值→UNCERTAIN)
⑤ 路由降采样(常值→首尾点;缓变→LTTB/SDT;含噪缓变→PAA+LTTB;
   阶跃→LTTB/DP 保沿;速变→抗混叠FIR+抽取;复合→趋势两段式;
   UNCERTAIN→SDT 误差有界收紧)
⑥ 评价与选点(代理指标、率失真扫描、Kneedle 拐点、硬约束回退;预留 E2E 接口)
```

降采样强度三模式(`downsample.spec.mode`):`fixed_cr`(定点数)/
`error_bounded`(MaxAE 上限,达标率 100% 保证)/ `auto`(默认,率失真扫描 +
拐点自动选工作点,硬约束越限回退保守模式)。

## 端到端预测感知压缩(v0.2,`telemetry_preproc/e2e/`)

逐参数独立选工作点的局限:参数之间不共享预算,噪声地板类参数按中间代理
指标守着 CR≈1.3-2.5×,吃掉大量存储,而对下游预测重要的参数未必分到足够
保真度。v0.2 把设计 §4.6 预留的 E2E 接口落地:**"下游预测精度衰减 ≤ ε"
作为全局约束,在总存储点数预算下求各参数压缩档位的最优分配** ——
常值/冗余参数压到极限,预算让给对预测贡献大的参数。

机制四件套:

1. **参数重要性(leave-one-out 敏感度)**:多参数联合 AR 岭回归代理
   (`JointArRidge`),对参数 j 做 Gram 矩阵分块剔除(不触碰原始数据、
   不重新扫描),系统预测退化量归一化即 w_j;
2. **逐参数"输出点数—预测衰减"曲线**:冻结的单参数自回归代理
   (`ArRidge`)在 原始/各压缩档重建 输入上滚动预测 holdout,NRMSE 差即
   δ_i(n)。模型只拟合一次,**任何压缩档位评估都无需重训**;δ 可 <0
   (压缩去噪反而提升预测,分配器会主动利用);
3. **全局预算分配**(`budget.py`):拉格朗日 λ 二分 + 跳步贪心补足
   (δ 曲线非凸/非单调,逐档贪心会卡死);另有纯贪心与 ε 反解最小预算;
4. **TimesFM 端到端验证**(`timesfm_eval.py`):`E2EEvaluator` 协议的正式
   实现 —— 同一 TimesFM 2.5(200M, PyTorch)分别在 原始/重建 序列上滚动
   多步预测 holdout,可经 `set_e2e_evaluator()` 注册进 pipeline 预留钩子,
   亦可批量评估参数组(全部窗口单次前向)。

公平性设计:所有对比在同一总存储点数下进行;TimesFM 的窗口/上下文/预测
长度/参考真值完全一致,唯一差异是输入历史来自哪种压缩分配。

运行(需 timesfm 环境):

```bash
PYTHONPATH=src python examples/run_e2e_budget.py --group xw --out out/e2e_budget
# 产物:report.json、summary.md、pareto_proxy.png、timesfm_per_param.png、
#       curves_sample.png(报告含等预算对比、预算—衰减 Pareto、ε 反解 B*)
```

### 实测结果(真实双星数据,TimesFM 2.5-200M 滚动 256 步预测验证)

组级 NRMSE 衰减(Δ = 重建输入预测 − 原始输入预测;**负值 = 压缩去噪使
预测更准**,全局分配可主动利用;下表"等预算"指与基线工作点相同的总存储):

| 参数组 | 基线工作点 @B_base | 比例缩放 @0.15B | 全局分配 @0.15B | 全局分配 @B_base |
|---|---|---|---|---|
| XW(17 参数) | −0.0196 | −0.1016 | **−0.1142** | **−0.1164** |
| TIANTA001(11 参数) | +0.0198 | +0.1620 | **−0.0653** | **−0.0804** |

- XW 组:等预测精度下存储降至基线的 **15%**(Δ 反而从 −0.020 → −0.114);
  最贵参数 TMZHDZT2073(基线 34033 点、CR 1.27)压到 **57 点**,预测精度
  反而大幅提升(δ=−0.62);
- TIANTA001 组:逐参数基线的工作点实际让预测**变差**(+0.020)—— 中间
  代理指标与预测任务脱节的直接证据;全局分配同预算下 −0.080;
- ε 反解:XW 组 B*=78553(基线 37%)、TIANTA001 组 B*=19151(基线 34%)
  即可达全预算分配的精度水平。

## 输出产物

```
out/<param_id>/
  ├── series.parquet        # 降采样序列(t/y/flags)
  ├── meta.json             # 决策记录(输入sha256、配置sha256、rule_version、
  │                         #   算法+参数+版本、verdict 特征快照、阶段耗时)
  ├── report.json           # 质量报告(压缩比/RMSE/MAE/MaxAE/极值率/趋势率/告警)
  └── ratedistortion.csv/png # auto 模式扫描结果与拐点标记
out/summary.csv|summary.md  # 批处理汇总
```

## 测试

```bash
python -m pytest tests/ -v
```

覆盖:各模块单测 + 端到端集成(6 类路由、error_bounded 达标率、硬约束回退
触发、阈值贴边 → UNCERTAIN 保守路径)。

## 实现说明(相对设计文档的工程决策)

- **代理任务与下游任务对齐(多步长直接预测)**:冻结 AR 代理最初做单步
  teacher-forced 预测,TimesFM 实测显示其在极端压缩档(0.3% 点数)系统性
  低估多步预测衰减(0.15 预算档分配输给比例缩放基线);改为多步长直接
  预测(horizons=[1,8,32,128,256],与 TimesFM horizon=256 对齐)后,
  全部预算档反超。教训:代理的"预测任务定义"必须与部署的下游任务同构,
  否则曲线在压缩谱两端失真。
- **曲线实测上移(部署模型 in-the-loop)**:代理曲线与 TimesFM 的去噪
  偏好仍不一致(中等预算档反扑);最终采用两级方案 —— 代理曲线只负责
  提供档位网格与退化参数识别,"输出点数—预测衰减"曲线由部署模型本身
  批量实测((参数,档位) 对扁平化一批前向,136 对仅数分钟),彻底消除
  代理-部署偏差。
- **Pareto 剪枝而非单调包络**:实测曲线的档位间噪声若用单调包络
  (running min)平滑,会把大档位的低 δ 记到小档位名下,成本与收益解耦
  (实测分配器退化到全最小档,v3 翻车);Pareto 剪枝直接删除被支配档位
  (存在点数更多且 δ 不劣者),保留的 (点数, δ) 对全部真实可达。
- **非凸曲线的跳步贪心**:δ 曲线在底部档位非单调(评估噪声/局部回弹),
  逐档贪心会卡在零增益步(实测只花掉 2.4% 预算);跳步贪心(当前档 →
  任意更高档的边际)解决。拉格朗日 λ 二分因 λ→档位映射存在悬崖,同样
  以跳步贪心收尾补足。

- **阶跃事件主动检测**:采样后的理想阶跃在 Hampel 窗内常表现为零 MAD +
  极少贴边点,仅靠"Hampel 连续异常段 ≥ m"无法可靠聚段;因此在清洗模块
  增加电平中值滤波 + 后窗口持续性的主动检测,其结果与设计规定的 Hampel 段
  持续性检验合并(step 事件均经持续性验证)。语义与设计 §4.3 一致。
- **Hampel MAD 快速近似**:两遍 `median_filter` 的标准近似;窗口零 MAD 时用
  全序列正 MAD 中位数做下限,纯常值序列不误报;数组两端与 GAP 段边界半窗内
  的窗不可信,不参与标记(nearest 填充会把中位数拉向边值,振动类成片误杀)。
- **降采样工作信号中和被剔除野值**(设计 §5 踩坑1 的落实):清洗只打标记不改
  数据,但 `DownsampleContext` 对被剔除点做段内线性中和供算法消费 —— 否则
  LTTB 会把野值钉成"重要点"直接保留进输出。评分仍按掩码排除这些点。
- **STEP 保沿强制点对**:跳变两侧相邻样本不同时保留时,分段线性重建把沿插成
  斜坡,误差可达半跳变幅度;STEP 路由在内层算法(LTTB/DP)输出上强制并入
  持续性检验通过的阶跃边沿点对(`*_steppreserve` 包装),保沿后重建精确。
- **抗混叠段边处理**:filtfilt 默认奇延拓对摆动中段起始一阶连续,残余段边瞬态
  区(≈3×ntaps)用原始值拼接(通带内 FIR≈原始,差异≈纹波);error_bounded
  下通带外能量无法保全时(decim=1 仍越限)恒等回退 —— 误差有界优先于压缩比。
- **hf_ratio 分母语义**:Welch 逐窗线性去趋势(窗内斜坡不泄漏成宽带高频),
  分母取全序列方差(缓变趋势周期常远大于窗长,逐窗去均值会把它藏掉);
  谱特征在最长无 GAP 弧段上估计(空洞相位跳变污染 PSD)。
- **复合判别残差显著性下限**(classify.composite_min_resid_std=5%):缓变类
  空洞边界中值滤波伪影残差 ~1-2%,真复合 ≥15%,低于下限不参与判别。
- **Kneedle**:率失真曲线取归一化后"距首末弦垂直偏差最大"的点(最大弦偏差/
  肘点法),对凸减与 S 型曲线均落在边际收益转折处;全档零误差的退化曲线
  (如保沿后的阶跃类)直接取最大压缩档。
- **误差有界二分的恒等可达性**:LTTB 在 target≥N 时恒等返回,PAA 在预缩放
  退化时跳过均值 —— 噪声/振动类信号少丢一个样本都可能越限,二分终点必须可达。

## 里程碑对应

M1 数据模型/io/合成生成器 → M2 timeline/cleaning/classify → M3 downsample/
router → M4 evaluate/率失真/报告 → M5 pipeline/CLI/demo/测试。
