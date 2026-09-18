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
