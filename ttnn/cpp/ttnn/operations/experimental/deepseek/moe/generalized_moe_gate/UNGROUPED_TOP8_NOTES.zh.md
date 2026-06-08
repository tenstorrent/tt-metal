# `generalized_moe_gate` 全局 top-8(ungrouped)开发记录

**目标。** 让 `ttnn.experimental.deepseek.moe.generalized_moe_gate` 返回 **256 个专家(8 组 × 32)
的真·全局 top-8**(按 bias 校正后的分数排序,返回归一化的非 bias 分数)—— 而**不是** DeepSeek 的
分组路由(top-2 求和 → top-4 组 → 128 里取 top-8)。必须是**单个融合的片上算子**(µs 级),不能用慢的
通用 `ttnn.topk`。

**状态:已完成并验证**(2026-06-04,WH B0),对照 flattened `torch.topk` golden、tie-robust 测试,
`seed × {sigmoid} × {batch 1,2}` 全部通过。Golden + 测试:
`models/demos/deepseek_v3/{tt/generalized_moe_gate/op.py, tests/test_generalized_moe_gate.py}`。

本次只改**设备 kernel**(WH B0)。BH 分支若要构建,需要同样的改动。

---

## 1. 最终架构

核心思路:已验证的"4 组合并"积木(`merge4_top8`)能从 4 个组正确产出一个 sorted-8。**跑两次再合并**:

```
topA = top8(组 0-3)          ← 排布组 0-3,合并
topB = top8(组 4-7)          ← 排布组 4-7,合并
全局 top-8 = 全排序(topA ∪ topB)   ← 16 个候选,再归一化
```

数学上精确:全局 top-8 ⊆ topA ∪ topB(任一组对 top-8 的贡献 ≤ 8)。

### 流水线(`GMG_UNGROUPED_TOP8` 下,见 `compute_kernel_api/generalized_moe_gate.h`)

```
sum_top2 → step0(转置:组 g → DEST 行 g) → [ungrouped 块] → step2(转置) → pack
```

ungrouped 块。SFPU 合并只能可靠寻址 **DEST 行 0-7**(见 §3),所以同一时刻只有一半能待在"合并槽"里。
FPU(`copy4rows`,即一次 MOVD2B→MOVB2D 的 4 行平拷)把暂时不用的另一半**暂存到行 8-15** —— SFPU 碰不到、
但 FPU 能:

| 步 | 操作 | 效果 |
|----|------|------|
| 1 | `copy4rows<4,8>` | 存组 4-7 源:行 4-7 → 行 8-11 |
| 2 | `step1_hi<d2b_dst=0>` + `merge4_top8<read=0, store={0,2}>` | topA = top8(组 0-3) → 列 {0,2} |
| 3 | `copy4rows<0,12>` | 停 topA:行 0-3 → 行 12-15 |
| 4 | `copy4rows<8,4>` | 复原组 4-7:行 8-11 → 行 4-7 |
| 5 | `step1_hi<d2b_dst=4>` + `merge4_top8<read=0, store={4,6}>` | topB = top8(组 4-7) → 列 {4,6} |
| 6 | `copy4rows<12,0>` | 复原 topA:行 12-15 → 行 0-3 |
| 7 | `finalize_ungrouped` | 16 个候选做完整 bitonic 排序 → 全局 top-8 + 归一化 |

每个 `copy4rows` 用**不相交的 SrcB 暂存窗口**(16/20/24/28)—— 见 §4 Bug 0。
topA 落在行-列 {0,2}(行 0-3),topB 落在 {4,6}(行 4-7):行不相交,所以第 6 步能把两者都放回而不冲突。

### `finalize_ungrouped`(最难搞对的一段)

topA{0,2} + topB{4,6} = 16 个候选值。**不要**用两-run 的 bitonic *合并*(`ph3_st4_to_1` 或
`merge4_runs_raw`)—— 它们期望特定的 run lane 朝向,而 `merge4_top8` 的输出不是那个朝向,会得到
"部分合并"或"重复"的错误结果(见 §4 Bug 3)。正确做法:把 16 个值当**未排序的 16-向量**装载
(topA→LREG0/1,topB→LREG2/3;idx 用 LO16、score 用 HI16 放 LREG4-7,并开 `SFPCONFIG(0x4,0xF,1)`
索引追踪),跑**完整**的 `bitonic_top8_ph0_to_ph3<idir=false>`。完整排序是**朝向无关的** —— 绕开了
所有 lane 布局的坑。然后 `store8_even_cols_split` + 标准归一化尾。

---

## 2. 用到的 DEST / SFPU 布局事实

- DEST 区域(单 face 算子):`scores=0, indices=64, bias=128, interm=192`(单位 = DEST 行;64 行 = 1 tile;
  `dst_tile_offset=64`)。一个存好的 run:idx 在 `indices`(LO16)、score 在 `scores`(HI16)、bias 在
  `bias` —— 即 `merge4_top8` 的存储约定(`store_lo`/`store_hi`)。
- `step0` 之后:组 `g` 在 DEST **行 g**(行 0-3 = 组 0-3,行 4-7 = 组 4-7)。`step1_hi<d2b_dst=4>` 读行
  4-7 → 产出 top8(组 4-7);单-half 测试已验证精确。
- 一个 `merge4_top8` 输出的 sorted-8 占**两列** {store_lo, store_hi}(LREG0→lo,LREG1→hi),各 4 个值。

---

## 3. 决定整个设计的硬约束

**SFPU 的 `SFPLOAD`/`SFPSTORE` 只能可靠寻址 tile 0-3 的行 0-7**(offset 0,2,4,6)。offset ≥ 8(以及
≥ 256)会 wrap / 读到陈旧值。所以 SFPU 合并被限制在行 0-7。但 FPU **可以**寻址行 8-15(`MOVB2D` 的
`b2d_base` 重定位、`MOVD2B`、`TRNSPSRCB` 都已验证)。正是这个不对称催生了 FPU `copy4rows` 的"暂存腾挪":
行 8-15 是另一半唯一的避难所,而且只有 FPU 能把数据放进去 / 取回来。

`TRNSPSRCB` = 对 SrcB 行 16-31 做原地 16×16 转置。`step1_hi` 把 4 行源装进 SrcB 16-19 和 28-31,转置,
再用 `MOVB2D` 读偶数转置行 → 得到可排序的 run。

---

## 4. 遇到的 bug 及解法

全程有效的方法论:**用二分诊断隔离;每次跑前硬复位(`tt-smi -glx_reset`)保证确定性;只信
`output[:,0,:8]` / 真实测试的索引**(scratch DEST 是 reset 相关的)。`GMG_DIAG_TOPA` / `GMG_DIAG_TOPB`
隔离宏 + `GMG_DUMP_AFTER_*` dump 点(用 `test_dump_sum_top2_layout` 读)驱动了每一步定位。

### Bug 0 —— 连续 `copy4rows` 的 SrcB 串扰(疑似,预防性修复)
连续的 `copy4rows` 共用 SrcB 行 16-19;后一条的 `MOVB2D` 可能读到前一条的 SrcB。**解法:** 每条
`copy4rows` 用不相交的 SrcB 窗口(16/20/24/28)。(后来发现这不是真正的故障 —— 换窗口结果逐位不变 ——
但这个改动正确且便宜,予以保留。)

### Bug 1 —— 假的"topB 坏了"(是诊断在骗人):缺 `SETRWC`
`GMG_DIAG_TOPB` 报告 topB 混进了 group-1(topA)的专家。二分(完全跳过 topA → topB 干净;再逐步加回
topA 各步)定位到 **`restore-topA`(`copy4rows<12,0>`)恰好在 SFPU 读出之前跑**。根因:
**`_gmg_copy_topk_run` / `_gmg_normalize_run` 缺了 `merge4_top8` / `finalize` 开头都有的
`TTI_SETRWC(...,SET_D)`**。FPU MOP(`copy4rows`,其最后一条 `MOVB2D` 用 `ADDR_MOD_2` = Dst base +64)
会把 **Dst RWC 计数器推进 +64/tile**;紧随其后的 SFPU `SFPLOAD` 就读到 `offset + 残留RWC` → 错位的行。
**规则:任何紧跟 FPU MOP 之后读 DEST 的 SFPU 算子,必须先复位 Dst RWC。** topB(和 topA)在 DEST 里
一直是对的 —— 只是缺 SETRWC 的*读出*被偏移了。解法:给这两个 helper 补 `SETRWC`。

### Bug 2 —— dump 证实 `log1.txt`(没 reset)是陈旧数据
没 reset 的一次跑出全 0 / 陈旧输出;只有 `tt-smi -glx_reset` 之后那次可信。强化了硬复位纪律。

### Bug 3 —— 两-run 合并的朝向问题(耗时最久)
在 topA、topB 单独都已正确(`GMG_DIAG_TOPA`/`TOPB` 在所有数据含 batch 2 上验证)的前提下,`finalize` 仍在
边界处错选约 1 个专家。依次试过:
- 单个 `ph3_st4_to_1`,topB hi→LREG2 / lo→LREG3:**topB 高端被丢**。
- 交换 topB lo/hi:**"每个 run 各取前 4"**(没有交叉比较)—— 更接近但仍错。
- 把 topA→{0,4}、topB→{2,6} 重排后调 `_gmg_merge4_runs_raw`:**重复**(`[71,71,114,114,…]`)。

根因:`merge4_top8` 输出的 run 的 lane 朝向,与两-run *合并*原语(`ph3_st4_to_1`、`merge4_runs_raw`)
期望的不一致(它们要的是 step1-转置格式的 run)。**最终有效解法:别再合并两个已排序 run —— 把 16 个候选
全部装载后做完整排序**(`bitonic_top8_ph0_to_ph3`),它不做任何朝向假设。见 §1 finalize。

一个有用的排错点:这些失败**不是平局** —— 通过重算真实 bias rank 验证(被丢的专家是 rank-5,比 cutoff 高
0.03 = 3-4 个 bf16 步,不是等值平局)。**别没核实 rank 差就把锅甩给"平局"** —— 要看该量级下与 bf16 步长
的对比。

---

## 5. 保留的诊断(供接下来的泛化用)

默认全 OFF。在 `device/kernels/generalized_moe_gate_kernel.cpp` 里切换:
- `GMG_DUMP_AFTER_SUM_TOP2` / `GMG_DUMP_AFTER_STEP0` / `GMG_DUMP_AFTER_STEP1` —— 在该阶段停下,把
  `bias`(→output)+ `indices`(→output_indices)按 16×16 face pack 出来;用
  `test_generalized_moe_gate.py::test_dump_sum_top2_layout` 读(rigged 输入:idx=arange(256),
  bias[g,j]=j+g)。
- `GMG_DIAG_TOPA` / `GMG_DIAG_TOPB` —— 在 ungrouped 路径里只输出 topA(或 topB);在 `op.py` 里配一个
  top8(组 0-3)(或组 4-7)的 golden,把某一半从 finalize 里隔离出来。helper `_gmg_copy_topk_run` /
  `_gmg_normalize_run` 已带 Bug-1 的 `SETRWC` 修复。

---

## 6. 泛化注意(512 专家 / softmax / top-n = 6,10)

- **512 专家**(16 组 × 32):打破现有布局假设(组 g 在行 g;行 0-7 可被 SFPU 寻址;单 face)。16 组超出单
  face —— 很可能要多 tile,且不止两次 `merge4`(例如 4 个 quarter 合并 + 更宽的 finalize)。
- **top-n = 6/10**:bitonic 网络(`merge4_top8`、`ph0_to_ph3`、finalize 排序)是按 k=8 设计的;改 k 要动
  bitonic 的阶段数。
- **softmax**(相对当前 sigmoid):归一化尾目前是普通的求和归一化;softmax 需要 exp + max 稳定化。

---

## 关键引用

- ungrouped 编排:`device/kernel_includes/tt_metal/include/compute_kernel_api/generalized_moe_gate.h`
- SFPU 核心:`device/kernel_includes/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_generalized_moe_gate_topk_single_face.h`
  (`store8_even_cols_split`、`ph3_st4_to_1`、`ph0_to_ph3`、`reverse_sort_order`、`sum_top2`、
  `merge4_top8`、`merge4_runs_raw`、`top8`、`copy_topk_run`、`normalize_run`、`finalize_ungrouped`)
- 转置 / copy4rows(FPU):`device/kernel_includes/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_generalized_moe_gate_transpose_dest_single_face.h`
  (`step0`、`step1`、`step1_hi`、`copy4rows`、`step2`)
- dump pack:`device/unified_kernels/generalized_moe_gate.hpp`
- golden + 测试:`models/demos/deepseek_v3/{tt/generalized_moe_gate/op.py, tests/test_generalized_moe_gate.py}`

---

> 英文版见 [`UNGROUPED_TOP8_NOTES.md`](UNGROUPED_TOP8_NOTES.md)(内容一致)。
