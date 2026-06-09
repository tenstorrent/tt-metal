# 输出 softmax(对选中的 top-k)for `generalized_moe_gate` —— 设计与实现

状态:**✅ 完成**,256 单 op 路径和 512 combine 路径都支持,k = 4/6/8。由新开关 `output_softmax` 控制
(默认 false → 维持线性归一化,deepseek 路不受影响)。测试参数化 `output_softmax ∈ {False, True}` ×
`topk ∈ {8, 6, 4}`。(英文:`OUTPUT_SOFTMAX_NOTES.md`。这里是**输出** softmax(对保留的 k 个);**打分**
softmax(对*全部* expert、在开头)是另一个仍待办的任务 —— 见 `SOFTMAX_NOTES.md`。)

## 算什么

- `output_softmax = false`(默认):输出权重 = `s_i / Σ_sel(s_j) * scale`(对选中 score 的线性重归一化,
  即原行为)。
- `output_softmax = true`:输出权重 = `exp(s_i) / Σ_sel(exp(s_j)) * scale` —— **对选中的 top-k 做 softmax**
  (Mixtral 风格)。选择逻辑不变(仍按 `sigmoid(x)+bias` 排序);只有最后的归一化不同。**仍乘 scaling_factor**。

## 关键技巧 —— softmax = exp + 现有的线性归一化

`linear_normalize(exp(s)) = exp(s_i) / Σ exp(s_j) = softmax(s)`。所以**不需要新的归约**:只要对选中的 score
做 `exp`,再跑**现有的** 求和 → 倒数 → 乘 尾部。整个实现就是在 `_generalized_moe_gate_finalize_ungrouped`
里插一个 `exp`。

## 实现(`_generalized_moe_gate_finalize_ungrouped<…, topk, output_softmax>`)

```
merge16_core();  store8();              // 排好序的全局 top-8 落在 scores/indices {0,4}
if constexpr (output_softmax) {         // <-- 唯一新增的步骤
    TTI_SETRWC(... SET_D);              // 复位 Dst RWC,让 dst_reg base 与 TTI/归一化的 base 对齐
    dst_reg[(scores_offset+0)/2] = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(dst_reg[(scores_offset+0)/2]);  // rank 0-3
    dst_reg[(scores_offset+4)/2] = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(dst_reg[(scores_offset+4)/2]);  // rank 4-7
}
// top-n mask(清 rank >= topk)—— 不变
// 归一化尾部(Σ scores{0,4} → 倒数 → 乘)—— 不变;现在求的是 Σexp(s) → softmax
```

### 为什么 exp 必须在 top-n mask 之前(关键顺序)

流程是 **exp(全 8 个) → mask 清 rank ≥ k → 归一化**。若先 mask,被丢弃的 rank 会是 `0`,而 `exp(0) = 1` ——
k 个假的 1 会污染 `Σexp`,且被丢弃的槽位会输出非零 softmax 权重。先 exp 再 mask,意味着 mask 清掉的是**已经
exp 过**的值,于是被丢弃的 rank 对 `Σexp` 贡献 0、输出 0。这就是 `if constexpr (output_softmax)` 块放在
`if constexpr (topk …)` mask 块**上方**的原因。已对 k=4(exp 后整行清 offset-4)和 k=6(exp 后按 lane 清)
验证 —— 都测过。

### 数值 —— 不需要减 max

score 是 `[0,1]` 的 sigmoid 值(gate `sigmoid` 开,或输入已 sigmoid 过),所以 `exp ∈ [1,e]` —— 不溢出。
softmax 平移不变,跳过常规的 `exp(x − max)` 得到完全相同的结果,还省一次 reduce。用
`_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>`(21-bit,约 3 FP32 ULP)。

### 稀疏 lane 不是问题(曾担心)

排好序的 8 个在每个 `{0,4}` 行只占 4 个 lane(0,8,16,24;见 `TOPN_NOTES.zh.md`);其余 28 个 lane 是排序残留。
我们对**整个** 32-lane 行做 exp,残留也被 exp 了。这曾是个顾虑(`exp(残留)` 会不会进 `Σexp`?)—— **已验证无害**:
归一化的 `SFPTRANSP` + reduce 只求那 8 个有效 lane 的和,残留被排除(k=8 softmax 与 torch golden 在容差内吻合)。

### dst_reg 寻址

`dst_reg[k]` ↔ TTI 地址 `k*SFP_DESTREG_STRIDE`(= `k*2`),默认 mod-0(SrcB),与归一化对 score 的 mod-0
`TTI_SFPLOAD` 一致。所以 `scores+0`(TTI 地址 0)= `dst_reg[0]`,`scores+4`(地址 4)= `dst_reg[2]`。先复位
Dst RWC,让 sfpi base 与 TTI base 对齐(和 top-n mask 同一个坑)。

## Plumbing(`output_softmax`,一个 named compile-time arg)

`op.py` / nanobind(`output_softmax=False`)→ `generalized_moe_gate` op 入口 → `device_operation::invoke`
→ `operation_attributes_t.output_softmax` → `program_descriptor_builder`(`{"moe_gate_softmax", …}`)→
`ComputeCTArgs::output_softmax` → `generalized_moe_gate_kernel.cpp`
(`get_named_compile_time_arg_val("moe_gate_softmax")`)→ gate 模板 / `combine_finalize<is_32bit, topk,
output_softmax>` → `finalize_ungrouped`。`hash_moe_gate_program_structure` 把 `named_compile_time_args`
也 hash 进去,所以每个 (topk, output_softmax) 组合都有自己独立的编译程序。

关键文件:`ckernel_sfpu_generalized_moe_gate_topk_single_face.h`(exp 块 + `#include "ckernel_sfpu_exp.h"`)、
`compute_kernel_api/generalized_moe_gate.h`(gate 模板 + `combine_finalize`)、
`unified_kernels/generalized_moe_gate.hpp`(`CTArgs::output_softmax` 传给两条路),以及常规的
op/device/descriptor/nanobind/op.py 一套。

## Golden 与测试

`op.py` golden:`weights = exp(topk_scores) if output_softmax else topk_scores`,再 `weights / Σweights
* scaling`。`test_generalized_moe_gate`(256)和 `test_generalized_moe_gate_512_global`(512)都参数化
`output_softmax`;256 测试的自洽检查也分支到 softmax 形式。

## 不是这个

对**全部** expert、在开头做的打分 softmax(需要全局 `Σexp`)—— 待办,见 `SOFTMAX_NOTES.md`。
