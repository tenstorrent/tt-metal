# Generalized MoE Gate —— 可配置 top-n(k ≤ 8):设计、历程、踩坑与解决

状态:**✅ 支持 top-4 / top-6 / top-8**,256 单 op 路径和 512 combine 路径都支持,平台 **Wormhole B0**。
`test_generalized_moe_gate` 和 `test_generalized_moe_gate_512_global` 都参数化 `topk ∈ {8, 6, 4}`。通用的
per-lane mask 顺带也覆盖 k=5/7(未测)。**k<4 和 k>8 暂不支持。**(英文版:`TOPN_NOTES.md`。)

---

## 1. 目标与约束

- 把路由专家数 `k` 做成参数,而不是写死的 8。DeepSeek 用 k=8;我们还要 k = 4 / 6(后续 10)。
- **往下(k < 8)比往上(k > 8)容易得多**:流水线前段已经产出排好序的**全局 top-8**,所以 k ≤ 8 时我们
  **完整保留** 32→top-8 的全部机制,只改**最后合并/归一化那一段尾巴**。往上(k=10)需要的候选比当前 merge
  保留的更多,推后做。
- **单 op**(softmax 后续要融进来)。目标平台 **Wormhole B0**,`fp32_dest_acc_en = false`(16-bit DEST),
  `dst_full_sync_en = true`。

## 2. 思路 —— 归一化之前把 rank ≥ k 的清零

`merge16_core` + `store8` 之后,全局 **top-8 已按降序** 落进 DEST。归一化会把每个保留的 score 除以保留 score 之和,
再乘 `scaling_factor`。所以 top-n 就是:

> **在归一化之前,把每个 rank ≥ k 的 score(和 idx)清 0。** 分母于是自动变成 top-n 之和,被丢弃的槽位输出 `0`。
> k=8 → 空操作。

finalize 之前的一切都不动。输出仍是 rank `r` 在第 `r` 列;ranks `k..7` 输出 `(0, 0)`。

## 3. 可用配方(`_generalized_moe_gate_finalize_ungrouped<…, topk>`)

排好序的 8 个落在 **scores/indices 区的 SFPU/TTI 偏移 `{0, 4}`**:offset 0 = ranks 0-3,offset 4 = ranks 4-7
(行内 lane 布局见 §4)。三种情况:

```
topk == 8 :  空操作(完整 top-8)。

topk <= 4 :  整行丢掉 offset-4(ranks 4-7):
               TTI_SFPSTORE(LCONST_0, scores_offset + 4)
               TTI_SFPSTORE(LCONST_0, indices_offset + 4)

4 < topk < 8 (k=5/6/7) :  对 offset-4 行做 per-lane mask。rank (4+j) 在 tileid 16*j
                          (ranks 4,5,6,7 -> tileid 0,16,32,48)。丢 ranks k..7 = tileid >= 16*(topk-4)
                          的 lane;留 ranks 4..k-1。
   constexpr int drop_thr = 16 * (topk - 4);                 // k=5->16, k=6->32, k=7->48
   constexpr int sc4 = (scores_offset  + 4) / 2;             // dst_reg 索引(见 §4):scores+4  -> 2
   constexpr int ix4 = (indices_offset + 4) / 2;             //                       indices+4 -> 34
   TTI_SETRWC(..., SET_D);                                   // 复位 Dst RWC,让 dst_reg base 与 TTI base 对齐
   vFloat sc = dst_reg[sc4]; v_if(vConstTileId >= drop_thr){ sc = 0.0f; } v_endif; dst_reg[sc4] = sc;
   vFloat ix = dst_reg[ix4]; v_if(vConstTileId >= drop_thr){ ix = 0.0f; } v_endif; dst_reg[ix4] = ix;
```

512 **combine** 路径要同样修:`combine_finalize` 调 `finalize_ungrouped` 时原先写死默认 k=8。把 `topk` 串进
`combine_finalize<is_32bit, topk>`,512 也就可配了。

## 4. DEST / SFPU 布局事实(WH B0,已验证)

- `store8` 后,降序 top-8 占 **scores/indices 偏移 `{0, 4}`**(TTI 地址):**offset 0 = ranks 0-3,offset 4
  = ranks 4-7**;输出把 rank `r` 打包到第 `r` 列。
- **一个 offset 行内的 4 个 rank 不是相邻的 —— 而是每隔 8 个 lane 分布一个**("even_cols" 存储 → lanes
  0, 8, 16, 24)。由于 `vConstTileId = 2 * lane`,rank `(4+j)` 落在 **tileid `16*j`**:ranks 4,5,6,7 →
  tileid 0, 16, 32, 48。**这个步长 16 是 mask 阈值最关键(也最反直觉)的事实。**
- **sfpi `dst_reg[ix]` 寻址的是 TTI 地址 `ix * SFP_DESTREG_STRIDE`(= `ix*2`)**,默认 mod-0(`SrcB`)读写。
  所以 TTI 地址 `A` 的字段是 `dst_reg[A/2]`:`scores+4`(地址 4)= `dst_reg[2]`,`indices+4`(地址 68)
  = `dst_reg[34]`。mod-0 读和 normalize 对 scores 的 mod-0 `TTI_SFPLOAD` 一致,且是**原样 bit 透传,16-bit
  id 无损往返**(如 337 = 0x0151 完好)。
- `vConstTileId` = 每 lane `[0, 2, 4, …, 62]`(tile-id 常量寄存器,sfpi 第 15 号)。sfpi 通过 `sfpreadlreg`
  物化它;裸 `TTI_SFPMOV`/`TTI_SFPIADD` 直接读 15 号 **不会**给出 per-lane 值。

## 5. 踩坑与解决(按踩到的顺序)

| # | 现象 | 根因 → 解决 |
|---|------|-------------|
| 1 | 裸 TTI per-lane mask 把 offset-4 **整行**清零(top-n 退化成 top-4) | `TTI_SFPMOV(LTILEID→LREG)` 再 `TTI_SFPIADD(…CC_GTE0)`,乃至 `SFPIADD` 直接以 `lreg_c = LTILEID` 为源,都**读不出 per-lane 的 tileid** —— CC 变成全使能,于是 `SFPMOV LCONST_0` 把每个 lane 都清了。(`SFPMOV` 是认 CC 的 —— 由 `ckernel_sfpu_dropout.h` 证实;bug 在 tileid 读取。)→ **用 `sfpi v_if(vConstTileId >= thr)`**,它经 `sfpreadlreg` 正确物化常量寄存器。 |
| 2 | sfpi mask **完全没效果** —— 仍输出完整 top-8 | `dst_reg[ix]` 寻址 TTI 地址 **`ix*2`**(`SFP_DESTREG_STRIDE = 2`)。`dst_reg[scores_offset+4] = dst_reg[4]` 写到了 **地址 8**,不是 `scores+4`(地址 4),mask 写进了死区,`scores+4` 没动。→ **`dst_reg[(scores_offset+4)/2] = dst_reg[2]`**。 |
| 3 | mask **多清**了 rank(k=6 跑成 top-5,而且阈值 4/8/16 都给 top-5) | lane 布局模型错了。我先以为 4 个 rank 在相邻 lane(tileid 步长 2),又改成 4、8。其实是**每隔 8 个 lane → tileid 步长 16**(ranks 4-7 在 tileid 0,16,32,48)。步长 16 时,任何 `drop_thr ∈ {4,8,16}` 都一样地丢掉 ranks 5,6,7(都 ≥ 各自阈值),**看起来像结果卡住**。→ **`drop_thr = 16*(topk-4)`**(k=6 → 32)。 |
| 4 | (假线索)"不同阈值结果相同 ⇒ kernel 缓存陈旧" | 清了 `~/.cache/tt-metal-cache` **毫无变化** —— 缓存是好的,改头文件**确实会**重编。结果相同是真的(坑 3):步长 16 让 ≤16 的阈值表现一致。教训:别赖缓存,用 dump 把布局钉死。 |
| 5 | 最终怎么钉死步长 | **诊断 dump:** 在 mask 分支里把 `scores+4` 整行覆写成 `int32_to_float(vConstTileId)`(不做 mask),读输出 cols 4-7。它们恰好回来是比例 **0 : 1 : 2 : 3** → 等差(均匀步长);结合"阈值 ≤16 会丢 rank5" → 绝对步长 = 16。一次 dump 顶 ~4 次盲猜阈值。 |
| 6 | 被丢弃的 **idx** 没清零(score 清了) | **带谓词的** sfpi store `v_if(...){ dst_reg[ix4] = vUInt(0); }` **没落地**,**`vUInt`** 的读-改-写也是**空操作**。→ 把 idx 行按 **`vFloat`**(默认 mod-0 `SrcB` 原样 bit 透传,和 score 一样)读写:清掉被丢弃 lane,且保留的 id 逐 bit 不变。 |

## 6. 关键文件与函数

- **`.../tt_llk/.../ckernel_sfpu_generalized_moe_gate_topk_single_face.h`** —— `_generalized_moe_gate_finalize_ungrouped`
  加了 `template <…, uint32_t topk = 8>`,以及 `store8` 之后、normalize 尾巴之前的三分支"清 rank ≥ k"块(§3)。
- **`.../compute_kernel_api/generalized_moe_gate.h`** —— `generalized_moe_gate<…, topk>`(gate 模板)和
  `generalized_moe_gate_combine_finalize<is_32bit, topk>` 都把 `topk` 串给 `finalize_ungrouped`。
- **`device/unified_kernels/generalized_moe_gate.hpp`** —— 256 路径传 `CTArgs::topk`;combine 路径传
  `combine_finalize<false, CTArgs::topk>`。
- **`topk`(named compile-time arg)端到端串接:** `op.py` / nanobind(`topk=8`)→ `generalized_moe_gate`
  op 入口 → `device_operation::invoke` → `operation_attributes_t.topk` → `program_descriptor_builder`
  (`{"moe_gate_topk", attrs.topk}`)→ `ComputeCTArgs::topk` → `generalized_moe_gate_kernel.cpp`
  (`get_named_compile_time_arg_val("moe_gate_topk")`)。`hash_moe_gate_program_structure` 把
  `named_compile_time_args` 也 hash 进去,所以每个 `k` 得到**自己独立的编译程序**(不会缓存撞车)。

## 7. 测试

`models/demos/deepseek_v3/tests/test_generalized_moe_gate.py`:
- `test_generalized_moe_gate` —— 256/128 路径,参数化 `topk ∈ {8, 6, 4}` × sigmoid × seed × batch。
- `test_generalized_moe_gate_512_global` —— 512 combine,参数化 `topk ∈ {8, 6, 4}`。
- `golden(…, topk)` 用 `torch.topk(bias_flat, topk)` + 对 top-k 归一化;两个测试都把输出切到
  `[:, 0, :topk]`(ranks 0..k-1;被丢弃的 rank 现在 score 和 idx 都是 `0`)。

## 8. 待办

1. **Top-10(k > 8):** 难的方向。流水线目前只保留 top-**8**(`merge16_core` → `store8`),rank 7 之后没东西可
   "解除遮罩"。需要让 merge 保留 ≥10 个候选(更宽的 store + top-16 排序,或第二趟 merge),再对 10 个归一化。
2. **k < 4**(k = 1/2/3):offset-0 行(ranks 0-3)也需要 lane mask —— 同样的 `drop_thr` 思路,套在
   `scores/indices_offset + 0` 上,用那一行的 rank→tileid 映射。
3. **Softmax / sqrt-softplus** 归一化变体(见 `SOFTMAX_NOTES.md`)。
