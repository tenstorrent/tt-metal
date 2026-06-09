# Generalized MoE Gate — 512-expert combine(A2):设计、开发历程、踩坑与解法

状态:**✅ 已完成。** `generalized_moe_gate` 在 **Wormhole B0** 上、用**一个 op** 对 **512 个 expert** 算出真正的
**全局 top-8**(k=8)。`test_generalized_moe_gate_512_global` 全参数通过;256 路径无回归。
(英文版:`COMBINE_512_NOTES.md`。)

---

## 1. 目标与约束

- 把融合的单 op gate 从 256 推广到 **256 / 384 / 512** 个 expert,仍是**一个 op**,算真正的**全局 top-8**(k=8)。
  256 保持快速单 op(~2.48 µs)。Kimi = 384,Qwen = 512。
- **必须是单 op** —— softmax 之后要融合进来,所以不能拆成两个 op。
- 目标架构:**Wormhole B0**。`fp32_dest_acc_en = false`(16-bit DEST);`dst_full_sync_en = true`。

## 2. 架构 —— 每个 block 产出一个 "run" + combine

- **输入布局(slice)**:每个 256-block → 它自己 32×32 tile 的 face 0;logits/bias 按 `num_blocks` 个 tile/core
  分片。`num_blocks = ceil(N/256)`(512 → 2)。
- **每个 block(`produce_run`)**:跑已验证的 256 ungrouped pipeline,到 **可再合并的 top-8 RUN** 为止
  (`merge16_to_run`,跳过 normalize/step2)。一个 "run" = 8 个 expert 的 `(bias, idx, score)`:
  - `bias` = 排序键(sigmoid_score + bias 项),
  - `idx`  = 全局 expert id(uint16),
  - `score`= sigmoid 值(输出权重)。
- **Combine**:让两个 block 的 run **同时驻留**在 DEST 里,block1 在 SFPU 偏移 `{0,2}`、block0 在 `{4,6}`,
  然后跑已验证的 `combine_finalize`(`merge16_core` 在 `{0,2,4,6}` 读 16 个候选,全 bitonic 排序 → 全局 top-8,
  normalize,step2 → 输出)。

**难点**在于让两个 run 同时驻留在 `merge16_core` 能读的 SFPU "math" 布局里 —— block0 的 run 要在 block1 被产出
期间存活,并落到 `{4,6}`。最终可行的答案是 **merge-only acquire**(见 §3)。

## 3. 可行的配方

```
num_blocks == 2(combine 路径):

  # --- 两个 block 都 stash 到 L1(各走已验证的往返) ---
  process_block_to_run<0>()    # block0 -> L1 run CB 第 0 页
  process_block_to_run<1>()    # block1 -> L1 run CB 第 1 页
    # process_block_to_run<b>:
    #   copy_tile(input_indices, b, 1)                 # 每 block 的全局 indices(tile b = arange + b*256)
    #   produce_run<...,0,2,idx_offset=0>              # run 在 math {0,2}
    #   relocate_run<0,2,0,4>                          # {0,2} -> {0,4}(step2 按 finalize 的 {0,4} 布局设计)
    #   step2_only<false>                              # math -> standard(现在转置 3 个 tile:score+idx+BIAS)
    #   逐字段:pack_untilize_dest(tile_dst_rt_offset = 0/1/2)  # standard DEST -> row-major L1

  # --- 在 merge acquire 之前 tilize 全部字段(tilize 自管 DEST) ---
  hw_startup(run_scores_cb, cb_tilize);  tilize run_scores ×num_blocks -> cb_tilize p0,p1   (bf16)
  (复用)                                 tilize run_bias   ×num_blocks -> cb_tilize p2,p3   (bf16)
  hw_startup(run_idx_cb, cb_tilize_idx); tilize run_idx    ×num_blocks -> cb_tilize_idx p0,p1 (uint16)

  # --- merge-only acquire:里面没有 produce_run ---
  tile_regs_acquire()
  for (run, dst) in [(block1, {0,2}), (block0, {4,6})]:
     for field in [score(HI16), idx(LO16), bias(mode0)]:
        reconfig_data_format_srca(cb_tilize / cb_tilize_idx)
        transpose_wh_init_short(...)
        transpose_wh_tile(cb_tilize[page], 0, 3)                 # standard tiled -> interm(DEST tile 3),math {0,4}
        place_field_from_interm<field, dst_lo, dst_hi, src=0,4>()# SFPU 行/列选择性拷贝 interm{0,4} -> home{dst}
  combine_init()
  UNPACK(set_srcb_dummy_valid())                                 # 在所有 transpose 之后、step2 之前
  combine_finalize()                                             # merge16 {0,2}+{4,6} -> top8 + normalize + step2
  tile_regs_commit(); pack tile0->scores_out, tile1->idx_out
```

**为什么用 merge-only acquire**:`produce_run` 会留下 SFPU/SrcB/addrmod 状态,**污染同一个 acquire 里的
`transpose_wh`**。先把两个 block 都 stash,再在一个**没有 produce_run** 的新 acquire 里做 restore+merge,
还原就在干净状态下跑(和已验证通过的 256 stash 隔离同样的状态)。

## 4. DEST 布局事实(WH B0,已验证)

- 区域:`scores @ 偏移 0`、`indices @ 64`、`bias @ 128`、`interm @ 192`(`dst_tile_offset = 64`)。
  `copy_tile`/`pack_tile`/`transpose_wh_tile` 的 tile 下标 k ↔ SFPU 偏移 k*64(tile 0/1/2/3)。
- 一个 run 占 **2 个 SFPU 偏移对** `{store_lo, store_hi}`(各 4 个候选)。`merge16_core` 读 `{0,2,4,6}`
  (两个 run:`{0,2}` 和 `{4,6}`)。
- **`merge16_core` 的 `{0,2,4,6}` 是 face 的"行"**(每个 SFPLOAD 读一行的 lane),不是列。
- **拼接字段编码**:一个候选是 `idx(LO16) | score(HI16)` 拼在一个 32-bit SFPU LREG 里;**分开存**为
  `score → scores 区(HI16)`、`idx → indices 区(LO16)`、`bias → bias 区(mode 0 / 全 16-bit)`。
  `merge16_core` 重新读 `idx(LO16)` + `score(HI16)` 拼回来;它**按 `bias` 排序**,`idx|score` 随交换被 index-track。
- `fp32_dest_acc_en = false`(16-bit dest)。`dst_full_sync_en = true`。
- **bf16 ↔ raw 16-bit**:DEST 的数据类型由 **CB 的编译期格式 metadata**(`datatype_to_dataformat_converter`)
  决定,不是 page size。`UInt16` 走整数路径;`RawUInt16` 走 float16 pack 路径(corruption 陷阱 —— 别用它存 id)。

## 5. 踩坑与解法(按踩到的顺序)

| # | 症状 | 根因 → 解法 |
|---|------|-------------|
| 1 | `tilize_block` 立刻 hang | MATH↔PACK 的 DST 信号量没初始化 → **第一次 tilize 前调 `compute_kernel_hw_startup(icb, ocb)`**。 |
| 2 | L1 stash dump 全 0 | PACK 只能读 **standard** tile 布局,而 run 在 SFPU **"math"/转置** 布局 → pack 读到空格子。在 `pack_untilize` 前插 **`step2_only`(math→standard)**(还原时再 `transpose_wh` standard→math)。 |
| 3 | scores/idx/bias 全 pack 成 **scores** | `pack_untilize_dest` 选 DEST tile 用的是**运行时 `tile_dst_rt_offset`(最后一个参数)**,不是第 3 个位置参数(那是 `block_c_index`)。用 `pack_untilize_dest<1,1>(cb, 1, 0, 16, 4, 0/1/2)`。 |
| 4 | idx(uint16)回来是垃圾 | bf16 的 scores 路径把**运行时 unpack 格式留成了 bf16**,而 WH 上 `tilize_uninit` 不完全恢复,所以 idx 的 `tilize_block` 把 raw uint16 当 bf16 解。→ 给 idx 的 tilize 一个**自己的 `compute_kernel_hw_startup(run_idx_cb, cb_tilize_idx)`**(UInt16 CB)。CB 编译期格式一直是对的;坏的是**运行时**格式。**bf16 不能当 raw-bit 载体**(denormal flush —— id 0-255 = 0x00xx 是 subnormal)。 |
| 5 | 还原的 run 是 **`[a,b,a,b]` 2-周期重复** | `step2` 是为 FINALIZE 布局设计的(`store8_even_cols` 在偏移 `{0,4}`);作用在 `produce_run` 的 `{0,2}` run 上会按错 stride、把后半 2-周期混叠。→ **`step2` 之前先 `relocate_run<0,2,0,4>`**(对齐到 `{0,4}`)。 |
| 6 | place / transpose 还原全 0 | **`llk_unpack_set_srcb_dummy_valid()` 放在 `transpose_wh` 之前**会让它的 TRNSPSRCB 读到 dummy SrcB → 全 0。transpose_wh 本身**不需要** srcb-dummy-valid;它放在**所有 transpose 之后**、紧挨着需要它的 `step2`(`combine_finalize`)。(另:transpose_wh 的 `idst` **没有** tile 限制 —— 能写任意 DEST tile;之前"写不到 tile 2/3"就是这个 srcb bug。) |
| 7 | combine:垃圾 + hang,然后**选反了半** | `produce_run` + restore 在**同一个 acquire** —— produce_run 的 SFPU/SrcB 状态污染了同 acquire 的 transpose_wh。→ **merge-only acquire**(两个 block 都 stash,restore+merge 里没有 produce_run)。 |
| 8 | combine:merge 选了**错的 8 个**(dev 最大 key == gold 最小 key) | **bias 排序键被 2-周期破坏**,而 scores+idx 没事:`step2` 用 `num_tiles=2`,只把 scores(tile0)+idx(tile1) 转置成 standard,**没转 bias(tile2)** → bias 以 math 布局被 pack → 往返损坏。256 输出路径从不读 bias(normalize 只读 scores),所以一直发现不了,直到 merge 按 bias 排序才暴露。→ **`step2_configure_mop<3>`**(转置 tile 0,1,2)。对 256/finalize 输出无害(它们只 pack tile 0,1)。 |

附:in-kernel 的 `idx += b*256` 是 **no-op**(`sfpi l_reg[]` 的 SSA 写不回 raw TTI_SFPSTORE 读的物理 LREG;
`TTI_SFPIADD ARG_IMM` 也无变化)→ 改用**每 block 的 `input_indices` tile**(tile b = `arange + b*256`)绕开。

## 6. 关键文件与函数

- **`device/unified_kernels/generalized_moe_gate.hpp`** —— TRISC op。`process_block_to_run<b>()`(把一个 block
  stash 到 L1)、多 block combine 路径(merge-only acquire)、256 路径。
- **`.../compute_kernel_api/generalized_moe_gate.h`** —— ALWI wrapper:`generalized_moe_gate<...,produce_run,...>`、
  `relocate_run`、`step2_only`、`combine_init`、`combine_finalize`、`place_field_from_interm<field,dst_lo,dst_hi,
  src_lo,src_hi>`。
- **`.../tt_llk/.../ckernel_sfpu_generalized_moe_gate_topk_single_face.h`** —— `merge16_to_run`、`merge16_core`、
  `copy_topk_run`(relocate)、`normalize_run`、`place_field_from_interm`。
- **`.../tt_llk/.../llk_math_generalized_moe_gate_transpose_dest_single_face.h`** —— step0/step1/step2 转置。
  **`step2_init` 现在用 `step2_configure_mop<3>`**(坑 8)。
- **`device/generalized_moe_gate_program_descriptor_builder.cpp`** —— CB:`run_scores/idx/bias_cb`(5/6/7,
  L1 stash,`num_blocks` 个 tile)、`cb_tilize`(8,bf16,`2*num_blocks` 个 tile)、`cb_tilize_idx`(9,uint16,
  `num_blocks` 个 tile)。

## 7. 测试 / 调试宏(`generalized_moe_gate_kernel.cpp`)与测试

保留的宏:`GMG_UNGROUPED_TOP8`(默认 ON)、`GMG_DIAG_BLOCK`(combine 路径里逐 block A1 验证)、
`GMG_DUMP_AFTER_SUM_TOP2/STEP0/STEP1` 和 `GMG_DIAG_TOPA/TOPB`(阶段探针)。combine bring-up 的隔离宏
(`GMG_TEST_STASH/PARK/PARK2/PRODUCE_RUN`、`GMG_DUMP_OCCUPANCY`、`GMG_COMBINE_DIAG`)和死 helper
(`place_run_at`、`unpack_run_to_regions[_transpose]`)在 combine 落地后已删。测试(在
`models/demos/deepseek_v3/tests/test_generalized_moe_gate.py`):`test_generalized_moe_gate`(256/128)、
`test_generalized_moe_gate_512_global`(combine)、`test_generalized_moe_gate_512_per_block`、
`test_dump_stash_run` / `test_dump_combine_run`(纯调试用的完整 16×16 区域 dump —— `test_dump_combine_run` 就是
靠读整个 32×32 face 最终定位 bias bug 的工具)。

## 8. 剩余工作(都不阻塞 512)

1. **Kimi 384**:`num_blocks=2`,block1 = 256-383(+128 padding)。多半只是 op.py/test 的事 —— 把 padding expert
   的 key 设得很低,让它们永不被选中;kernel 的 combine 应该不用改。
2. **Top-n(k = 4/6/10)**:目前 k=8 写死在 `merge16_to_run`/`finalize`/bitonic 排序里,需要把 k 参数化。
3. **Softmax / sqrt-softplus** 归一化变体(见 `SOFTMAX_NOTES.md`)。
4. **>512**:需要 combine **树**(现在是单次 2-run merge)。
5. **性能。**
