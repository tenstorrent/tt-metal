# DeepSeek MoE Gate —— SFPU bitonic sort 逐步解析

对应代码（BH / WH 只差 `ADDR_MOD_7` vs `ADDR_MOD_3`）：

| 层 | 文件 |
| --- | --- |
| compute kernel | `device/unified_kernels/deepseek_moe_gate.hpp` |
| compute API（阶段编排） | `device/kernel_includes/tt_metal/include/compute_kernel_api/deepseek_moe_gate.h` |
| SFPU 排序主体 | `device/kernel_includes/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_deepseek_moe_gate_topk_single_face.h` |
| FPU transpose（跨 lane 搬运） | `.../llk_lib/llk_math_deepseek_moe_gate_transpose_dest_single_face.h` |
| FPU copy + bias add | `.../llk_lib/llk_math_deepseek_moe_gate_eltwise_binary.h` |

`generalized_moe_gate`（GMG）里的 `ckernel_sfpu_generalized_moe_gate_topk_single_face.h` 是同一份代码的参数化版本，
`_gmg_merge4_top8` / `_gmg_merge4_runs_raw` 就是本文第二部分的 merge 被抽出来的可复用块。

> **可运行的 Python 等价实现**：`docs/bitonic_top8_sim.py`（纯标准库）。
> 它逐条模拟 SFPSWAP / SFPTRANSP，把第 3 节「一个 group 的 32 个 expert → 有序 top-8」的每一步
> LREG 内容和逻辑序列都打出来，最后和 `sorted()` 对拍。
> 三种输入：`python3 bitonic_top8_sim.py` 用 key = 1..32 顺序摆放；
> `... shuffle` 换成 1..32 的一个固定置换（(a)/(b) 两半的 top-8 交错，merge 那步不退化）；
> `... random [SEED]` 用随机置换，seed 会打出来方便复现。

---

## 0. 算子在算什么

一个 core 处理一个 token：

- 输入 `[16, 16]` 的一个 face = **256 个 expert 的 router logits**，逻辑上是 `[8 groups, 32 experts]`
- `n_group = 8`，`topk_group = 4`，`top_k = 8`（DeepSeek V3 配置）
- 流程（golden 见 `models/demos/deepseek_v3_b1/micro_ops/deepseek_moe_gate/op.py`）：
  1. `score = sigmoid(logit)`（可选），`key = score + bias`
  2. 每组 32 个 expert 按 `key` 降序排序 → 取该组 top-8
  3. 每组 `key` 的 top-2 求和 → 8 个组分数 → 排序取 **top-4 groups**
  4. 这 4 组的 top-8（共 32 个候选）合并 → 全局 **top-8**
  5. 用**原始 score（不含 bias）**归一化：`score / (Σscore + eps) * scaling_factor`

第 2 步和第 4 步就是本文要讲的两部分 bitonic sort。

---

## 1. 数据模型（读代码前必须先建立的心智模型）

### 1.1 DEST 里的 4 个 tile

```cpp
constexpr uint32_t dst_tile_offset = 64;              // 1 tile = 64 dest rows
constexpr uint32_t scores_offset  = 0;                // tile0: 原始 score（sigmoid 后、不加 bias）
constexpr uint32_t indices_offset = 64;               // tile1: expert 全局 index
constexpr uint32_t bias_offset    = 128;              // tile2: key = score + bias（排序键）
constexpr uint32_t interm_offset  = 192;              // tile3: 中间量（top2 sum / 暂存 payload）
```

`tile0` 和 `tile2` 由一条自定义 FPU MOP 同时产生：`MOVA2D`（SrcA → DEST+0，拷贝 score）+
`ELWADD`（dst_math_offset = 2×64 → DEST+128，写 score+bias）。indices 由 `copy_tile` 提前放进 tile1。

### 1.2 SFPU 的 32 lane 布局

SFPU 一条向量 = 32 lane = **4 行 × 8 列**。

- **8 个 lane 列 = 8 个并行的排序实例 = 8 个 expert group**
- 一个 LREG 对某个实例只贡献 **4 个值**（4 行各一个）
- 因此：`2 个 LREG = 一条长度 8 的 run`，`4 个 LREG = 长度 16 的序列`

一条 sorted run 在 DEST 里按「列对」存：`LREG0 → +0`、`LREG1 → +4`，读回来就是 rank0..rank7。
（这一点有 GMG 的 dump 佐证，`edited_files/ckernel_sfpu_generalized_moe_gate_topk_single_face.h:461-478` 里
打印的 4×8 矩阵，沿 lane 0 竖着读 LREG0 行0-3 再 LREG1 行0-3，正好是降序的 8 个数。）

### 1.3 两套「坐标系」——SFPTRANSP 的作用

设序列位置为 `p`（0..15），LREG 号 `L`（0..3），SFPU 行号 `r`（0..3）：

| 状态 | 位置映射 | `SWAP(L0,L1)` 的比较距离 | `SWAP(L0,L2)` 的距离 |
| --- | --- | --- | --- |
| canonical（DEST 存取用的） | `p = 4L + r` | 4 | 8 |
| transposed（`SFPTRANSP` 之后） | `p = 4r + L` | 1 | 2 |

`SFPSWAP` 只能在两个 LREG 之间比较，**不能在行之间、更不能跨 lane**。
所以 `SFPTRANSP`（对 LREG0-3 与 LREG4-7 各做一次 4×4 转置）就是用来切换坐标系的：
需要距离 1/2 的比较器就切到 transposed，需要距离 4/8 的就切回 canonical。
**跨 lane 完全做不到**（这是 SFPU 的硬限制），只能靠 FPU 的 `MOVD2B/TRNSPSRCB/MOVB2D` 或 `SFPSHFT2` 的 lane 旋转。

### 1.4 payload：一个寄存器同时带 index 和 score

`TTI_SFPCONFIG(0x4, 0xF, 1)` 打开 **index tracking**：`SFPSWAP` 交换 `LREG0-3` 时，
`LREG4-7` 会跟着一起换（配对关系 `L0↔L4, L1↔L5, L2↔L6, L3↔L7`）。

关键 trick 在 `bitonic_topk_load16_concat_indices_single_face`：

```cpp
SFPLOAD(LREG4, LO16_ONLY, indices_offset + ...);   // 低 16 位 = expert index
SFPLOAD(LREG4, HI16_ONLY, scores_offset  + ...);   // 高 16 位 = 原始 bf16 score
```

一个 32-bit payload 寄存器 = `idx | score` 拼接，排序时一次搬两样东西。
代价：**`is_fp32_dest_acc_en` 必须为 false**（代码里有 `static_assert`），因为这依赖 16-bit DEST 打包。

### 1.5 SFPSWAP 的方向掩码

```
UNCONDITIONALLY : 无条件交换（不比较）
ALL_ROWS_MAX    : 4 行都是「第一个操作数取 max」
ROWS_01_MAX     : 行 0/1 取 max（降序），行 2/3 取 min（升序）
ROWS_02_MAX     : 行 0/2 取 max，行 1/3 取 min
```

bitonic 网络要求相邻子块交替方向，硬件正好用这些掩码表达：
在 transposed 坐标下行号 `r` 就是「4-block 编号」，所以
`ROWS_02_MAX` = 4-block 交替方向；在 canonical→transposed 的 8-block 视角下
`ROWS_01_MAX` = 前 8 个降序 / 后 8 个升序。方向不用分支，纯掩码。

---

## 2. 排序原语（4 个 phase）

`bitonic_top8_ph0_to_ph3()` = 对 16 个元素做**完整的 bitonic sort**，由 4 个 phase 组成。
下表里 T 表示进入/离开时的坐标系（0 = canonical，1 = transposed）：

| phase | 合并规模 | steps（比较距离） | 指令 | T 变化 |
| --- | --- | --- | --- | --- |
| ph0 `bitonic_topk_ph0_st1_to_1` | 2 | step1 (dist 1) | TRANSP + 2×SWAP | 0 → 1 |
| ph1 `bitonic_topk_ph1_st2_to_1` | 4 | step2 (2), step1 (1) | 4×SWAP + TRANSP | 1 → 0 |
| ph2 `bitonic_topk_ph2_st3_to_1` | 8 | step3 (4), step2 (2), step1 (1) | 2×SWAP + TRANSP + 4×SWAP + TRANSP | 0 → 0 |
| ph3 `bitonic_top8_ph3_st4_to_1` | 16 | step4 (8), step3 (4), step2 (2), step1 (1) | 7×SWAP + 2×TRANSP | 0 → 0 |

逐条说明：

**ph0（造长度 2 的有序对）**
```cpp
TTI_SFPTRANSP;                                  // 切到 transposed，LREG 相邻 = 距离 1
SFPSWAP(LREG0, LREG1, ALL_ROWS_MAX);            // pair (4r+0, 4r+1) 降序
SFPSWAP(LREG3, LREG2, ALL_ROWS_MAX);            // 操作数反过来写 → pair (4r+2, 4r+3) 升序
```
两个 pair 方向相反，凑成长度 4 的 bitonic 序列。**「反着写操作数」= 反方向比较器**，这是全文反复用的手法。

**ph1（把 4-block 排好）**
```cpp
SFPSWAP(L0,L2, ROWS_02_MAX);  SFPSWAP(L1,L3, ROWS_02_MAX);   // step2: 距离 2
SFPSWAP(L0,L1, ROWS_02_MAX);  SFPSWAP(L2,L3, ROWS_02_MAX);   // step1: 距离 1
TTI_SFPTRANSP;                                                // 回 canonical
```
`ROWS_02_MAX` 让 4-block 0/2 降序、1/3 升序 → 相邻 4-block 反向，为 ph2 准备 bitonic-8。

**ph2（把 8-block 排好）**——有 `bitonic` 模板开关：
```cpp
// step3: 距离 4（canonical 下 LREG 相邻）
SFPSWAP(L0,L1, ALL_ROWS_MAX);
bitonic ? SFPSWAP(L3,L2, ALL_ROWS_MAX)    // 两个 8-block 反向 → 给 ph3 做 bitonic-16
        : SFPSWAP(L2,L3, ALL_ROWS_MAX);   // 两个 8-block 同向 → 各自独立排好（"not bitonic"）
TTI_SFPTRANSP;
swap_mode = bitonic ? ROWS_01_MAX : ALL_ROWS_MAX;
SFPSWAP(L0,L2,mode); SFPSWAP(L1,L3,mode);   // step2
SFPSWAP(L0,L1,mode); SFPSWAP(L2,L3,mode);   // step1
```

**ph3（16 的 bitonic merge，为 top8 做了裁剪）**
```cpp
SFPSWAP(L0,L2, ALL); SFPSWAP(L1,L3, ALL);   // step4: 距离 8 → top8 落到 L0/L1，bottom8 落到 L2/L3
SFPSWAP(L0,L1, ALL);                        // step3: 只做上半！L2/L3 反正要丢
TTI_SFPTRANSP;
SFPSWAP(L0,L2, ALL); SFPSWAP(L1,L3, ALL);   // step2
SFPSWAP(L0,L1, ALL); SFPSWAP(L2,L3, ALL);   // step1
TTI_SFPTRANSP;                              // 回 canonical，L0/L1 = 有序 top-8
```
`dir` 模板参数为 `ArgMin` 时所有操作数顺序整体反写 → 得到**升序**结果（后面要用）。

**`reverse_sort_order()`** —— 把一条 run 整体倒过来：
```cpp
TTI_SFPTRANSP;
SFPSWAP(L0,L3, UNCONDITIONALLY);   // 无条件交换 = 行 0↔3
SFPSWAP(L1,L2, UNCONDITIONALLY);   // 行 1↔2
TTI_SFPTRANSP;
```
它只反转每个 LREG 内部的行序。所以调用者**同时把两半的地址也反着 load**
（例如 `LREG2 ← +6`、`LREG3 ← +2`），两者合起来才是完整的 8 元素反转。

---

## 3. 第一部分：每组 32 个 expert → 有序 top-8

函数：`_deepseek_moe_gate_sum_top2()`

一组 32 个 expert 分两次 load：地址 `{0,4,8,12}` 叫 **even columns**，`{2,6,10,14}` 叫 **odd columns**，
各 16 个值。8 个 lane 列 = 8 个 group **同时**做，一遍走完 256 个 expert。

```cpp
TTI_SETRWC(..., SET_D);          // 复位 Dst RWC，否则前面 FPU MOP 留下的 +64 偏移会污染 SFPLOAD 地址
TTI_SFPCONFIG(0x4, 0xF, 1);      // 打开 index tracking

// ---- (a) even 16 个 → 降序 ----
bitonic_topk_load16_concat_indices_single_face<..., 0>();   // L0-3 = key, L4-7 = idx|score
bitonic_top8_ph0_to_ph3<..., idir=false>();                 // 完整 bitonic sort of 16（降序）
bitonic_topk_store8_even_cols_concatted_indices_single_face();
                                 // 只存 top-8：L0,L1 → bias+0,+4 ; L4,L5 → interm+0,+4

// ---- (b) odd 16 个 → 升序 ----
bitonic_topk_load16_concat_indices_single_face<..., 2>();
bitonic_top8_ph0_to_ph3<..., !idir>();                      // 升序！L0/L1 里是升序 top-8

// ---- (c) 两条 run 合并 ----
bitonic_topk_load8_even_cols_concatted_indices_single_face();  // 把 (a) 的降序 run 读回 L0/L1
                                 // 此时 L0/L1 = 降序 8，L2/L3 = 升序 8 → 长度 16 的 bitonic 序列
bitonic_top8_ph3_st4_to_1<idir, true>();                       // 一次 ph3 merge → 32 个数的 top-8
bitonic_topk_store8_even_cols_split_indices_single_face();
                                 // L0,L1 → bias+0,+4 ; L4,L5 拆开：LO16→indices, HI16→scores
```

**为什么 (c) 只跑 ph3 而不是完整的 32 元素排序？**
代码注释写得很清楚：`Instead of a full phase 4, we rerun phase 3 since we are only comparing top8 values`。
32 个数的全局 top-8 一定落在「前 16 的 top-8」∪「后 16 的 top-8」里面，
所以两个 16 各自排完只留 top-8，再对这 16 个候选做一次 bitonic merge 就够了。
一半的元素在 (a)(b) 结束时就被丢掉，省掉了整个 phase 4。

**(b) 为什么要升序？** bitonic merge 要求输入是 bitonic 序列（先升后降或先降后升）。
让第二条 run 直接排成升序，接上第一条的降序，天然就是 bitonic，省掉一次 `reverse_sort_order`。

**尾巴：top-2 求和 + 广播**
```cpp
TTI_SFPTRANSP;
SFPADD(L0 = L0 + L1);        // transposed 后 L0 行0=rank0、L1 行0=rank1 → 行0 得到 rank0+rank1
TTI_SFPNOP;
TTI_SFPTRANSP;

TTI_SFPCONFIG(0, LREG14, 0); // 把 LREG14 配成「行0 广播到 4 行」
TTI_SFPMOV(0, LREG14, LREG0, 0);
SFPSTORE(L0, interm+0);  SFPSTORE(L0, interm+4);   // 组分数沿位置轴复制 8 份
```
广播是给下一阶段用的：转置之后，「每个 lane 一个组分数」会变成「每个 lane 都拿到全部 8 个组分数」。

**这一部分结束时 DEST 的状态**：lane k = group k，位置 0..7 = 该组 top-8，
`bias+0/+4` 存 key、`indices/scores +0/+4` 存 idx|score、`interm+0/+4` 存被广播的 top-2 和。

---

## 4. 中间过渡：选出 top-4 groups

这一步 **必须借 FPU**，因为要在 lane 之间搬数据。

**(1) `transpose_dest_single_face_step0`**（`num_tiles = 4`，4 个 DEST tile 全转）
用 `MOVD2B` → `TRNSPSRCB` → `MOVB2D` 对 face 做转置，把
「lane = group，位置 = rank」换成「lane = rank，位置 = group」。
效果：**每个 lane 现在都持有全部 8 个组分数**（靠 3 节末尾那次广播），
而 lane k 的 payload 是「各组的第 k 名」。

**(2) `_deepseek_moe_gate_sort_top4_groups()`**
```cpp
L0,L1 ← interm+0,+4        // 8 个组分数（每个 lane 都一样）
L4,L5 ← indices/scores     // payload A = idx|score
L2,L3 ← L0,L1 的拷贝        // 同一份 key 复制一次
L6,L7 ← bias+0,+4          // payload B = key 值

bitonic_topk_ph0_st1_to_1<true,false>();
bitonic_topk_ph1_st2_to_1<false,true>();
bitonic_topk_ph2_st3_to_1<true, /*bitonic=*/false>();   // ← 注意第二个模板参数是 false

SFPSTORE(L4, LO16, indices+0);  SFPSTORE(L4, HI16, scores+0);  SFPSTORE(L6, bias+0);
```
- 只跑 ph0+ph1+ph2、且 `bitonic=false`，得到的是**两个各自独立降序的 8-block**，不是 bitonic-16。
- key 复制成两份、payload 分两份，是为了绕开 index tracking 「一个 key 只能配一个 payload」的限制：
  `(L0,L1)+(L4,L5)` 带 idx|score，`(L2,L3)+(L6,L7)` 带 key 值，两边排出来的名次完全一致。
- 结束时是 canonical 坐标，`LREG4` = 位置 0..3 = **排名前 4 的组**。只 store `+0` 这一列，
  后 4 个组直接被丢弃 —— 这就是 `topk_group = 4` 的落地方式。
- 一共存下 4 组 × 8 名 = 32 个候选。

**(3) `transpose_dest_single_face_step1`**（`num_tiles = 3`）
再转一次，把 4 条 top-8 run 摆成第二部分要的形状：run 落在 DEST 列 `{0,4}` 和 `{2,6}`，
且分别位于 **lane 0 和 lane 1**（GMG 的 dump 直接印证了这一点：
`LREG0+1 is descending, LREG2+3 is ascending, lane0=lane6, lane1=lane7`，
lane 2-5 是无关数据）。

---

## 5. 第二部分：4 组的 top-8 合并成最终 top-8

函数：`_deepseek_moe_gate_top8()`（GMG 里抽成了 `_gmg_merge4_runs_raw` / `_gmg_merge4_top8`）

4 条 run 的摆放：

```
            lane 0            lane 1
列 {0,4}    run A (组1)       run B (组2)      → LREG0 / LREG1
列 {2,6}    run C (组3)       run D (组4)      → LREG2 / LREG3
```

### Stage 1：lane 内合并，4 条 run → 2 条

```cpp
TTI_SETRWC(..., SET_D);

// 先读 {+6,+2} 这条（注意地址是反的：+6 进 L2，+2 进 L3）
SFPLOAD(L2, bias+6);  SFPLOAD(L3, bias+2);
SFPLOAD(L6/L7, indices LO16 + scores HI16 @ +6/+2);
reverse_sort_order();          // 配合反序 load，整条 run 完全倒过来 → 升序

// 再读 {+0,+4} 这条（原样降序）
SFPLOAD(L0, bias+0);  SFPLOAD(L1, bias+4);
SFPLOAD(L4/L5, ... @ +0/+4);

bitonic_top8_ph3_st4_to_1<idir=false, true>();   // 降序8 + 升序8 = bitonic16 → 有序 top-8
bitonic_topk_store8_even_cols_concatted_indices_single_face();   // 暂存到 bias/interm +0,+4
```
这一步 lane 0 得到 `top8(A ∪ C)`，lane 1 得到 `top8(B ∪ D)` ——
对应源码注释 `Combine and sort 4 groups of 8 values to 2 groups of 8 values`。

### Stage 2：跨 lane 合并，2 条 run → 1 条

lane 0 和 lane 1 的结果必须碰面，而 `SFPSWAP`/`SFPTRANSP` 都跨不了 lane，
所以用 `SFPSHFT2` 的**子向量 lane 旋转**把 lane 1 的 run 搬到 lane 0：

```cpp
TTI_SFPSHFT2(0, L0, L3, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);   // 每行 8 lane 循环右移 1 位
TTI_SFPSHFT2(0, L1, L2, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
TTI_SFPCONFIG(0, 0xF, 1);      // 关掉 index tracking：HW bug，开着时对 LREG4-7 操作会出错
TTI_SFPSHFT2(0, L4, L7, ...);  // payload 手动跟着搬
TTI_SFPSHFT2(0, L5, L6, ...);
TTI_SFPCONFIG(0x4, 0xF, 1);    // 重新打开

reverse_sort_order();          // 搬过来的这条转成升序
bitonic_topk_load8_even_cols_concatted_indices_single_face();   // 把 Stage 1 暂存的那条读回 L0/L1

// 只做 step 4！
TTI_SFPSWAP(L0, L2, ALL_ROWS_MAX);
TTI_SFPSWAP(L1, L3, ALL_ROWS_MAX);
```
`step4` 是 bitonic-16 merge 的第一级（距离 8 的比较器），它的作用正是
**把 16 个数切成「上 8」和「下 8」**：`L0/L1` 拿到全部 top-8，`L2/L3` 拿到剩下 8 个。
因为最终输出不要求有序（Python 侧对 index 排序后再比对），所以 step3/2/1 全部省掉。
源码注释：`Step 4 Only, we need top8 but it doesn't have to be sorted`。

存回：`L4/L5` 拆成 `LO16 → indices+0/+4`、`HI16 → scores+0/+4`。

### 归一化尾巴

```cpp
SFPLOAD(L0, scores+0);  SFPLOAD(L1, scores+4);   // 取的是原始 score，不是 key
SFPADD(L0 = L0 + L1);                            // 4 个部分和
TTI_SFPTRANSP;
SFPADD(L0 = L0+L1);  SFPADD(L2 = L2+L3);  SFPADD(L0 = L0+L2);   // 树形规约 → 行0 = Σ(top8)

TTI_SFPCONFIG(0, 0xF, 1);                        // recip 用的寄存器由编译器分配，先关 tracking 保平安
l0 = 1 / (l0 + eps) * scale;                     // sfpi 写的倒数

TTI_SFPCONFIG(0, LREG14, 0);                     // 行0 广播到 4 行
SFPLOAD(L0, scores+0);  SFPLOAD(L1, scores+4);
SFPMUL(L0 *= LREG14);   SFPMUL(L1 *= LREG14);
SFPSTORE → scores+0, scores+4;
```

最后 `transpose_dest_single_face_step2`（`num_tiles = 2`）把 scores / indices 两个 tile
摆成输出 tile 的 `1×16` 形状，pack 出去（只有前 8 个有效）。

---

## 6. 小结与注意点

**两部分的对称性**：其实是同一个 `bitonic_top8_ph3_st4_to_1` 被用了三次，
只是「输入怎么变成 bitonic 序列」的手法不同：

| 场景 | 两条 run 怎么凑成 bitonic | merge 深度 |
| --- | --- | --- |
| 第一部分 (c)：组内 16+16 | 第二条直接**排成升序**（`!idir`） | 完整 ph3（要有序输出） |
| 第二部分 Stage 1：lane 内 8+8 | **反序 load + `reverse_sort_order`** | 完整 ph3（结果还要再 merge） |
| 第二部分 Stage 2：跨 lane 8+8 | `SFPSHFT2` 旋转 + `reverse_sort_order` | **只做 step4**（不要求有序） |

**性能上省掉的东西**：
- 32 个数不做完整排序，两个 16 各留 top-8 → 省掉 phase 4
- 最后一级 merge 只做 step4 → 省掉 step3/2/1
- 每级 merge 里 `L2/L3` 的下半部分不再排序（ph3 的 step3 只做 `SWAP(L0,L1)`）

**踩坑点**：
- 每个 SFPU 阶段开头都要 `TTI_SETRWC(..., SET_D)`：前面的 FPU MOP 会把 Dst RWC 推进 +64/tile，
  不复位的话 `SFPLOAD/SFPSTORE` 的立即数偏移全部偏掉。
- `is_fp32_dest_acc_en` 必须为 false —— `idx|score` 拼接依赖 16-bit DEST 打包。
- `SFPSHFT2` 前后必须关/开 index tracking（HW bug 会影响 LREG4-7）。
- 排序键是 `score + bias`，但归一化用的是**不含 bias 的原始 score**，两者一路靠
  `bias tile` / `payload HI16` 分开携带。
- 跨 lane 只有两条路：FPU transpose（`MOVD2B/TRNSPSRCB/MOVB2D`）或 `SFPSHFT2` 的 lane 旋转；
  `SFPSWAP`/`SFPTRANSP` 都只在 LREG 与行之间动。
