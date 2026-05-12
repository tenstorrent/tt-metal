# Type-1 Migration Sweep Design — 37 blocks across 15 partial-migration kernels

Branch: `astancov/eltwise_run7_refined`
Base: `d1d1f9246c5` (`[partial-audit] Type-1 migration targets across 31 B-classified kernels`)
Helper-library state: post-commit-8 (`3b0cc6026e8` — Doxygen + caller-init contract spec on chain helper headers).
Stance: **Type-1 only.** No helper extensions, no Type-2 / Type-3 work, no helper API touches.

---

## Section A — Scope summary

This sweep migrates the **37 Type-1 raw-LLK blocks** identified in `partial_kernel_audit.md` Section 3 (kernel-IDs T1.01 – T1.37) across **15 partial-migration kernels** (the audit's "files with at least one Type-1 block" subset of the 47-file B enumeration). Every block in scope is a sequential `[CB-reader] → [DEST compute] → [CB-writer]` shape that the run7-refined `eltwise_chain` already accepts — migration is mechanical.

**Out of scope (deferred):**
- **Type-2 helper extensions** — MASK-INJECT (`MaskInject<COND, …>`), HELD-CB (`HeldCbBinaryFpu`), MULTI-DEST per-iter, ASYMMETRIC-INDEX recovery (`BinaryFpuPerTileScalarB`), POWER-MULTISTAGE op-struct, MOREH `binary_max_tile` op-struct, `mask_tile_to_cb` family. These would unlock further migrations in 17+ moreh kernels but are explicitly deferred per the user's "Type 1 only" lock (see Section F).
- **Type-3 patterns** — `tilize_block`/`untilize_block` (ssm_prefix_scan), `transpose_wh` / `topk_*` (deepseek_grouped_gate), PACK-thread SFPU (moe_compute / moe_gpt), matmul (moe_*). Permanently out of `eltwise_chain` scope.
- **Two run7 Q4 regressed kernels** (`eltwise_binary_scalar.cpp` no-act branch; `deepseek_grouped_gate.cpp::add_bias`) stay raw-LLK per design v6 Section E. Recovery requires `BlockBinaryFpuScalarB<…>` / `BinaryFpuPerTileScalarB<…>` element types — future helper-extension run.

### Quick map: commit → block IDs → est. LOC delta → test cluster

| # | Subject | Block IDs | Files (count) | Est. LOC Δ (-/+) | Pytest cluster | Risk tier |
|---|---------|-----------|----------------|-------------------|-----------------|-----------|
| 1 | `eltwise sweep: migrate deepseek_grouped_gate scale block` | T1.37 | 1 | -18 / +8 | `tests/ttnn/nightly/unit_tests/operations/reduction/test_deepseek_grouped_gate.py` | Green |
| 2 | `eltwise sweep: migrate moreh_norm/ord_other/moreh_norm_nc IS_ZERO branch` | T1.17 | 1 | -22 / +12 | `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_norm.py` | Yellow (`--dev`) — UnaryNe runtime-param dispatch verification |
| 3 | `eltwise sweep: migrate moreh_softmax_backward {c,h,h_large,w,w_large} Type-1 blocks` | T1.01 – T1.10 | 5 | -76 / +52 | `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py` (covers backward) | Green |
| 4 | `eltwise sweep: migrate moreh_softmax forward {c,h,w}_large exp/rexp/log/recip blocks` | T1.11 – T1.16 | 3 | -42 / +60 | `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py` | Green |
| 5 | `eltwise sweep: migrate moreh_nll_loss / moreh_nll_loss_backward Type-1 blocks` | T1.25 – T1.31 | 2 | -110 / +55 | `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_nll_loss.py` | Green |
| 6 | `eltwise sweep: migrate moreh_adam / moreh_adamw bias-correction Type-1 blocks` | T1.32 – T1.36 | 2 | -60 / +40 | `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_adam.py` + `test_moreh_adamw.py` | Yellow (`--dev`) — Power runtime-param dispatch |
| 7 | `eltwise sweep: migrate moreh_layer_norm_backward gamma_beta_grad cb_y mul block` | T1.24 | 1 | -22 / +12 | `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py` (covers backward) | Green |
| 8 | `eltwise sweep: migrate moreh_layer_norm_backward input_grad_small bcast blocks` | T1.22, T1.23 | 1 | -44 / +24 | `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py` | Yellow (`--dev`) — bcast templating verification |
| 9 | `eltwise sweep: migrate moreh_layer_norm_backward input_grad_large bcast blocks` | T1.18 – T1.21 | 1 | -88 / +48 | `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py` | Yellow (`--dev`) — multi-block, bcast templating, T1.21 needs in-commit tile_idx audit |

**Total: 9 commits, 37 Type-1 blocks, 15 distinct kernels, est. -482 / +311 LOC net.**

### Regression bar

- **Per-commit gate:** the touched-kernel pytest cluster (the column above) green via `scripts/run_safe_pytest.sh --run-all <test_path>`. Yellow-tier commits run with `--dev` first, then re-run plain to confirm not relying on dev asserts.
- **End-of-pipeline gate (after commit 9):**
  - Full ~74 migrated-kernel pytest cluster green — the union of every Section 6 cluster of `coverage_audit.md` (data_movement/bcast, eltwise/binary, eltwise/binary_ng, eltwise/ternary, eltwise/unary, eltwise/unary_backward, copy/typecast, dropout, deepseek_grouped_gate, ssm/prefix_scan, moe_*, bcast_to/*, all moreh_* tests).
  - 401-test eltwise helper suite (`scripts/run_safe_pytest.sh --run-all ttnn/cpp/ttnn/kernel_lib/tests/eltwise/test_eltwise.py`).
  - No new failures introduced versus the pre-sweep baseline at `d1d1f9246c5`.

---

## Section B — Commit grouping plan

### Commit 1 — `eltwise sweep: migrate deepseek_grouped_gate scale block`

**Block IDs covered:** T1.37 (single block).

**Files touched:**
- `ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/compute/deepseek_grouped_gate.cpp` — lines 326–343 (`scale` block).

**Concrete change shape (per block):**

T1.37 — `scale` block (lines 326–343 of `deepseek_grouped_gate.cpp`). Current shape: per-tile raw `acquire / mul_tiles_bcast_scalar(cb_normalized_scores, cb_route_scale_scalar, …) / commit / wait / pack(cb_out_weights) / release`. Migration produces a single chain:

```
BinaryFpu<cb_normalized_scores, cb_route_scale_scalar, cb_out_weights,
          BinaryFpuOp::Mul, BroadcastDim::Scalar,
          BinaryDataFormatReconfig::InputAndOutput,
          CopyTilePolicy::WaitAndPop /* A */, CopyTilePolicy::NoWaitNoPop /* B held */,
          CbIndexMode::FirstTile, Dst::D0, /*EnableFp32DestAcc=*/false>{}
+ PackTile<cb_out_weights, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
```

Driven by `eltwise_chain(num_tiles, …)`. Caller scope already issues `cb_wait_front(cb_route_scale_scalar, 1)` before the block (preserves `NoWaitNoPop` policy on B-side); existing `cb_pop_front(cb_route_scale_scalar, 1)` after the chain is preserved (helper does not own held-CB lifecycle on `NoWaitNoPop` operands). `compute_kernel_hw_startup(cb_normalized_scores, cb_route_scale_scalar, cb_out_weights)` already emitted at MAIN-top per kernel-existing scaffolding (the file has multiple stages — the call near the scale stage stays).

**Risk assessment:**
- Single block; lowest-risk. The chain element shape is exercised by `bcast_hw.cpp` (Class A in coverage_audit) and `bcast_w.cpp` (Class A) — same `BinaryFpu(Bcast=Scalar) + Pack` pattern with mixed wait policies. Helper coverage solid.
- DST capacity: 1 tile per chain iter (DEST slot 0 only) — well within the bf16 8-tile / fp32 4-tile ceiling.
- `NoWaitNoPop` on B-side requires the caller-side `cb_wait_front(cb_route_scale_scalar, 1)` to remain in place. Verified by reading the audit's note ("scalar held across stage").
- No interaction with the kernel's other regressed `add_bias` block (lines 39–68) — that block stays raw per design v6 Section E.

**Acceptance criteria:**
- `scripts/run_safe_pytest.sh --run-all tests/ttnn/nightly/unit_tests/operations/reduction/test_deepseek_grouped_gate.py` — 2 tests passing.
- Risk tier: **Green** (no `--dev` flag).

---

### Commit 2 — `eltwise sweep: migrate moreh_norm/ord_other/moreh_norm_nc IS_ZERO branch`

**Block IDs covered:** T1.17 (single block, one file, gated on `#ifdef IS_ZERO`).

**Files touched:**
- `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_nc/kernels/moreh_norm_nc_kernel.cpp` — lines 39–60 (IS_ZERO branch under `#ifdef IS_ZERO`).

**Concrete change shape:**

T1.17 — IS_ZERO branch f(x) computation. Current raw shape: `acquire / copy_tile(cb_x, 0, dst0) / unary_ne_tile(dst0, 0u) / (#ifdef MINUS_INF) negative_tile(dst0) / commit / wait / pack(cb_val) / release`. Migration:

```
CopyTile<cb_x, Dst::D0, CopyTilePolicy::WaitAndPop>{}
+ UnaryNe<Dst::D0>{0u}                                            // member-exec runtime param
+ OptionalChainElement<MINUS_INF, Negative<Dst::D0>>{}            // compile-time conditional
+ PackTile<cb_val, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
```

Driven by `eltwise_chain(num_tiles, …)`. The audit confirmed that `eltwise_predicates.hpp:61` defines `UnaryNe` via `ELTWISE_DECLARE_UNARY_PARAM(UnaryNe, unary_ne)` macro (lines 53–62 of that header), and the chain-side dispatch routes `apply()` through member-`exec(uint32_t)` — verified at `eltwise_chain.inl:888,965`. The migration log's "UnaryNe runtime-param dispatch GAP" entry is a stale claim closed in run7-refined. `OptionalChainElement<MINUS_INF, Negative<Dst::D0>>` follows the U5 pattern from `where_tss_kernel.cpp:49-52` (commit 7 of run7 refinement) and `logit_kernel.cpp:44`.

**Risk assessment:**
- Yellow tier — though `UnaryNe` member-exec is verified to exist (helper-side), the migration log's stale claim warrants a smoke test under `--dev` to confirm runtime-param dispatch fires correctly with the chain v1 dispatcher in the IS_ZERO `#ifdef` macro permutation.
- Cross-`#ifdef` matrix to verify: `(IS_ZERO ∈ {set, unset}) × (MINUS_INF ∈ {set, unset})` = 4 build configurations. The kernel file at lines 39–60 is gated on `#ifdef IS_ZERO`, so only the IS_ZERO branch is touched; the !IS_ZERO branch (lines 66–82) is already-migrated (Class A within this kernel).
- DST: single slot (`Dst::D0`) — well within bf16 8-tile capacity.
- The audit explicitly recommends `--dev` for this commit (Section 6 risk hotspot).

**Acceptance criteria:**
- `scripts/run_safe_pytest.sh --dev --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_norm.py` — 6 tests passing under `--dev`.
- Re-run without `--dev` (`scripts/run_safe_pytest.sh --run-all <path>`) to confirm no `--dev`-only assert masks the result.
- Risk tier: **Yellow** (`--dev`).

---

### Commit 3 — `eltwise sweep: migrate moreh_softmax_backward {c,h,h_large,w,w_large} Type-1 blocks`

**Block IDs covered:** T1.01, T1.02, T1.03, T1.04, T1.05, T1.06, T1.07, T1.08, T1.09, T1.10 (10 blocks across 5 files).

**Files touched:**
- `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_c_large.cpp` — line 102 (T1.01: `exp_tile_to_cb`), line 182 (T1.02: `mul_tiles_and_negative_to_cb`).
- `…/moreh_softmax_backward_h.cpp` — line 129 (T1.03: `exp_tile_to_cb` with runtime tile-idx `w`), line 204 (T1.04: `mul_tiles_and_negative_to_cb` with runtime A-side idx `h`).
- `…/moreh_softmax_backward_h_large.cpp` — line 131 (T1.05), line 227 (T1.06).
- `…/moreh_softmax_backward_w.cpp` — line 128 (T1.07: `exp_tile_to_cb` with runtime `w`), line 203 (T1.08: `mul_tiles_and_negative_to_cb` with runtime `h`).
- `…/moreh_softmax_backward_w_large.cpp` — line 131 (T1.09), line 225 (T1.10).

**Concrete change shape (representative — applied per block):**

T1.01 / T1.05 / T1.09 (`exp_tile_to_cb(cb_y, cb_exp)` — no runtime tile-idx, B-side held):
```
CopyTile<cb_y, Dst::D0, CopyTilePolicy::WaitAndPop>{}
+ Exp<Approx::Exact, Approx::Fast, Dst::D0>{}
+ PackTile<cb_exp, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
```

T1.02 / T1.06 / T1.10 (`mul_tiles_and_negative_to_cb(cb_dy_m_sum, cb_y, cb_dx)` — both inputs popped):
```
BinaryFpu<cb_dy_m_sum, cb_y, cb_dx,
          BinaryFpuOp::Mul, BroadcastDim::None,
          BinaryDataFormatReconfig::InputAndOutput,
          CopyTilePolicy::WaitAndPop, CopyTilePolicy::WaitAndPop,
          CbIndexMode::FirstTile, Dst::D0>{}
+ Negative<Dst::D0>{}
+ PackTile<cb_dx, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
```

T1.03 / T1.07 (`exp_tile_to_cb(cb_y, cb_exp, w, /*dst=*/0, /*pop=*/0)` — runtime tile-idx `w`, B-side `cb_y` held by outer scope):
```
auto copy_elt = CopyTile<cb_y, Dst::D0, CopyTilePolicy::NoWaitNoPop, CbIndexMode::Pinned>{};
copy_elt.cb_tile_idx = w;
auto pack_elt = PackTile<cb_exp, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{};
eltwise_chain(1, copy_elt, Exp<>{}, pack_elt);
```
(`cb_tile_idx` is the `CopyTile` runtime member field per the design v6 Q17 pattern — same idiom moreh_bin_chain uses with `Pinned` mode.)

T1.04 / T1.08 (`mul_tiles_and_negative_to_cb(cb_y, cb_inter2, cb_dx, h, 0, /*pop0=*/0, /*pop1=*/1)` — A-side runtime `h`, B-side held, asymmetric per-side policies):
```
auto bin_elt = BinaryFpu<cb_y, cb_inter2, cb_dx,
                         BinaryFpuOp::Mul, BroadcastDim::None,
                         BinaryDataFormatReconfig::InputAndOutput,
                         CopyTilePolicy::NoWaitNoPop /* A held by outer */,
                         CopyTilePolicy::WaitAndPop  /* B popped per iter */,
                         CbIndexMode::Pinned, Dst::D0>{};
bin_elt.a_tile_idx = h;
bin_elt.b_tile_idx = 0;
eltwise_chain(1, bin_elt, Negative<>{}, PackTile<cb_dx, …>{});
```

(Asymmetric per-side `APolicy/BPolicy` is preserved by design v6 Q4 — only `AIndex/BIndex` collapsed to single `Index`. Per-side runtime tile_idx values stay independent — see `eltwise_chain.inl:330-331`. With `Index=Pinned`, both sides read their member field; `b_tile_idx=0` is semantically identical to `FirstTile`.)

**Risk assessment:**
- T1.04 / T1.08 use the asymmetric per-side policy (`A=NoWaitNoPop, B=WaitAndPop`) AND runtime-pinned A-side index. This is the moreh_bin_chain_rt idiom from `moreh_softmax_backward_h.cpp` lines 57–67, already exercised in production with `Pinned` mode. The pattern is the canonical post-Q4-collapse asymmetric expression — tested by every existing moreh_bin_chain_rt callsite.
- Shared test cluster (`test_moreh_softmax.py`, lines 297, 330, 363, 399, 466) covers all 5 backward kernel variants — high coverage density.
- `compute_kernel_hw_startup` placement: each kernel currently issues it at MAIN-top; the migration does not move it (in-loop chain calls per moreh row-5 of D5 placement table — outer `binary_op_init_common` already covers).
- Bundling rationale: 10 blocks, 5 kernels share **one** test file (`test_moreh_softmax.py`). Audit Section 6 explicitly recommends bundling. Bisect ergonomics preserved by the per-file commit-message body listing each file's block count.

**Acceptance criteria:**
- `scripts/run_safe_pytest.sh --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py` — 12 tests passing (covers both forward and backward variants).
- Risk tier: **Green**.

---

### Commit 4 — `eltwise sweep: migrate moreh_softmax forward {c,h,w}_large exp/rexp/log/recip blocks`

**Block IDs covered:** T1.11, T1.12, T1.13, T1.14, T1.15, T1.16 (6 IDs covering up to 10 callsites; T1.15/T1.16 are 4 callsites each in their respective files).

**Files touched:**
- `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_c_large.cpp` — line 128 (T1.11: SOFTMAX `exp_tile_to_cb`), line 140 (T1.12: SOFTMIN `rexp_tile_to_cb`), line 160 (T1.13: LOG `log_tile_to_cb`), line 163 (T1.14: else `recip_tile_to_cb`).
- `…/moreh_softmax_h_large.cpp` — lines 156, 169, 263, 288 (T1.15: `exp_tile_to_cb` / `rexp_tile_to_cb`).
- `…/moreh_softmax_w_large.cpp` — lines 190, 203, 297, 322 (T1.16: same pattern).

**Concrete change shape (representative):**

T1.11 / T1.15 / T1.16 `exp_tile_to_cb(cb_in, cb_out)` (and per-variant `rexp_tile_to_cb` adds `Negative` upstream):
```
CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitAndPop>{}
+ Exp<Approx::Exact, Approx::Fast, Dst::D0>{}
+ PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
```

T1.12 `rexp_tile_to_cb(cb_tmp, cb_exps)` — `exp(-x)`:
```
CopyTile<cb_tmp, Dst::D0, CopyTilePolicy::WaitAndPop>{}
+ Negative<Dst::D0>{}
+ Exp<Approx::Exact, Approx::Fast, Dst::D0>{}
+ PackTile<cb_exps, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
```

T1.13 `log_tile_to_cb(cb_add, cb_recipsumexps)`:
```
CopyTile<cb_add, …, WaitAndPop>{} + Log<>{} + PackTile<cb_recipsumexps, …>{}
```

T1.14 `recip_tile_to_cb(cb_add, cb_recipsumexps)`:
```
CopyTile<cb_add, …, WaitAndPop>{} + Recip<>{} + PackTile<cb_recipsumexps, …>{}
```

The audit suggests introducing a local `moreh_unary_chain<Sfpu, CbIn, CbOut>` template wrapper to consolidate the ~10 callsites uniformly (mirror of existing `moreh_bin_chain`/`moreh_copy_chain` wrappers). **Adopt this consolidation** — it lands the wrapper template in the `_c_large.cpp` file (or a shared moreh-helper header if existing — verify in-commit), and re-uses the wrapper across `_h_large.cpp` and `_w_large.cpp`. Net LOC delta is more favorable than per-callsite expansion.

**Risk assessment:**
- All blocks are 3-element chains (`CopyTile + Sfpu + PackTile`) — the most-tested chain shape in the helper suite (every Class A/E unary kernel exercises it; test coverage is exhaustive).
- Negative as the second element of a chain (T1.12 rexp variant): exercised by `moreh_softmax_backward_c_large.cpp:182` `mul_tiles_and_negative_to_cb` (Class B chain shape after Commit 3 lands) and by `logsigmoid_kernel.cpp` (`CopyTile + Negative + Exp + LogSigmoid + Pack` Class E in coverage_audit Section 2). Pattern is exercised.
- Test cluster: shared with Commit 3 (`test_moreh_softmax.py` covers both forward and backward). Backward landed first (Commit 3), so Commit 4 lands on a green base.

**Acceptance criteria:**
- `scripts/run_safe_pytest.sh --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py` — 12 tests passing.
- Risk tier: **Green**.

---

### Commit 5 — `eltwise sweep: migrate moreh_nll_loss / moreh_nll_loss_backward Type-1 blocks`

**Block IDs covered:** T1.25, T1.26, T1.27, T1.28, T1.29, T1.30, T1.31 (7 blocks across 2 files).

**Files touched:**
- `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/moreh_nll_loss_step2_kernel.cpp` — lines 43–61 (T1.28: negative block), lines 63–80 (T1.29: cb_tmp1 * cb_tmp_weight), lines 82–97 (T1.30: cb_tmp3 * cb_divisor_recip with bcast scalar), lines 106–130 (T1.31: no-WEIGHT branch). Audit notes that T1.28–T1.31 collapse into the `for (b)` body's helper-driven rewrite — the four-block migration replaces the entire per-iter sequence.
- `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/moreh_nll_loss_backward_kernel.cpp` — lines 48–63 (T1.25: cb_tmp_weight * cb_output_grad → -result → cb_tmp2), lines 65–79 (T1.26: cb_tmp2 * cb_tmp1 → cb_input_grad with B held), lines 82–100 (T1.27: no-DIVISOR branch).

**Concrete change shape (representative):**

T1.28 (negative block):
```
CopyTile<cb_tmp_input, Dst::D0, CopyTilePolicy::WaitAndPop>{}
+ Negative<Dst::D0>{}
+ PackTile<cb_tmp1, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
```

T1.29 (cb_tmp1 * cb_tmp_weight):
```
BinaryFpu<cb_tmp1, cb_tmp_weight, cb_tmp3,
          BinaryFpuOp::Mul, BroadcastDim::None,
          BinaryDataFormatReconfig::InputAndOutput,
          CopyTilePolicy::WaitAndPop, CopyTilePolicy::WaitAndPop,
          CbIndexMode::FirstTile, Dst::D0>{}
+ PackTile<cb_tmp3, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
```

T1.30 / T1.31 (cb_tmp3 * cb_divisor_recip with bcast scalar):
```
BinaryFpu<cb_tmp3, cb_divisor_recip, cb_output,
          BinaryFpuOp::Mul, BroadcastDim::Scalar,
          BinaryDataFormatReconfig::InputAndOutput,
          CopyTilePolicy::WaitAndPop, CopyTilePolicy::NoWaitNoPop /* held divisor */,
          CbIndexMode::FirstTile, Dst::D0>{}
+ PackTile<cb_output, …>{}
```

T1.25 (cb_tmp2 = -(cb_tmp_weight * cb_output_grad), bcast scalar B):
```
BinaryFpu<cb_tmp_weight, cb_output_grad, cb_tmp2,
          BinaryFpuOp::Mul, BroadcastDim::Scalar,
          …, CopyTilePolicy::WaitAndPop, CopyTilePolicy::NoWaitNoPop /* output_grad held */,
          CbIndexMode::FirstTile, Dst::D0>{}
+ Negative<Dst::D0>{}
+ PackTile<cb_tmp2, …>{}
```

T1.26 (cb_input_grad = cb_tmp2 * cb_tmp1 with B held):
```
BinaryFpu<cb_tmp2, cb_tmp1, cb_input_grad,
          BinaryFpuOp::Mul, BroadcastDim::Scalar,
          …, CopyTilePolicy::WaitAndPop, CopyTilePolicy::NoWaitNoPop /* tmp1 held */,
          CbIndexMode::FirstTile, Dst::D0>{}
+ PackTile<cb_input_grad, …>{}
```

T1.27 (no-DIVISOR shape; same as T1.25 but pack to cb_input_grad).

**Risk assessment:**
- Cross-`#ifdef` matrix: `(WEIGHT ∈ {set, unset}) × (DIVISOR ∈ {set, unset})` = 4 build configurations per file. Each Type-1 block lives under specific gating; the migration must preserve the exact `#ifdef` shape. Per-block ifdef scope is documented in audit Section 2.13.
- T1.28–T1.31 form a connected DEST-handoff chain: under the original raw shape, the held-DEST flow into the next block's pack determines correctness. The audit explicitly notes the migration of these 4 blocks together "replaces the entire `for (b)` body" — keep them in one commit (any partial migration would leave a dangling DEST in cb_tmp1 / cb_tmp3 between iterations).
- `init_sfpu(cb_output_grad, cb_input_grad)` boot in `moreh_nll_loss_backward_kernel.cpp:27` (alternate engine-boot variant per audit Section 5) is caller-owned and stays — does not interact with the chain elements.
- Test cluster: 6 tests in `test_moreh_nll_loss.py` cover forward (line 173, 190, 227) + backward (line 251, 267, 303).

**Acceptance criteria:**
- `scripts/run_safe_pytest.sh --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_nll_loss.py` — 6 tests passing across all `(WEIGHT, DIVISOR)` permutations the test suite covers.
- Risk tier: **Green**.

---

### Commit 6 — `eltwise sweep: migrate moreh_adam / moreh_adamw bias-correction Type-1 blocks`

**Block IDs covered:** T1.32, T1.33, T1.34 (moreh_adam), T1.35, T1.36 (moreh_adamw) — 5 blocks across 2 files.

**Files touched:**
- `ttnn/cpp/ttnn/operations/moreh/moreh_adam/device/kernels/moreh_adam.cpp` — lines 270–281 (T1.32: `cb_tmp1 = pow(beta2, step)`), lines 378–389 (T1.33: `cb_tmp2 = pow(beta1, step)`), lines 318–328 (T1.34: AMSGRAD copy, conditional on `#ifdef AMSGRAD`).
- `ttnn/cpp/ttnn/operations/moreh/moreh_adamw/device/kernels/moreh_adamw.cpp` — lines 281–292 (T1.35: `cb_tmp1 = 1/(1 - cb_beta2_exponent)`), lines 359–371 (T1.36: `cb_tmp2 = 1/(1 - cb_beta1_exponent)`).

**Concrete change shape:**

T1.32 (cb_tmp1 = pow(beta2, step) — A-side pinned at scalar tile_idx=`beta2_tile=2`, runtime exponent `step`):
```
auto copy_elt = CopyTile<cb_scalar_args, Dst::D0,
                          CopyTilePolicy::WaitNoPop /* held — popped at end of kernel */,
                          CbIndexMode::Pinned>{};
copy_elt.cb_tile_idx = beta2_tile;  // = 2
auto power_elt = Power<Dst::D0>{};
power_elt.exponent = step;  // runtime
eltwise_chain(1, copy_elt, power_elt, PackTile<cb_tmp1, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
```

(`Power<>` member field `exponent` confirmed at `eltwise_math.hpp:80-86`. Member-`exec(uint32_t)` dispatch is the same path as `UnaryNe` — `eltwise_chain.inl:888`.)

T1.33 (same shape, A-side `beta1_tile=1`).

T1.34 (AMSGRAD copy `cb_max_exp_avg_sq_out = tmp_cb_max_exp_avg_sq`):
```
CopyTile<tmp_cb_max_exp_avg_sq, Dst::D0, CopyTilePolicy::WaitAndPop>{}
+ PackTile<cb_max_exp_avg_sq_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
```
(Functionally identical to the existing `moreh_copy_chain` wrapper invocations elsewhere in the file — substitute via that wrapper.)

T1.35 / T1.36 (cb_tmp{1,2} = 1/(1 - cb_beta{2,1}_exponent) — single-acquire-window 2-stage compute, no held-CB):
```
BinaryFpu<cb_one, cb_beta2_exponent, cb_tmp1,
          BinaryFpuOp::Sub, BroadcastDim::None,
          BinaryDataFormatReconfig::InputAndOutput,
          CopyTilePolicy::NoWaitNoPop /* cb_one held */, CopyTilePolicy::WaitAndPop,
          CbIndexMode::FirstTile, Dst::D0>{}
+ Recip<Dst::D0>{}
+ PackTile<cb_tmp1, …>{}
```

(Audit confirmed at line 281: only `cb_reserve_back` precedes — no `cb_pop_front` on cb_tmp1 — so it is a single-acquire-window compute, no held-CB on tmp1. The cb_one is presumably allocated once at MAIN-top via `FillScalar(1.0)` upstream and held; `NoWaitNoPop` policy reflects that.)

**Risk assessment:**
- **Yellow tier — Power runtime-param dispatch.** T1.32 / T1.33 are the first production callsites of the chain-driven `Power<>` element with a runtime exponent member. Same dispatch family as UnaryNe (member-`exec(uint32_t)` routed through `eltwise_chain.inl:888`), but no production kernel exercises it today. Smoke-test under `--dev` to confirm.
- T1.34 (AMSGRAD copy) is gated on `#ifdef AMSGRAD` — verify both build configurations.
- moreh_adam.cpp has substantial held-CB / held-DEST raw blocks (lines 284–298, 302–315, 332–357, 360–374, 392–406) that stay raw (Type 2). The Type-1 migration leaves the surrounding scaffolding unchanged.
- DST: 1 slot per chain (Power and the Sub+Recip both live in slot 0). Bf16 8-tile capacity safely held.
- `compute_kernel_hw_startup` placement: kernel uses outer `binary_op_init_common` covering all chain calls (D5 row-5 of placement table) — no per-block hw_startup needed.
- Test isolation: `test_moreh_adam.py` (3 tests) and `test_moreh_adamw.py` (4 tests) are independent files; either kernel could fail without affecting the other. Bundling justified by parallel structure (both use the same bias-correction pattern).

**Acceptance criteria:**
- `scripts/run_safe_pytest.sh --dev --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_adam.py` — 3 tests passing under `--dev` (Power smoke).
- `scripts/run_safe_pytest.sh --dev --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_adamw.py` — 4 tests passing.
- Re-run both without `--dev` to confirm not relying on dev asserts.
- Risk tier: **Yellow** (`--dev` for Power smoke).

---

### Commit 7 — `eltwise sweep: migrate moreh_layer_norm_backward gamma_beta_grad cb_y mul block`

**Block IDs covered:** T1.24 (single block).

**Files touched:**
- `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp` — lines 191–213 (cb_y = cb_xmm * cb_rstd, no-mask path).

**Concrete change shape:**

T1.24 (cb_y = cb_xmm * cb_rstd with bcast):
```
BinaryFpu<cb_xmm, cb_rstd, cb_y,
          BinaryFpuOp::Mul,
          /* Bcast: */ is_lastdim_layernorm ? BroadcastDim::Cols : BroadcastDim::Scalar,
          BinaryDataFormatReconfig::InputAndOutput,
          CopyTilePolicy::WaitAndPop, CopyTilePolicy::WaitAndPop,
          CbIndexMode::FirstTile, Dst::D0>{}
+ PackTile<cb_y, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
```

The bcast-mode template parameter is selected at compile time on `is_lastdim_layernorm` (compile-time constant per the kernel's host-defined macro). If the constant is templated as a `bool`, use `if constexpr` to instantiate the chain.

**Risk assessment:**
- Single Type-1 block. Surrounding mask blocks (lines 82–113, 150–189) and held-CB blocks (lines 131–146, 237–268) stay raw (Type 2).
- Bcast templating: `BroadcastDim::Cols` vs `BroadcastDim::Scalar` selected on a host macro / compile-time bool. The kernel currently selects via `mul_tiles_bcast_cols` vs `mul_tiles_bcast_scalar` raw call — same template-arg axis. No new templating risk.
- DST: 1 slot.
- Isolation: `test_moreh_layer_norm.py` covers both forward (`test_moreh_layer_norm`) and backward (`test_moreh_layer_norm_backward`, `test_moreh_layer_norm_backward_with_gamma_or_beta`). 8 test rows total.

**Acceptance criteria:**
- `scripts/run_safe_pytest.sh --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py` — 8 tests passing.
- Risk tier: **Green** (single block, established bcast-templating pattern).

---

### Commit 8 — `eltwise sweep: migrate moreh_layer_norm_backward input_grad_small bcast blocks`

**Block IDs covered:** T1.22, T1.23 (2 blocks, single file).

**Files touched:**
- `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_small_kernel.cpp` — lines 140–159 (T1.22: cb_y mul bcast no mask), lines 377–398 (T1.23: cb_ndymdysum sub bcast).

**Concrete change shape:**

T1.22 (cb_y = cb_xmm * cb_rstd):
```
BinaryFpu<cb_xmm, cb_rstd, cb_y,
          BinaryFpuOp::Mul,
          is_lastdim_layernorm ? BroadcastDim::Cols : BroadcastDim::Scalar,
          BinaryDataFormatReconfig::InputAndOutput,
          CopyTilePolicy::WaitAndPop, CopyTilePolicy::WaitAndPop,
          CbIndexMode::FirstTile, Dst::D0>{}
+ PackTile<cb_y, …>{}
```

T1.23 (cb_ndymdysum = cb_ndy - cb_dysum):
```
BinaryFpu<cb_ndy, cb_dysum, cb_ndymdysum,
          BinaryFpuOp::Sub,
          is_lastdim_layernorm ? BroadcastDim::Cols : BroadcastDim::Scalar,
          BinaryDataFormatReconfig::InputAndOutput,
          CopyTilePolicy::WaitAndPop, CopyTilePolicy::WaitAndPop,
          CbIndexMode::FirstTile, Dst::D0>{}
+ PackTile<cb_ndymdysum, …>{}
```

**Risk assessment:**
- Yellow tier — bcast-mode `Cols` is exercised in commit 3's bcast paths but `Sub` + bcast (T1.23) is a less-common combination than `Mul` + bcast. Verify under `--dev` smoke.
- The kernel has multiple still-raw blocks (lines 101–138 mask, 162–211 mask-dy, 255–307 held-CB, 364–375 Q4-asymmetric, 400–419 Q4-asymmetric) staying raw per Type 2.
- Block T1.22 is structurally similar to commit 7's T1.24 (cb_y = mul bcast no mask) — pattern already proven by commit 7 landing first.
- DST: 1 slot per block.

**Acceptance criteria:**
- `scripts/run_safe_pytest.sh --dev --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py` — 8 tests passing under `--dev`.
- Re-run plain.
- Risk tier: **Yellow** (`--dev` for Sub+bcast smoke).

---

### Commit 9 — `eltwise sweep: migrate moreh_layer_norm_backward input_grad_large bcast blocks`

**Block IDs covered:** T1.18, T1.19, T1.20, T1.21 (4 blocks, single file).

**Files touched:**
- `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_large_kernel.cpp` — lines 77–98 (T1.18: cb_xmm sub bcast no mask), lines 451–477 (T1.19: cb_ndymdysum sub bcast), lines 514–540 (T1.20: cb_y mul bcast no mask), lines 542–562 (T1.21: cb_yydysum mul bcast both Pinned 0).

**Concrete change shape:**

T1.18 (cb_xmm = cb_x - cb_mean, bcast):
```
BinaryFpu<cb_x, cb_mean, cb_xmm,
          BinaryFpuOp::Sub,
          is_lastdim_layernorm ? BroadcastDim::Cols : BroadcastDim::Scalar,
          BinaryDataFormatReconfig::InputAndOutput,
          CopyTilePolicy::WaitAndPop, CopyTilePolicy::NoWaitNoPop /* mean held */,
          CbIndexMode::FirstTile, Dst::D0>{}
+ PackTile<cb_xmm, …>{}
```

T1.19 (cb_ndymdysum = cb_ndy - cb_dysum, bcast) — same shape as T1.23 of Commit 8.

T1.20 (cb_y = cb_xmm * cb_rstd, bcast no mask) — same shape as T1.22 of Commit 8.

T1.21 (cb_yydysum = cb_y * cb_ydysum, bcast, BOTH index 0 → uniform `Pinned`-with-0 ≡ FirstTile):
```
auto bin_elt = BinaryFpu<cb_y, cb_ydysum, cb_yydysum,
                          BinaryFpuOp::Mul,
                          is_lastdim_layernorm ? BroadcastDim::Cols : BroadcastDim::Scalar,
                          BinaryDataFormatReconfig::InputAndOutput,
                          CopyTilePolicy::WaitAndPop, CopyTilePolicy::WaitAndPop,
                          CbIndexMode::FirstTile, Dst::D0>{};
eltwise_chain(num_tiles, bin_elt, PackTile<cb_yydysum, …>{});
```

**Risk assessment:**
- **Yellow tier — multi-block plus T1.21 in-commit verification required.** The audit Section 6 hotspot list explicitly flags T1.21:

  > T1.21 `moreh_layer_norm_backward_input_grad_large_kernel.cpp` L542-562 — A-side needs `tile_idx=0` (Pinned) and B-side `tile_idx=0` (Pinned). Confirm the Wt-loop variable doesn't leak in. If the actual kernel uses `tile_idx=wt` for cb_y (loop var), this is Q4-asymmetric (Type 2). Verify via re-read at landing time.

  **Mandatory in-commit check:** before applying T1.21, the implementer must re-read the kernel at lines 542–562 and verify both A-side and B-side `tile_idx=0` (no `wt` loop variable in the index slot). If the re-read finds A-side uses runtime `wt`, drop T1.21 from this commit (Type 2 — defer to future ASYMMETRIC-INDEX helper extension), keeping T1.18/T1.19/T1.20.
- Multi-block in one file requires careful sequencing — apply T1.18 first (top of file), then T1.19/T1.20/T1.21 in source order.
- Bcast templating already smoke-tested by commits 7 and 8 landing first.
- DST: 1 slot per block.

**Acceptance criteria:**
- Pre-edit re-read of lines 542–562 to confirm T1.21 index uniformity (per audit hotspot).
- `scripts/run_safe_pytest.sh --dev --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py` — 8 tests passing under `--dev`.
- Re-run plain to confirm no `--dev`-only assert masks the result.
- Risk tier: **Yellow** (`--dev` + in-commit re-read).

---

## Section C — Sequencing & dependencies

### Order rationale

1. **Commit 1 (deepseek_grouped_gate scale)** — single block, smallest, lowest risk. First green commit establishes confidence in the sweep methodology.
2. **Commit 2 (moreh_norm_nc IS_ZERO)** — single block but Yellow due to UnaryNe runtime-param dispatch first-production-callsite verification. Land second so the verification result feeds into Commit 6's Yellow Power-dispatch verification (similar dispatch family).
3. **Commit 3 (moreh_softmax_backward x5)** — high-volume bundled commit, all Green (chain shapes well-exercised). Lands the first multi-file bundle.
4. **Commit 4 (moreh_softmax forward x3)** — same test cluster as Commit 3 (`test_moreh_softmax.py`). Lands on the green base from Commit 3.
5. **Commit 5 (moreh_nll_loss + backward)** — independent test cluster (`test_moreh_nll_loss.py`). Bundled together because the 4 step2 blocks (T1.28–T1.31) form a connected DEST handoff and must land atomically.
6. **Commit 6 (moreh_adam + moreh_adamw)** — Yellow tier for Power runtime-param dispatch verification. Independent test files (`test_moreh_adam.py`, `test_moreh_adamw.py`).
7. **Commit 7 (moreh_layer_norm_backward gamma_beta_grad)** — single block, Green, establishes the bcast-templating pattern for the layer_norm_backward family.
8. **Commit 8 (moreh_layer_norm_backward input_grad_small)** — Yellow, lands on Commit 7's green base. Verifies Sub+bcast.
9. **Commit 9 (moreh_layer_norm_backward input_grad_large)** — Yellow, multi-block, T1.21 in-commit re-read. Lands last.

### Hard dependencies

| Commit | Hard predecessor | Reason |
|--------|------------------|--------|
| 4 | 3 | Same test file (`test_moreh_softmax.py`). Bisect cleanliness — landing 3 first means a 4-failure points to forward kernels only. |
| 8 | 7 | Same test file (`test_moreh_layer_norm.py`); bcast-templating pattern smoke-tested by 7 first. |
| 9 | 8 | Same test file; multi-block on top of established 2-block pattern from 8. |

### Soft dependencies (recommended order, not strict)

| Commit | Soft predecessor | Reason |
|--------|------------------|--------|
| 6 | 2 | Both Yellow with runtime-param dispatch concern. Landing 2 first builds confidence in the dispatch path before exercising it again with `Power<>`. |

### Parallelizable pairs (different files, different test clusters; serial preferred for simpler bisect)

| Commit pair | Parallelizable | Note |
|-------------|----------------|------|
| 1 ↔ 2 | Yes | `test_deepseek_grouped_gate.py` vs `test_moreh_norm.py` — disjoint clusters. |
| 5 ↔ 7 | Yes | `test_moreh_nll_loss.py` vs `test_moreh_layer_norm.py` — disjoint. |
| 5 ↔ 6 | Yes | `test_moreh_nll_loss.py` vs `test_moreh_adam.py`/`test_moreh_adamw.py` — disjoint. |

The implementer may serialize all 9 commits for simpler bisect — recommend serial. Documenting parallelizable pairs is for emergency hot-fix scenarios.

### No block-level inter-commit dependencies

All Type-1 blocks within a single commit are file-isolated or contiguously located in the same file; **no block-N requires a prior block-M migration to be correct.** This is a property of Type-1 (mechanical rewrite — no shared-state across blocks at the helper level). The lone exception is T1.28–T1.31 in Commit 5, which form a connected DEST handoff and must land atomically — explicitly bundled in one commit.

### Files-touched matrix (overlap risk)

No two commits touch the same kernel file. Overlap risk = 0.

---

## Section D — Per-commit acceptance test plan

| # | Pytest command | Expected pass count | Tier flag |
|---|----------------|---------------------|-----------|
| 1 | `scripts/run_safe_pytest.sh --run-all tests/ttnn/nightly/unit_tests/operations/reduction/test_deepseek_grouped_gate.py` | 2 tests | Green |
| 2 | `scripts/run_safe_pytest.sh --dev --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_norm.py`; then re-run without `--dev` | 6 tests (each run) | Yellow |
| 3 | `scripts/run_safe_pytest.sh --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py` | 12 tests | Green |
| 4 | `scripts/run_safe_pytest.sh --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py` | 12 tests | Green |
| 5 | `scripts/run_safe_pytest.sh --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_nll_loss.py` | 6 tests | Green |
| 6 | `scripts/run_safe_pytest.sh --dev --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_adam.py` (3) and `test_moreh_adamw.py` (4); re-run plain | 7 tests total (each run) | Yellow |
| 7 | `scripts/run_safe_pytest.sh --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py` | 8 tests | Green |
| 8 | `scripts/run_safe_pytest.sh --dev --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py`; re-run plain | 8 tests (each run) | Yellow |
| 9 | `scripts/run_safe_pytest.sh --dev --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py`; re-run plain | 8 tests (each run) | Yellow |

### End-of-pipeline gate (after commit 9)

1. **Migrated-kernel pytest cluster** — run every test file in the union of pre-sweep B-classified kernels (15 of which are touched + 22 which are Type-2/Type-3 only and must continue passing untouched). Expected: same pass-count delta as baseline at `d1d1f9246c5`.
2. **401-test eltwise helper suite** — `scripts/run_safe_pytest.sh --run-all ttnn/cpp/ttnn/kernel_lib/tests/eltwise/test_eltwise.py`. Expected: 401 passes (no regressions; this sweep does not touch helper sources).
3. **Per-commit-2/6/8/9 retro confirmation** — re-run yellow commits without `--dev` to confirm no asserts mask failures.

---

## Section E — Misclassification note

### T1.17 (`moreh_norm_nc` IS_ZERO branch) is migratable

`migration_log.md` previously claimed UnaryNe runtime-param dispatch was a chain v1 GAP. Re-reading the helper sources at `astancov/eltwise_run7_refined` HEAD `d1d1f9246c5`:

- `eltwise_predicates.hpp:53-62` defines the macro `ELTWISE_DECLARE_UNARY_PARAM` that emits each runtime-param SFPU op (`UnaryEq`, `UnaryNe`, `UnaryGt`, `UnaryGe`, `UnaryLt`, `UnaryLe`) with a `param0` member field and an `exec(uint32_t /*i*/) const` body that calls `unary_<op>_tile(to_u32(Slot), param0)`.
- `eltwise_chain.inl:888,965` confirm the chain-side dispatcher routes `apply()` to the member-`exec(uint32_t)` path when an element exposes a non-static `exec(uint32_t)` — the same path used by `Power<>`'s runtime exponent and by the `_tile` SFPU calls that take a runtime param.

Therefore the GAP claim in `migration_log.md` is stale (from pre-run7 helper state), and T1.17 IS Type-1 migratable in this sweep. The sweep includes it as Commit 2, with Yellow-tier `--dev` smoke to verify the runtime-param path fires correctly under the chain v1 dispatcher in the IS_ZERO macro permutation.

If Commit 2's `--dev` smoke uncovers a runtime regression (unlikely given the dispatch-path verification), the disposition is to drop T1.17 from this sweep, document a freshly-discovered regression in `agents/migration_log.md`, and continue with Commit 3 onward. No other commit depends on T1.17.

---

## Section F — Out of scope (defer)

### File-level B kernels skipped from this sweep (22 of 47-file enumeration)

Per audit Section 6 "Kernels to skip entirely from this sweep":

- All 5 `binary_ng/kernels/*` (entries 1–5: `eltwise_binary_no_bcast`, `eltwise_binary_sfpu_no_bcast`, `eltwise_binary_sfpu_scalar`, `eltwise_where_sfpu`, `eltwise_where_sfpu_scalar`) — Type-2 (MACRO-INJECT, MULTI-DEST) + Type-3 (OUTER-FREQ-LOOP) only. No Type-1 blocks.
- All 6 `binary_ng/kernels_ng/*` (entries 6–11) — Type-2 only (activations + bcast-preamble).
- All 3 ternary `_bcast` (entries 12–14) — Type-2 only (OUTER-FREQ-LOOP).
- All 3 `bcast_to/*` (entries 15–17) — Type-2 only (MULTI-AXIS-EARLY-EXIT).
- Both `moe_compute` / `moe_gpt` (entries 18–19) — Type-3 (matmul + PACK-thread SFPU).
- `ssm_prefix_scan` (entry 21) — Type-3 (tilize/untilize).
- `moreh_clip_grad_norm_step1` (entry 24) — Type-2 only (MASK-INJECT + POWER-MULTISTAGE + HELD-CB).
- Both `moreh_layer_norm_{small,large}_kernel` (entries 25, 26) — Type-2 only (MASK-INJECT + MULTI-DEST + non-standard `_with_dt(…, is_lastdim_layernorm)` init).
- `moreh_mean_nc` (entry 30) — Type-2 only (RUNTIME-CB + HELD-CB).
- `moreh_sgd` (entry 39) — Type-2 only (RUNTIME-CB + RUNTIME-POP).
- `moreh_norm_h`, `moreh_norm_w` (entries 33, 35), `moreh_norm/ord_other/moreh_norm_h`, `moreh_norm/ord_other/moreh_norm_w` (entries 36, 38) — Type-2 only (MASK-INJECT + HELD-CB; ord_other variants also need POWER-MULTISTAGE).
- `moreh_norm_other` (entry 34) — Type-2 only (already-migrated chain blocks remain; rest blocked by POWER-MULTISTAGE + HELD-CB).

### Type-2 helper extensions explicitly deferred

The following 3 helper extensions, if they landed in a future helper-extension run, would unlock further migrations across these skipped kernels. **Not planned in this sweep.**

| Extension | Unlock count | Note |
|-----------|--------------|------|
| `MaskInject<COND, MaskCb, MaskIdx>` chain element | 17+ moreh kernels (all `moreh_norm_*`, `moreh_layer_norm_*`, `moreh_softmax_*` mid-loop mask, `moreh_clip_grad_norm_step1`, `moreh_softmax_backward_{h,h_large,w,w_large}` mask paths) | Single biggest unlock per audit Section 4. |
| `HeldCbBinaryFpu<…>` element with pop-then-push pack policy | 13+ moreh kernels (all `moreh_norm_*` add fold, `moreh_adam`/`moreh_adamw` recurrence on cb_tmp1/cb_tmp2, `moreh_layer_norm_*` xmm2sum/dyadd/ydyadd folds, `moreh_softmax_c_large` cb_max recurrence, `moreh_mean_nc` cb_intermed0 fold) | Second biggest. |
| `BinaryFpuPerTileScalarB<…>` with hardcoded `A=BlockIter, B=FirstTile` | 4+ kernels (`deepseek_grouped_gate::add_bias`, `eltwise_binary_scalar.cpp` no-act branch, `moreh_layer_norm_backward_input_grad_small` L364-375 / L400-419) | Recovery path for design v6 Q4-collapse regressions. |

### 2 Q4 regressed kernels stay raw-LLK per design v6 Section E

- `eltwise_binary_scalar.cpp` no-activations branch (lines 71–103).
- `deepseek_grouped_gate.cpp::add_bias` (lines 39–68).

These are the user-accepted cost of the v6 Q4 collapse (`AIndex/BIndex` → single `Index`). Recovery requires `BlockBinaryFpuScalarB<…>` / `BinaryFpuPerTileScalarB<…>` element types — explicitly out of scope for this sweep.

---

## Section G — Open questions (resolved)

### Q1 — Which kernels need `--dev` `static_assert` / `LLK_ASSERT` smoke runs before commit?

**Answer:** Yellow-tier commits (Commit 2, 6, 8, 9). Specifically:

| Commit | Reason for `--dev` |
|--------|---------------------|
| 2 (moreh_norm_nc IS_ZERO) | First production callsite of `UnaryNe` member-exec runtime-param dispatch under the chain v1 dispatcher. |
| 6 (moreh_adam / moreh_adamw) | First production callsites of `Power<>` member-exec runtime-exponent dispatch (T1.32, T1.33). |
| 8 (moreh_layer_norm_backward input_grad_small) | Sub+bcast (T1.23) is a less-common chain shape than Mul+bcast; `--dev` smoke confirms `BinaryFpuOp::Sub` + `BroadcastDim::Cols/Scalar` interaction. |
| 9 (moreh_layer_norm_backward input_grad_large) | Multi-block (4 blocks) plus T1.21 in-commit re-read requirement. `--dev` smoke catches any block-interaction regression. |

Green-tier commits (1, 3, 4, 5, 7) run plain. Their chain shapes are well-exercised by existing Class A / Class E kernels (verified via coverage_audit Section 4 per-feature usage histogram — `BinaryFpu`, `CopyTile`, `PackTile`, `Negative`, `Exp`, `Recip`, `Log` all in the top-tier usage band).

### Q2 — Should this sweep introduce new test rows in `test_eltwise.py` to cover newly-migrated patterns?

**Answer: NO.** The migrated kernels are already covered by their own production pytests (`test_moreh_*.py`, `test_deepseek_grouped_gate.py`). Helper-side coverage is already comprehensive in `test_eltwise.py` (401 tests) — every chain element used in this sweep (`CopyTile`, `BinaryFpu`, `PackTile`, `Negative`, `Exp`, `Recip`, `Log`, `UnaryNe`, `Power`, `OptionalChainElement`) has dedicated test rows per coverage_audit Section 4.

**Exception:** if Commit 2 (UnaryNe) or Commit 6 (Power) Yellow `--dev` smoke uncovers a dispatch regression, the implementer adds a new chain test row to cover the failing dispatch path before fixing forward. This is a contingency, not a default.

### Q3 — Sequencing of moreh_softmax_backward family — single bundled commit vs split per kernel?

**Answer: SINGLE BUNDLED COMMIT (Commit 3).** All 5 backward kernels (`_c_large`, `_h`, `_h_large`, `_w`, `_w_large`) share `test_moreh_softmax.py` (the audit incorrectly cited a `test_moreh_softmax_backward.py` — verified at landing time the correct path is `test_moreh_softmax.py` covering both forward and backward). 10 blocks, 5 files, mechanical chain-shape rewrites — bundling matches audit Section 6 recommendation and gives a single test invocation to gate the 10 blocks. Splitting per-kernel would inflate to 5 commits with the same test file each — net negative for bisect ergonomics.

The same logic applies to `moreh_softmax` forward (Commit 4) — bundle 3 kernels in one commit.

---

## Final ordered commit list (recap)

| # | Subject | Block IDs | Risk |
|---|---------|-----------|------|
| 1 | `eltwise sweep: migrate deepseek_grouped_gate scale block` | T1.37 | Green |
| 2 | `eltwise sweep: migrate moreh_norm/ord_other/moreh_norm_nc IS_ZERO branch` | T1.17 | Yellow |
| 3 | `eltwise sweep: migrate moreh_softmax_backward {c,h,h_large,w,w_large} Type-1 blocks` | T1.01–T1.10 | Green |
| 4 | `eltwise sweep: migrate moreh_softmax forward {c,h,w}_large exp/rexp/log/recip blocks` | T1.11–T1.16 | Green |
| 5 | `eltwise sweep: migrate moreh_nll_loss / moreh_nll_loss_backward Type-1 blocks` | T1.25–T1.31 | Green |
| 6 | `eltwise sweep: migrate moreh_adam / moreh_adamw bias-correction Type-1 blocks` | T1.32–T1.36 | Yellow |
| 7 | `eltwise sweep: migrate moreh_layer_norm_backward gamma_beta_grad cb_y mul block` | T1.24 | Green |
| 8 | `eltwise sweep: migrate moreh_layer_norm_backward input_grad_small bcast blocks` | T1.22, T1.23 | Yellow |
| 9 | `eltwise sweep: migrate moreh_layer_norm_backward input_grad_large bcast blocks` | T1.18–T1.21 | Yellow |

**Total: 9 commits, 37 Type-1 blocks, 15 distinct kernels.** Per-commit gate: touched-kernel pytest cluster green. End-of-pipeline gate: full ~74 migrated-kernel pytest sweep + 401-test eltwise suite green.
