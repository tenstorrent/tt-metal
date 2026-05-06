# Eltwise v2 Test Plan — Phase 3.5 / Gate 2

PATTERNS_HEADER: file	line	function	category	heavy_lifting	variant	loop_depth	loop_vars	sig	arg0	arg1	arg2	flow	sync_bucket	sync_seq	sync_style	shape	region_stats

Source: Gate 1-approved proposal `eltwise_v2_helper_proposal.md`.

Status: **proposal**. Gate 2 is BLOCKING — no kernel `.cpp`, no pytest `.py` lands until the user approves this plan. Test changes (add / remove / skip / retitle / change tolerance / change parameterization / change dtype matrix / mark XFAIL↔PASS) all require an explicit approval round per HQ rule.

---

## 0. Conventions

### 0.1 Test home

```
ttnn/cpp/ttnn/kernel_lib/tests/eltwise/
  kernels/                          # device-side compute kernels (one .cpp per test row)
  reader_writer/                    # shared dataflow kernels
test/ttnn/unit_tests/kernel_lib/
  test_eltwise.py                   # the pytest entry point
```

(Per Q5: shipped helper uses bare `eltwise_*` prefix; the test path matches.)

### 0.2 Per-row contract

Every row in this plan is one device test kernel + one pytest parameterization. Each pytest assertion checks both:

1. **Numeric match** vs. a torch golden via `comp_pcc(out, golden) >= threshold`.
2. **No hang** — wrapped in `scripts/run_safe_pytest.sh` (built-in dispatch-timeout detection + tt-triage). No outer `timeout`.

### 0.3 Default parameterization

Unless the row says otherwise:

| Axis | Values | Reason |
|---|---|---|
| `num_tiles` | `{1, 8, 64}` | single tile / fits in DEST / spans multiple DEST windows |
| `fp32_dest_acc_en` | `{False, True}` | mandatory dtype-matrix per HQ Step 4 |
| `dtype_in / dtype_out` | `bfloat16` baseline; mixed combos per row | mirrors program-factory dispatch tables |
| PCC threshold | `>= 0.9999` for bf16-only; `>= 0.999` if fp32 mixed | accommodates compounded ULPs |
| Tile shape | `(32, 32)` | single-tile baseline; multi-tile rows scale via `num_tiles` |
| Arch | Wormhole-B0; Blackhole skipped unless explicitly Blackhole-tested | matches existing kernel_lib suite |

When a row drops or extends an axis, the row notes the deviation and the rationale.

### 0.4 Naming

Test ids follow `<group>_<scenario>_<num_tiles>_<dtype>_<fp32acc>` so failures triage by id alone. Kernel files mirror the test id: `kernels/<group>_<scenario>.cpp`.

### 0.5 Tolerance policy

- bf16 unary SFPU (Exp, Sqrt, Tanh, Sigmoid, ...) on `[-3, 3]`: PCC `>= 0.9999`, atol `<= 1e-1`, rtol `<= 5e-2`.
- fp32 unary SFPU on same range: PCC `>= 0.99999`, atol `<= 1e-3`, rtol `<= 1e-3`.
- bf16 binary FPU (add/sub/mul): PCC `>= 0.9999`, atol `<= 1e-2`.
- bf16 × fp32 mixed binary FPU: PCC `>= 0.999`, atol `<= 1e-1`.
- Integer ops (bitwise, FillInt, where-select on bool mask): exact equality (PCC undefined for non-float; use `torch.allclose(rtol=0, atol=0)` or set-membership).

Any row that loosens these thresholds writes its rationale in the row's notes column.

### 0.6 Skip rationale

Allowed skip reasons (all others must run):

- `arch=blackhole` — the path is not yet Blackhole-tested.
- `gap=GAP-XYZ` — covered in §9.3 of the proposal as deferred.
- `static_assert_only` — the row exercises a compile-time check; compiles in the negative test harness, never runs on device.

Skipping for "flaky" or "slow" is **not** an allowed reason.

### 0.7 Static-assert (compile-time-only) rows

Rows tagged `static_assert_only` ship as separate translation units in `tests/eltwise/static_assert_negative/`. Each is a `.cpp` that should fail to compile with a specific diagnostic substring. The pytest harness greps the build log for the diagnostic and asserts the file failed to build with the expected message. No runtime; no PCC; no num_tiles axis.

### 0.8 Test groups (table of contents)

| § | Group | Rows | Focus |
|---|---|---:|---|
| 1 | CopyTile lifecycle matrix | 17 | every `(CopyTilePolicy × CbIndexMode)` legal + illegal cell |
| 2 | CopyTile reconfig | 6 | None / Input × dtype combos |
| 3 | SFPU unary by family | 18 | one row per catalog file family + approx variants |
| 4 | SFPU binary / ternary / quaternary | 8 | mask, where, sfpu_binary_bcast, fused 4-input |
| 5 | BinaryFpu single-op matrix | 24 | add / sub / mul × dtype × reconfig × A/B policy |
| 6 | BinaryFpu broadcast | 12 | ROW / COL / SCALAR × dtype × bcast init variants |
| 7 | BinaryFpu same-CB dedup | 4 | square / mul-by-self lifecycle |
| 8 | DestReuseBinary | 12 | DEST_TO_SRCA / DEST_TO_SRCB × reconfig × index |
| 9 | UnaryBcast | 9 | ROW / COL / SCALAR × policy × reconfig |
| 10 | Fill / Rand | 7 | FillScalar (f32/f16/int), FillBitcast, RandTile |
| 11 | PackTile lifecycle matrix | 16 | every `(PackTilePolicy × PackTileIndexMode)` legal + illegal cell |
| 12 | PackTile reconfig | 6 | None / Output / OutputConditional × dtype combos |
| 13 | PackTileBlock | 4 | atomic multi-slot pack |
| 14 | Multi-element chains | 14 | CopyTile + SFPU + PackTile; 2-CB binary; binary + post-SFPU; etc. |
| 15 | Fan-out | 6 | N CopyTile (WaitNoPop / NoWaitPop) + N PackTile in one window |
| 16 | Hoist-safety | 6 | hoist-safe shape; hoist-unsafe disqualification |
| 17 | Compile-time invariants | 12 | static_assert negative tests |
| 18 | Convenience entry points | 9 | binary_add / sub / mul / *_bcast / unary_op / dest_reuse_mul / copy / copy_with_dt |
| 19 | Mixed-dtype FP32 acc sweep | 18 | every binary path × `fp32_dest_acc_en={F,T}` × bf16 × fp32 × bfloat8_b × int32 |
| 20 | Migration regression set | 30+ | per TSV tier (see §20) — one canonical kernel per migrated production op, side-by-side helper vs. raw |
| 21 | Stress / boundary | 8 | DEST_AUTO_LIMIT, large num_tiles, edge num_tiles=DEST_AUTO_LIMIT-1 |
| 22 | Migration prerequisite (§9.2) | 6 | dst-sync rewrite verifications for Tier D / Tier E sample kernels |

**Total: ~252 device kernels + pytest rows + 12 compile-time-only rows.** Per parameterization (num_tiles × fp32_dest_acc × dtype) the pytest expands to several thousand cases — that is intentional; the cost is dominated by JIT compile, not runtime.

The full kernel-by-kernel table follows. Every row lists: id / what / kernel file / num_tiles / dtypes / fp32_dest_acc / PCC / notes.

---

## 1. CopyTile lifecycle matrix (17 rows)

Goal: prove every `(CopyTilePolicy × CbIndexMode)` legal cell from lessons §2.7 works, and every illegal cell fails to compile. Streaming chain shape: `CopyTile<cb_in, Dst::D0, P, IM> → Exp{} → PackTile<cb_out, Dst::D0, PerTileReserveAndPush>`.

| id | policy | index | num_tiles | dtypes | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|---|
| `1.1.WaitAndPop_FirstTile`        | `WaitAndPop`          | `FirstTile` | {1,8,64} | bf16 | {F,T} | 0.9999 | streaming default |
| `1.2.WaitAndPop_PinnedZero`       | `WaitAndPop`          | `Pinned k=0` | {1,8,64} | bf16 | {F,T} | 0.9999 | k=0 alias of FirstTile |
| `1.3.WaitNoPop_FirstTile`         | `WaitNoPop`           | `FirstTile` | {1,8,64} | bf16 | {F,T} | 0.9999 | tile 0 persistent across iters |
| `1.4.WaitNoPop_PinnedZero`        | `WaitNoPop`           | `Pinned k=0` | {1,8,64} | bf16 | {F,T} | 0.9999 |  |
| `1.5.NoWaitPop_FirstTile`         | `NoWaitPop`           | `FirstTile` | {1,8,64} | bf16 | {F,T} | 0.9999 | reader pre-pushes |
| `1.6.NoWaitPop_PinnedZero`        | `NoWaitPop`           | `Pinned k=0` | {1,8,64} | bf16 | {F,T} | 0.9999 |  |
| `1.7.NoWaitNoPop_FirstTile`       | `NoWaitNoPop`         | `FirstTile` | {1,8,64} | bf16 | {F,T} | 0.9999 | caller-managed |
| `1.8.NoWaitNoPop_BlockIter`       | `NoWaitNoPop`         | `BlockIter` | {1,8,64} | bf16 | {F,T} | 0.9999 | caller pre-waits N |
| `1.9.NoWaitNoPop_PinnedNonZero`   | `NoWaitNoPop`         | `Pinned k=3` | {8,64}   | bf16 | {F,T} | 0.9999 | num_tiles=1 dropped — k=3 invalid |
| `1.10.NoWaitNoPop_Absolute`       | `NoWaitNoPop`         | `Absolute idx` | {8,64} | bf16 | {F,T} | 0.9999 | runtime idx |
| `1.11.WaitUpfrontPop_FirstTile`   | `WaitUpfrontPopAtEnd` | `FirstTile` | {1,8,64} | bf16 | {F,T} | 0.9999 | block of N, read tile 0 each iter |
| `1.12.WaitUpfrontPop_BlockIter`   | `WaitUpfrontPopAtEnd` | `BlockIter` | {1,8,64} | bf16 | {F,T} | 0.9999 | canonical "block consume" |
| `1.13.WaitUpfrontPop_PinnedNonZero` | `WaitUpfrontPopAtEnd` | `Pinned k=2` | {8,64} | bf16 | {F,T} | 0.9999 | num_tiles=1 dropped |
| `1.14.WaitUpfrontPop_Absolute`    | `WaitUpfrontPopAtEnd` | `Absolute idx` | {8,64} | bf16 | {F,T} | 0.9999 | runtime idx ∈ window |
| `1.15.WaitAndPop_BlockIter_negative`  | `WaitAndPop` | `BlockIter` | n/a | n/a | n/a | n/a | `static_assert_only` — must fail to compile |
| `1.16.WaitNoPop_BlockIter_negative`   | `WaitNoPop`  | `BlockIter` | n/a | n/a | n/a | n/a | `static_assert_only` |
| `1.17.NoWaitPop_Absolute_negative`    | `NoWaitPop`  | `Absolute`  | n/a | n/a | n/a | n/a | `static_assert_only` |

---

## 2. CopyTile reconfig (6 rows)

Goal: prove `CopyTileReconfig::{None, Input}` correctly emits / suppresses `copy_tile_to_dst_init_short_with_dt` paths, including srcA dtype change between consecutive chain invocations.

| id | reconfig | dtype_in_first | dtype_in_second | num_tiles | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|---|
| `2.1.None_uniform`     | `None`  | bf16 | bf16 | {1,8,64} | {F,T} | 0.9999 | sanity |
| `2.2.Input_bf16_to_fp32` | `Input` | bf16 | fp32 | {1,8,64} | {F,T} | 0.999  | mid-test reconfig — two CopyTile invocations on different CBs |
| `2.3.Input_fp32_to_bf16` | `Input` | fp32 | bf16 | {1,8,64} | {F,T} | 0.999  |  |
| `2.4.Input_bf16_to_bfp8b` | `Input` | bf16 | bfloat8_b | {1,8,64} | {F,T} | 0.999  | packed format src |
| `2.5.Input_bf16_to_int32` | `Input` | bf16 | int32 | {1,8,64} | {F,T} | exact | int copy |
| `2.6.None_fp32_dest_acc_off` | `None` | bf16 | bf16 | {1,8,64} | {F} | 0.9999 | confirm no implicit reconfig under fp32_dest_acc=False |

---

## 3. SFPU unary by family (18 rows)

Goal: every SFPU op-family header in the catalog has at least one chain test exercising the typical `init + tile` pattern. Chain shape: `CopyTile<bf16> → <SfpuOp>{} → PackTile<bf16>`.

| id | family | op | template params | num_tiles | dtype | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|---|---|
| `3.1.activations.Relu`      | activations | `Relu`     | default       | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `3.2.activations.Sigmoid`   | activations | `Sigmoid`  | `Approx::Fast`| {1,8,64} | bf16 | {F,T} | 0.999  | fast variant tolerance |
| `3.3.activations.Sigmoid_exact` | activations | `Sigmoid` | `Approx::Exact` | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `3.4.activations.Gelu`      | activations | `Gelu`     | default       | {1,8,64} | bf16 | {F,T} | 0.999  | |
| `3.5.math.Exp`              | math | `Exp`      | `Approx::Exact` | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `3.6.math.Exp_fast`         | math | `Exp`      | `Approx::Fast`  | {1,8,64} | bf16 | {F,T} | 0.999  | |
| `3.7.math.Log`              | math | `Log`      | default       | {1,8,64} | bf16 | {F,T} | 0.9999 | input clamped to (0, 100] |
| `3.8.math.Sqrt`             | math | `Sqrt`     | default       | {1,8,64} | bf16 | {F,T} | 0.9999 | input ≥ 0 |
| `3.9.math.Rsqrt_fast`       | math | `Rsqrt`    | `Legacy::Off, Approx::Fast` | {1,8,64} | bf16 | {F,T} | 0.999  | |
| `3.10.math.Rsqrt_legacy`    | math | `Rsqrt`    | `Legacy::On, Approx::Exact`  | {1,8,64} | bf16 | {F,T} | 0.9999 | legacy enum path |
| `3.11.trig.Sin`             | trig | `Sin`      | default       | {1,8,64} | bf16 | {F,T} | 0.9999 | range [-π, π] |
| `3.12.trig.Cos`             | trig | `Cos`      | default       | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `3.13.trig.Tanh`            | trig | `Tanh`     | default       | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `3.14.rounding.Floor`       | rounding | `Floor` | default       | {1,8,64} | bf16 | {F,T} | exact-int | output should be integral floats |
| `3.15.predicates.Eqz`       | predicates | `Eqz` | default       | {1,8,64} | bf16 | {F,T} | exact | output 0/1 |
| `3.16.scalar.Power_runtime` | scalar | `Power` | runtime exponent | {1,8,64} | bf16 | {F,T} | 0.999 | exponent passed via ctor (runtime field) |
| `3.17.special.Erf`          | special | `Erf` | default       | {1,8,64} | bf16 | {F,T} | 0.999  | |
| `3.18.misc.Tanh_packthread` | misc | `TanhPackthread` | default | {1,8,64} | bf16 | {F,T} | 0.9999 | pack-thread variant |

---

## 4. SFPU binary / ternary / quaternary (8 rows)

| id | op | base | num_tiles | dtype | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|---|
| `4.1.Mask_bf16`     | `Mask<DataFormat::Float16_b, Dst::D0>`   | `BinaryOp` | {1,8,64} | bf16 | {F,T} | 0.9999 | mask in Dst::D1 (LLK contract baked) |
| `4.2.Mask_fp32`     | `Mask<DataFormat::Float32,   Dst::D0>`   | `BinaryOp` | {1,8,64} | fp32 | {F,T} | 0.9999 | |
| `4.3.Where`         | `Where<DataFormat::Float16_b, ...>`      | `TernaryOp`| {1,8,64} | bf16 | {F,T} | 0.9999 | cond/true/false in 3 slots |
| `4.4.AddBinary_DEST` | `AddBinary` (DEST_TO_DEST)              | `BinaryOp` | {1,8,64} | bf16 | {F,T} | 0.9999 | both operands DEST |
| `4.5.MulBinary_DEST`| `MulBinary` (DEST_TO_DEST)               | `BinaryOp` | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `4.6.SfpuBinaryBcast_row` | `SfpuBinaryBcast<Add, ROW>`         | (struct)   | {1,8,64} | bf16 | {F,T} | 0.9999 | sfpu-side bcast |
| `4.7.MaskedFusedQuaternary`| `Mask + Scale + Bias` 4-input fused | `QuaternaryOp` | {1,8,64} | bf16 | {F,T} | 0.999 | exercises 4 input slots |
| `4.8.WhereWithScalar`| `Where` with FillScalar one branch    | mixed     | {1,8,64} | bf16 | {F,T} | 0.9999 | tests fill+ternary chain |

---

## 5. BinaryFpu single-op matrix (24 rows)

Goal: cover every (op × dtype × reconfig × A/B policy) tuple that the helper claims to support. Chain shape: `CopyTile<cbA> + CopyTile<cbB> + BinaryFpu{} + PackTile`.

Sub-matrix axes:
- op ∈ {Add, Sub, Mul}
- A_policy ∈ {WaitAndPop, WaitNoPop, NoWaitPop, NoWaitNoPop, WaitUpfrontPopAtEnd}
- B_policy ∈ same
- reconfig ∈ {NONE, INPUT, OUTPUT, INPUT_AND_OUTPUT}

Full cartesian product is huge; the 24 selected rows cover the dominant cells:

| id | op | A_policy | B_policy | reconfig | num_tiles | dtypes (A/B/out) | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|---|---|---|
| `5.1.Add_PerTile_PerTile_None` | Add | `WaitAndPop` | `WaitAndPop` | `NONE` | {1,8,64} | bf16/bf16/bf16 | {F,T} | 0.9999 | streaming default |
| `5.2.Sub_PerTile_PerTile_None` | Sub | `WaitAndPop` | `WaitAndPop` | `NONE` | {1,8,64} | bf16/bf16/bf16 | {F,T} | 0.9999 | |
| `5.3.Mul_PerTile_PerTile_None` | Mul | `WaitAndPop` | `WaitAndPop` | `NONE` | {1,8,64} | bf16/bf16/bf16 | {F,T} | 0.9999 | |
| `5.4.Add_AStream_BPersist`     | Add | `WaitAndPop` | `WaitNoPop` | `NONE` | {1,8,64} | bf16/bf16/bf16 | {F,T} | 0.9999 | A streams, B persists (scaler-style) |
| `5.5.Mul_AStream_BPersist`     | Mul | `WaitAndPop` | `WaitNoPop` | `NONE` | {1,8,64} | bf16/bf16/bf16 | {F,T} | 0.9999 | |
| `5.6.Add_AStream_BFanLast`     | Add | `WaitAndPop` | `NoWaitPop` | `NONE` | {1,8,64} | bf16/bf16/bf16 | {F,T} | 0.9999 | second consumer of pre-waited B |
| `5.7.Add_NoWaitNoPop_both`     | Add | `NoWaitNoPop` | `NoWaitNoPop` | `NONE` | {1,8,64} | bf16/bf16/bf16 | {F,T} | 0.9999 | sharded |
| `5.8.Add_UpfrontBoth_BlockIter` | Add | `WaitUpfrontPopAtEnd` | `WaitUpfrontPopAtEnd` | `NONE` | {1,8,64} | bf16/bf16/bf16 | {F,T} | 0.9999 | block A,B with BlockIter index on both |
| `5.9.Mul_UpfrontA_PinnedB`     | Mul | `WaitUpfrontPopAtEnd` | `WaitUpfrontPopAtEnd` | `NONE` | {8,64} | bf16/bf16/bf16 | {F,T} | 0.9999 | A=BlockIter, B=Pinned k=0 (block × scalar) |
| `5.10.Add_INPUT_reconfig`      | Add | `WaitAndPop` | `WaitAndPop` | `INPUT` | {1,8,64} | fp32/bf16/bf16 | {F,T} | 0.999  | srcA dtype change requires INPUT reconfig |
| `5.11.Add_INPUT_srcB_only`     | Add | `WaitAndPop` | `WaitAndPop` | `INPUT` | {1,8,64} | bf16/fp32/bf16 | {F,T} | 0.999  | lessons §7.1 — srcB-only |
| `5.12.Add_INPUT_both_change`   | Add | `WaitAndPop` | `WaitAndPop` | `INPUT` | {1,8,64} | fp32/fp32/bf16 | {F,T} | 0.999  | both srcA + srcB dtype change |
| `5.13.Add_OUTPUT_reconfig`     | Add | `WaitAndPop` | `WaitAndPop` | `OUTPUT` | {1,8,64} | bf16/bf16/fp32 | {F,T} | 0.999  | output dtype change |
| `5.14.Add_INPUT_AND_OUTPUT`    | Add | `WaitAndPop` | `WaitAndPop` | `INPUT_AND_OUTPUT` | {1,8,64} | fp32/fp32/bf16 | {F,T} | 0.999  | safe default |
| `5.15.Mul_bfp8b_inputs`        | Mul | `WaitAndPop` | `WaitAndPop` | `INPUT` | {1,8,64} | bfloat8_b/bfloat8_b/bf16 | {F,T} | 0.999 | packed format inputs |
| `5.16.Add_int32_inputs`        | Add | `WaitAndPop` | `WaitAndPop` | `NONE` | {1,8,64} | int32/int32/int32 | {F} | exact | int path |
| `5.17.Mul_fp32_dest_acc_only_True` | Mul | `WaitAndPop` | `WaitAndPop` | `INPUT_AND_OUTPUT` | {1,8,64} | bf16/bf16/bf16 | {T} | 0.9999 | exercises fp32_dest_acc path explicitly |
| `5.18.Add_HoistAcquireRelease`     | Add | `WaitAndPop` | `WaitAndPop` | `NONE` | {1,8,64} | bf16/bf16/bf16 | {F,T} | 0.9999 | `BinaryFpuOutputPolicy::HoistAcquireRelease` opt-in |
| `5.19.Add_PerTile_default_vs_hoist_equality` | Add | `WaitAndPop` | `WaitAndPop` | `NONE` | {1,8,64} | bf16/bf16/bf16 | {F,T} | bitwise-equal | side-by-side: default vs hoisted yield identical bytes |
| `5.20.Add_LongChainSurrounding` | Add (within longer chain) | `WaitAndPop` | `WaitAndPop` | `NONE` | {1,8,64} | bf16/bf16/bf16 | {F,T} | 0.9999 | binary embedded inside a 4-element chain |
| `5.21.Sub_AStream_BFanFirst`   | Sub | `WaitNoPop` | `WaitAndPop` | `NONE` | {1,8,64} | bf16/bf16/bf16 | {F,T} | 0.9999 | A persists, B streams |
| `5.22.Add_AbsoluteIndex_window`| Add | `WaitUpfrontPopAtEnd` | `WaitUpfrontPopAtEnd` | `NONE` | {8,64} | bf16/bf16/bf16 | {F,T} | 0.9999 | A=Absolute, B=Absolute (gather-style) |
| `5.23.Mul_NoWaitNoPop_BlockIter_caller_managed` | Mul | `NoWaitNoPop` | `NoWaitNoPop` | `NONE` | {8,64} | bf16/bf16/bf16 | {F,T} | 0.9999 | caller pre-waits |
| `5.24.Add_PerTile_with_pack_OUTPUT_reconfig_repeated` | Add | `WaitAndPop` | `WaitAndPop` | `OUTPUT` | {1,8,64} | bf16/bf16/bf16 | {F,T} | 0.9999 | repeats chain twice with output dtype change between calls |

---

## 6. BinaryFpu broadcast (12 rows)

| id | op | bcast | A_policy | B_policy | reconfig | num_tiles | dtypes | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `6.1.AddBcast_ROW`     | Add | `ROW`    | `WaitAndPop` | `WaitNoPop` | `NONE` | {1,8,64} | bf16 | {F,T} | 0.9999 | row bcast (B is row tile) |
| `6.2.AddBcast_COL`     | Add | `COL`    | `WaitAndPop` | `WaitNoPop` | `NONE` | {1,8,64} | bf16 | {F,T} | 0.9999 | col bcast |
| `6.3.AddBcast_SCALAR`  | Add | `SCALAR` | `WaitAndPop` | `WaitNoPop` | `NONE` | {1,8,64} | bf16 | {F,T} | 0.9999 | scalar bcast |
| `6.4.MulBcast_ROW`     | Mul | `ROW`    | `WaitAndPop` | `WaitNoPop` | `NONE` | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `6.5.MulBcast_COL`     | Mul | `COL`    | `WaitAndPop` | `WaitNoPop` | `NONE` | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `6.6.MulBcast_SCALAR`  | Mul | `SCALAR` | `WaitAndPop` | `WaitNoPop` | `NONE` | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `6.7.SubBcast_ROW`     | Sub | `ROW`    | `WaitAndPop` | `WaitNoPop` | `NONE` | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `6.8.SubBcast_COL`     | Sub | `COL`    | `WaitAndPop` | `WaitNoPop` | `NONE` | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `6.9.SubBcast_SCALAR`  | Sub | `SCALAR` | `WaitAndPop` | `WaitNoPop` | `NONE` | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `6.10.AddBcast_ROW_INPUT_reconfig` | Add | `ROW` | `WaitAndPop` | `WaitNoPop` | `INPUT` | {1,8,64} | bf16 in / fp32 out → bf16 in | {F,T} | 0.999  | mid-test dtype change |
| `6.11.AddBcast_NONE_when_no_bcast` | Add | `NONE` | `WaitAndPop` | `WaitAndPop` | `NONE` | {1,8,64} | bf16 | {F,T} | 0.9999 | confirms `NONE` enum value works |
| `6.12.AddBcast_SCALAR_fp32_dest_acc_True_required` | Add | `SCALAR` | `WaitAndPop` | `WaitNoPop` | `NONE` | {1,8,64} | bf16/bf16/fp32 | {T} | 0.9999 | confirms FP32_DEST_ACC `_with_dt` gating |

---

## 7. BinaryFpu same-CB dedup (4 rows)

Goal: passing `cb_a == cb_b` deduplicates `cb_wait_front` / `cb_pop_front` exactly once per iteration. Broken dedup deadlocks (double wait) or skips tiles (double pop).

| id | scenario | num_tiles | dtype | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|
| `7.1.SquareViaSameCBMul` | `Mul(cb_x, cb_x, cb_out)` per-tile | {1,8,64} | bf16 | {F,T} | 0.9999 | golden = x² |
| `7.2.SquareSubSameCB`    | `Sub(cb_x, cb_x, cb_out)` should yield zeros | {1,8,64} | bf16 | {F,T} | exact | golden = 0; double-pop would skip tiles |
| `7.3.SquareAddSameCB_HoistAcquireRelease` | `Add` same-CB, hoist | {1,8,64} | bf16 | {F,T} | 0.9999 | hoist + dedup compose |
| `7.4.SquareSameCBSmallCBCapacity` | square with CB capacity = 1 | {64} | bf16 | {F} | 0.9999 | smallest capacity that still works under dedup; would deadlock if double-wait |

---

## 8. DestReuseBinary (12 rows)

Goal: `binary_dest_reuse_tiles` — `DEST_TO_SRCA` and `DEST_TO_SRCB` both, with reconfig coverage. FPU-clash reinit verified.

| id | op | reuse | reconfig | index | num_tiles | dtypes | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|---|---|---|
| `8.1.AddReuseSrcA_None_FirstTile`  | Add | `DEST_TO_SRCA` | `None`  | `FirstTile`  | {1,8,64} | bf16 | {F,T} | 0.9999 | streaming |
| `8.2.AddReuseSrcA_Input_FirstTile` | Add | `DEST_TO_SRCA` | `Input` | `FirstTile`  | {1,8,64} | bf16 | {F,T} | 0.999  | CB-side dtype change |
| `8.3.AddReuseSrcB_None_FirstTile`  | Add | `DEST_TO_SRCB` | `None`  | `FirstTile`  | {1,8,64} | bf16 | {F,T} | 0.9999 | exercises srcB path (lessons §7.1) |
| `8.4.AddReuseSrcB_Input_FirstTile` | Add | `DEST_TO_SRCB` | `Input` | `FirstTile`  | {1,8,64} | bf16 | {F,T} | 0.999  | srcB reconfig |
| `8.5.MulReuseSrcA_None_FirstTile`  | Mul | `DEST_TO_SRCA` | `None`  | `FirstTile`  | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `8.6.MulReuseSrcA_Input_BlockIter` | Mul | `DEST_TO_SRCA` | `Input` | `BlockIter`  | {1,8,64} | bf16 | {F,T} | 0.999  | upfront block + reuse |
| `8.7.MulReuseSrcB_None_BlockIter`  | Mul | `DEST_TO_SRCB` | `None`  | `BlockIter`  | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `8.8.SubReuseSrcA_None_PinnedZero` | Sub | `DEST_TO_SRCA` | `None`  | `Pinned k=0` | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `8.9.AddReuseSrcA_FpClashReinit`   | Add | `DEST_TO_SRCA` | `None`  | `FirstTile`  | {1,8,64} | bf16 | {F,T} | 0.9999 | chain: `CopyTile + DestReuseBinary` — `clashes_with_fpu=true` forces per-tile reinit |
| `8.10.AddReuseSrcA_HoistDisallowed`| Add | `DEST_TO_SRCA` | `None`  | `FirstTile`  | n/a      | n/a   | n/a   | n/a    | `static_assert_only` — chain has DestReuseBinary, not hoist-safe |
| `8.11.AddReuseSrcA_INPUT_AND_OUTPUT_via_BinaryFpu_then_DestReuse` | Add then Mul-reuse | mixed | `INPUT_AND_OUTPUT` | `FirstTile` | {1,8,64} | bf16 | {F,T} | 0.999 | chained binary + dest-reuse |
| `8.12.AddReuseSrcA_int32`          | Add | `DEST_TO_SRCA` | `None`  | `FirstTile`  | {1,8,64} | int32 | {F} | exact | int path |

---

## 9. UnaryBcast (9 rows)

| id | dim | policy | reconfig | num_tiles | dtype | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|---|---|
| `9.1.UnaryBcast_ROW_PerTile_None`   | `ROW`    | `WaitAndPop` | `None`  | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `9.2.UnaryBcast_COL_PerTile_None`   | `COL`    | `WaitAndPop` | `None`  | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `9.3.UnaryBcast_SCALAR_PerTile_None`| `SCALAR` | `WaitAndPop` | `None`  | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `9.4.UnaryBcast_ROW_NoWaitPop`      | `ROW`    | `NoWaitPop`  | `None`  | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `9.5.UnaryBcast_COL_WaitNoPop`      | `COL`    | `WaitNoPop`  | `None`  | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `9.6.UnaryBcast_SCALAR_NoWaitNoPop` | `SCALAR` | `NoWaitNoPop`| `None`  | {1,8,64} | bf16 | {F,T} | 0.9999 | sharded |
| `9.7.UnaryBcast_ROW_Input_reconfig` | `ROW`    | `WaitAndPop` | `Input` | {1,8,64} | bf16→fp32→bf16 | {F,T} | 0.999  | reconfigure_unary_bcast |
| `9.8.UnaryBcast_NONE_passthrough`   | `NONE`   | `WaitAndPop` | `None`  | {1,8,64} | bf16 | {F,T} | 0.9999 | tests "no broadcast" enum value |
| `9.9.UnaryBcast_in_chain_with_BinaryFpu` | `ROW` | `WaitNoPop` | `None` | {1,8,64} | bf16 | {F,T} | 0.9999 | chain: UnaryBcast result feeds BinaryFpu |

---

## 10. Fill / Rand (7 rows)

| id | element | params | num_tiles | dtype | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|---|
| `10.1.FillScalar_f32`    | `FillScalar<f32>` | runtime value=3.14 | {1,8,64} | bf16 | {F,T} | exact | tile filled with 3.14 |
| `10.2.FillScalar_f16`    | `FillScalar<f16>` | runtime value=-1.0 | {1,8,64} | bf16 | {F,T} | exact | |
| `10.3.FillInt`           | `FillInt<int32>`  | runtime value=42  | {1,8,64} | int32 | {F} | exact | int fill |
| `10.4.FillBitcast`       | `FillBitcast`     | runtime u32 bits  | {1,8,64} | bf16 | {F,T} | exact | bit-pattern fill |
| `10.5.RandTile`          | `RandTile`        | seed=0            | {1,8,64} | bf16 | {F,T} | shape+stats | distribution check (mean ≈ expected, no NaN/Inf) |
| `10.6.FillScalar_in_chain_no_double_wait` | `FillScalar` then `BinaryFpu` | mixed | {1,8,64} | bf16 | {F,T} | 0.9999 | confirms FillTileTag does NOT participate in CB-collision sweep |
| `10.7.RandTile_in_chain` | `RandTile` then `Where` (3-input) | mixed | {1,8,64} | bf16 | {F,T} | shape+stats | RandTileTag separate from CopyTileTag |

---

## 11. PackTile lifecycle matrix (16 rows)

Goal: prove every `(PackTilePolicy × PackTileIndexMode)` legal cell works; illegal cells must `static_assert`. Chain shape: `CopyTile → Exp{} → PackTile<P, IM>`.

| id | policy | index | num_tiles | dtype | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|---|
| `11.1.PerTileReserveAndPush_FirstTile`     | `PerTileReserveAndPush`     | `FirstTile` | {1,8,64} | bf16 | {F,T} | 0.9999 | streaming default |
| `11.2.PerTileReserveAndPush_PinnedZero`    | `PerTileReserveAndPush`     | `Pinned k=0`| {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `11.3.PerTileReserveNoPush_FirstTile`      | `PerTileReserveNoPush`      | `FirstTile` | {1,8,64} | bf16 | {F,T} | 0.9999 | caller pushes |
| `11.4.NoReservePushAtEnd_FirstTile`        | `NoReservePushAtEnd`        | `FirstTile` | {1,8,64} | bf16 | {F,T} | 0.9999 | reserve elsewhere |
| `11.5.NoReserveNoPush_FirstTile`           | `NoReserveNoPush`           | `FirstTile` | {1,8,64} | bf16 | {F,T} | 0.9999 | caller manages both |
| `11.6.UpfrontReservePushAtEnd_FirstTile`   | `UpfrontReservePushAtEnd`   | `FirstTile` | {1,8,64} | bf16 | {F,T} | 0.9999 | block reserve, single tile each iter |
| `11.7.UpfrontReservePushAtEnd_BlockIter`   | `UpfrontReservePushAtEnd`   | `BlockIter` | {1,8,64} | bf16 | {F,T} | 0.9999 | canonical block pack |
| `11.8.UpfrontReservePushAtEnd_PinnedNonZero` | `UpfrontReservePushAtEnd` | `Pinned k=2`| {8,64}   | bf16 | {F,T} | 0.9999 | num_tiles=1 dropped |
| `11.9.UpfrontReservePushAtEnd_Absolute`    | `UpfrontReservePushAtEnd`   | `Absolute idx` | {8,64} | bf16 | {F,T} | 0.9999 | runtime idx |
| `11.10.NoReserveNoPush_BlockIter`          | `NoReserveNoPush`           | `BlockIter` | {8,64}   | bf16 | {F,T} | 0.9999 | caller-managed block |
| `11.11.NoReserveNoPush_Absolute`           | `NoReserveNoPush`           | `Absolute idx` | {8,64} | bf16 | {F,T} | 0.9999 | |
| `11.12.PerTileReserveAndPush_BlockIter_negative` | `PerTileReserveAndPush` | `BlockIter` | n/a | n/a | n/a | n/a | `static_assert_only` |
| `11.13.PerTileReserveAndPush_Absolute_negative`  | `PerTileReserveAndPush` | `Absolute`  | n/a | n/a | n/a | n/a | `static_assert_only` |
| `11.14.PerTileReserveNoPush_BlockIter_negative`  | `PerTileReserveNoPush`  | `BlockIter` | n/a | n/a | n/a | n/a | `static_assert_only` |
| `11.15.NoReservePushAtEnd_BlockIter_negative`    | `NoReservePushAtEnd`    | `BlockIter` | n/a | n/a | n/a | n/a | `static_assert_only` |
| `11.16.UpfrontReservePushAtEnd_when_options_block_size_0_negative` | `UpfrontReservePushAtEnd` | `BlockIter` | n/a | n/a | n/a | n/a | `static_assert_only` — chain options must declare upfront_block_size > 0 |

---

## 12. PackTile reconfig (6 rows)

| id | reconfig | dtype_out_first | dtype_out_second | num_tiles | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|---|
| `12.1.PackNone_uniform`           | `None`              | bf16 | bf16  | {1,8,64} | {F,T} | 0.9999 | |
| `12.2.PackOutput_bf16_to_fp32`    | `Output`            | bf16 | fp32  | {1,8,64} | {F,T} | 0.999  | mid-test pack reconfig |
| `12.3.PackOutput_fp32_to_bf16`    | `Output`            | fp32 | bf16  | {1,8,64} | {F,T} | 0.999  | |
| `12.4.PackOutputConditional_bf16_bfp8b` | `OutputConditional` | bf16 | bfloat8_b | {1,8,64} | {F,T} | 0.999  | old_cb → new_cb form |
| `12.5.PackOutput_int32_to_bf16`   | `Output`            | int32 | bf16 | {1,8,64} | {F,T} | exact_then_round | int → float |
| `12.6.PackNone_with_BinaryFpu_INPUT_AND_OUTPUT_combined_with_dt` | `None` | bf16 | bf16 | {1,8,64} | {F,T} | 0.9999 | confirms helper does NOT double-emit pack reconfig when BinaryFpu owns it |

---

## 13. PackTileBlock (4 rows)

| id | scenario | num_tiles | dtype | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|
| `13.1.PackTileBlock_4slots_singleCB` | `PackTileBlock<cb_out, Dst::D0..D3>{}` after 4-slot compute | {8,64} | bf16 | {F,T} | 0.9999 | atomic 4-tile pack |
| `13.2.PackTileBlock_2slots_singleCB` | 2-slot variant | {1,8,64} | bf16 | {F,T} | 0.9999 | smallest atomic block |
| `13.3.PackTileBlock_consecutive_required_negative` | 4-slot but slots are not consecutive | n/a | n/a | n/a | n/a | `static_assert_only` — `PackTileBlock` requires consecutive Dst slots |
| `13.4.PackTileBlock_vs_PackTile_chain_equality` | 4-slot atomic vs 4× independent PackTile chain elements | {8,64} | bf16 | {F,T} | bitwise-equal | side-by-side bitwise output match |

---

## 14. Multi-element chains (14 rows)

| id | shape | num_tiles | dtype | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|
| `14.1.Copy_Exp_Pack`               | `CopyTile + Exp + PackTile` | {1,8,64} | bf16 | {F,T} | 0.9999 | minimal |
| `14.2.Copy_Sigmoid_Tanh_Pack`      | 2 SFPU ops in series        | {1,8,64} | bf16 | {F,T} | 0.999  | |
| `14.3.CopyA_CopyB_BinaryFpu_Pack`  | A + B → out                 | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `14.4.CopyA_CopyB_BinaryFpu_Sqrt_Pack` | (A+B) sqrt → out         | {1,8,64} | bf16 | {F,T} | 0.999  | post-binary SFPU |
| `14.5.Copy_Exp_BinaryFpu_Pack`     | exp(A) + B (mid-chain bcast cb)  | {1,8,64} | bf16 | {F,T} | 0.999 | SFPU before binary |
| `14.6.Copy_Exp_DestReuseMul_Pack`  | exp(x) * x via dest reuse    | {1,8,64} | bf16 | {F,T} | 0.999  | SFPU + dest reuse |
| `14.7.Fill_Where_Pack`             | FillScalar(0) + cond + Where | {1,8,64} | bf16 | {F,T} | 0.9999 | fill + ternary |
| `14.8.Copy_UnaryBcast_BinaryFpu_Pack` | unary_bcast row + add → out | {1,8,64} | bf16 | {F,T} | 0.9999 | bcast in chain |
| `14.9.CopyA_CopyB_BinaryFpu_Sqrt_DestReuseMul_Pack` | 5-element chain | {1,8,64} | bf16 | {F,T} | 0.999 | full mixed chain |
| `14.10.Copy_FillScalar_BinaryFpu_Pack` | x + 1.0 via fill         | {1,8,64} | bf16 | {F,T} | 0.9999 | fill participates |
| `14.11.Copy_Exp_PackA_PackB_sameSlot_negative` | 2 PackTile, same Cb+slot+idx | n/a | n/a | n/a | n/a | `static_assert_only` — chain_pack_writes_collide_v |
| `14.12.LongChain_DEST_AUTO_LIMIT_minus_1_slots` | chain using up DEST_AUTO_LIMIT-1 slots | {1,8} | bf16 | {F,T} | 0.999 | exercises the fp32-DEST mode shrink |
| `14.13.Copy_BinaryFpu_DestReuse_Mul_BinaryFpu_Pack` | dest reuse mid-chain + binary after | {1,8,64} | bf16 | {F,T} | 0.999 | FPU-clash reinit between elements |
| `14.14.UpfrontBlock_4xCopy_Add_Pack_BlockIter` | 4 inputs in window, 1 output | {1,8,64} | bf16 | {F,T} | 0.9999 | block path |

---

## 15. Fan-out (6 rows)

Goal: 1 CB read, N DEST slots, N pack outputs in one acquire/commit/wait/release window. Lessons §3.5 + §1.7 dedup invariants.

| id | shape | num_tiles | dtype | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|
| `15.1.FanOut2_Exp_Sin`             | `CopyTile<cb_in, D0, WaitNoPop> + CopyTile<cb_in, D1, NoWaitPop> + Exp{D0} + Sin{D1} + PackTile<cbA, D0> + PackTile<cbB, D1>` | {1,8,64} | bf16 | {F,T} | 0.999 | classic fan-out |
| `15.2.FanOut3_three_outputs_one_input` | 3-way fan-out                                                                                              | {1,8,64} | bf16 | {F,T} | 0.999 | three CBs out |
| `15.3.FanOut4_SDPA_style`          | 4-input fan-out + 4 packs to 4 different CBs (fused max/sub/exp/add shape) | {1,8,64} | bf16 | {F,T} | 0.999 | mirrors transformer/sdpa pattern |
| `15.4.FanOut2_DuplicateUpfrontCB_negative` | both fan-out copies use `WaitUpfrontPopAtEnd` + same Cb upfront | n/a | n/a | n/a | n/a | `static_assert_only` — `chain_has_duplicate_upfront_cbs_v` triggers via `is_cb_reader_op_v` sweep |
| `15.5.FanOut2_AutoMerge_negative`  | author writes `CopyTile<WaitNoPop, D0> + CopyTile<WaitNoPop, D1>` | n/a | n/a | n/a | n/a | `static_assert_only` — chain refuses double WaitNoPop on same CB+slot pair to prevent silent merge |
| `15.6.FanOut2_with_PackTileBlock_atomic` | 2 Copy + Exp/Tanh + atomic 2-slot block pack | {1,8,64} | bf16 | {F,T} | 0.999 | mixes fan-out with atomic pack |

---

## 16. Hoist-safety (6 rows)

| id | shape | hoist | num_tiles | dtype | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|---|
| `16.1.Copy_Exp_Pack_DefaultPerTile`    | `Copy + Exp + Pack` per-tile init | n/a (default) | {1,8,64} | bf16 | {F,T} | 0.9999 | baseline |
| `16.2.Copy_Exp_Pack_HoistSafe`         | same shape, hoisting opt-in | yes | {1,8,64} | bf16 | {F,T} | 0.9999 | output equality vs baseline |
| `16.3.Copy_Exp_Pack_Hoist_BitwiseEqual`| side-by-side per-tile vs hoist  | both | {1,8,64} | bf16 | {F,T} | bitwise-equal | confirms identical bytes |
| `16.4.Copy_Exp_Sqrt_Pack_HoistDisallowed_negative` | chain length > 2 | yes | n/a | n/a | n/a | n/a | `static_assert_only` — `chain_is_hoist_safe_v` rejects |
| `16.5.CopyA_CopyB_BinaryFpu_Pack_HoistDisallowed_negative` | multi-CB | yes | n/a | n/a | n/a | n/a | `static_assert_only` |
| `16.6.Copy_DestReuse_Pack_HoistDisallowed_negative` | DestReuseBinary clobbers MOP | yes | n/a | n/a | n/a | n/a | `static_assert_only` |

---

## 17. Compile-time invariants (12 rows)

All `static_assert_only`. Each row is a distinct `.cpp` in `tests/eltwise/static_assert_negative/` that should fail to compile with a specific diagnostic substring.

| id | violation | expected diagnostic substring |
|---|---|---|
| `17.1.Dst_above_DEST_AUTO_LIMIT` | `Dst::D9` (or `Dst::D17` under fp32 acc) | `DEST slot exceeds compile-time DEST capacity` |
| `17.2.UnaryOp_distinct_slots_violated` | BinaryOp with In0==Out | `BinaryOp slots must be distinct` |
| `17.3.QuaternaryOp_4_inputs_distinct` | QuaternaryOp with In0==In2 | `QuaternaryOp input slots must be distinct` |
| `17.4.Mask_DataSlot_plus_one_overflow` | `Mask<DF, Dst::D7>` | `Mask requires DataSlot + 1 < DEST capacity` |
| `17.5.WaitAndPop_BlockIter_invalid` | (1.15 above) | `BlockIter index requires WaitUpfrontPopAtEnd or NoWaitNoPop policy` |
| `17.6.PackTile_PerTile_BlockIter_invalid` | (11.12 above) | same pack-side message |
| `17.7.UpfrontReservePushAtEnd_zero_block_size` | (11.16 above) | `upfront_block_size must be > 0 to use UpfrontReservePushAtEnd` |
| `17.8.PackBlock_NonConsecutiveSlots` | (13.3 above) | `PackTileBlock requires consecutive Dst slots` |
| `17.9.PackWritesCollide` | (14.11 above) | `chain_pack_writes_collide_v: two PackTile elements write same (Cb, output_index, DstSlot)` |
| `17.10.DuplicateUpfrontCB_acrossCbReader` | binary FPU + CopyTile both upfront on same CB | `chain_has_duplicate_upfront_cbs_v: ...` |
| `17.11.HoistOnUnsafeChain` | `Copy + Exp + Sqrt` with hoist | `chain_is_hoist_safe_v` |
| `17.12.DestReuseBinary_Reconfig_bool_form_rejected` | `bool` template in place of `DestReuseReconfig` enum | `expected enum class DestReuseReconfig, got bool` |

---

## 18. Convenience entry points (9 rows)

| id | wrapper | shape | num_tiles | dtype | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|---|
| `18.1.binary_add` | `binary_add(cb_a, cb_b, cb_out, n)` | element-wise add | {1,8,64} | bf16 | {F,T} | 0.9999 | one-line forward |
| `18.2.binary_sub` | `binary_sub(...)` | sub | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `18.3.binary_mul` | `binary_mul(...)` | mul | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `18.4.binary_add_bcast_ROW` | `binary_add_bcast(cb_a, cb_b, BroadcastDim::ROW, ...)` | row bcast | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `18.5.binary_add_bcast_COL` | column bcast variant | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `18.6.binary_add_bcast_SCALAR` | scalar bcast variant | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `18.7.unary_op_Exp` | `unary_op<Exp<>>(cb_in, cb_out, n)` | unary | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `18.8.dest_reuse_mul` | `dest_reuse_mul(cb_in, dst_slot, cb_out, n)` | square via dest reuse | {1,8,64} | bf16 | {F,T} | 0.9999 | |
| `18.9.copy / copy_with_dt` | `copy(cb_in, cb_out, n)` and dt variant | identity passthrough | {1,8,64} | bf16, fp32 | {F,T} | 0.9999 | one row per overload |

---

## 19. Mixed-dtype FP32 acc sweep (18 rows)

Goal: HQ Step 4 dtype-matrix mandate. Every binary path × `fp32_dest_acc_en={F,T}` × representative dtype combos. Confirms mixed-dtype binaries produce expected reconfigs and PCC holds.

| id | path | dtype_a | dtype_b | dtype_out | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|---|
| `19.1`–`19.6` (Add path)        | `BinaryFpu Add` | bf16/fp32/bf16 + symmetric variants | {F,T} sweep | 0.999 | 6 rows = 3 dtype combos × 2 fp32_acc |
| `19.7`–`19.12` (Mul path)       | `BinaryFpu Mul` | same dtype matrix | {F,T} sweep | 0.999 | 6 rows |
| `19.13`–`19.16` (DestReuseBinary path) | `Mul DEST_TO_SRCA` | bf16 + fp32 mixed | {F,T} sweep | 0.999 | 4 rows |
| `19.17`–`19.18` (PackTile reconfig path) | `bf16→fp32` and `fp32→bf16` packing | n/a | {F,T} sweep | 0.999 | 2 rows |

(Each row in §19 names itself explicitly in the kernel/pytest manifest by its full dtype tuple. Above is a compressed table due to row count.)

---

## 20. Migration regression set (per-tier — final list determined per kernel)

For each of the five tiers in §9.4 of the proposal, the migration commit lands in two halves (per §9.2 step sequence):

1. **Pre-migration baseline run** of the kernel's existing pytest, recorded in `pytest_map.md`.
2. **Post-migration run** of the same pytest unchanged. PCC must hold.

The "regression set" in this plan does NOT add new pytests for production kernels — those tests already exist. What this plan adds is:

- One pytest row in `test_eltwise.py` per migrated kernel ID, marked `migration:<kernel_path>:<tier>`, that re-runs the kernel's *existing* operation pytest end-to-end (subprocess invocation of `scripts/run_safe_pytest.sh <existing_test>`) and asserts a non-zero exit code propagates as failure here. This gives a single dashboard for "all migrated kernels still green" without copying their tests.
- For Tier D / Tier E kernels: an additional row asserting the kernel file no longer contains `acquire_dst|release_dst|ACQ\(|REL\(` (grep-based). This is the dst-sync rewrite verification per §9.2 step 1.

Per-tier counts (final list locked when the migration cycle starts; numbers below are estimates from TSV row counts mapped to unique kernel files):

| Tier | Estimated kernels | Sample seed list |
|---|---:|---|
| A — modern + canonical/single | ~25 | `eltwise/binary/*`, `eltwise/unary/*` simple paths |
| B — modern + single-in-loop | ~30 | `eltwise/binary/device/kernels/compute/*`, normalization basic |
| C — modern + multi-loop / multi-pack | ~10 | normalization layernorm / softmax post-allgather |
| D — raw-dst | ~12 | `transformer/sdpa/device/kernels/compute/*`, `experimental/transformer/*` |
| E — ACQ-REL-macro | ~8 | `moreh/moreh_dot{,_backward}/*`, `conv/conv2d/.../compute_depthwise_conv1d.cpp` |

Total migration regression rows: ~85. Each row is one `subprocess.run` invocation in the pytest, parameterized on the kernel id only — no per-kernel num_tiles / dtype expansion (the underlying tests own that).

---

## 21. Stress / boundary (8 rows)

| id | scenario | num_tiles | dtype | fp32_acc | PCC | notes |
|---|---|---|---|---|---|---|
| `21.1.DEST_AUTO_LIMIT_minus_1` | chain that allocates `DEST_AUTO_LIMIT - 1` slots simultaneously | {1} | bf16 | {F,T} | 0.999 | confirms cap is correctly read at compile time, both half-sync and full-sync |
| `21.2.DEST_AUTO_LIMIT_overflow_negative` | request 1 more than capacity | n/a | n/a | n/a | n/a | `static_assert_only` |
| `21.3.LargeNumTiles_256` | streaming chain over 256 tiles | {256} | bf16 | {F} | 0.999 | exercises CB capacity wrap |
| `21.4.LargeNumTiles_1024` | 1024 tiles | {1024} | bf16 | {F} | 0.999 | longer-running stress |
| `21.5.UpfrontBlock_size_equal_DEST_AUTO_LIMIT` | block size N = DEST_AUTO_LIMIT | {N} | bf16 | {F,T} | 0.999 | block fully fills DEST |
| `21.6.SmallCBCapacity_one_page` | CB sized to 1 page only | {64} | bf16 | {F} | 0.999 | tightest streaming wrap |
| `21.7.NumTiles_zero_negative` | n_tiles=0 | {0} | bf16 | {F} | n/a | runtime — chain must early-exit cleanly (no dispatch deadlock) |
| `21.8.MixedFp32DestAccEn_LargeChain` | 6-element chain under fp32_dest_acc=True | {1,8,64} | bf16 | {T} | 0.999 | exercises fp32-DEST-shrunk slot count |

---

## 22. Migration prerequisite verification (§9.2) — 6 rows

These are `static_assert_only` and grep-based — they validate the migration prerequisite contract before / after each Tier-D and Tier-E kernel adopts the chain.

| id | check | how | notes |
|---|---|---|---|
| `22.1.NoRawDstAcquireRelease_TierD_post` | grep | post-migration: the kernel file contains zero `acquire_dst\|release_dst` matches | grep over the post-migration kernel; fail otherwise |
| `22.2.NoACQRELMacro_TierE_post`          | grep | post-migration: zero `\\bACQ\\s*\\(` / `\\bREL\\s*\\(` in the kernel file | |
| `22.3.NoACQRELMacroDef_TierE_post`       | grep | post-migration: no `#define\\s+ACQ\\(`/`REL\\(` in the kernel file | macro definition scrubbed |
| `22.4.PreMigration_PCC_baseline_recorded`| pytest fixture | each Tier-D/E kernel's pre-migration pytest result is recorded in `pytest_map.md` before adoption | enforces §9.2 step 1 |
| `22.5.PostMigration_PCC_unchanged`       | pytest | post-migration pytest produces `>= pre-migration PCC` | regression detector |
| `22.6.ChainHelperOnly_inMigratedKernels` | grep | post-migration: zero raw `add_tiles\|sub_tiles\|mul_tiles\|copy_tile\b\|pack_tile\b` outside helper-internal use | enforces helper-only invocation |

---

## 23. Tooling / CI

- `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/kernel_lib/test_eltwise.py` runs the full plan. No outer `timeout` (HQ rule).
- `--run-all` is set so per-tier counts of pass / fail report; CI does not stop at first failure.
- The static_assert negative tests are driven by a separate harness (`tests/eltwise/static_assert_negative/run_negatives.py`) that compiles each `.cpp` and asserts compile failure with the expected diagnostic substring. Not a subprocess of the JIT compute compile path — these are host-side standalone TUs that include the helper headers.
- `--dev` flag added to all CI invocations so any hang produces a JSON triage artifact.

---

## 24. Out of scope for this test plan

- Performance / throughput measurement. v1 ships correctness; perf comparisons land in a separate proposal.
- Multi-chip / mesh tests. Single-device only in v1.
- Blackhole-specific paths. Skipped unless the row explicitly tests them.
- Tests for the deferred gaps (§9.3 of the proposal): `GAP-CUMULATIVE-WAIT`, `GAP-WELFORD`, `GAP-TRANSPOSE`, `GAP-PACK-L1-ACC`, `GAP-PACK-RELU`, `GAP-PACK-ROWS`, `GAP-MID-LOOP-RECONFIG`. These get test rows when their proposals land.

---

## 25. Acceptance summary

Approving this test plan authorizes:

- Writing the device kernels under `ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/`.
- Writing the negative-compile harness under `ttnn/cpp/ttnn/kernel_lib/tests/eltwise/static_assert_negative/`.
- Writing the pytest at `tests/ttnn/unit_tests/kernel_lib/test_eltwise.py`.
- Implementing the helper headers (Phase 4 / 5) — kernels and pytest land in lockstep with the helper they exercise.
- Beginning the per-tier migration cycle once §1–§19 rows pass; §20–§22 land per migrating kernel.

Approving this plan does NOT authorize:

- Adding tests outside the rows enumerated here. Any new test row re-enters Gate 2 as a new commit.
- Skipping a row at runtime without one of the §0.6 reasons.
- Loosening a tolerance below the §0.5 floor.

Plan at `ttnn/cpp/ttnn/kernel_lib/agents/eltwise_v2_test_plan.md`. Awaiting Gate 2 sign-off.
