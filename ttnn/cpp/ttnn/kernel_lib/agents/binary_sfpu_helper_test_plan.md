# Test Plan — `BinarySfpu` op-struct coverage + chain composition pattern

Status: **AWAITING SIGN-OFF** (Gate 2).
Companion to `binary_sfpu_helper_proposal.md` v2 (Gate 1: approved).
Pipeline phase: Phase 4 sub-stage 2c / 2d (validation).

---

## 1. Test layout

Three test kernels + their pytest fns. One regression sweep on the existing baseline.

### 1.1 `binary_sfpu_compose.cpp` — composition smoke test (Step 0)

The mandatory first gate. Validates that **the existing chain helper already supports the composition** of:
- `CopyTile<cb_a, … , Dst::D0, CopyTileReconfig::Input>` (A → even DEST slots)
- `CopyTile<cb_b, … , Dst::D1, CopyTileReconfig::Input>` (B → odd DEST slots; srca fold reconfigs cb_a → cb_b)
- `AddBinary<Dst::D0, Dst::D1, Dst::D0>` (existing DEST-only SFPU bin)
- `PackTile<cb_out, Dst::D0, PerBlockReserveAndPush, BlockIter, …>` (pack even slots)

…at `chain_lane_width = 2`, with per-block streaming.

If this kernel fails to JIT-compile, fails PCC, or hangs:
- Most likely cause: the chain's prev-CB fold doesn't emit `copy_tile_to_dst_init_short_with_dt(cb_a, cb_b)` between the two `CopyTile` elements, OR `chain_lane_width=2` doesn't propagate to per-element exec the way the static reasoning suggests.
- Either is a **local plumbing fix** in `eltwise_chain.inl` — NOT a new chain element. Document the fix, land it before Commit 2.

### 1.2 `binary_sfpu_per_family.cpp` — op-struct coverage per representative family

Same chain shape as 1.1, parameterised by SFPU op via a kernel-side define. One test invocation per representative struct family:

| Family | Representative struct | Selected via define |
|---|---|---|
| Float bin (no template, no runtime arg) | `AddBinary` | `SFPU_OP=Add` |
| Float bin (paired init: shared init across whole bitwise family) | `BitwiseAndBinary<DataFormat::Float16_b>` | `SFPU_OP=BitwiseAnd` |
| Int-dtype templated | `AddIntBinary<DataFormat::Int32>` | `SFPU_OP=AddInt32` |
| UInt-dtype templated | `BinaryMaxUint32` | `SFPU_OP=MaxUint32` |
| Runtime ctor arg | `QuantBinary` (ctor: `zero_point`) | `SFPU_OP=Quant` |
| Comparison (returns 0/1) | `LtBinary` | `SFPU_OP=Lt` |

Exercises the union of:
- bare struct (no template),
- `DataFormat`-templated struct,
- struct with runtime ctor arg,
- comparison op (output range different from inputs — proves PCC infrastructure handles 0/1 output).

If a new op-struct fails to compile or returns wrong values, it's caught at family level, not waiting until full migration.

### 1.3 `binary_sfpu_per_side_idx.cpp` — composition + per-side regime (reuses existing capability)

Same shape as 1.1 but A=`WaitAndPopPerBlock` + B=`WaitUpfrontPopAtEnd` with `BIndex=BlockIter`. Verifies the per-side-idx dispatch (just landed) works when the chain has the two-CopyTile + DEST-only SFPU + PackTile composition. Sanity check that the SFPU composition doesn't have a different code path that bypasses per-side resolution.

### 1.4 Regression — no test changes

The existing `tests/ttnn/unit_tests/kernel_lib/test_eltwise.py` baseline (591 passed + 7 skipped, after the per-side-idx extension) must remain exactly that — no test added, removed, retitled, or XFAIL-flipped on the existing surface.

## 2. Parameterization

### 2.1 binary_sfpu_compose (1D chain)

| Axis | Values | Rationale |
|---|---|---|
| `num_tiles` | `4, 8, 16, 64` | divisible by all block sizes; covers single + multi DEST windows |
| `block_size` | `1, 2, 4` | `1` is the degenerate "no-block" case; `2` and `4` exercise `chain_lane_width=2 * BlockSize ∈ {4, 8}` |
| `fp32_dest_acc` | `False, True` | per HQ; verifies `DEST_AUTO_LIMIT` shrink-by-half halves max block size automatically |
| `dtype` | `bfloat16` | bf16 path |

Total: `4 × 3 × 2 = 24` cases. Skip cases where `BlockSize=4` + `fp32_dest_acc=True` (would exceed DEST_AUTO_LIMIT=4 at chain_lane_width=2).

### 2.2 binary_sfpu_per_family

| Axis | Values | Rationale |
|---|---|---|
| `op_family` | 6 values (table in 1.2) | one per representative struct family |
| `num_tiles` | `4, 16` | single + multi window |
| `block_size` | `2` | fixed — coverage axis is on op_family, not block |
| `fp32_dest_acc` | per-family default (bf16 → False; int32 → False; float32 ops → True where required) | follows host-side dispatch in `get_sfpu_init_fn` |
| `dtype` | per-family (bf16 / int32 / uint32 / float32) | dtype selected to match op family's valid input |

Total: `6 op_family × 2 num_tiles × 1 block_size = 12` cases, dtype/fp32_dest_acc fixed per family.

For ops with a runtime arg (Quant family): pick one zero-point value (`zp=128`) and verify against torch quantization golden.

### 2.3 binary_sfpu_per_side_idx (1D chain, mixed-regime)

| Axis | Values | Rationale |
|---|---|---|
| `direction` | `a_local` (A=PerBlock, B=Upfront), `b_local` (A=Upfront, B=PerBlock) | proves per-side resolution applies under composition |
| `num_tiles` | `4, 16` | single + multi window |
| `block_size` | `2, 4` | both lane combos |
| `fp32_dest_acc` | `False, True` | dtype path |

Total: `2 × 2 × 2 × 2 = 16` cases (minus `block_size=4 + fp32_dest_acc=True` → 12).

### 2.4 Grand total

24 + 12 + 12 = **48 new test cases**.

## 3. PCC thresholds

- bf16 paths: `comp_pcc ≥ 0.9999`.
- int32 / uint32 ops: exact equality (`torch.equal` after dtype-cast) — these are integer ops, no FP rounding.
- Comparison ops (LT/GT/etc.): exact equality on the 0/1 mask.
- Quant family: `comp_pcc ≥ 0.999` (slightly looser — quantize involves rounding).

## 4. Skip rationale

No silent skips. Cases that exceed DEST_AUTO_LIMIT are skipped *programmatically* with `pytest.skip()` and the skip message names the constraint (`BlockSize * chain_lane_width > DEST_AUTO_LIMIT`).

Blackhole: not targeted; existing test_eltwise.py auto-skips.

## 5. Implementation notes (test kernels)

Each test kernel:
- Same CB layout as existing eltwise tests: `c_0` (A), `c_1` (B), `c_16` (out).
- Drives the chain via `eltwise_chain<BLOCK_SIZE>(num_tiles, …)`.
- Reuses the existing `_run_binary_with_kernel` harness — no new reader needed for 1.1 / 1.3. 1.2 needs a per-family adapter (handles dtype + runtime args).
- For ops with runtime ctor args (Quant family): pass via kernel runtime arg slot.

The pytest harness adds `test_binary_sfpu_compose`, `test_binary_sfpu_per_family[op_family]`, `test_binary_sfpu_per_side_idx[direction]` following the pattern of `test_binary_fpu_per_side_idx_a_local` (already in tree).

## 6. Acceptance criteria

- All 48 new cases pass per the PCC thresholds in §3.
- `tests/ttnn/unit_tests/kernel_lib/test_eltwise.py` baseline (591 passed + 7 skipped) **unchanged**.
- `tests/ttnn/unit_tests/operations/eltwise/test_add.py` (110p1s) unchanged.
- `test_binary_ng_bcast_fp32_dest_acc.py + test_binary_ng_program_cache.py` (22p) unchanged.
- `tests/ttnn/unit_tests/operations/fused/test_group_norm.py` (189p17s) unchanged.
- `tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_post_allgather.py` (79p) unchanged.
- `tests/ttnn/nightly/unit_tests/operations/fused/test_rmsnorm.py` (29p) unchanged.

## 7. Run sequence (post-Gate-2 sign-off)

1. Write `binary_sfpu_compose.cpp` + pytest. **Run it first.** If it fails, fix the chain plumbing locally (`emit_pre_element_transitions` / fold) before any op-struct adds.
2. Add ~25 op-structs to `eltwise_binary_sfpu.hpp` in a single commit (mechanical).
3. Write `binary_sfpu_per_family.cpp` + pytest. Confirm each new struct family passes.
4. Write `binary_sfpu_per_side_idx.cpp` + pytest. Confirm composition + per-side dispatch.
5. Run the full kernel_lib suite + regression set (§6 list). Confirm baselines unchanged.
6. Per-kernel migration commits (NOT part of this Gate-2 scope) follow once helper layer is green: `eltwise_binary_sfpu_no_bcast.cpp` first.

Each numbered step is its own commit. If a step's test fails, fix the helper or the test — never relax thresholds, never skip a case to make it green.

## 8. Out of scope

- The host-side `binary_ng_utils.cpp` mapping change to emit a struct identifier define is documented as a follow-up; test plan covers the helper layer assuming the kernel can already reference the right struct (via local mapping in the kernel file, as the FPU scalar kernel does).
- WHERE (ternary) — separate proposal.
- Activations-branch raw-LLK paths.
- Legacy `eltwise_binary_sfpu_kernel.cpp` (runtime block size).

---

Test plan at `ttnn/cpp/ttnn/kernel_lib/agents/binary_sfpu_helper_test_plan.md`. Awaiting sign-off.
