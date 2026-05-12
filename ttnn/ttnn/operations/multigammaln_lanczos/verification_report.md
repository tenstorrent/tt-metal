# Verification Report: multigammaln_lanczos

## Code Review

The kernel and program descriptor are well-aligned with `op_design.md` — design conformance (algorithm, data pipeline topology, work distribution, no inter-core comm) is clean. Helper usage matches the design's rationale for being "helper-light" (24 unique scalars × 4 lgammas + 4-DST-slot budget make `sfpu_chain` structurally inapplicable). API surface uses the current primitives (`TensorAccessor`, `kernel_main()` syntax, `api/dataflow/dataflow_api.h` includes, `copy_dest_values<DataFormat::Float32>` not the deprecated overload).

### Fixes applied

| # | Issue | Fix | File:line |
|---|-------|-----|----------|
| 1 | `cb_input_tiles` and `cb_accumulator` are both Float32 and are read-back-into-DST via `copy_tile`, but `unpack_to_dest_mode` was not set. Under `fp32_dest_acc_en=True` the unpacker's default mode routes through SrcA/SrcB at TF32 (~10-bit mantissa) precision. Output max-abs error was ~0.13 against a torch fp64 reference. | Set `unpack_to_dest_mode[cb_input_tiles] = unpack_to_dest_mode[cb_accumulator] = ttnn.UnpackToDestMode.UnpackToDestFp32` in the compute config (program descriptor). Verified by re-running the precision baseline: max-abs dropped from ~0.13 → ~0.005 (24×), mean-abs from ~0.031 → ~0.0008 (39×), relative RMS from ~1.7e-3 → ~5.3e-5 (32×). All 24 tests pass. No kernel-side changes required. | `multigammaln_lanczos_program_descriptor.py:166-180` |

### Reviewed and confirmed correct (no changes needed)

- **Algorithm**: matches `op_design.md` § Overview — 4 Lanczos-lgamma evaluations against `a − 0.5·k` for `k = 0..3`, plus `3·log(π)`, with pole zeroing at `a = 1` and `a = 2`. Algebraic re-grouping (`+5.5 shift` collapsed into `+4.5` and the `LANCZOS_OFFSET = 3.581…` constant) is faithful to the design.
- **Data pipeline topology**: reader (NCRISC, NoC0) → `cb_input_tiles` → compute → `cb_accumulator` (intra-tile RMW) → `cb_output_tiles` → writer (BRISC, NoC1). Exactly the design's diagram.
- **Work distribution**: `split_work_to_cores` on the full compute grid; per-core RT args walk group_1 then group_2 (matches `tests/ttnn/unit_tests/operations/debug/test_generic_op.py`'s canonical pattern). Compute uses a single KernelDescriptor with per-core `num_tiles` as an RT arg (avoiding two KernelDescriptors for uneven splits — simpler and works).
- **CB sync**: balances per `op_design.md` § "CB sync — push count = wait count". 1=1 for input, 1=1 for output, 5=5 for accumulator (1 init-zero + 4 lgamma packs vs. 4 reloads + 1 finalisation read).
- **CB sizing**: 2 pages per CB; `cb_accumulator` at 2 pages is the minimum that avoids the front+back ping-pong deadlock (`op_design.md` K6 #3).
- **Helper non-use rationale**: `sfpu_chain` requires compile-time-fixed scalars and a stride budget that does not fit the 4-DST-slot fp32 layout with 24 unique scalars. `binary_op_helpers::add` etc. operate at CB level, not DST level — using them for the in-DST Lanczos accumulation would round-trip every intermediate through L1. The raw `*_binary_tile` / `add_unary_tile` calls are the right primitive here; `op_design.md` API Mapping documents each non-use with a file:line citation.
- **Pole mask**: applied to a copy of `a` (always finite on the safe domain), not to the lgamma result — avoids the `NaN × 0 = NaN` failure mode at the polynomial poles. Verified by `test_multigammaln_lanczos_pole_x_equals_2`.
- **`pack_reconfig_data_format` between `cb_accumulator` and `cb_output_tiles`**: kept even though both CBs are fp32, because the packer's bound CB id changes (`op_design.md` K6 #4). Conservative-but-correct.
- **Redundant-looking SFPU `*_tile_init` calls** inside the inner Lanczos loop: every one is preceded by a different SFPU op family that overwrites SFPU programmable state. Each init is therefore necessary; the apparent redundancy is the price of correct alternation between `copy_dest_values`, `binop_with_scalar`, `recip_tile`, `log_tile`, `unary_ne_tile`, and the binary-DST ops.
- **Reader / writer**: use `TensorAccessor` (not deprecated `InterleavedAddrGen`), single-tile streaming with explicit barriers, standard `kernel_main()` entry point, includes via `api/dataflow/dataflow_api.h`. Matches `tests/ttnn/unit_tests/operations/debug/test_generic_op.py`'s canonical idiom.

### Deferred to refinements (architectural — out of verification scope)

- Compute-config exposure (Refinement 1) — Phase 0 hard-codes by design.
- bfloat16 / bfloat8_b dtype widening (Refinement 2) — Phase 0 rejects.
- Non-tile-aligned shapes (Refinement 3) — Phase 0 rejects.
- ROW_MAJOR input (Refinement 4) — no in-kernel tilize path.
- Sharded layouts (Refinement 5) — no real model demand yet; op is compute-bound.

## Precision Baseline

After the UnpackToDestFp32 fix (`multigammaln_lanczos_program_descriptor.py:166-180`):

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|-----|-------------|--------------|------------------|
| `(1,1,32,32)` | 0.9999999961 | 0.00485 | 0.000793 | 5.19e-5 |
| `(1,1,64,64)` | 0.9999999960 | 0.00505 | 0.000809 | 5.27e-5 |
| `(1,1,256,256)` | 0.9999999957 | 0.00518 | 0.000829 | 5.41e-5 |
| `(2,4,64,128)` | 0.9999999957 | 0.00518 | 0.000829 | 5.41e-5 |

Pre-fix baseline (for reference — what Phase 0 looked like before the verifier touched it):

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|-----|-------------|--------------|------------------|
| `(1,1,32,32)` | 0.99999954 | 0.117 | 0.0312 | 1.69e-3 |
| `(1,1,64,64)` | 0.99999956 | 0.128 | 0.0312 | 1.67e-3 |
| `(1,1,256,256)` | 0.99999955 | 0.136 | 0.0308 | 1.66e-3 |
| `(2,4,64,128)` | 0.99999955 | 0.136 | 0.0308 | 1.66e-3 |

**Assessment**: errors are consistent with a Lanczos-6 polynomial evaluated at fp32 with all internal round-trips at full fp32 precision. The max-abs scales weakly with tile count (small numerical noise across more elements), and the mean-abs is essentially flat across the four shapes — the per-element math is stable. PCC > 0.99999995 across all shapes is well above the Phase 0 budget (PCC > 0.99). Max absolute error of ~0.005 on outputs whose magnitude is `O(10)` for the largest inputs in the safe domain corresponds to relative errors of ~5e-4.

**Recommended tolerances** for downstream callers:

- PCC ≥ 0.999 (current measurement is ~5 nines tighter than that).
- `rtol = 5e-3`, `atol = 1e-2` is comfortable on the safe domain.
- For strict acceptance gates, keep the existing `rtol = 0.1, atol = 0.5` until refinements lock further precision tightening.

## Test Results

- **Acceptance tests** (`test_multigammaln_lanczos.py`): 14 / 14 passing. Covers single-tile, multi-tile-W, multi-tile-H, non-square `64×128` and `128×64`, multi-batch `(2,4,64,128)`, positional + keyword call styles, the `x = 2.0` pole case, and four negative validation cases.
- **Precision baseline** (`test_multigammaln_lanczos_precision_baseline.py`): 4 / 4 passing. Asserts PCC ≥ 0.999 plus regression gates `max_abs < 0.05` and `rel_rms < 5e-4` to trip immediately on an accidental UnpackToDestFp32 regression.
- **Extended coverage** (`test_multigammaln_lanczos_extended.py`): 6 / 6 passing. Covers L1-interleaved input (acceptance test only used DRAM), three constant-input value-domain edges (pole at 2.0, near-pole 2.5, high-end 10.0), a linspace pattern over the safe domain, and a 65-tile shape that forces both `core_group_1` and `core_group_2` non-empty in `split_work_to_cores`.

**Total: 24 / 24 passing.** Test files use `scripts/run_safe_pytest.sh`; runtime is under 1 s end-to-end.

## Recommendations

### Priority refinement targets

1. **Refinement 1 — Expose `compute_kernel_config`** (highest user value): callers should be able to pick HiFi2 / no-fp32-dest-acc for throughput, while the default preserves Phase 0 precision exactly. Do **not** let callers accidentally disable `UnpackToDestFp32` — the numerical_stability analysis and this verification both confirm it is the single highest-leverage precision lever for this op (24× max-abs improvement). Either keep it always-on regardless of `compute_kernel_config`, or surface it behind a separate "I know what I'm doing" flag.

2. **Refinement 2 — bfloat16 inputs**: most of the plumbing is already in place (page sizes come from `tensor.buffer_page_size()`). Only the validator and the explicit `cb_accumulator` page size need attention — `cb_accumulator` must stay Float32 + UnpackToDestFp32 to preserve the running-sum precision.

3. **Refinement 3 — Non-tile-aligned shapes**: simplest as a host-side pad-then-call wrapper. The kernel does not need changes.

### Tradeoffs surfaced by the companion analyses

- **Precision vs L1 footprint**: `cb_accumulator` is Float32 (4096 B/page × 2 pages = 8 KB) — this is fixed by the algorithm (Float32 round-trip preserves the running sum). Halving the page size by going to bfloat16 in the accumulator would re-introduce the precision leak. Don't.
- **Precision vs throughput**: `math_approx_mode = false` runs `recip_tile` (24/element) and `log_tile` (8/element) in high-precision variants. Switching to `Approx::Fast` for `Log` would buy throughput but at a cost in precision that I have not characterised in this verifier pass. If a future refinement adds it, re-run the precision baseline.
- **Memory bandwidth vs compute**: the kernel is compute-bound today (see `data_transfer.md` § 7), so the standard reader/writer single-tile streaming with double-buffered CBs is correct. No reason to switch to multi-tile-batch reads at this point.

### Locked-in invariants for downstream agents

- `cb_accumulator` and `cb_input_tiles` must keep `UnpackToDestMode::UnpackToDestFp32` whenever `fp32_dest_acc_en=True`. The precision baseline's `max_abs < 0.05` assertion will trip if this is dropped.
- `cb_accumulator` must stay at **2 pages** — 1 page deadlocks the front+back ping-pong.
- The DST budget is **4 slots** under fp32_dest_acc + half-sync — do not add a 5th DST-resident intermediate without re-deriving the slot map in `op_design.md` § "DST slot register".
- The pole mask must be applied to `a` (in D2), not to the lgamma result (D0) — see `op_design.md` K6 #7.
