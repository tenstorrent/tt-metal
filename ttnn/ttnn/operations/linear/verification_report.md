# Verification Report: linear

## Code Review

### Design conformance
The implementation matches the design in `op_design.md` on every binding dimension:

- **Algorithm**: helper-driven `matmul_block` (no manual K-loop), with `LastBlockTarget::Out` in the no-bias path and `LastBlockTarget::Interm` + `add_bias_bcast_rows` in the bias path. Matches the design exactly.
- **Data pipeline topology**: NCRISC reader → CBs → TRISC compute → CB → BRISC writer → DRAM. No deviation.
- **Parallelization**: single Tensix at `(0, 0)` as designed.
- **Inter-core communication**: none specified, none implemented.

CB sync is correct: every CB's reader-push count equals the consumer's wait/pop count. Manual verification matches the table in `op_design.md` lines 158-164. Helpers own the input/weight/output CB lifecycle; bias CB lifecycle is caller-owned per `bias_add_helpers.hpp:93-97`.

### Helper usage
- ✅ `matmul_block` is invoked end-to-end for the K-reduction (not a manual `mm_init`/`matmul_block` LLK loop).
- ✅ `add_bias_bcast_rows` handles the bias add (no manual `add_tiles_bcast_rows` loop).
- ✅ `LastBlockTarget::Interm` correctly threads `cb_partials` between matmul and bias add.
- ✅ `OutputLayout::SubblockMajor` matches across both helpers (otherwise the bias helper would consume `cb_partials` in the wrong order — gotcha #8 in the design).
- ✅ `compute_kernel_hw_startup` is called once at the top of the kernel before any helper.

### Code-review fixes applied
- **`linear_compute.cpp` lines 88, 98**: switched raw `cb_wait_front(cb_bias_tiles, Nt)` / `cb_pop_front(cb_bias_tiles, Nt)` to `bias_buf.wait_front(Nt)` / `bias_buf.pop_front(Nt)` to match the canonical `bmm_large_block_zm_fused_bias_activation.cpp` style and use the `experimental::CircularBuffer` object that's already constructed in scope. Pure consistency improvement; tests still pass.

### Items deferred to refinements (architectural changes)
- **Per-tile `noc_async_read_barrier()` in the reader**: each `noc_async_read_tile` is followed by an immediate barrier (`linear_reader.cpp:55-57, 68-70, 81-83`). A bulk pattern (issue all reads in one CB-fill loop, single barrier at the end) would amortize the round-trip stall. At Phase 0's single-block, single-core sizes the matmul helper waits on the entire `Mt*Kt` block before consuming, so streaming individual tiles does not unlock earlier compute work. This is a real perf win but it pairs cleanly with the multi-core / sender-pattern refinement, where the standard idiom is bulk read + barrier per block. Documented as a refinement-relevant note in `data_transfer.md` §8 rather than fixed in Phase 0.
- **`compute_kernel_config` not exposed**: hard-coded HiFi4 + fp32_dest_acc_en is a reasonable Phase 0 default but should become a kwarg before this op is used as a general drop-in. Filed as Refinement 2.
- **`cb_partials` is bf16, not fp32**: the bias path costs one extra bf16 rounding step relative to the no-bias path. Promoting `cb_partials` to fp32 + configuring `UnpackToDestMode::UnpackToDestFp32` is the next precision lever. Filed as Refinement 3.

## Precision Baseline

Measured by `test_linear_precision_baseline.py` against the PyTorch reference, comparing in float32 after `ttnn.to_torch`. PCC threshold asserted at 0.999; achieved values reported below.

| Shape (M,K,N) | Bias | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|---------------|:----:|------|-------------|--------------|------------------|
| (32, 32, 32)    | no  | 0.99999930 | 0.125  | 0.001255 | 0.001192 |
| (128, 128, 128) | no  | 0.99999947 | 0.250  | 0.002594 | 0.001034 |
| (128, 256, 128) | no  | 0.99999946 | 0.250  | 0.003880 | 0.001043 |
| (256, 256, 256) | no  | 0.99999945 | 0.250  | 0.003790 | 0.001050 |
| (32, 32, 32)    | yes | 0.99999775 | 0.1875 | 0.003949 | 0.002206 |
| (128, 128, 128) | yes | 0.99999868 | 0.250  | 0.005333 | 0.001657 |
| (128, 256, 128) | yes | 0.99999845 | 0.250  | 0.008554 | 0.001814 |
| (256, 256, 256) | yes | 0.99999826 | 0.500  | 0.009020 | 0.001925 |

**Assessment**: PCC ≥ 0.99999775 across the full shape × bias matrix — well above any practical "matmul correctness" bar. Mean absolute error tracks `O(K^0.5)` as expected for an unbiased K-accumulation in fp32 DEST; max absolute error is bounded by 0.5 even at 256×256×256 with bias, consistent with the "fp32 K-accumulation + 1-2 bf16 rounding events" envelope analyzed in `numerical_stability.md`. The bias path's max error is exactly 2× the no-bias max error at the largest shape, which is the signature of the extra `cb_partials` bf16 round-trip.

**Recommended tolerances** for downstream tests:

- PCC: `>= 0.999` (safely covers worst-case 0.99999775)
- `rtol`: `0.02`, `atol`: `0.15` (matches the existing acceptance suite — no need to tighten)

## Test Results

| Suite | Cases | Result |
|-------|-------|--------|
| `test_linear.py` (acceptance) | 19 | 19/19 PASS |
| `test_linear_extended.py` | 8 | 8/8 PASS |
| `test_linear_precision_baseline.py` | 8 | 8/8 PASS |
| **Total** | **35** | **35/35 PASS** |

Run: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/linear/` — 21.23s, no hangs, no warnings under default mode.

## Recommendations

Synthesizing the data transfer and numerical stability analyses, the priorities for refinement are:

1. **Multi-core (Refinement 1)** — biggest end-to-end win. The data transfer profile is already optimal at 1.0× read amplification; the gating perf lever is parallelizing the `[Mt, Nt]` output grid. The standard 2D-multicast bmm pattern preserves 1.0× amplification across N cores via sender/receiver multicast (see data_transfer.md §9). This refinement also naturally subsumes the per-tile barrier inefficiency by switching the reader to the standard sender pattern (bulk read + barrier per block).

2. **Compute config exposure (Refinement 2)** — small implementation cost, big practical-deployment win. Numerical stability flags `HiFi4 + fp32_dest_acc_en=True` as a known-bad combination on Wormhole B0 (#38306), and HiFi4 is over-conservative for bf16-input matmul where HiFi2 already covers full mantissa precision (see numerical_stability.md "Math Fidelity Profile"). Exposing `compute_kernel_config` lets users dial down to HiFi2/HiFi3 for throughput or to skirt the WH B0 bug. Trivial code change (one kwarg + plumbing into the descriptor); high value.

3. **fp32 partials in bias path (Refinement 3)** — narrow but cleanly-isolated precision lever. Today's bias path costs one bf16 rounding step at `cb_partials`; promoting it to fp32 + `UnpackToDestMode::UnpackToDestFp32` doubles that CB's L1 footprint but pushes the post-matmul precision through to the bias add intact. At Phase 0 sizes the L1 cost is acceptable; needs re-evaluation when paired with K-blocking (Refinement 4).

**Tradeoffs noted**:
- Refinements 3 and 4 conflict on L1 pressure: fp32 `cb_partials` doubles its size, which is fine without K-blocking but tight when K-blocking is added (the partials CB then needs to hold a full output block, not a per-K-block subset). Plan for them in that order.
- Refinement 1 and Refinement 11 (memory pressure) are the same axis — multi-core changes the per-core CB sizing, so memory-pressure auditing must wait until multi-core is in.

**Known limitations to address**:
- Activation fusion (Refinement 6) is exposed by the helpers but unused — straightforward to add when needed.
- ROW_MAJOR input acceptance (Refinement 9) saves a host-side `.to_layout(TILE)` call, valuable for end-to-end model integration.
- Batched matmul (Refinement 7) — `MatmulBlockShape::batch` is a free helper feature; today's validation rejects leading dims != `[1, 1]`.
