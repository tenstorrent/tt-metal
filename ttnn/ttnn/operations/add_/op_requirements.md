# Operation Requirements: add_

## Definition
- **Formula**: `a <- a + b` (`activations=None`); `a <- max(a + b, 0)` (`activations=[RELU]`). The sum is rounded once to the operands' dtype, then clamped. Written into `input_tensor_a`'s own buffer; `input_tensor_b` is untouched.
- **PyTorch Reference**: `input.add_(other)`, and with RELU `input.add_(other).relu_()` (specification `reference`: float64 sum of the stored operands, RELU applied).
- **Import Path**: `from ttnn.operations.add_ import add_`
- **Function Signature**:
  ```python
  def add_(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *,
           activations=None, compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor
  ```
- **Specification**: `eval/specs/add_/spec.py` (governs: RULES R1–R9, REJECTIONS X1, NUMERIC 1 ulp bf16 / 0 ulp fp32).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Backend**: this run is Quasar on ttsim (no clock). A perf refinement here is gated on a **structural count** read off the op by a test or probe, not on device-ns; its gain in time is unconfirmed until a timed backend (`ttsim-perf`) or hardware measures it.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True] (the default compute config)
- **SUPPORTED layout**: [TILE]
- **SUPPORTED placement**: memory_config = [L1], sharding = [height] (regime `sharded_resident`: three rings borrowed from the caller's shards, 0 B NoC, 0 B DRAM, 0 B op-owned L1)
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, h_non_aligned}, rank = [4], broadcast = [none]
- **SUPPORTED op-specific axes**: activation = ["relu"] (packer ReLU, `PackRelu::Zero`)
- **Cores**: the caller's height-shard grid (cores whose shard holds no tile get no kernels, R8); thread plan `THREAD_PLAN = (1, 4, 1)`, C clamped to divide the shard's tile count
- **Compute config**: `default_compute_kernel_config()` = HiFi4, `fp32_dest_acc_en=True`, approx off; fidelity / approx mode passed through, never gated (R3)
- **Golden baseline**: 22 / 22 supported cells passing (14 PHASE0 cells + 8 numeric regression cases), 850 xfail_expected, 288 invalid_skipped, X1 rejection passing; 0 supported_fail / xpass_drift / xfail_wrong_mode

### [ ] Refinement 1 — Interleaved placement (DRAM and L1) + the gate-only axes

**Goal**:
- Build the design's `interleaved_streaming` regime (op_design.md, Regimes) so both operands interleaved in DRAM or in L1 are served natively:
  - add `"DRAM"` to `SUPPORTED["memory_config"]` and `"interleaved"` to `SUPPORTED["sharding"]` (L1 is already there);
  - in the same pass, open the axes that are gate-only on the existing path: add `None` to `SUPPORTED["activation"]` (CT arg `pack_relu = 0`, already wired), `"w_non_aligned"` to `SUPPORTED["alignment"]`, and `2` and `3` to `SUPPORTED["rank"]`. All of these go in both regimes, `sharded_resident` and `interleaved_streaming`.
- The interleaved path:
  - deals tiles over the whole grid with `split_work_to_cores(..., row_wise=True)`;
  - reuses the compute kernel unchanged (same helper call on allocated rings `cb_a_tiles`, `cb_b_tiles`, `cb_result_tiles`, depth `ring_depth_blocks · neo_block_tiles · C`);
  - replaces only the credit-only reader and writer: a reader issuing one tagged implicit-sync `noc.async_read<NocOptions::TXN_ID>` per tile from `a`'s and `b`'s TensorAccessors, and a writer issuing one tagged `noc.async_write<NocOptions::TXN_ID>` per tile into `a`'s accessor. In-place stays safe because a core writes only the pages it read. Both kernels end with `noc_async_*_set_trid(0, noc.get_noc_id())`.
- No tensor is allocated, and there is still one `generic_op` per call.

**Implementation skill**: /interleaved-parallel

**Verifier notes**:
- Hardest generality item, so it goes first. The numeric refinement (2) then extends both regimes once. Refinement 3 (perf) tunes the L1-interleaved deal this refinement creates.
- Build it to the performance-conformance bar, not correct-only:
  - the whole grid engaged;
  - both NoC halves streaming with implicit sync, with no per-tile reserve/barrier/push handshake;
  - ring depth ≥ 2 blocks;
  - the block knobs (`neo_block_tiles`, `ring_depth_blocks`) and the thread plan as single host constants.
- Thread plan:
  - Per the target page, DRAM interleaved is bound by data movement, so the plan is (1, 1, 1).
  - State the plan per regime as one named constant, with its bound.
  - If the L1 path takes C > 1, the every-thread-ran test (`tag_threads`) must cover it.
- Add the one-device-session test for the interleaved path (implicit-sync tag handback): `add_` after another program, and `ttnn.to_layout(..., ROW_MAJOR_LAYOUT)` after `add_`, in one session. The golden suite (one process per cell) cannot see a sticky tag.
- Gate-only axes: a verifier probe (`tests/ttnn/unit_tests/operations/add_/probes/probe_002.py`, SUPPORTED widened in the probe only) ran them on the existing sharded path and they pass: activation None, W ∈ {50, 47, 17, 100}, ranks 2 and 3, all ≤ 0.5 ulp, result in place. For the sharded regime this is a SUPPORTED/gate change plus tests. The interleaved regime must cover them as well.
- Native only: never wrap the op in `ttnn.to_memory_config` / interleaved↔sharded conversions. That would be a second dispatch and is forbidden by the one-dispatch rule.
- L1 ledger: add the three allocated rings as rows, with an axis account for each (they stream over the per-core tile walk and span only `neo_block_tiles · C · ring_depth_blocks`). Add the interleaved data-movement budget: `a` 2 DRAM crossings (read + write), `b` 1.

**Done when**: every golden cell with `memory_config ∈ {DRAM, L1}`, `sharding ∈ {interleaved, height}` (minus INVALID), `activation ∈ {None, relu}`, `alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}`, `rank ∈ {2, 3, 4}` at bfloat16 / `fp32_dest_acc_en=True` passes. `verify_supported` shows 0 supported_fail / xpass_drift / xfail_wrong_mode. The interleaved one-session test and the plan test pass. Phase 0 tests stay green.

### [ ] Refinement 2 — Numerical configurability: float32 and the 16-bit accumulator

**Goal**:
- Add `ttnn.float32` to `SUPPORTED["dtype"]` and `False` to `SUPPORTED["fp32_dest_acc_en"]`, in both regimes.
- Add rule R1's `{"dtype": ttnn.float32, "fp32_dest_acc_en": False}` to `EXCLUSIONS` (refused with `ExcludedCell`).
- Ring page formats follow the tensor dtype; `compute_kernel_config.fp32_dest_acc_en` maps to `enable_32_bit_dest`.
- The float32 cells are graded at **0 ulp**:
  - measure the FPU add first;
  - if it is not exact (the FPU source registers may not hold full fp32), switch the float32 helper call to the SFPU binary add (`binary_sfpu<AddBinary, …>`, op_design.md "Other TARGET axes"), with `UnpackToDest` for the 32-bit pages and `DataFormatReconfig::Disabled` on that chain.
- bfloat16 keeps the FPU path in both DEST widths.
- `default_compute_kernel_config()` keeps `fp32_dest_acc_en=True`.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Comes after Refinement 1 so the dtype work lands once on both regimes' rings. This is a difficulty ordering, not a hard dependency.
- Use `UnpackToDest` for float32 pages only.
- Under `UnpackToDest` every operand of a DEST section lands before the first SFPU op. That keeps the in-place hazard safe: tile i is still packed only after tile i is unpacked. Keep it that way if the block walk changes.
- The packer ReLU must still clamp a float32 result: check the `all_negative`-style data on float32.
- Run the helper at C = 1 and at the plan's C for the SFPU path (target page: verified for eltwise helpers, but the SFPU binary variant on borrowed rings has not been run under a plan).

**Done when**: every golden cell of `dtype ∈ {bfloat16, float32}` × `fp32_dest_acc_en ∈ {True, False}` (minus the R1 exclusion) passes across every placement / activation / alignment / rank that Refinement 1 supports. `{float32, fp32_dest_acc_en=False}` is refused. `verify_supported` loud categories are 0, and prior tests stay green.

### [ ] Refinement 3 — Bank-local deal for L1-interleaved operands

**Type**: perf

**Goal**:
- Target the `{memory_config: L1, sharding: interleaved}` cells that Refinement 1 builds (no LOOSE_CASE is perf-flagged, so the verifier selected this region). The config is bfloat16 (and float32), `fp32_dest_acc_en=True`, both activations.
- With a contiguous `split_work_to_cores` deal, ~31/32 of each core's pages sit in another core's L1 bank, so every tile costs 3 cross-core NoC transfers (read `a`, read `b`, write `a`).
- An L1-interleaved buffer deals page i round-robin over the device's L1 banks, one bank per worker core. A verifier probe (`probes/probe_003.py`) found 32 banks on the 8×4 grid, with `ttnn._ttnn.reports.get_buffer_pages` giving each page's bank and core.
- So deal each core exactly the pages its own bank holds (a strided deal, start bank read off the buffer). All three transfers per tile then become self-aimed, and in-place stays safe because a core still writes only what it read.
- This is a knob-turn of the core-assignment knob (master.md: work distribution / `noc_placement` family, ⭐⭐ T2). The compute kernel is unchanged. No SUPPORTED change.

**Done when**:
- **Structural gate**, read off the op by a test or probe that compares the host deal with the buffer's page→core map: on L1-interleaved cells, cross-core (non-self-aimed) NoC transfers per output tile drop from ≈ 3 · 31/32 (Refinement 1's deal; record the measured before value) to **0**. Per-core tile counts stay balanced within one tile.
- The golden suite is green.
- No regression on the config-spanning guard set, one representative per kernel path × placement: height-sharded L1, DRAM interleaved, L1 interleaved; bfloat16 and float32. The structural counts of the other placements must be unchanged (sharded 0 B NoC; DRAM `a` 2 / `b` 1 crossings).
- The gain in time is **unconfirmed** on ttsim; say so in the changelog and do not report a simulator timing as a speedup.

## Omitted values

None: every value of `TARGET − SUPPORTED` is named in a refinement above. (`layout` and `broadcast` have no missing value. INVALID `{DRAM, height}` is structural. No `INVALID_FOR_ARCH` entry exists; `bfloat8_b` is not in TARGET.)
