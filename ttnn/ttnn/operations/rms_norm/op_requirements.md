# Operation Requirements: rms_norm

## Definition
- **Formula**: `out = x * rsqrt(mean(x², dim=-1) + eps) [* gamma]`, mean over the true (unpadded) width W
- **PyTorch Reference**: `def ref(x, gamma=None, eps=1e-6): y = x.float() / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps); return y * gamma.float().reshape(-1) if gamma is not None else y`
- **Import Path**: `from ttnn.operations.rms_norm import rms_norm`
- **Function Signature**: `rms_norm(input_tensor: ttnn.Tensor, *, gamma: Optional[ttnn.Tensor] = None, epsilon: float = 1e-6, compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N`. Partial follow-ups are named `Refinement Nb`, `Nc`, … (never `N.5`), and ordered immediately after their parent. The parser matches exactly `Refinement \d+[a-z]?`.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16]; **fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE, ROW_MAJOR]
- **SUPPORTED shape-derived axes**: alignment = all three, rank = 2/3/4
- **SUPPORTED op-specific axes**: gamma_mode ∈ {gamma, no_gamma}; gamma_dtype ∈ {float32, bfloat16, none}; gamma_layout ∈ {ROW_MAJOR, none}; memory_layout = INTERLEAVED
- **Cores**: full grid, row-parallel (`split_work_to_cores`); R1 row-resident / R2 two-pass streamed
- **Compute config**: HiFi4 + fp32_dest_acc_en default (config passed through)
- **Golden baseline**: 436 / 436 supported cells pass

### [ ] Refinement 1 — Precision surface + TILE gamma (perf-flagged-profile contract, part 1)

**Goal**: the perf-profile loose cases (`_PERF_BASE`: bf16, `fp32_dest_acc_en=False`, HiFi2, bf16 **TILE** gamma) are the config every perf refinement targets. Land that whole contract:
- add `False` to `SUPPORTED["fp32_dest_acc_en"]` (fp32 input + `False` stays an `ExcludedCell`); drop `cb_sq`/`cb_xn`/`cb_stat` to bf16 pages when DEST is 16-bit and keep fp32 when it is on;
- add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` (TILE-only; bf8b + non-aligned is INVALID; bf8b+RM is INVALID) and `ttnn.bfloat8_b` to `gamma_dtype`;
- add `ttnn.TILE_LAYOUT` to `SUPPORTED["gamma_layout"]`: the reader reads gamma tiles directly (row 0 of the tile is the broadcast row), no scatter;
- `math_fidelity` stays a passthrough.
Cells that fail out of the box go to `EXCLUSIONS`, not their own refinement.

**Implementation skill**: /numeric-formats-metal, /memory-layouts

**Verifier notes**: the numerics bundle and gamma layout are one refinement because the perf-1 target needs both. Knob-turn class (descriptor-level CB formats + reader gamma path). Keep the multi-core row split and the segment-batched reader/writer on the new paths — don't ship a naive TILE-gamma path that perf 1 will rewrite. The untilize helper cannot output bf8b (RM is INVALID for bf8b, so no conflict).

**Done when**: `fp32_dest_acc_en=False`, bf8b and TILE/bf8b gamma `xfail_expected` cells move to `supported_pass`; the perf-profile loose cases (interleaved) pass at PCC 0.9995; zero `xpass_drift`.

### [ ] Refinement 2 — W-split across cores with cross-core partial combine (perf-flagged-profile contract, part 2)

**Goal**: shapes whose independent axes under-fill the grid or whose row exceeds L1 (decode `32×{1024..7168}`, `1×16384`, `1×32768`, `64×12288`, W-heavy resilience shapes) currently run on few cores and/or in the 2×-DRAM-read R2 regime. Split the dependent W axis across cores (logical width sharding of the interleaved tensor): each core keeps its W-slice resident (x read once), computes a partial Σx² tile per row, partials are combined across the cores of a row group (`ttnn.Mcast` + `McastArgs` sender/receiver + semaphores; the `cb_stat`/`Accumulate` slot is the combine point), then each core normalizes its slice. Keep both regimes available behind the host predicate in `pick_blocking`. The output of `cb_stat` is the existing slot, so the loop nest is unchanged.

**Implementation skill**: none in inventory — state the topology explicitly: partial reduce → gather/all-gather of `n_w` fp32 (or bf16 when DEST is 16-bit) partial tiles per row block → local add → inv_rms → normalize. Reference `op_design.md` regime R3 and the master.md mcast pattern.

**Verifier notes**: scheme-change, stands alone. Ordered second because the perf-1 contract includes the decode shapes, which are only fast with a filled grid. Must land on the Refinement 1 config (bf16 / `fp32_dest_acc_en=False`). Folds in the ledger finding: record the new combine CBs in `l1_ledger.md` with their axis accounting (spans `n_w`, streams over rows).

**Done when**: all Phase 0 + R1 cells still pass; the three wide `_WIDE` loose cases plus the decode perf cases run on > Rb-limited core counts (verify core count in the test) with device-ns for `32×5120`-class cells measurably below Phase 0; golden green.

### [ ] Refinement 3 — Speed up the perf-flagged profile

**Type**: perf

**Goal**: no loose case carries an `attention:` flag; free-select the region: the `_PERF_BASE` interleaved cases (bf16, HiFi2, `fp32_dest_acc_en=False`, TILE gamma), prefill `8192×{1024,2304,5120,7168}` (bandwidth-bound — x/out crossing DRAM once, gamma re-read per core) and decode `32×{1024..7168}`. Levers from `ttnn/ttnn/operations/examples/master.md`: block-size / buffer-depth co-tune (`G_CAP` 16→32, `Rb` > 1 for wide `Wb`, `x_depth` / `OUT_DEPTH`), fix the `Sb=1` degeneration for prime `Wt` via a ragged tail segment, the fused scale chain (gamma multiply without the `cb_xn` round trip), and gamma mcast from one injector (R6) for tall cases. Chunk granularity floor: whole tiles minimum, coarser amortizes. Prefer the cheap knob-tunes first; gamma mcast is its own T-higher lever.

**Done when**: measured device-ns improves on the targeted perf-case shapes (e.g. `8192×5120`, `32×5120`) toward their `achievable_ns` (scaled by AICLK), the 0.9995 PCC gate holds, golden suite green, no regression on the guard set (one representative per kernel path × layout × placement: R1 TILE, R1 RM, R2 streamed, R3 W-split).

### [ ] Refinement 4 — HEIGHT_SHARDED in place + output `memory_config`

**Goal**: add `HEIGHT_SHARDED` to `SUPPORTED["memory_layout"]` and honor `memory_config` for the output (sharded input → sharded output). Local shard = whole row block; back `cb_x` / `cb_out_tiles` with `ttnn.cb_descriptor_from_sharded_tensor` (zero-copy, no reader NoC re-read of the local shard, `Rb` = shard rows). Includes the `_SHARDED`/resilience HEIGHT cells; ragged shards must work.

**Verifier notes**: knob-turn / CB-placement change over R1; no new math and no cross-core traffic. The native path means the shard is consumed in place — never re-read through a `TensorAccessor`. No existing skill covers sharding placement (don't attach `/memory-layouts`). RM input × TILE gamma × sharded is INVALID in the spec and stays skipped.

**Done when**: HEIGHT_SHARDED `xfail_expected` cells pass; dataflow check confirms no `TensorAccessor` read of the local shard.

### [ ] Refinement 5 — WIDTH_SHARDED and BLOCK_SHARDED with cross-core Σx² combine

**Goal**: add `WIDTH_SHARDED` and `BLOCK_SHARDED` to `SUPPORTED["memory_layout"]`. The reduced axis is cut across cores, so reuse Refinement 2's partial-combine topology with the slice pre-placed in L1 (CB backed by the shard); block-sharded combines along each grid row. Output inherits the shard spec.

**Verifier notes**: scheme-change in kind, but it extends Refinement 2's combine rather than adding a new one — hence it must follow Refinement 2 (true dependency). Ordered last among generality because it sits on R2 and R4. No inventory skill; state the topology (partial Σx² + gather/all-gather along the core row).

**Done when**: WIDTH/BLOCK_SHARDED `xfail_expected` cells (including the resilience shapes with ragged/prime tile counts) pass; the sharded perf loose cases pass at 0.9995.

### [ ] Refinement 6 — Speed up the sharded perf cases

**Type**: perf

**Goal**: the sharded `_perf_case` entries (WIDTH_SHARDED decode, BLOCK_SHARDED prefill/640-token cases) against their `achievable_ns`. Levers from `master.md`: overlap of the combine with compute, mcast vs all-gather topology, subblock/inv-rms chain, fidelity HiFi2. No SUPPORTED change.

**Done when**: measured device-ns improves on the sharded perf-case shapes, PCC gate 0.9995 holds, golden green, no regression on the config-spanning guard set.

## TARGET − SUPPORTED coverage
| Axis value | Where |
|---|---|
| fp32_dest_acc_en=False | Refinement 1 |
| dtype bfloat8_b | Refinement 1 (bf8b+RM, bf8b+non-aligned are INVALID) |
| gamma_dtype bfloat8_b | Refinement 1 |
| gamma_layout TILE | Refinement 1 |
| memory_layout HEIGHT_SHARDED | Refinement 4 |
| memory_layout WIDTH_SHARDED / BLOCK_SHARDED | Refinement 5 (depends on 2) |
