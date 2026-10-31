# Operation Requirements: layer_norm_tile

## Definition
- **Formula**: `x' = x + residual` (if given); `mean[r] = (1/W) sum_j x'[r,j]`; `c = x' - mean`; `var[r] = (1/W) sum_j c[r,j]^2`; `out[r,j] = c[r,j] * rsqrt(var[r] + epsilon) * gamma[j] + beta[j]` (W = logical last dim, 768 under the contract)
- **PyTorch Reference**:
  ```python
  def pytorch_layer_norm(x, gamma, beta, residual=None, *, epsilon=1e-5):
      xf = x.float() if residual is None else x.float() + residual.float()
      return torch.nn.functional.layer_norm(
          xf, (x.shape[-1],), weight=gamma.float().reshape(-1), bias=beta.float().reshape(-1), eps=epsilon
      ).to(x.dtype)
  ```
- **Import Path**: `from ttnn.operations.layer_norm import layer_norm` (re-export of `ttnn.operations.layer_norm_tile`)
- **Function Signature**:
  ```python
  layer_norm(
      input_tensor: ttnn.Tensor,                    # (B, 1, S, 768) bf16 TILE, interleaved (L1 per contract)
      gamma: ttnn.Tensor,                           # (768,) / (1,1,1,768) TILE, or (1,1,24,32) / (1,1,1,768) ROW_MAJOR
      beta: ttnn.Tensor,                            # same forms as gamma, independently
      residual_input_tensor: ttnn.Tensor = None,    # same shape / dtype / layout as input_tensor
      *,
      epsilon: float = 1e-5,
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None,  # None -> HiFi4, fp32_dest_acc_en=True
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory - the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, ... (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements - a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

**Queue basis.** `eval/golden_tests/layer_norm_tile/feature_spec.py` does not exist, so TARGET is the design's
"Expected TARGET axes" table (`op_design.md`), which restates the prompt contract. Every TARGET value is in
SUPPORTED, so `TARGET - SUPPORTED` is empty and there is no generality candidate. Every refinement below is a
measured perf refinement. No `LOOSE_CASES` entry exists to carry an `attention:` perf flag, so the regions are
selected from the verifier's measurements (`verification_report.md`, "Performance baseline").

**Target config for every perf refinement**: the op default (`compute_kernel_config=None`: HiFi4,
`fp32_dest_acc_en=True`), bf16 TILE input and residual in L1 interleaved, both affine forms (`tile_dram`,
`rm_l1`). All of it is already in SUPPORTED.

**Measurement protocol**: `scripts/run_safe_pytest.sh --run-all --profile
tests/ttnn/unit_tests/operations/layer_norm_tile/test_layer_norm_tile_perf.py`. Read `DEVICE KERNEL DURATION
[ns]`. Each case dispatches twice; take the min per case, then the median over at least 3 invocations of the
file. Run-to-run spread is 3-10%. R = 64 with residual is bimodal (~26 or ~33 us with unchanged code), so do
not claim a win or a regression on that cell from fewer than 5 invocations. The **guard set** is that file's
matrix: 5 shapes (32 / 64 / 128 / 248 / 256 tile-rows) x {tile_dram, rm_l1} x {no residual, residual} x
{default, fp32_acc_off, lofi}. It covers both affine paths, both intermediate formats, both residual
producers (reader at R = 32, writer at R >= 64), the split-NoC x path (rm_l1, no residual, R >= 128), and
aligned and non-aligned S. Stage attribution: `test_layer_norm_tile_ablation.py --profile`
(`LAYER_NORM_ABLATE_*` defines; see its docstring).

### [x] Phase 0 - Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment = {tile_aligned, h_non_aligned}
- **SUPPORTED op-specific axes**: fp32_dest_acc_en = {True, False}; residual = {none, present}; affine_layout = {TILE, ROW_MAJOR, mixed}; memory_layout = {INTERLEAVED} (L1 or DRAM buffer)
- **EXCLUSIONS**: none
- **Regime**: `row_resident`:
  - Tile-rows are split over the full grid (`split_work_to_cores(..., row_wise=True)`), with the whole W resident.
  - The compute block is one tile-row (`BLOCK_HEIGHT_MAX = 1`). The reader runs ahead up to `input_depth <= INPUT_DEPTH_MAX = 4` tile-rows.
  - From `R >= RESIDUAL_SPLIT_MIN_TILE_ROWS (64)` the residual stream is fetched by the writer RISC on NoC1; below that, by the reader beside x.
  - Without residual, with gamma/beta in L1, cores owning >= 2 tile-rows alternate x reads over NoC0 / NoC1 (`DM_DYNAMIC_NOC`).
  - The writer fetches gamma/beta row 0 once per core.
  - Input, residual and output each cross the NoC once.
- **Cores**: multi-core, `min(R, 110)` on Blackhole p100a, `R = B * ceil(S/32)`
- **Compute config**: `default_compute_kernel_config()` = HiFi4, `fp32_dest_acc_en=True`; `math_fidelity` / `math_approx_mode` passed through ungated
- **Golden baseline**: no golden suite exists. Local: 232/232 (acceptance 80, extended 12, precision 16, perf 60, ablation 64). Acceptance + extended 92/92 under `--dev`.
- **Device time** (default config, median of 3, us; rm_l1 nores / rm_l1 res / tile_dram nores / tile_dram res):

  | R (tile-rows) | rm_l1 nores | rm_l1 res | tile_dram nores | tile_dram res |
  |---|---|---|---|---|
  | 32 | 14.0 | 17.6 | 14.0 | 17.9 |
  | 64 | 19.2 | ~26-33 (bimodal) | 19.8 | ~27-35 (bimodal) |
  | 128 | 31.5 | 37.2 | 33.6 | 41.1 |
  | 248 | 40.3 | 53.1 | 47.2 | 54.4 |
  | 256 | 41.0 | 59.6 | 49.7 | 57.3 |

- **Pre-queue perf work**: the first queue's Refinement 2 levers (a) "balance the two NoCs" and (d) "Overlap lamp" landed in Phase 0 (implementer, after the first verifier pass): R=128 res 59.8 -> 37.2 us, R=256 res 96.5 -> 59.6 us, R=256 nores 60.2 -> 41.0 us. The queue below is re-filed against that baseline.

### [ ] Refinement 1 - `w_split_combine`: split W across cores with a cross-core statistic combine

**Type**: perf

**Goal**: build the design's deferred `w_split_combine` regime (`op_design.md`, Regimes / Traffic ranking rank 3-4) for the shapes where `row_resident` under-fills or raggedly fills the 110-core grid:
- `B*S = 1024` gives R = 32: 32 cores busy, 78 idle.
- `B*S = 2048` gives R = 64: 46 cores idle.
- `B*S = 4096` gives R = 128: 18 cores carry 2 tile-rows and set the makespan.

At R = 32 the op is compute-bound. Stage ablation: `compute_only` 10.2 of 13.9 us, `reads_only` at the 2.6 us floor. Net compute is about 7.6 us per 24-tile row (9.4 us with residual).

Split each tile-row's W across `k` cores (`k` a divisor of `Wt = 24`), optionally grouping `g` tile-rows per core, so `(R/g) * k` fills the grid. Examples:
- R = 32: `g=1, k=3` gives 96 cores, each core a 1 x 8-tile block.
- R = 64: `g=2, k=3` gives 96 cores, each 2 x 8 tiles.
- R = 128: `g=4, k=3` gives 96 cores, each 4 x 8 tiles; the makespan drops from 2 rows to 4/3 row-equivalents.

Each core reduces its slice to col-0 partials. Two combine rounds per block (row-sum partials for the mean, then centered sum-of-squares partials) give every core of a `k`-group the full-row statistic. `1/W` stays the logical W, never the slice width. Each core then normalizes its slice against its own gamma/beta slice.

Relevant `ttnn/ttnn/operations/examples/master.md` entries:
- `width_split`.
- `tensix_all_reduce`: root reduce plus one multicast back for a 1-D line with a tiny payload. The design's alternative is a unicast all-gather where every core combines.
- `tensix_all_reduce_compute`: FPU `acc_to_dest` combine in fp32 DEST.
- `mcast_topology`: transport through `ttnn.Mcast` + `McastArgs`, never raw multicast + semaphores.
- `noc_placement`: keep a group's `k` cores adjacent, in one grid row.

No SUPPORTED change.

**Verifier notes**: scheme-change (a dependent axis cut across cores), so it stands alone. It goes first because every later refinement tunes the per-core block this one redefines.

The precision gate the design put on this regime is met: fp32 DEST PCC 0.999998 / rel-RMS 0.19%, bf16 DEST PCC 0.99999 / rel-RMS 0.52%. The combine only reorders the two sums.

Knobs and sizing:
- `k`, `g` and the regime choice are new `derive_blocking()` knobs with one source of truth. Selection stays one `generic_op` dispatch per call.
- Every CB that spans W (`cb_input_tiles`, `cb_residual_tiles`, the `work` group, `cb_gamma_rows`, `cb_beta_rows`, `cb_output_tiles`) scales with the slice width through `_cb_plan()`.

Keep the Phase 0 NoC decisions working in the new regime, or re-measure them per regime:
- The residual on the writer RISC / NoC1 from `RESIDUAL_SPLIT_MIN_TILE_ROWS`.
- The per-core x NoC split from `X_NOC_SPLIT_MIN_CORE_ROWS`.
- Reader run-ahead `input_depth`.
- `block_height = 1` for compute.

The combine's rendezvous adds a cross-core dependency on NoC traffic, so DM placement matters more than in Phase 0. Keep the stall-shadow rows from the design:
- The writer fetches only its gamma/beta slice.
- The next tile-row's local partial runs while a combine is in flight.

Pick `(g, k)` per R by measurement. A shape class that does not win stays on `row_resident`, with the reason recorded in the regime table. Update `l1_ledger.md` with the new partial / receive CBs and the cross-core traffic of `2 (k-1) T_acc` per core per tile-row. Add acceptance cases pinned to the new regime: aligned and non-aligned S, residual on and off, both affine forms, `fp32_dest_acc_en` on and off.

**Done when**:
- Device-ns improves (median of 3) on `(32, 1, 32, 768)` and `(2, 1, 512, 768)`, with and without residual, both affine forms, at the default config.
- Any R = 64 / 128 shape moved to the new regime also improves (R = 64 residual judged over >= 5 invocations).
- No regression across the guard set.
- All layer_norm_tile tests green.
- Precision-baseline PCC and rel-RMS within 10% of the Phase 0 numbers.

### [ ] Refinement 2 - `affine_mcast`: fetch gamma/beta once and multicast them

**Type**: perf

**Goal**: build the design's deferred `affine_mcast` regime row. One injector reads each gamma/beta row-0 slice once and multicasts it into `cb_gamma_rows` / `cb_beta_rows` on every core that needs it (`ttnn.Mcast` + `McastArgs`). This replaces today's per-core fetch, where every active core reads the same 24 TILE pages.

With TILE gamma/beta in DRAM, the fetch is no longer off the critical path:
- It costs time directly: R=128 res, tile_dram 41.1 vs rm_l1 37.2 us.
- It is why `split_x_nocs` is gated on `affine_in_l1`. Dynamic-NoC mode slows the writer's DRAM fetch (+5 us, implementer), so DRAM-affine shapes lose the x NoC split: R=248 nores 47.2 vs 40.3 us, R=256 nores 49.7 vs 41.0 us.

Target: `(256, 1, 16, 768)` and `(124, 1, 33, 768)` without residual, and `(8, 1, 512, 768)` with residual, all with tile_dram affine at the default config.

Expected end state:
- The tile_dram cells close on the rm_l1 cells.
- The `affine_in_l1` gate on `split_x_nocs` is lifted, or re-justified by measurement.

Relevant `master.md` entries:
- `shared_input_reuse` (read once, broadcast).
- `mcast_topology` (transport and topology).
- `noc_placement`.

No SUPPORTED change.

**Verifier notes**: reuse multicast, a new dataflow topology (one injector plus a broadcast), so it stands alone (T3). It follows Refinement 1 because `w_split_combine` changes which gamma/beta slice each core needs (`Wt/k` tiles per `k`-group position), and the mcast topology must be built on that final slice map.

Re-measure the tile_dram vs rm_l1 gap after Refinement 1 lands. The per-core DRAM fetch shrinks by `k` there, which may close part of it. If the gap on all three target cells is then within run-to-run noise, record the measurement and close this refinement with no code change.

Validate by building the variant and measuring device-ns, not by a remove-the-fetch ablation, which under-shows a broadcast's value. The mcast rendezvous must not land on the R = 32 critical path: gamma/beta are first needed at `scale_block`, and the stall-shadow rule from the design holds. Record the measured choice and the new affine traffic in `l1_ledger.md`.

**Done when**:
- Device-ns improves (median of 3) on the three target cells (tile_dram, default config).
- No regression across the guard set, including the rm_l1 cells and R = 32.
- All layer_norm_tile tests green.
- Alternatively, the post-Refinement-1 gap measures within noise and is recorded as such.

### [ ] Refinement 3 - Overlap the first-row reads with compute on multi-row cores

**Type**: perf

**Goal**: on cores that own several tile-rows, reads and compute still largely serialize. Stage ablation (rm_l1, default config):

| Shape | `full` (us) | `reads_only` (us) | `compute_only` (us) | Overlapped ideal |
|---|---|---|---|---|
| R=128, no residual | 30.5 | 10.7 | 19.8 | ~20 us |
| R=256, no residual | 39.4 | 14.4 | 29.3 | ~29 us |
| R=256, residual | 55.6 | 31.6 | 34.3 | ~35 us |

Compute cannot start `mean_block` until a whole 24-tile row has landed, and every core issues its first row at the same moment, so the first-row arrival is set by grid-wide contention.

Target: `(8, 1, 512, 768)` and `(256, 1, 16, 768)`, with and without residual, both affine forms, at the default config.

Levers (T2, each measured; the implementer picks):
- **Finer reader->compute handoff on the first row.** Push sub-row chunks so the first compute phase starts on landed tiles. It needs a compute-side wait granularity below one row: `mean_block` must keep x resident for `center_block`.
- **Bank stagger.** Rotate each core's start column in its row walk (`bank_stagger`), so the first wave does not pile onto the same banks.
- **Transaction-id barriers.** Keep the next row's reads on the wire across the current row's barrier (`trid_double_issue`). Co-tune with `input_depth`.
- **The R=64 residual bimodality** (~26 vs ~33 us with unchanged code). Attribute it with the ablation harness before choosing a lever; a placement- or bank-dependent stall is the likely cause.

Relevant `master.md` entries: `double_buffer`, `trid_double_issue`, `bank_stagger`, `split_reader`, `noc_placement`.

No SUPPORTED change.

**Verifier notes**: T2 knob and transfer-shape levers on the existing block surface, so several fit one phase. It follows Refinements 1 and 2 because both change which cores read what (per-core block shape, affine traffic, the dynamic-NoC gate). Gate on headroom: re-run the ablation first. If `full` is within ~10% of `max(reads_only, compute_only)` on the targets after Refinements 1-2, record that and close with no change.

Every new chunk size, stagger or trid depth is a `derive_blocking()` knob with one source. The L1 ledger's `cb_input_tiles` / `cb_residual_tiles` run-ahead window (`input_depth` rows vs a consumer live set of 1 row / 1 tile) is co-tuned here; record the measured choice in `l1_ledger.md`.

**Done when**:
- Device-ns improves (median of 3) on both target shapes, both residual modes, both affine forms, at the default config.
- No regression across the guard set; the R=64 residual cell is judged over >= 5 invocations.
- All layer_norm_tile tests green.

### [ ] Refinement 4 - Trim the per-tile-row compute chain

**Type**: perf

**Goal**: lower the per-tile-row compute cost: net ~7.6 us per 24-tile row without residual, ~9.4 us with residual (stage ablation, R=32). It is the floor for every shape once Refinements 1-3 have spread and overlapped data movement, and it is the whole cost at R=32.

Levers, each measured:
- (a) **Format reconfig.** Drop the helpers' data-format reconfig at phase boundaries where formats provably do not change (`compute_block_size`, "second lever", up to 1.19x). `center` -> `sumsq` -> `normalize` -> `scale` all read and write `T_acc` CBs. `mean_block` / `rstd_block` pass `INPUT_AND_OUTPUT` explicitly.
- (b) **SFPU scope.** Scope the `rstd_block` post-op SFPU work (`mul_unary_tile`, `add_unary_tile`, `rsqrt_tile`) to column 0 of the stat tile (`sfpu_tile_scope`, `c_skip`). It is one tile per tile-row, so measure before keeping it.
- (c) **Residual add cost.** The add is 24 tiles per tile-row, yet residual adds ~1.8 us of compute at R=32 (`compute_only` 12.0 vs 10.2). Attribute it before choosing a lever (`add` init/reconfig vs `cb_xsum` round trip).
- (d) **SFPU init hoisting.** Hoist the `binop_with_scalar_tile_init` / `rsqrt_tile_init` re-inits out of the per-output post-op where the helper contract allows.

Reference points:
- LoFi saves 1.7 us at R=32 (12%), so fidelity is a minor term.
- `fp32_dest_acc_en=False` is within noise.
- Do **not** fuse normalize/scale/shift through FPU DEST-reuse: `compute_fusion` measures that as a 0.82x loss for FPU-after-FPU.

Target: `(32, 1, 32, 768)` and `(16, 1, 128, 768)`, with and without residual, both affine forms, default and `fp32_acc_off`. No SUPPORTED change.

**Verifier notes**: compute-local T1/T2 levers, independent of the topology, so it goes last and blocks nothing. Lever (a) is legal only where the format is genuinely constant across the boundary; `cb_input_tiles` / `cb_residual_tiles` -> `T_acc` transitions keep their reconfig. Precision must not move: (a), (b) and (d) change only MMIO / lane scope / init placement.

**Done when**:
- Device-ns improves (median of 3) on the target shapes at both configs (R = 64 residual over >= 5 invocations).
- No regression across the guard set.
- Precision-baseline PCC and rel-RMS unchanged within 2%.
- All layer_norm_tile tests green.
