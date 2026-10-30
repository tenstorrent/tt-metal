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
file. Run-to-run spread is 3-10%, worst on the 248/256 tile-row residual cases. The **guard set** is that
file's matrix: 5 shapes (32 / 64 / 128 / 248 / 256 tile-rows) x {tile_dram, rm_l1} x {no residual, residual}
x {default, fp32_acc_off, lofi}. It covers both affine reader paths, both intermediate formats, the
single-block and multi-block paths, and aligned and non-aligned S.

### [x] Phase 0 - Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment = {tile_aligned, h_non_aligned}
- **SUPPORTED op-specific axes**: fp32_dest_acc_en = {True, False}; residual = {none, present}; affine_layout = {TILE, ROW_MAJOR, mixed}; memory_layout = {INTERLEAVED} (L1 or DRAM buffer)
- **EXCLUSIONS**: none
- **Regime**: `row_resident`. Tile-rows are split over the full grid (`split_work_to_cores(..., row_wise=True)`) with the whole W resident. Block = the whole per-core assignment when it fits `block_height_cap(1)`; otherwise multi-block with `input_depth = 2`. Input, residual and output each cross the NoC once. The writer fetches gamma/beta row 0 once per core.
- **Cores**: multi-core, `min(R, 110)` on Blackhole p100a, with `R = B * ceil(S/32)`
- **Compute config**: `default_compute_kernel_config()` = HiFi4, `fp32_dest_acc_en=True`; `math_fidelity` / `math_approx_mode` passed through ungated
- **Golden baseline**: no golden suite exists. Acceptance 80/80 (prod + `--dev`), extended 12/12 (prod + `--dev`), precision baseline 16/16, perf guard set 60/60
- **Device time (default config, rm_l1, no residual / residual)**: R=32 14.1 / 17.7 us; R=64 21.1 / 29.0 us; R=128 33.3 / 59.8 us; R=248 54.9 / 93.6 us; R=256 60.2 / 96.5 us

### [ ] Refinement 1 - `w_split_combine`: split W across cores with a cross-core statistic combine

**Type**: perf

**Goal**: build the design's deferred `w_split_combine` regime (`op_design.md`, Regimes / Traffic ranking rank 3-4). Every contract shape under-fills the 110-core grid or is ragged on it:
- `B*S = 1024` gives R = 32: 32 cores busy, 78 idle.
- `B*S = 2048` gives R = 64: 64 cores busy.
- `B*S = 4096` gives R = 128: 18 cores carry 2 tile-rows and set the makespan.

At R = 32 the op is compute-bound: about 11 us of the 14 us per tile-row is compute, and the DM-only stub floor is 2.8 us without the affine fetch.

Split each tile-row's W across `k` cores (`k` a divisor of `Wt = 24`), optionally grouping `g` tile-rows per core, so `(R/g) * k` fills the grid. Examples:
- R = 32: `g=1, k=3` gives 96 cores, each core a 1 x 8-tile block.
- R = 64: `g=2, k=3` gives 96 cores, each core 2 x 8 tiles.
- R = 128: `g=4, k=3` gives 96 cores, each core 4 x 8 tiles. The makespan drops from 2 rows to 4/3 row-equivalents.

Each core reduces its slice to col-0 partials. Two combine rounds per block (row-sum partials for the mean, then centered sum-of-squares partials) give every core of a `k`-group the full-row statistic. `1/W` stays the logical W, never the slice width. Each core then normalizes its resident slice against its own gamma/beta slice.

Relevant `ttnn/ttnn/operations/examples/master.md` entries:
- `width_split` (the occupancy evidence).
- `tensix_all_reduce`. For a 1-D line with a tiny payload, root reduce plus one multicast back is the recommended shape. The design's alternative is a unicast all-gather where every core computes the combine.
- `tensix_all_reduce_compute` (FPU `acc_to_dest` combine in fp32 DEST).
- `mcast_topology` (transport through `ttnn.Mcast` + `McastArgs`, never raw multicast + semaphores).
- `noc_placement` (keep a group's `k` cores adjacent, in one grid row).

No SUPPORTED change.

**Verifier notes**: scheme-change (a dependent axis cut across cores), so it stands alone. It goes first because Refinements 2 and 3 tune per-core shapes that this regime redefines. The design gated this regime on a `row_resident` precision baseline. That baseline now exists: fp32 DEST PCC 0.999998 / rel-RMS 0.19%, bf16 DEST PCC 0.99999 / rel-RMS 0.52%. The combine only reorders the two sums.

`k`, `g` and the regime choice are new `derive_blocking()` knobs with one source of truth (CT/RT args derived from them; no restated literals). Regime selection stays one `generic_op` dispatch per call. Every CB that spans W (`cb_input_tiles`, `cb_residual_tiles`, the `work` group, `cb_gamma_rows`, `cb_beta_rows`, `cb_output_tiles`) must scale with the slice width, not `Wt`. `_cb_plan()` is the single place to change that.

Keep the stall-shadow rows from the design:
- The writer fetches only its gamma/beta slice.
- The next tile-row's local partial runs while a combine is in flight.

Pick `(g, k)` per R by measurement. If a shape class does not win, keep it on `row_resident` and record why in the regime table. Update `l1_ledger.md` with the new partial / receive CBs and the cross-core traffic of `2 (k-1) T_acc` per core per tile-row. Add acceptance cases pinned to the new regime: aligned and non-aligned S, residual on and off, both affine forms, `fp32_dest_acc_en` on and off.

**Done when**: device-ns improves (median of 3) on `(32, 1, 32, 768)` and `(2, 1, 512, 768)`, with and without residual, both affine forms, at the default config. Any R = 64 / 128 shape moved to the new regime also improves. No regression across the guard set. All layer_norm_tile tests are green, with precision-baseline PCC and rel-RMS within 10% of the Phase 0 numbers.

### [ ] Refinement 2 - Raise NoC throughput on the data-movement-bound shapes

**Type**: perf

**Goal**: the 128+ tile-row shapes, and every residual shape, are bound by data movement, not compute. Evidence:
- The implementer's DM-only stub (compute stubbed, affine fetch removed) runs `(8, 1, 512, 768)` at 15.5 us without residual and 37.9 us with residual. The full op takes 33.3 / 59.8 us.
- Adding the residual costs +26 us at R = 128 but only +3.6 us at R = 32 (rm_l1).

All input and residual reads go through one reader RISC on NoC0, which carries 2/3 of the bytes when a residual is present. NoC1 carries only the output writes. Aggregate throughput on the residual case is about 500 GB/s, against about 810 GB/s without residual.

Levers (T2 knob-turns on the existing block surface; each to be measured):
- (a) **Balance the two NoCs.** Move one stream (the residual, or half of each tile-row's reads) onto the writer RISC / NoC1 (`split_reader`). Verifier ablation: issuing NoC1 reads from the NCRISC reader fails in the default dedicated-NoC mode, so this needs either the second RISC or dynamic-NoC mode.
- (b) **Bank stagger.** Rotate each core's start column in its tile-row walk (`bank_stagger`). Row `g` starts on bank `24g mod 110`, which takes only 55 distinct values, so pairs of cores start on the same bank.
- (c) **Barrier batching / transaction ids** across the input and residual streams (`trid_double_issue`, `double_buffer`).
- (d) **The design's "Overlap" lamp**, per regime. Verifier ablation (`block_height=1, input_depth=2` forced) with residual: +3% at R=128 (59.0 -> 57.3 us), +7% at R=248 (101.2 -> 93.9 us) and +3% at R=256 (97.5 -> 94.5 us). Without residual it is neutral at R=128/248 but -19% at R=256 (56.0 -> 66.6 us). So the choice must be a regime-dependent knob, not a global flip.

Target: `(8, 1, 512, 768)` and `(256, 1, 16, 768)` with residual, both affine forms, default config. Relevant `master.md` entries: `split_reader`, `noc_placement`, `bank_stagger`, `trid_double_issue`, `double_buffer`. No SUPPORTED change.

**Verifier notes**: T2, several levers per phase. It follows Refinement 1 because the per-core block shape (`g x Wt/k`) changes there, and the same NoC levers must cover both regimes. Folded L1-ledger finding: on the residual path, `cb_input_tiles` and `cb_residual_tiles` are sized to the whole block (`input_depth * block_height * Wt`) while their consumer-side live set is one tile (`add_residual_block` streams them). The ledger's stated reason is whole-block reader run-ahead. Co-tune that window with levers (c)/(d) and record the measured choice in `l1_ledger.md`. Every new chunk, depth or stagger value is a `derive_blocking()` knob with one source.

**Done when**: device-ns improves (median of 3) on both target shapes for both affine forms at the default config. No regression across the guard set, including no-residual R = 256 and DRAM-interleaved input. All layer_norm_tile tests green.

### [ ] Refinement 3 - Trim the per-tile-row compute chain

**Type**: perf

**Goal**: lower the per-tile-row compute cost (about 11 us at R = 32), which is the floor for every shape once Refinements 1 and 2 have spread and overlapped data movement. Levers, each measured:
- (a) **Format reconfig.** Drop the helpers' data-format reconfig at phase boundaries where formats provably do not change (`compute_block_size`, "second lever", up to 1.19x). `center` -> `sumsq` -> `normalize` -> `scale` all read and write `T_acc` CBs. `mean_block`/`rstd_block` pass `INPUT_AND_OUTPUT` explicitly.
- (b) **SFPU scope.** Scope the `rstd_block` post-op SFPU work (`mul_unary_tile`, `add_unary_tile`, `rsqrt_tile`) to column 0 of the stat tile (`sfpu_tile_scope`, `c_skip`, ~1.9x on the SFPU call). It is one tile per tile-row, so measure before keeping it.
- (c) **Residual add cost.** The add is 24 tiles per tile-row, yet residual costs +3.6 to +4.8 us at R = 32. Attribute that between the reader and the add phase before choosing a lever.
- (d) **SFPU init hoisting.** Hoist the `binop_with_scalar_tile_init` / `rsqrt_tile_init` re-inits out of the per-output post-op where the helper contract allows.

Reference points:
- LoFi saves 1.6 us at R = 32 (12%), so fidelity is a minor term.
- `fp32_dest_acc_en=False` saves 0.2-2.3 us, mostly within run-to-run noise.
- Do **not** fuse normalize/scale/shift through FPU DEST-reuse: `compute_fusion` measures that as a 0.82x loss for FPU-after-FPU.

Target: `(32, 1, 32, 768)` and `(16, 1, 128, 768)`, with and without residual, both affine forms, default and `fp32_acc_off`. No SUPPORTED change.

**Verifier notes**: compute-local T1/T2 levers, independent of the topology, so it goes last and blocks nothing. Lever (a) is legal only where the format is genuinely constant across the boundary. `cb_input_tiles` / `cb_residual_tiles` -> `T_acc` transitions must keep their reconfig. Precision must not move: (a) and (b) change only MMIO / lane scope.

**Done when**: device-ns improves (median of 3) on the target shapes at both configs. No regression across the guard set. Precision-baseline PCC and rel-RMS unchanged within 2%. All layer_norm_tile tests green.
