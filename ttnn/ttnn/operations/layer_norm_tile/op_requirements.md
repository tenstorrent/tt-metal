# Operation Requirements: layer_norm_tile

## Definition
- **Formula**: `mean[r] = (1/W) sum_j x[r,j]`; `xc = x - mean`; `var[r] = (1/W) sum_j xc[r,j]^2`; `out[r,j] = gamma[j] * xc[r,j] * rsqrt(var[r] + epsilon) + beta[j]` (W is the logical last dim, 768 under the contract)
- **PyTorch Reference**:
  ```python
  def pytorch_layer_norm(x, gamma, beta, *, epsilon=1e-5):
      return torch.nn.functional.layer_norm(
          x.float(), (x.shape[-1],), weight=gamma.float().reshape(-1), bias=beta.float().reshape(-1), eps=epsilon
      ).to(x.dtype)
  ```
- **Import Path**: `from ttnn.operations.layer_norm import layer_norm` (re-export of `ttnn.operations.layer_norm_tile`)
- **Function Signature**:
  ```python
  layer_norm(
      input_tensor: ttnn.Tensor,            # (B, 1, S, 768) bf16 TILE, interleaved (L1 per contract)
      gamma: ttnn.Tensor,                   # (768,) TILE in DRAM, or (1, 1, 24, 32) / (1, 1, 1, 768) ROW_MAJOR in L1
      beta: ttnn.Tensor,                    # same forms as gamma
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

Queue basis. `eval/golden_tests/layer_norm_tile/feature_spec.py` does not exist, so TARGET is taken from the
design's TARGET table (`op_design.md`, "feature_spec.py status"), which restates the prompt contract. Every
TARGET value is already in SUPPORTED, so `TARGET - SUPPORTED` is empty and there is no generality candidate.
All refinements are measured perf refinements. No `LOOSE_CASES` entry exists to carry an `attention:` perf flag,
so the verifier selected the regions from its own ablations (`verification_report.md`, "Performance baseline").
The target config for every perf refinement is the op default (`compute_kernel_config=None`: HiFi4,
`fp32_dest_acc_en=True`), bf16 TILE input in L1 interleaved, both affine forms unless a refinement names one.

Measurement protocol for every perf refinement: `scripts/run_safe_pytest.sh --profile
tests/ttnn/unit_tests/operations/layer_norm_tile/test_layer_norm_tile_perf.py`, read `DEVICE KERNEL DURATION
[ns]`, and take the median of at least 3 runs. Run-to-run spread on the 110-core shapes is 5-10%. The
config-spanning guard set is that file's matrix: R = 32 / 64 / 128 / 248 x {tile_dram, rm_l1} x {default,
fp32_acc_off, lofi}. It covers both affine reader paths, both intermediate formats, single-block and multi-block
cores, and aligned and non-aligned S.

### [x] Phase 0 - Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment = {tile_aligned, h_non_aligned}
- **SUPPORTED op-specific axes**: fp32_dest_acc_en = {True, False}; affine_layout = {TILE, ROW_MAJOR}; memory_layout = {INTERLEAVED} (L1 or DRAM buffer)
- **EXCLUSIONS**: none
- **Regime**: `row-split-resident-w`: tile-rows split over the full grid (`split_work_to_cores(..., row_wise=True)`), whole W row resident, input and output each cross the NoC once, gamma/beta row 0 read once per core
- **Cores**: multi-core, min(R, 110) on Blackhole p100a, with R = B * ceil(S/32)
- **Compute config**: `default_compute_kernel_config()` = HiFi4, fp32_dest_acc_en=True; math_fidelity / math_approx_mode passed through ungated
- **Golden baseline**: no golden suite exists. Acceptance 41/41, extended 7/7, precision baseline 16/16, perf smoke 24/24
- **Device time (default config, tile_dram / rm_l1)**: R=32: 14.1 / 14.2 us; R=64: 21.5 / 20.4 us; R=128: 41.6-43.7 / 31.4-34.5 us; R=248: 52.9-54.0 / 46.0-54.5 us

### [ ] Refinement 1 - W-split cross-core combine for under-filled grids

**Type**: perf

**Goal**: build the design's deferred `w-split-combine` regime for shapes whose tile-rows leave most of the grid idle (`tensor_row_tiles <= num_cores / 2`; under the contract this is the B*S = 1024 class: R = 32 for power-of-two S, and R = 33..55 for non-aligned S such as `(11, 1, 100, 768)` R=44 and `(3, 1, 500, 768)` R=48). Today R = 32 runs on 32 of 110 cores. Compute is 10.2 us of the 14.1 us per tile-row (verifier ablation). Split each tile-row's `w` across `k` cores, where `k` is the largest divisor of `tensor_w_tiles` with `k * R <= num_cores` (k = 3 at R = 32, so 96 cores). Each core reduces its slice to column-0 partial tiles. Two combine rounds per tile-row (mean, then sum of squares) gather the `k` partials and multicast the result back. Each core then normalizes its resident slice with its own gamma/beta slice. 1/W uses the logical W, never the slice width. Target: the R = 32 shapes `(32, 1, 32, 768)` and `(2, 1, 512, 768)`, both affine forms, default config. Relevant `ttnn/ttnn/operations/examples/master.md` entries: `width_split`, `tensix_all_reduce`, `mcast_topology` (transport through `ttnn.Mcast` + `McastArgs`), `noc_placement` (keep each row's `k` cores adjacent). No SUPPORTED change.

**Verifier notes**: scheme-change (a dependent axis is split cross-core), so it stands alone and goes first: Refinements 2 and 3 change the affine fetch and the core assignment that this regime redefines. The design's precision gate on this regime is met: the Phase 0 baseline is PCC 0.999998 / rel-RMS 0.19% (fp32 DEST) and PCC 0.999992 / rel-RMS 0.51% (bf16 DEST). The combine only reorders the summation. `k` and the slice width must be `select_blocking()` knobs with one source of truth. Every CB that spans `w` (`cb_input_tiles`, `cb_centered`, `cb_gamma_rows`, `cb_beta_rows`, `cb_output_tiles`) scales with the slice, not the full row. Update `l1_ledger.md`: new partial / combined-stat CBs, cross-core traffic of about `R * k * 4 KB` per call, and the design's stall-shadow row (`xc * gamma` can run while waiting on peers' partials). The regime-selection function picks between `row-split-resident-w` and `w-split-combine` per call; it is still one `generic_op` dispatch. Add acceptance cases pinned to the new path: aligned and non-aligned S, both affine forms, `fp32_dest_acc_en` on and off.

**Done when**: device-ns improves (median of 3) on both R = 32 target shapes for both affine forms at the default config; R = 44 / 48 non-aligned shapes run on the new regime and do not regress; no regression across the guard set; acceptance, extended and precision-baseline tests green, with PCC and rel-RMS within 10% of the Phase 0 baseline numbers.

### [ ] Refinement 2 - Read gamma/beta once and multicast (TILE-DRAM affine form)

**Type**: perf

**Goal**: replace the per-core TILE-DRAM affine fetch with a read-once + multicast of the reuse-shared gamma/beta row-0 data. Today every participating core issues 24 x 576 B DRAM reads per tensor, so P = 110 cores put about 5300 requests on the few DRAM banks holding 24 pages. Measured cost: the DM-only ablation at R = 128 is 32.2 us with TILE-DRAM affine vs 17.2 us with RM-L1 affine. Full-op R = 128 is 41.6-43.7 us vs 31.4-34.5 us. One injector, or one per slice group under Refinement 1's `w-split-combine`, reads the vectors once and multicasts the row-0 tiles into `cb_gamma_rows` / `cb_beta_rows` on its receivers. Target: `(8, 1, 512, 768)` and `(124, 1, 33, 768)` with `tile_dram` affine at the default config. Relevant `master.md` entries: `shared_input_reuse`, `mcast_topology`. Use the multicast helper (`ttnn.Mcast` + `attach()` on host, `McastArgs` `sender(noc).send()` / `receiver(noc).receive()` in the kernel), not raw `noc_async_write_multicast` + semaphores. No SUPPORTED change.

**Verifier notes**: reuse-multicast is a new dataflow topology, so it stands alone. It comes after Refinement 1 so the multicast groups are built once, on the final slice structure (a `w`-slice group needs only its slice of gamma/beta). Validate by building the variant and measuring device-ns, not by inferring from an ablation. Keep the fetch issued after block-0 input reads, so it stays in the stall shadow of P1..P5. Measure whether the RM-L1 form also benefits. Apply multicast to RM-L1 only on a measured win: at R = 32 the per-core RM fetch is already hidden. This reverses the design's `affine-read-once-mcast: rejected` row on measured evidence (`op_design.md`, "Post-implementation deltas"). Record the new regime status and traffic in `l1_ledger.md`.

**Done when**: device-ns improves (median of 3) on both TILE-DRAM target shapes at the default config; no regression on R = 32 / 64 TILE-DRAM or any RM-L1 guard entry; no regression across the guard set; all layer_norm_tile tests green.

### [ ] Refinement 3 - Overlap and localize input/output data movement

**Type**: perf

**Goal**: cut the data-movement term that currently serializes with compute. On one-tile-row cores (R <= 110), the reader moves the whole 24-tile row behind one barrier, then compute runs, then the writer waits for all 24 output tiles. The verifier ablation shows full time is about DM-only plus compute-only (R = 64 RM-L1: 9.0 + 10.2 us vs 19.1-20.9 us full). Aggregate L1-to-L1 NoC throughput saturates at about 700-900 GB/s, so the DM term grows with core count (R = 32 / 64 / 128 / 248: 4.5 / 9.0 / 17.2 / 27.0 us). Two levers, both knob-turns on the existing block surface. (a) Overlap: push the input row in sub-row chunks so P1 starts on the first chunk, and drain the output in sub-row batches so writes overlap P7 (the design's "Writer batch" lamp). Co-tune the read chunk, the write batch and the CB depths. (b) Locality: the L1-interleaved input/output pages of tile-row r sit on 24 consecutive L1 banks. Assign rows to cores near those banks, or stagger each core's walk, so less traffic crosses the mesh. Relevant `master.md` entries: `double_buffer`, `noc_placement`, `bank_stagger`, `trid_double_issue`, `compute_block_size` (block-size floor: whole tiles minimum, coarser amortizes). Target: `(16, 1, 128, 768)` R = 64 and `(8, 1, 512, 768)` R = 128 with `rm_l1` affine at the default config (`rm_l1` is the faster form, so production should pass it; see the report). No SUPPORTED change.

**Verifier notes**: T2 knob-turns, so several levers fit in one phase. It follows Refinement 1 because the locality assignment must keep each row's `w`-split cores adjacent, and both regimes need the same treatment. Every new chunk, batch or depth value is a `select_blocking()` knob with one source (CT arg to every kernel that uses it), and the ledger rows for `cb_input_tiles` / `cb_output_tiles` must follow. If P1 starts on partial input, it needs a streaming / accumulating reduce policy over the resident block, and `cb_input_tiles` must still hold the whole row for P2. Revisit `BLOCK_ROWS_MAX` (1 today) once overlap exists: the implementer's 2 -> 1 measurement was taken without it.

**Done when**: device-ns improves (median of 3) on both target shapes at the default config; no regression across the guard set (including R = 248 multi-block cores and DRAM-interleaved input); all layer_norm_tile tests green.

### [ ] Refinement 4 - Trim per-tile-row compute cost

**Type**: perf

**Goal**: lower the 10.2 us per-tile-row compute time (verifier compute-only ablation), which sets the floor for every shape once data movement is overlapped. Candidate levers, each to be measured. (a) Scope the P4 finalize SFPU work (`mul_unary_tile`, `add_unary_tile`, `rsqrt_tile`) to column 0 of the stat tile instead of all 32 vectors; the result lives only in column 0 (`master.md` `sfpu_tile_scope`, col-0 via an even-parity stride, about 1.9x on the SFPU op). (b) Drop helper data-format reconfig at phase boundaries where the formats provably do not change (`compute_block_size`, "second lever"; P5 to P6 keeps `cb_centered` as srcA and pack target). (c) Re-measure `block_rows` 1 vs 2 against the new overlap from Refinement 3; phase-init amortization is about 0.3 us per phase per block (`compute_block_size`). Reference points: LoFi saves about 11% (R = 32: 12.5 vs 14.1 us), so fidelity is not the main cost; `fp32_dest_acc_en=False` saves about 1%. Target: `(32, 1, 32, 768)` and `(16, 1, 128, 768)`, both affine forms, default config and `fp32_acc_off`. No SUPPORTED change.

**Verifier notes**: compute-local and independent of the topology, so it goes last and does not block anything. Levers (a) and (b) are T1/T2. Do not fuse P5..P7 through FPU DEST-reuse: `compute_fusion` measures it as a loss for FPU-after-FPU, and it would also require fully row-replicated gamma/beta tiles. Precision must not move: (a) changes only which lanes the SFPU touches, and (b) must leave formats identical.

**Done when**: device-ns improves (median of 3) on the target shapes at both configs; no regression across the guard set; precision-baseline PCC and rel-RMS unchanged within 2%; all layer_norm_tile tests green.
