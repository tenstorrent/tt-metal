# Operation Requirements: layer_norm_adaln

## Definition
- **Formula**: `mean_i = Σ_j x_ij / W`; `var_i = Σ_j (x_ij − mean_i)² / W`; `y_ij = (x_ij − mean_i) / sqrt(var_i + ε)`; `out_ij = y_ij · w_j + b_j` when weight and bias are supplied, else `out = y`.
- **PyTorch Reference**:
  ```python
  def layer_norm_adaln(x, weight=None, bias=None, epsilon=1e-6):
      y = torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), eps=epsilon)
      if weight is not None:
          y = y * weight.float().reshape(-1) + bias.float().reshape(-1)
      return y.to(x.dtype)
  ```
- **Import Path**: `from ttnn.operations.layer_norm_adaln import layer_norm_adaln`
- **Function Signature**: `layer_norm_adaln(input_tensor: ttnn.Tensor, weight: ttnn.Tensor = None, bias: ttnn.Tensor = None, *, epsilon: float = 1e-6, compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Queue shape**: `TARGET − SUPPORTED = ∅` on every axis (dtype, fp32_dest_acc_en, layout, alignment, rank, affine, affine_dtype, affine_layout, memory_layout) and `xfail_expected` is empty, so there are **no generality refinements**. Per the 2:1 cadence rule, once generality is exhausted the remaining phases are all perf. `LOOSE_CASES` is empty (no perf-flagged shape), so regions are free-selected among cells > 32 tiles (`eval/prompts/perf_rules.txt`); the FLUX per-chip shapes `(1,1,256,3072)`, `(1,1,2048,3072)`, `(1,1,2304,3072)` are the prompt's named perf targets. `ttnn/ttnn/operations/examples/master.md` is absent on this branch — levers come from the design's perf lamps (`op_design.md` § Perf lamps) and the measurements in `verification_report.md`.
>
> **Measurement**: `tests/ttnn/unit_tests/operations/layer_norm_adaln/test_layer_norm_adaln_perf.py` (DEVICE KERNEL DURATION, avg of 5, fp32 DEST / HiFi4; run with `--no-precompile`). A difference within ~2–3% is noise.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True, False]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment = tile_aligned, rank = 4
- **SUPPORTED op-specific axes**: affine ∈ {no_affine, dynamic_wb}, affine_dtype ∈ {bfloat16, none}, affine_layout ∈ {TILE, none}, memory_layout = INTERLEAVED (DRAM)
- **Cores**: full grid — rows split over `num_row_groups`, W split over `num_w_slices = S` cores per group with a rotating-mcast all-gather of per-slice stats + Chan combine (`select_blocking`). 120/120 cores at 2048/2304×3072, 96 at 256×3072.
- **Compute config**: caller's `compute_kernel_config` honored (default HiFi4 + fp32 DEST); 16-bit DEST uses a pivot-relative `[p, d, M2]` payload (verifier fix).
- **Golden baseline**: 16 / 16 supported cells passing, 48 INVALID skipped, 37 regression cells passing, 0 loud categories.
- **Perf baseline (µs, no-affine / affine)**: 256×3072 37.4 / 39.1; 2048×3072 81.4 / 108.3; 2304×3072 88.2 / 110.8.

### [ ] Refinement 1 — Co-tune the block surface: S selection + read/compute overlap

**Type**: perf

**Goal**: speed up the FLUX shapes by turning knobs the planner already exposed — no new topology, no SUPPORTED change.
- **256×3072 (and 32×3072)**: `select_blocking` minimizes tiles per busiest core only, so it picks S=12 (8 tiles/core, 37.5 / 38.9 µs) although a forced S=6 measures 29.6 / 33.0 µs and S=4 28.2 / 36.0 µs. Give the cost model a measured S-dependent term (per-peer combine work + rotating mcast rounds per block, i.e. ∝ `S·bh`) alongside `core_h_tiles·bw`, or retune `MIN_SLICE_TILES` / `GATHER_TILE_CAP` (design lamp "Grid synchronization"). Keep `select_blocking` pure and grid-parameterized; update the pinned-selection test's expectations to whatever is measured fastest.
- **2048 / 2304×3072**: one block per core with `mean_block` waiting `WaitUpfrontNoPop` on the whole block serializes each core's entire DRAM read (64 tiles) in front of its compute, and its compute in front of most of its writes (design lamp "Overlap (input)"; the design planned a cumulative per-row wait). Levers on the existing knobs: per-row cumulative waits in `mean_block`, and/or `block_height < core_h_tiles` with `INPUT_BLOCK_DEPTH = 2` so block b+1 lands while block b computes (≈ +128 KB L1 at bh=1, bw=32 — fits; the multi-block path is already built and lockstep-safe because rows are assigned per group).

**Verifier notes**: first because it is cheap (host cost model + policy/depth knobs) and gives the later refinements an honest baseline. Measured S sweep is in `verification_report.md` § Performance Baseline. Whatever the overlap change does to the per-block exchange count (more blocks ⇒ more rendezvous) must be measured, not assumed. The 16-bit-DEST path's extra `dev_sum` pass and 3-stat payload ride the same knobs — include one 16-bit-DEST cell in the guard set.

**Done when**: measured device-ns improves on 256×3072 and on 2048×3072 / 2304×3072 (affine and no-affine) beyond the ~3% noise band; golden suite green; precision baseline test green; no regression across the config-spanning guard set (one representative per kernel path: `rows_x_wslices` × {affine, no-affine} × {fp32 DEST, 16-bit DEST}; the `rows_only` regime only has ≤ 32-tile cells, which must still pass the golden suite).

### [ ] Refinement 2 — Cut the per-row compute cost: SFPU combine and the normalize tail

**Type**: perf

**Goal**: on 2048×3072 / 2304×3072 with affine (108.3 / 110.8 µs vs a ~50 µs DRAM floor) the cores are compute-bound. Two measured hot regions:
- **`combine_block`** costs ≈ 1 µs per (row, peer): every peer step re-issues `reconfig_data_format_srca` + `copy_init` + SFPU inits on full 32×32 tiles of which only column 0 is meaningful (2048 at S=12, bh=7: 183 µs). Levers: hoist inits/reconfigs out of the peer loops, restrict SFPU work to the faces holding column 0, and/or the design's "Stats transport format" lamp (compact column records instead of full acc tiles). A cheaper combine also makes higher S viable at 2048 (S=12 gives 56 tiles/core vs 64) — re-run Refinement 1's selection afterwards.
- **`normalize_block`**: 3 FPU ops per tile without affine, 5 with (affine adds ≈ 27 µs at 2048 on 64 tiles/core); every DestReuse op moves DEST→srcA. Levers: the design's "Normalize chain length" lamp (fold the shift / rstd / affine into fewer FPU ops or onto SFPU ops on DEST), dev_sq + normalize init/reconfig hoisting.
No SUPPORTED change.

**Verifier notes**: after Refinement 1 because R1 changes `bh`/S, which sets how many (row, peer) combine steps exist. Moving the normalize tail to SFPU on fp32 DEST should also remove the ≈ −0.5% / −1% systematic magnitude bias recorded in `verification_report.md` (DestReuse tf32 truncation) — the precision baseline must not get worse in any cell, including the 16-bit-DEST offset-50 cells (whose combine is the `dev_sum_pass` branch and must keep carrying the slice mean pivot-relative).

**Done when**: measured device-ns improves on 2048×3072 and 2304×3072 (affine and no-affine) and on 256×3072 beyond the noise band; golden suite and the precision baseline test green (no cell's rel-RMS worse than recorded); no regression across the guard set (one representative per kernel path × {affine, no-affine} × {fp32 DEST, 16-bit DEST}).

### [ ] Refinement 3 — Load-balanced row × W work split

**Type**: perf

**Goal**: at 2048×3072 the busiest core does 64 tiles while the mean is 51.2 (64 tile rows over 40 groups of S=3: 24 groups take 2 rows, 16 take 1); 2304×3072 is 64 vs 57.6. Reduce the busiest-core work toward `tensor_h_tiles·tensor_w_tiles / grid` by changing the work-distribution scheme — e.g. unequal W slices per group with an `n_j`-weighted Chan combine (the current combine assumes equal `n_slice`, a design mechanism cap), or heterogeneous group widths S per group (each group still a 1×S run inside one grid row, lockstep within a group). This is a **scheme-change** (new partition + combine weights), so it stands alone. No SUPPORTED change.

**Verifier notes**: gated on headroom — re-measure after Refinements 1–2. If the busiest core is no longer compute-bound (e.g. the op is DRAM-bound or a cheaper combine made S=12/bh=7 the winner at 56 tiles/core), record that in the changelog and mark this refinement done with the measurement as evidence instead of building it. The mcast group rectangle cap (`S | grid_x`, groups inside one grid row) and group lockstep invariants in `op_design.md` § Mechanism caps still bind.

**Done when**: measured device-ns improves on 2048×3072 and 2304×3072 beyond the noise band (or the headroom measurement above shows the lever is a no-op); golden suite and precision baseline green; no regression across the guard set (one representative per kernel path × {affine, no-affine} × {fp32 DEST, 16-bit DEST}).
