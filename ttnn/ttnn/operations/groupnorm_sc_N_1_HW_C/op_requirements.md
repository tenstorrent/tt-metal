# Operation Requirements: groupnorm_sc_N_1_HW_C

## Definition
- **Formula**: `y[n,0,s,c] = (x[n,0,s,c] − mean(n,g)) * rstd(n,g) * gamma[c] + beta[c]`,
  `g = c // (C/G)`, `mean(n,g) = E[x]` over the HW × (C/G) slab,
  `rstd(n,g) = 1/sqrt(E[(x−mean)²] + eps)`
- **PyTorch Reference**: `torch.nn.functional.group_norm` on the (N, C, HW) permutation —
  see `eval/golden_tests/groupnorm_sc_N_1_HW_C/helpers.py:pytorch_groupnorm_sc_N_1_HW_C`
- **Import Path**: `from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C`
- **Function Signature**:
  `groupnorm_sc_N_1_HW_C(input_tensor: ttnn.Tensor, num_groups: int, *, gamma: ttnn.Tensor = None, beta: ttnn.Tensor = None, eps: float = 1e-5, memory_config: ttnn.MemoryConfig = None) -> ttnn.Tensor`
  (output always TILE_LAYOUT, dtype == input dtype)

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE, ROW_MAJOR] (RM converted host-side)
- **SUPPORTED shape-derived axes**: alignment=tile_aligned only; groups_alignment=aligned only
- **SUPPORTED op-specific axes**: affine ∈ {gamma_beta, gamma_only, no_affine};
  affine_dtype=[bfloat16]; affine_layout=[ROW_MAJOR, TILE] (TILE promoted by verifier on probe evidence)
- **Cores**: single (0,0); per-(n,g) groups processed sequentially, 3 streaming passes each
- **Compute config**: hard-coded defaults (no compute_kernel_config exposed)
- **Golden baseline**: 300 / 7236 cells passing, 3385 xfail_expected, 3551 invalid_skipped,
  0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode (per `verifier_report.json`)

### [x] Refinement 1 — Numerical configurability + multi-core distribution

**Goal**: add `ttnn.float32`, `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` AND to
`SUPPORTED["affine_dtype"]` (covers mixed-precision: bf16 activations + f32/bf8b gamma/beta),
expose `compute_kernel_config: ttnn.ComputeKernelConfig` on the entry point, and set
intermediate-CB precision (cb_mean / cb_var / cb_centered / cb_xhat / cb_scaled fp32 when
`fp32_dest_acc_en`, incl. `UnpackToDestFp32` tagging where applicable). Distribute (n,g)
groups across the multi-core grid in the same pass — the work is embarrassingly parallel over
interleaved DRAM (group slabs disjoint; no inter-core communication; gamma/beta read by each core).
Cells failing out of the box land in `EXCLUSIONS`, not in their own refinement.

**Implementation skill**: /numeric-formats-metal, /interleaved-parallel

**Verifier notes**: lands first — Refinements 2–3 reuse the dtype-driven CB-format derivation
and per-dtype tolerances (already in `helpers.py:TOLERANCES`). bf8b+ROW_MAJOR is INVALID
(skipped), so layout interaction is free. Work unit = one (n,g) group; N·G groups is small for
the SD shapes (N·32) but ample for the grid. Multi-core does not shrink per-core CBs — see
verification_report.md L1 note.

**Done when**: golden xfail cells whose only missing axes are dtype / affine_dtype
(~2 600 marginal cells) pass; verifier CLI loud categories all 0.

### [x] Refinement 2 — Non-tile-aligned shapes (HW and C tails)

**Goal**: add `"hw_non_aligned"` and `"c_non_aligned"` to `SUPPORTED["alignment"]`. Tail tiles
in HW need row masking in the statistics passes; tail tiles in C need column masking — the
streaming N_grp scaler becomes `1/sqrt(HW·Cg)` on logical (not padded) sizes. REDUCE_SCALAR has
no partial-scaler support (`prepare_partial_reduce_scalers` rejects it), so either restructure
stats as COL+ROW reduce pairs with `ReducePartialScaler::last_tile_at(1)` or zero-mask the tail
tiles via fill helpers before the reduce. Padding rows/cols of the output may stay garbage-free
(zero-mask after pass 3 not required by tests; only logical region is checked).

**Verifier notes**: do after Refinement 1 (the masking work must already handle all dtypes —
bf8b xfail TOLERANCES are wide for the masked-tail rounding). Keep `eps` handling unchanged;
the affine paths are unaffected (gamma/beta tails are zero-padded host-side). INPUTS exercises
hw-only, c-only, and 8 hard cells where C tails couple with group straddle (those flip only
after Refinement 3 lands; until then they remain xfail on groups_alignment).

**Done when**: golden cells with alignment ∈ {hw_non_aligned, c_non_aligned} and
groups_alignment=aligned (~700 marginal cells) pass.

### [x] Refinement 3 — Non-tile-aligned group widths (SD / SDXL regime)

**Goal**: add `"non_aligned"` to `SUPPORTED["groups_alignment"]` — groups straddle tile
boundaries (C/G ∈ {10, 20, 24, 30, 40, ...} with num_groups=32), the dominant Stable Diffusion
GroupNorm regime (~1 815 marginal cells). The reduction unit changes shape: a group slab is no
longer a whole tile column band, so per-(n,g) statistics need per-group column masks within
shared tiles, and pass 3 must apply different (mean, rstd) pairs to different column ranges of
one tile. Design seam: per-group column masking + COL/ROW-reduce stat refactor
(REDUCE_SCALAR scaler is applied twice — unusable with partial column masks; see
op_design.md Risk row "(Cg % 32)≠0").

**Verifier notes**: hardest refinement — algorithmic, no skill covers it; the masking
machinery from Refinement 2 is a prerequisite, do not start before it. Avoid bucketing failures
under new taggers — `groups_alignment` is the structural axis and already exists. Also flips
the 8 hardest INPUTS cells (C tail AND group straddle) provided Refinement 2 has landed.

**Done when**: golden cells with groups_alignment=non_aligned (incl. the coupled-alignment
cells) pass; full suite shows 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode.
