# Operation Requirements: rms_norm

## Definition
- **Formula**: `out[..., h, w] = x[..., h, w] · rsqrt( (1/W)·Σ_{j<W} x[..., h, j]² + eps ) · gamma[w]`
- **PyTorch Reference**:
  ```python
  def rms_norm(x, gamma=None, eps=1e-6):
      var = x.pow(2).mean(dim=-1, keepdim=True)
      out = x * torch.rsqrt(var + eps)
      return out * gamma if gamma is not None else out
  ```
- **Import Path**: `from ttnn.operations.rms_norm import rms_norm`
- **Function Signature**:
  ```python
  rms_norm(
      input_tensor: ttnn.Tensor,
      *,
      gamma: ttnn.Tensor | None = None,
      epsilon: float = 1e-6,
      compute_kernel_config: ttnn.ComputeConfigDescriptor | None = None,
      memory_config: ttnn.MemoryConfig | None = None,
  ) -> ttnn.Tensor
  ```

## Axis gap (TARGET − SUPPORTED)

| Axis | TARGET | SUPPORTED (now) | Missing | Disposition |
|------|--------|-----------------|---------|-------------|
| dtype | float32, bfloat16, bfloat8_b | bfloat16 | float32, bfloat8_b | Refinement 2 |
| fp32_dest_acc_en | True, False | True | False | Refinement 2 |
| layout | TILE, ROW_MAJOR | TILE | ROW_MAJOR | Refinement 3 |
| alignment | tile_aligned, w_non_aligned, h_non_aligned | tile_aligned | w_non_aligned, h_non_aligned | Refinement 3 |
| rank | 2, 3, 4 | 2, 3, 4 | — | complete |
| gamma_mode | gamma, no_gamma | gamma, no_gamma | — | complete |
| gamma_dtype | float32, bfloat16, bfloat8_b | bfloat16 (+float32 for no_gamma canonical, EXCLUDED when gamma present) | gamma-present float32, bfloat8_b | Refinement 2 |
| gamma_layout | TILE, ROW_MAJOR | TILE | ROW_MAJOR | Refinement 3 |

INVALID (in `feature_spec.py`) absorbs `{bf8b, ROW_MAJOR}` (both tensors) and the
no_gamma canonicalizations — no refinement needed for those.

## Phases

> **Non-regression rule**: every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean SUPPORTED was not updated to match
> the kernel — fix by editing SUPPORTED.
> **Checkbox protocol**: `[x]` complete + green; `[~]` real work landed, ≥1 named
> axis value deferred; `[ ]` nothing usable produced.
> **Refinement 1 is a hard gate**: do not start Refinement 2/3 until it is clean.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED alignment**: tile_aligned only
- **SUPPORTED rank**: [2, 3, 4]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED gamma**: gamma_mode {gamma, no_gamma}; gamma_dtype bf16 (TILE)
- **Regimes**: A (row-parallel, single full row resident) — **correct**;
  B (wide-W cross-core all-gather) — **broken**, see Refinement 1.
- **Cores**: multi-core (embarrassingly parallel Regime A already wired)
- **Compute config**: HiFi4 + fp32_dest_acc_en=True default; config forwarded
- **Golden baseline**: 22 / (22+21 supported) passing — the 21 failures are all
  Regime B (Refinement 1).

### [ ] Refinement 1 — Fix Regime B cross-core all-gather correctness (BLOCKER)

**Goal**: move the 21 wide / few-row `supported_fail` cells (every shape whose
`Ht_total < grid` or whose full row exceeds the L1 resident budget, e.g.
`128x512`, `1x1x32x4096`, `1x1x32x8192`, `1x1x128x4096`, `2x1x64x4096`,
`1x32x4096`, `1x32x8192`, `32x4096`, `128x8192`, and the LOOSE cases
`1x1x32x16384`, `1x1x32x32768`, `1x1x64x12288`) from failing to passing. **No
SUPPORTED axis is added** — Regime B is selected by shape/L1 fit, not an axis;
this is a pure correctness fix of an existing code path.

**Repro**:
```python
import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
x = torch.ones((1,1,32,4096))                      # Regime B (Ht_total=1 < grid)
ti = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
out = ttnn.to_torch(rms_norm(ti)).float()
# expected ~1.0 everywhere; actual ~1.4142  (Σx² summed only 0.5×)
```

**Symptom (exact)**: Regime B output is too large by `sqrt(2·num_chunks)`; the
gathered/combined Σx² underflows by **1/(2·num_chunks)**, where
`num_chunks = ceil(Wt_s / reduce_block)`. Measured summed-fraction table:
`(num_chunks=1)→0.5`, `(2)→0.25`, `(3)→0.166`. Regime A with identical PASS-1
parameters returns the exact value, so PASS-1 reduce-accumulate is **not** the
bug — it is isolated to the Regime-B-only path.

**Where to look**:
- `kernels/rms_norm_reader_mcast.cpp` — the K-round rotating all-gather
  (`SenderPipe`/`ReceiverPipe`, Staging::Counter, EXCLUDE_SRC + local self-copy).
- `kernels/rms_norm_compute.cpp` — the K-partial combine (`copy` slot0 + K-1
  `add`s) and the **double producer/consumer handshake on `cb_partial_sumsq`**:
  compute produces it in PASS-1 → reader consumes it as the mcast source →
  compute *re-produces* it in the combine → compute consumes it in finalize, all
  on a CB sized to 2 pages. The clean dependence of the error on a *compute-side*
  chunk count that the gather/combine never see strongly implicates a
  cross-thread CB staleness / ordering bug at this handshake (e.g. finalize or
  the mcast reading a stale / partially-accumulated `cb_partial_sumsq`, or the
  combine summing the wrong slots). Verify push==wait on `cb_partial_sumsq` and
  `cb_partials_gathered` across all three threads, and that the reader's
  `cb_wait_front(cb_partial_sumsq,1)` observes the *fully* accumulated PASS-1
  result before it is mcast.
- Cross-check the all-gather actually delivers all K distinct partials (use a
  per-shard-distinguishable input, not all-ones) and that `num_dests`/EXCLUDE_SRC
  accounting matches the rectangle.

**Reference material**: `op_design.md` §"Tensix-to-Tensix contract" and its §9
silent-hang checklist (virtual coords, barrier-before-signal, `num_dests`,
semaphores on the union, never-mcast-to-self);
`tt_metal/.../references/cross_core_reduction_design.md`;
`ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp` (SenderPipe/ReceiverPipe semantics,
Counter staging).

**Implementation skill**: none — cross-core reductions with a real data
dependency (mcast + semaphores + all-gather) are explicitly outside every
current skill's scope (`/interleaved-parallel` covers only embarrassingly-
parallel interleaved work, no cross-core data deps).

**Verifier notes**: hard blocker — runs first; Refinements 2 and 3 stay frozen
until this is green. While here, also fix the deferred `cb_normalized`/`cb_gamma`
= `Wt` sizing (see verification_report.md): it makes the per-core L1 footprint
scale with `Wt`, so `RESIDENT_BUDGET_TILES` (560) understates pressure and the
A/B heuristic can mis-select. The design's sanctioned pass-2 optimization
(streaming fused Col→Row multiply per `REDUCE_BLOCK`, dropping the
`cb_normalized` round-trip) makes both the budget and the wide-row single-core
case sound. **Done when** every Regime B cell in the `supported_fail` bucket
(incl. all three LOOSE cross-core cases) passes at the bf16 tolerance band.

### [ ] Refinement 2 — Numerical configurability expansion

**Goal**: add `ttnn.float32` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`, add
`False` to `SUPPORTED["fp32_dest_acc_en"]`, and extend gamma precision
(gamma-present `float32` — move it out of `EXCLUSIONS` — and `bfloat8_b`). Wire
`compute_kernel_config` through to the compute kernel descriptor and set
intermediate-CB formats / `UnpackToDestFp32` tagging from the dtype. Cells that
fail out of the box land in `EXCLUSIONS`, not their own refinement — in
particular **`{dtype: float32, fp32_dest_acc_en: False}`** (fp32 needs fp32
accumulation; this is the documented EXCLUSION per the prompt) and
**`bfloat8_b + non_tile_aligned`** if it appears. This also clears the 15
`no_axes_found` float32 `test_regression.py` cases.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: lands after Refinement 1 (the dtype-aware CB-format
derivation introduced here is reused by Refinement 3's ROW_MAJOR legs, and any
fp32 Σx² intermediate it introduces should sit on a *correct* Regime B). Keep
all float dtypes in one descriptor-level refinement — do not split bf8b out.

### [ ] Refinement 3 — ROW_MAJOR layout + non-tile-aligned shapes (native)

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]` and to
`SUPPORTED["gamma_layout"]` (gamma is supplied ROW_MAJOR `(1,1,1,W)` per the
prompt), and add `w_non_aligned` + `h_non_aligned` to `SUPPORTED["alignment"]`.
All handled **natively in the kernel** — a tilize-wrapped reader/writer for the
RM legs (math stays on tiles) and last-tile zero-pad/mask in the reader or
compute so the RMS denominator counts only valid (non-padding) elements along W.
The prompt's MUST is explicit: **no host-side `ttnn.to_layout` / `tilize` /
`untilize` / `pad` / `slice`** — `SUPPORTED` must reflect real in-kernel
capability. Output layout must match input layout.

**Implementation skill**: /memory-layouts

**Verifier notes**: bundle layout + alignment — they are the same reader/compute
data-access-boundary rewrite (RM access path and edge-tile masking touch the same
code). Depends on Refinement 2 (RM legs must carry the dtype set introduced
there) and on Refinement 1 (wide RM rows may route through Regime B). The
`w_non_aligned` vs `h_non_aligned` tagger split already exists so the W-mask and
H-mask paths report independently — if one mask path is harder, `[~]`-tick the
landed one and leave the other in `SUPPORTED` minus that value. Wrapping the op
in manipulation ops is a `[~]` partial-tick escape hatch only (name it in the
changelog and file the in-kernel follow-up) — it is **not** the default.
**Done when** the ROW_MAJOR and non-aligned golden cells pass natively at the
bf16 tolerance band, output layout matching input.
