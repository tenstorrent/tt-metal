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

### [x] Refinement 1 — Fix Regime B cross-core all-gather correctness (BLOCKER)

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

### [x] Refinement 2 — Numerical configurability expansion

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

### [x] Refinement 3 — ROW_MAJOR layout + non-tile-aligned shapes (native)

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

---

## Non-registry refinements (code-quality + performance)

> Refinements 4–6 add **no** `SUPPORTED` axis value — the TARGET universe is
> already fully covered (1683/1683 golden passing as of Refinement 3). They are
> structural / correctness-preserving / performance work, the same shape as
> Refinement 1 (a pure fix that added no axis). Consequently:
> - **No drift signal applies** — `verify_supported` must stay all-green with
>   0 `xpass_drift` / 0 `supported_fail` throughout; do not look for new cells.
> - **Non-regression is the hard gate**: the full golden suite stays
>   **1683/1683**, `test_rms_norm_precision_matrix.py`, `test_rms_norm_layout_matrix.py`,
>   `test_rms_norm_regime_b.py`, and `test_rms_norm.py` all stay green. Any red
>   cell means the refactor changed behavior — fix before ticking.
> - **Ordering / coupling**: R4 and R5 both rewrite the compute kernel(s) and
>   should be co-designed (do R5's simplification first, then fold the result
>   into R4's unification, OR land them as one combined kernel rewrite — the
>   implementer's call, recorded in `changelog.md`). R6 (perf tuning) lands
>   **last**, on the clean unified base, so its measurements reflect the final
>   structure.

### [x] Refinement 4 — Unify the kernel set (7 → 3–4 max; single compute non-negotiable)

**Goal**: collapse the current **7 kernels** to **3–4 total**. The **one
non-negotiable** is a **single compute kernel** — the two computes must merge.
Unifying the readers (and writers) is **preferred** and is the path to the 3–4
budget, but not mandatory in the same hard sense as the compute. Also make the
ROW_MAJOR path use the **same cross-core mcast all-gather** as TILE Regime B
instead of being row-parallel-only.

Current inventory (all in `kernels/`):
- `rms_norm_compute.cpp` (TILE math) **and** `rms_norm_compute_rm.cpp` (RM:
  tilize → identical TILE math → untilize) — **two computes** doing the same
  core arithmetic. Merge into one.
- `rms_norm_reader.cpp` (Regime A, TILE), `rms_norm_reader_mcast.cpp`
  (Regime B all-gather, TILE), `rms_norm_reader_rm.cpp` (RM sticks, no mcast).
- `rms_norm_writer.cpp` (TILE), `rms_norm_writer_rm.cpp` (RM sticks).

**Three concrete defects this removes**:
1. **Duplicated math.** `rms_norm_compute_rm.cpp:84-188` is byte-for-byte the same
   square→reduce→rsqrt→normalize as `rms_norm_compute.cpp` with a `tilize` prologue
   and `untilize` epilogue bolted on. The TILE-vs-RM difference is purely a
   data-access boundary (sticks vs tiles), which belongs in the dataflow kernels
   (or a constexpr-gated tilize/untilize wrap), **not** a forked compute. Unify to
   one compute parameterized by a `layout_is_rm` (or tilize-wrap) compile-time arg.
2. **Three forked readers.** `rms_norm_reader.cpp` (Regime A TILE),
   `rms_norm_reader_mcast.cpp` (Regime B all-gather TILE), and
   `rms_norm_reader_rm.cpp` (RM sticks, no mcast) are three kernels for what is one
   read-then-optionally-all-gather flow. Unify into **one reader** gated by
   compile-time args (`layout_is_rm`, `num_partials`) — a degenerate `num_partials==1`
   is the Regime-A no-gather case, sticks-vs-tiles is the layout branch. This is the
   bulk of getting to ≤4 kernels.
3. **RM never goes cross-core.** `_regime_rm_descriptor` is row-parallel only
   (`rms_norm_program_descriptor.py:134` short-circuits before the A/B heuristic),
   so a wide-W ROW_MAJOR row that exceeds one core's L1 budget has **no** Regime-B
   fallback — it either OOMs or under-parallelizes. After unification the RM legs
   must route through the **same** `_select_k` / mcast all-gather path (the
   tilize-wrapped sticks become the resident shard; the partial-Σx² combine is
   identical). This is also a prerequisite for R6 measuring RM perf on wide shapes.

**Where to look**:
- `kernels/rms_norm_compute.cpp` vs `kernels/rms_norm_compute_rm.cpp` — diff them;
  the shared body is the merge target. The RM-only pieces are the per-block
  `tilize<reduce_block, cb_rm_in, cb_input_resident>` prologue and the per-chunk
  `untilize<reduce_block, cb_out_tiled, cb_rm_out>` epilogue.
- `kernels/rms_norm_reader_mcast.cpp` — the `SenderPipe`/`ReceiverPipe` all-gather
  the RM path must adopt; `kernels/rms_norm_reader_rm.cpp` — the stick read +
  W-padding-zeroing the unified reader must preserve.
- `rms_norm_program_descriptor.py:131-175` (regime dispatch) and `_regime_rm_descriptor`
  / `_regime_a_descriptor` / `_regime_b_descriptor` — these likely collapse toward a
  single descriptor builder selecting reader/writer variant + tilize-wrap by layout,
  and selecting A/B by the same L1-fit heuristic for both layouts.

**Implementation skill**: /memory-layouts (the tilize-wrapped vs pure-TILE access
patterns and the "math stays on tiles" invariant) + the cross-core mcast machinery
from Refinement 1 (no skill — out of every skill's scope) for extending the
all-gather to RM.

**Verifier notes**: structural refactor — **golden must stay 1683/1683 and all
auxiliary suites green** before and after (it is the only correctness oracle here,
since SUPPORTED is unchanged). Co-design with R5. The unified compute must remain
byte-identical numerically for the bf16/TILE Phase-0 corner (the regression anchor).
Count the kernels in `kernels/` before/after and record the reduction in
`changelog.md`. **Done when**: (a) there is **exactly one compute kernel** (the
hard requirement) and `kernels/` holds **≤4** kernel files total (readers/writers
unified as far as is clean); (b) the ROW_MAJOR path exercises the mcast all-gather
on a wide-W RM shape (add a probe/test proving a ROW_MAJOR row routes through
Regime B); (c) the full suite is non-regressed (1683/1683).

### [x] Refinement 5 — Remove the redundant DEST-level reduce-accumulate chunking

**Goal**: eliminate the PASS-1 **DEST-level W-chunking + reduce-accumulate**
pattern from the compute kernel(s). The chunked `Accumulate(cb_partial_sumsq, c)`
loop exists to keep the squared block within DEST capacity and accumulate Σx²
across chunks — i.e. it is an **OOM/capacity workaround for wide W**. But wide-W
OOM is *already* solved structurally by the **distributed reduction** (Regime B
W-split across cores, Refinement 1): each core's resident shard `Wt_s` is chosen
to fit L1, and cross-core partials are combined by the all-gather. Carrying a
*second*, DEST-level chunking-and-accumulate inside the per-core reduce is
redundant and obscures the compute — DEST-level chunking makes no sense once the
shard is already L1-bounded.

**The pattern to remove** (both computes, identical):
```cpp
// rms_norm_compute.cpp:80, 91-123  (and rms_norm_compute_rm.cpp:80-112)
constexpr uint32_t num_chunks = (Wt + reduce_block - 1) / reduce_block;
for (uint32_t c = 0; c < num_chunks; ++c) {
    // square resident[base..base+cw) -> cb_squared
    // reduce<SUM,REDUCE_ROW>(..., Accumulate(cb_partial_sumsq, c));  // <-- accumulate across chunks
}
```

**Target**: a single reduce over the whole resident shard producing the local Σx²
directly — no `Accumulate`-across-chunks, no `num_chunks` loop on the reduce path.

**Why there is no DEST constraint to worry about** (do not reintroduce one): the
compute uses the **kernel-lib helpers** (`ckl::eltwise_chain` for the square,
`ckl::reduce` for the sum), which are **L1→L1** operations. The helper owns the
DEST lifecycle internally — it tiles its own work through DEST as needed, so the
`square`/`reduce` over the full shard width is a single helper call regardless of
how many tiles `Wt_s` is. The kernel does **not** see DEST capacity; the
`reduce_block` / `num_chunks` loop is **not** required by any hardware limit. It is
pure redundancy left over from a hand-rolled view of DEST, and OOM (bounding the
resident shard) is already handled one level up by the distributed W-split. Remove
the loop and pass the full shard to one `square` + one `reduce`.

**Where to look**: `kernels/rms_norm_compute.cpp:80-123`,
`kernels/rms_norm_compute_rm.cpp:80-112`; `reduce_helpers_compute.hpp` (confirm the
reduce helper accepts an arbitrary-width `ReduceInputBlockShape` and manages DEST
internally — so `Accumulate` is unnecessary); `op_design.md:62,223-224`
(REDUCE_BLOCK = `min(Wt_s, DEST_AUTO_LIMIT)` rationale — **this is the line item
being deleted**). Also re-check PASS-2's per-chunk normalize loop: if it shared the
same `num_chunks` structure only to mirror PASS-1, simplify it consistently (the
same L1→L1 helper reasoning applies to the normalize multiplies).

**Implementation skill**: /memory-budget-metal (confirm the per-core shard is
already L1-bounded by the distributed reduction, so no in-kernel reduction-chunking
pattern is needed at all).

**Verifier notes**: correctness-preserving simplification — golden stays
**1683/1683**, precision matrix unchanged within tolerance (bf16 byte-identical
anchor where possible). This is the natural partner of R4; if the unified compute
from R4 lands first, apply this simplification to the single kernel. **Done when**
the per-core PASS-1 reduce is a single `square` + single `reduce` over the full
shard (no `num_chunks` / `reduce_block` / `Accumulate` loop), the kernel is simpler,
and the full suite is non-regressed.

### [~] Refinement 6 — Performance: measure (tracy), tune the heuristic, exploit the forgotten knob

**Goal**: make the op **fast**, measured, not assumed. Concretely:
**minimize per-kernel device time, maximize active cores** (subject to that
actually helping — see the crossover below), and tune the in-op blocking / mcast
parameters against real measurements across both regimes.

**Sub-tasks**:

1. **Research the measurement methodology first.** Understand how kernel device
   time is measured via the **Tracy profiler** on this stack before changing any
   knob (`tt_metal/tools/profiler`, `TT_METAL_DEVICE_PROFILER`, the
   `.device_timing.jsonl` artifact the eval harness already emits, and the
   `device_run_seconds` / `device_timings` the runner records on sim). Write down
   the exact command/flow used to get per-kernel device time for a single
   `rms_norm` invocation. This is a prerequisite, not optional.

2. **Stand up perf tests for both regimes.** A `test_rms_norm_perf.py` (or probe
   harness) that sweeps representative shapes covering **Regime A** (many
   tile-rows, narrow/moderate W) and **Regime B** (few rows, wide W:
   4096/8192/16384/32768) × {bf16, fp32} × {±gamma} × {TILE, ROW_MAJOR}, capturing
   per-kernel device time and active-core count per case. These drive the tuning;
   commit the baseline numbers table (like `precision_matrix_results.md`).

3. **Exploit the forgotten perf knob — row-blocking (`BLOCK_HEIGHT > 1`).** This is
   explicitly flagged as an unimplemented pure-perf refinement in
   `op_design.md:229-232`: "one compute init/reconfig amortized over a taller block"
   and bigger (coalesced) mcasts. **Hard constraint from the design**: row-blocking
   must **NOT** reduce active core count — `BLOCK_HEIGHT` may grow only after every
   core already has work, and trades block height against occupancy under the same
   L1 budget AND against DEST capacity (halved when `fp32_dest_acc_en`). The design
   claims all CB sizings already reference `bh`, so this is a host-heuristic change
   plus larger `cb_input_resident` / `cb_partial_sumsq` / `cb_recip_rms` /
   `cb_partials_gathered` allocations — verify that claim against the current
   descriptor (post-R4/R5 it may have shifted). "Less inits, bigger mcasts" is the
   target win — measure it.

4. **Tune the A-vs-B crossover — when is the W-split actually worth it?** The
   current heuristic (`rms_norm_program_descriptor.py:152-166`) chooses Regime B
   whenever a rectangular partition **adds cores** over Regime A (`Ht_total * K >
   regime_a_cores`), with **no cost model for the mcast**. But:
   - **For OOM, the W-split is mandatory** — when a full row doesn't fit one core's
     L1 budget, Regime B (or, post-R4, the unified mcast path) is the only option.
     Keep that as a hard correctness floor.
   - **For performance, more cores is not automatically a win.** Setting up the
     cross-core all-gather (semaphores, mcast rounds, the partial combine) has real
     fixed cost; for a row that *fits* one core and is only moderately wide, paying
     that overhead to gain a few cores can be **slower** than staying in Regime A.
   Use the perf tests (sub-task 2) to **measure the crossover**: the W (or Wt/L1
   occupancy) at which W-split's parallelism win exceeds its mcast setup cost.
   Encode the result in the heuristic — e.g. only go B when the row doesn't fit L1
   (OOM-forced) **or** when measured Wt exceeds the empirical crossover — instead of
   the current "B whenever it adds any cores." Document the crossover number and the
   measurements behind it in `changelog.md`.

5. **In-op tuning of blocking / mcast parameters.** With the above in place, tune
   the surviving blocking knobs — `BLOCK_HEIGHT` (row-blocking) and `K` (the
   W-split factor), plus any L1-sizing parameter that outlived R5 — per
   regime/shape to minimize measured device time, and record the chosen policy.
   (Note: R5 removes the reduce-path `reduce_block` chunking, so it is not expected
   to remain a tuning knob; tune what actually remains in the unified kernel.)

**Where to look**: `rms_norm_program_descriptor.py:112-175,332-` (`_select_k`, the
A/B heuristic, `RESIDENT_BUDGET_TILES`, `_dest_limit`); `op_design.md:13-16` (P2
"use the whole grid"), `:62` (REDUCE_BLOCK), `:229-232` (row-blocking knob);
`tt_metal/tools/profiler` + `references/cross_core_reduction_design.md`.

**Implementation skill**: none directly — this is measurement + host-heuristic
tuning. Pull on /memory-budget-metal for the L1/DEST budget arithmetic that bounds
`BLOCK_HEIGHT`, and /interleaved-parallel for the grid-saturation reasoning.

**Verifier notes**: lands **last**, on the unified+simplified base (R4+R5), so the
numbers reflect the final kernel. This is a **non-registry** refinement — SUPPORTED
is unchanged, golden stays **1683/1683** (a perf change that breaks a golden cell is
a bug, not a tradeoff). Success is judged on the committed before/after device-time
table, not on the support matrix. **Done when**: (a) the Tracy measurement flow is
documented and reproducible; (b) a perf-test/baseline table exists for both regimes;
(c) row-blocking is implemented without reducing core count and its win is measured;
(d) the A/B heuristic has a measured crossover (not "B whenever it adds cores") with
OOM kept as a hard floor; (e) measured per-kernel device time improves over the
R5 baseline on the wide-W and many-row representative shapes, with numbers in
`changelog.md`.

### [~] Refinement 7 — Row-blocking (BLOCK_HEIGHT>1) for Regime A many-row, behind a `bh` CT arg

**Goal**: implement the deferred "forgotten knob" from Refinement 6 — process `bh`
tile-rows per work-unit in Regime A (the grid-saturated many-row case), to amortize
per-row helper init over a taller block. **No SUPPORTED axis** (non-registry, perf).

**What R6 already established (don't re-measure from scratch)**:
- Measured ceiling is **~10%** on the current kernel: a Regime-A rows/core scaling
  sweep (W=256) showed a near-flat marginal ~10-11us/row — the per-row cost is
  data-movement/compute bound, not init bound, so row-blocking only recovers the
  per-helper init overhead. (This is why R6 prioritized K-tuning, which gave 2-6x,
  over row-blocking.)
- Hard design constraint (op_design.md:229-232): `BLOCK_HEIGHT` may grow **only after
  every core already has work** — it must NOT reduce active core count. So it applies
  to Regime A when `Ht_total > total_cores` (cores already saturated, each owns
  multiple rows); group a core's owned rows into blocks of `bh`.

**Exact next levers for the implementer**:
1. Thread a `bh` compile-time arg through the unified compute kernel
   (`rms_norm_compute.cpp`), defaulting to 1 so all existing paths are byte-identical.
   The reduce `ckl::reduce<SUM,REDUCE_ROW>` over a `bh×Wt` block must emit `bh` output
   tiles (one Σx² column-vector per row) via `ReduceInputBlockShape::of(bh, Wt)`;
   FINALIZE produces `bh` recip tiles; PASS-2's Col-broadcast multiply must index the
   per-row recip with a `TileOffset` (row r uses recip tile r).
2. Size `cb_input_resident` / `cb_squared` / `cb_partial_sumsq` / `cb_recip_rms` to
   `bh*` their current page counts in `_regime_a_descriptor` (and Regime B's
   `cb_partials_gathered`/`cb_local_sumsq` if extending there). Verify against the
   byte-aware `L1_RESIDENT_BUDGET_BYTES` floor R6 added.
3. Host heuristic: pick `bh` = rows-per-core (or a divisor) only when
   `Ht_total > total_cores`, bounded by L1 (the byte budget) AND by DEST capacity
   (`_dest_limit(cfg)`, halved when fp32_dest_acc_en). Measure with
   `test_rms_norm_perf.py` (extend it with a bh sweep).
4. Bound the blast radius: land it for **TILE Regime A no-gamma first** (simplest),
   keep `bh=1` everywhere else, and gate every change on golden staying **1683/1683**.

**Done when**: row-blocking implemented behind `bh`, active core count never reduced,
golden 1683/1683 preserved, and the measured many-row Regime-A device time improves
(expected modest, ~10%, per R6's ceiling measurement) — numbers in changelog.md.

**Outcome (`[~]` partial)**: the `bh` knob is fully implemented, correct (golden
1683/1683 with it forced on; PCC ≥ 0.99999), and active-core-count-safe, BUT
measurement shows it is **net-negative** on Regime A (0.83–0.96x of bh=1 across
(1,1,{4096,8192,16384},256) and (1,1,{4096,8192},512)) — so it is **disabled by
default** (`_ENABLE_ROW_BLOCKING = False`) to avoid a production regression. Root
cause: `rms_norm` Regime A is **DRAM-bandwidth-bound** (~10us/tile-row read+write),
and row-blocking only amortizes per-row **compute init**, which is not the
bottleneck. Confirmed fundamental, not an impl defect: `ttnn-static-analyzer`
findings F1 (redundant per-row PASS-2 reconfig → fixed with `Reconfig::None`) and F3
(single-buffer serialization → fixed with a `2*bh*Wt` double buffer) were both
applied and re-measured — **still net-negative**. The "device time improves" clause
is therefore NOT met on Regime A. See changelog.md for the table. Follow-up below
redirects row-blocking to the place it can actually pay off (Regime B mcast
coalescing), per op_design.md:229-232 ("bigger coalesced mcasts").

### [x] Refinement 8 — Row-blocking for **Regime B** (coalesced mcast all-gather), not Regime A

**Goal**: realize the design's OTHER row-blocking benefit — **bigger (coalesced)
mcasts** (op_design.md:229-232) — in **Regime B**, where the cross-core all-gather has
a real fixed cost (R6: K mcasts + a K-tile combine, the dominant non-reduce term).
Regime A row-blocking is a confirmed dead end (R7: memory-bound, net-negative even
after the F1/F3 fixes), so do **not** revisit it; the win, if any, is in amortizing
the **mcast/semaphore fixed cost** over a taller block of rows in Regime B.

**Why this is the right target**: R7 proved row-blocking cannot help a memory-bound
phase (Regime A is pure read→compute→write). Regime B additionally pays a
per-row-group all-gather (semaphore handshake + K mcast rounds + K-tile combine) whose
**fixed** cost does NOT scale with the data — so grouping `bh` row-groups per core and
issuing ONE coalesced mcast of `bh` partials (instead of `bh` separate K-round
gathers) could amortize that fixed cost. This is the "bigger coalesced mcasts" the
design names, and it is structurally distinct from R7's compute-init amortization.

**Exact next levers for the implementer**:
1. The `bh` compute CT arg, `_regime_a_block_height`, the `_FORCE_BH` hook, and the
   row-blocked PASS-1/FINALIZE/PASS-2 structure already exist (R7) and are correct —
   reuse them. The new work is **host-side (Regime B band grouping)** + **reader-side
   (coalesced all-gather of `bh*K` partials)**, NOT new compute math.
2. In `_regime_b_descriptor` / `_regime_rm_b_descriptor`: group `bh` row-groups into one
   core's work-unit (each core then owns `bh` shards along H × one W-column shard),
   bounded by L1 (`cb_partials_gathered` grows to `bh*K`, `cb_local_sumsq`/`cb_input_resident`
   to `bh*`) and DEST. Keep active-core-count unchanged (bh only after the grid is full).
3. In `rms_norm_reader.cpp` (`num_partials > 1` path): issue ONE mcast carrying the
   `bh` local partials per round instead of `bh` separate gathers — coalesce the
   `sender.send` / `receiver.receive` payload to `bh * partial_tile_bytes`. Measure the
   mcast-round count and per-op device time vs the current per-row-group gather.
4. Measure with `test_rms_norm_perf.py` (extend the `_FORCE_BH` sweep to Regime B
   shapes: (1,1,32,8192), (1,1,32,16384), etc.). **Done when** the coalesced-mcast
   Regime-B device time improves over the R6/R7 baseline on wide-W shapes (or, if it
   too proves net-negative, document that row-blocking is fully exhausted as a lever and
   close it). Golden stays 1683/1683.

> **QUEUE-CLOSED DIRECTIVE (applies to every implementer from here on):**
> **Refinement 9 below is the FINAL refinement.** No implementer — including the
> one running Refinement 8 or 9 — may append a `### [ ] Refinement N` section to
> this file. The pattern of each phase spawning its own follow-up (R6→R7→R8) stops
> here. If you discover further worthwhile work, record it in `changelog.md` under a
> "Future ideas (not queued)" note ONLY; do not add it to `op_requirements.md`.

### [ ] Refinement 9 — All-reduce algorithm + combine compute (and conditional K re-tune)

**THIS IS THE LAST REFINEMENT. It has NO authority to add new refinements/follow-ups**
(see the queue-closed directive above). Any leftover idea goes to `changelog.md` as a
note, never as a new `Refinement N` section.

**Goal**: the cross-core Σx² all-reduce is currently *every core mcasting its partial
to every other core* (an all-to-all all-gather), and the combine that sums the gathered
partials is an eltwise `add` loop that round-trips L1 on every step. Both are likely
suboptimal. Explore better all-reduce **transports** and a better combine **compute**,
measure them with R6's perf harness, and keep whatever is fastest. **No SUPPORTED axis**
(non-registry, perf + structure). Golden stays **1683/1683** throughout.

**Part A — all-reduce transport (the topology is the lever).**
Current: `rms_norm_reader.cpp` (`num_partials > 1` path, ~lines 203-230) runs a K-round
rotating-sender **mcast all-gather** — `SenderPipe`/`ReceiverPipe`, `Staging::Counter`,
`EXCLUDE_SRC` — so all K cores mcast. That is O(K) mcasts per row-group and contends the
NoC. Prototype and measure at least these alternatives, per regime and across K:
1. **Ring all-reduce (unicast ring).** Each core unicasts its running partial to its ring
   neighbor; after K-1 unicast hops the full Σx² has circulated (reduce-scatter +
   all-gather, or a simple running-sum ring since the payload is a single tile). No
   all-to-all mcast, minimal NoC contention; latency grows with K but each step is tiny
   (1 tile). Likely best at large K.
2. **Gather-then-broadcast (one root).** One designated core reads (or receives) all K
   partials, sums them locally, then does a **single** mcast of the final Σx² to the
   group. Trades K mcasts-of-partials for 1 gather + 1 mcast-of-result. Likely best at
   small/medium K.
3. (Keep the current all-mcast as the baseline to beat.)
Check `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp` for an existing unicast/ring primitive
before hand-rolling `noc_async_*`; `references/cross_core_reduction_design.md` for the
topology trade-offs; honor the op_design.md §9 silent-hang checklist (virtual coords,
barrier-before-signal, `num_dests`, semaphores on the union, never-mcast-to-self) for
any new transport. Pick the winner per (regime, K) — it may be a small switch in the
descriptor/heuristic, not one global choice.

**Part B — the combine compute (stop the L1 round-trips).**
Current combine, `rms_norm_compute.cpp` ~lines 264-273:
```cpp
ckl::copy<cb_partials_gathered, cb_partial_sumsq>(EltwiseShape::tiles(1));   // slot 0
for (uint32_t k = 1; k < num_partials; ++k)
    ckl::add<cb_partials_gathered, cb_partial_sumsq, cb_partial_sumsq>(EltwiseShape::tiles(1)); // K-1 adds
```
Each `add` is an `eltwise_chain` that reads two operands from L1 and packs the result
back to L1 — **K-1 unnecessary L1 round-trips** to sum K single tiles. Replace with one
of (measure both if cheap):
- **DEST accumulation / FPU dest-reuse**: accumulate all K gathered tiles into a DEST
  register (FPU/SFPU accumulate, e.g. `add_tiles` with DST reuse, no intermediate pack),
  and **pack exactly once** at the end. Investigate `dest_helpers.hpp` and the
  reduce/eltwise helpers for a dest-accumulate path that does not pack per step.
- **L1 pack-accumulate**: use the pack stage's L1 accumulate to sum into a single L1
  destination without re-reading the running sum each step.
The combine is summing K column-tiles into one — a reduction. If the all-gather lays the
K partials out contiguously, a single `ckl::reduce<SUM>` over the K-tile block may express
the whole combine in one helper call (no copy + add loop at all). Prefer that if it
measures faster and stays correct (Refinement 1's correctness invariant: the reader must
hand over only the FULLY-accumulated local Σx² — preserve `cb_local_sumsq`).

**Part C — conditional heuristic re-tune (ONLY if Part A/B measured a win).**
R6 set the A↔B crossover (Wt ≥ 16) and a proxy-min-K under the *current* expensive
all-mcast cost. If the new transport/combine makes the all-reduce **cheaper**, that
changes the economics: bigger K (more cores, narrower shards) becomes more attractive
because the per-K mcast/combine cost you were paying to avoid is now smaller. So **if and
only if** Part A or B shows a measured improvement, re-tune R6's `_select_k` /
crossover / proxy-min-K against the new cost and record the new numbers. If no transport
beats the baseline, leave the heuristic untouched and say so.

**Where to look**: `rms_norm_reader.cpp:203-230` (all-gather), `rms_norm_compute.cpp:264-273`
(combine), `rms_norm_program_descriptor.py` (`_select_k`, crossover, the regime
descriptors), `mcast_pipe.hpp`, `dest_helpers.hpp`, `reduce_helpers_compute.hpp`,
`references/cross_core_reduction_design.md`, `op_design.md:229-232`. R6's
`test_rms_norm_perf.py` is the measurement harness — extend it with the transport/combine
variants on Regime B wide-W shapes ((1,1,32,{8192,16384,32768}), (1,1,64,12288)).

**Implementation skill**: none directly (cross-core all-reduce + DEST/L1 accumulate are
outside every skill's scope). Pull on /memory-budget-metal for the DEST/L1 budget
arithmetic and the L1-accumulate reduction pattern.

**Verifier notes**: terminal, non-registry perf+structure refinement — golden stays
**1683/1683**, precision matrix within tolerance (the combine is a numerical change:
re-check Regime B PCC, especially bf8b). Build on whatever R8 leaves (R8's coalesced
`bh*K` mcast and R9's transport choice interact — a ring/gather-bcast must still carry
the coalesced payload if R8 landed it). **Done when**: (a) at least the ring and
gather-then-broadcast transports are prototyped and measured against the all-mcast
baseline, with a per-(regime,K) winner selected (or a documented "baseline wins");
(b) the combine no longer does K-1 L1 round-trips (DEST-accumulate, L1-accumulate, or a
single reduce), measured; (c) Part C done iff a win was measured; (d) before/after
device-time numbers in `changelog.md`; (e) golden non-regressed; (f) **no new refinement
appended** — leftover ideas live in changelog.md only.
