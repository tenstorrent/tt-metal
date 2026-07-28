# Operation Requirements: rms_norm

## Definition

- **Formula**: `out[..., h, w] = x[..., h, w] * rsqrt( (1/W) * Σ_{w'=0}^{W-1} x[..., h, w']² + ε ) * gamma[w]`
  (reduction is over the **last** dimension only; `gamma` optional)
- **PyTorch Reference**:

  ```python
  def pytorch_rms_norm(input_tensor, *, gamma=None, epsilon=1e-6):
      original_dtype = input_tensor.dtype
      x = input_tensor.to(torch.float32)
      rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + epsilon)
      result = x / rms
      if gamma is not None:
          result = result * gamma.to(torch.float32).reshape(-1)
      return result.to(original_dtype)
  ```

- **Import Path**: `from ttnn.operations.rms_norm import rms_norm`
- **Function Signature**:

  ```python
  def rms_norm(
      input_tensor: ttnn.Tensor,
      *,
      gamma: Optional[ttnn.Tensor] = None,
      epsilon: float = 1e-6,
      compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None,
      memory_config: Optional[ttnn.MemoryConfig] = None,
      program_config: Optional[Any] = None,
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

---

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: `[ttnn.float32, ttnn.bfloat16]`
- **SUPPORTED layout**: `[TILE_LAYOUT, ROW_MAJOR_LAYOUT]` — both **native** (tilize-wrapped reader / untilize writer in-kernel, no host `to_layout`)
- **SUPPORTED shape-derived axes**: `alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}` (all three native, masked reduce + valid-stick writer); `rank ∈ {2, 3, 4}`
- **SUPPORTED op-specific axes**: `gamma_mode ∈ {gamma, no_gamma}`; `gamma_dtype ∈ {float32, bfloat16, "none"}`; `gamma_layout ∈ {TILE, ROW_MAJOR, "none"}` (both independent of the activation's)
- **SUPPORTED memory_layout**: `[INTERLEAVED]`
- **SUPPORTED fp32_dest_acc_en**: `[True]` (the maxed-out precision corner)
- **EXCLUSIONS**: `{dtype: float32, fp32_dest_acc_en: False}` — permanent op-side refusal
- **Cores**: **multi-core from day 1** — the independent tile-row axis is split over the full grid via `ttnn.split_work_to_cores(grid, ht_total, row_wise=True)`. Measured: 110/110 cores on the prefill shapes, **1 core on the decode shapes** (`ht_total == 1`), which is the standing perf gap.
- **Blocking**: one knob `L1_BLOCK_BUDGET_BYTES = 512 KB` → `TILE_BLOCK_BUDGET` → `WT_CHUNK` / `HT_BLOCK` / `NW`; depth knobs `X_DEPTH = OUT_DEPTH = GAMMA_DEPTH = 2`, `X_RESIDENT_DEPTH = 2`; predicate-guarded `X_RESIDENT` / `GAMMA_RESIDENT` residency with a bounded streaming fallback. Per-core CB total 408–920 KB against a 1 100 KB budget.
- **Compute config**: `default_compute_kernel_config()` = HiFi4 + `fp32_dest_acc_en=True` + `math_approx_mode=False`; `math_fidelity` accepted at any value and never gated
- **Golden baseline**: **755 / 755** supported cells passing; `supported_fail` = `xpass_drift` = `xfail_wrong_mode` = 0 (per `verifier_report.json`)

---

### [ ] Refinement 1 — Numerical configurability expansion (unlocks every perf target)

**Goal**: add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` **and** `SUPPORTED["gamma_dtype"]`, and add `False` to `SUPPORTED["fp32_dest_acc_en"]`, so the full float surface `{bfloat16, float32, bfloat8_b} × {fp32_dest_acc_en True, False}` is expressible. `compute_kernel_config` is already on the entry point and already forwarded verbatim to the compute-kernel descriptor — the work is (a) relaxing the `validate()` gate, (b) deriving the intermediate-CB formats and DEST budget correctly for the new cells, including `UnpackToDestFp32` tagging where it applies, and (c) keeping `default_compute_kernel_config()` returning `fp32_dest_acc_en=True` unchanged. Cells that fail out of the box — canonically `bfloat8_b + non_tile_aligned_dim` — land in **`EXCLUSIONS`**, not in their own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- **This is the queue's gating refinement, and the reason is mechanical, not stylistic.** All 13 perf-flagged `LOOSE_CASES` in `eval/golden_tests/rms_norm/feature_spec.py` pin `fp32_dest_acc_en=False` (+ `math_fidelity=HiFi2` via `extras`), so **every one of them is `xfail` today**. Until this lands, a perf slot could only measure a supported stand-in at `fp32_dest_acc_en=True` — a different DEST datapath from the one every `achievable_ns` reference was taken at. Nothing downstream may measure before this ships.
- Largest single axis gap in the suite: `fp32_dest_acc_en=False` alone touches **3 255** xfail cells; `dtype=bfloat8_b` 960 and `gamma_dtype=bfloat8_b` 860.
- Two op-file invariants must survive: `default_compute_kernel_config()` is read by `eval/golden_tests/rms_norm/axes.py:40-43` and asserted by `test_rms_norm.py::test_rms_norm_default_config_matches_factory`, so its `fp32_dest_acc_en=True` default is fixed; and the existing `EXCLUSIONS` entry `{float32, fp32_dest_acc_en: False}` becomes *reachable* the moment `False` joins SUPPORTED — it is the load-bearing gate that keeps an fp32 input off a bf16 DEST accumulator (`references/precision_convention.md`). Do not delete it.
- **Accuracy watch item, not a blocker.** With `fp32_dest_acc_en=False` the `AccumulateViaAdd` running sum lands in bf16 DEST. The perf loose cases carry a tighter-than-default soft gate (`extras["pcc_threshold"] = 0.9995`) at `W` up to 7168 (`NW = 7` chunks). `examples/row_reduce_accumulate` measures bf16 *accumulation* error growing with reduce width; the pairwise-add datapath this op already uses is the good one (1.4 ULP at 32 tiles vs 13.3 for a single folded reduce). If 0.9995 misses, the fix is the SFPU finalize from that example — never widening the gate.
- `feature_spec.INVALID` currently parks `{bfloat8_b, w_non_aligned}` and `{bfloat8_b, h_non_aligned}` (720 cells) as author-scoped skips; the verification report recommends re-homing them to EXCLUSIONS. This refinement is correct either way — if they move, it covers them; if not, they stay skipped.
- Worth *trying* while the skill is open (do not let it gate the refinement): fp32 currently delivers ≈11 effective mantissa bits because the two FPU multiplies truncate their operands through 19-bit SrcA/SrcB. `UnpackToDestFp32` on those inputs is the lever, and it is inside this skill's scope. Every fp32 cell passes today with ≥13× tolerance headroom, so this is upside, not a requirement.

**Done when**: `bfloat8_b` and `fp32_dest_acc_en=False` are in `SUPPORTED`, the 8 interleaved perf-flagged loose cases run as `supported_pass` (they need only this refinement), the golden suite is green with all three loud verifier categories at 0, and any residual failing cell is named in `EXCLUSIONS` rather than left failing.

---

### [ ] Refinement 2 — Cross-core `W`-split: partial-sum combine + `1/rms` multicast

**Goal**: split the **dependent** `W` axis across cores — each core reduces its own `W`-slice into a raw partial `Σx²` tile, the partials are combined across the grid, one core finalizes `rsqrt(mean + ε)` with `n_reduced = W` (the grand total), and the single `1/rms` tile is multicast back so every core can scale its slice. Add `ttnn.TensorMemoryLayout.WIDTH_SHARDED` and `ttnn.TensorMemoryLayout.BLOCK_SHARDED` to `SUPPORTED["memory_layout"]` (the same scheme with the data pre-placed), and engage the split for **interleaved** inputs whose independent row axis under-fills the grid.

**Verifier notes**:
- **Scheme-change, so it stands alone** — the new loop nest and the combine topology *are* the work. It is also the hardest thing in the queue, which is why it comes before the two easy axes: build the structure once and let the lighter refinements extend it, rather than building them against a structure that changes underneath them.
- **`op_design.md` designed this and deliberately left it reachable — read §4.2 and Lamp L1 before writing code.** Phase 0 already writes a **raw** partial into `cb_partials` (a CB distinct from `cb_rms_sum`) and already runs the finalize as a separate phase over a 1-tile-per-tile-row CB, so the combine slots between compute phases 3 and 4 without touching phases 1/2/5/6/7. Topology per §4.2: a line of cores along one grid **row** owns one tile-row's `W`.
- **Reference material** (no implementation skill covers cross-core sharding yet): `references/cross_core_reduction_design.md` for the combine; `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp` (`SenderPipe` / `ReceiverPipe`) + `kernel_lib/host/mcast_host.hpp` (`Mcast1D` / `McastConfig`) for the broadcast — do **not** hand-roll `noc_async_write_multicast` + semaphores. Measured guidance in `ttnn/ttnn/operations/examples/master.md`: use **`two_stage_grid_reduce`** when the grid is busy or the payload is 1 tile/core (both true here — 1.45–1.60× over a flat root, and the only low-variance variant), a flat **root** reduce for ≤4-core groups; **never all-gather the partials** (`tensix_all_reduce` measures unicast all-gather at 0.74×, the worst option). Never route the combine as a serpentine spanning two grid rows (`tensix_all_reduce_ring_transport`: ~47 µs trap).
- **Native sharded data access is what "WIDTH/BLOCK_SHARDED supported" means.** Each core's slice is already resident in its own L1, so the input CB must be placed on it with `ttnn.cb_descriptor_from_sharded_tensor` — zero-copy, no NoC read. Re-reading a core's own local shard through a `TensorAccessor` does not count as implementing the axis value even if the golden cells go green; `TensorAccessor` stays only for interleaved I/O.
- Unlocks 1 505 (`WIDTH_SHARDED`) + 1 502 (`BLOCK_SHARDED`) xfail cells and 5 perf-flagged loose cases. It also removes the reason for op_design's Lamp L2 (gamma broadcast) *in this regime*: under a `W`-split each core owns a **disjoint** gamma slice, so gamma is read exactly once in total. Do not file a separate gamma-broadcast refinement until it is clear whether a row-parallel path survives for tall shapes.
- **Build it performantly, not correct-only** — Refinement 3 optimizes exactly this path and will rewrite a naive stub. Hold it to the Phase 0 bar: fill the grid, batch both dataflow halves (reader ≥4–8 tiles per barrier, writer likewise), depth-2 streaming CBs, and keep every new block factor a single-source parameter (the existing `L1_BLOCK_BUDGET_BYTES` derivation chain), never an inlined constant.
- Golden-side shard specs come from `eval.sharding.auto_shard_config` / `shard_config`; the harness already requests a matching **sharded output** (`helpers.py:218-221` passes `memory_config=ttnn_input.memory_config()`), so `validate()` must stop refusing a sharded output `memory_config` for the schemes this refinement adds.
- Re-check `blk.cb_total_bytes` against `L1_CB_BUDGET_BYTES` on the sharded path: the resident shard now sits in L1 *on top of* the CBs. The existing halve-and-re-derive loop in `_derive_blocking` will absorb an overflow, but silently, by shrinking the block — make sure that is not what is paying for the shard.

**Done when**: `WIDTH_SHARDED` and `BLOCK_SHARDED` are in `SUPPORTED["memory_layout"]`, their golden cells and the 5 sharded perf-flagged loose cases pass, the interleaved wide/few-row `_WIDE` loose cases (`(1,1,32,16384)`, `(1,1,32,32768)`, `(1,1,64,12288)`) use **more than one core** (verifiable via the CORE COUNT column of `ops_perf_results` or `test_report_blocking`), all prior-phase tests still pass, and all three loud verifier categories stay at 0.

---

### [ ] Refinement 3 — Speed up the perf-flagged **decode** column

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` pins four decode profiles `(1, 1, 32, W)` for `W ∈ {1024, 2304, 5120, 7168}` at the fixed perf config (bf16 / TILE / `fp32_dest_acc_en=False` / `math_fidelity=HiFi2` / bf16 TILE gamma), each carrying an `extras["achievable_ns"]` reference measured on `blackhole_p150b` at `reference_aiclk_mhz = 1350`. **The headline is `(1, 1, 32, 7168)`**, whose entry additionally sets `minimum_expected_speedup = 7.0` with the comment *"expected to expose a decisively better architecture"* — at 1350 MHz its ≤ 104 259 ns reference and ≥ 7× requirement imply a **≤ 14 894 ns** goal. Clock-scale every reference before use: `scaled_ns = achievable_ns * 1350 / actual_aiclk_mhz`, and where `minimum_expected_speedup` is present the ceiling is `scaled_ns / minimum_expected_speedup`. Optimize **these exact shapes at that exact config** — never a `fp32_dest_acc_en=True` stand-in. Pick levers from `ttnn/ttnn/operations/examples/master.md`. No SUPPORTED change.

**Verifier notes** *(sizing + where the time actually is — the lever is yours)*:
- Measured baseline at the *default* config (bf16 / HiFi4 / fp32-on, blackhole_p150b @1350 MHz, DEVICE KERNEL DURATION), from `test_rms_norm_perf.py`: `32×1024` 12 172 ns (ref 9 149) · `32×2304` 21 382 (ref 17 003) · `32×5120` 42 439 (ref 75 825 — already ahead) · `32×7168` 57 474 (goal ≤ 14 894, **3.9× away**). Re-baseline at the perf config once Refinement 1 lands; these numbers are the wrong datapath and are for orientation only.
- **Structural cause, measured:** every decode shape has `ht_total == 1`, so the independent row axis offers exactly **one** unit of work and the op runs on **1 of 110 cores** (`test_report_blocking`). No block-size or buffer-depth knob reaches a 3.9× gap on one core — the win is grid occupancy, i.e. the `W`-split Refinement 2 builds. This phase is where that split gets *tuned* (how many cores per line, combine shape, chunk granularity per core, whether the whole `W` still needs chunking once it is spread).
- This is a ⭐⭐⭐ T3-scale phase — one hard lever plus its co-tunes, not a grab-bag. The natural co-tunes once the split exists: per-core chunk granularity (master.md's floor is **whole tiles minimum**, coarser amortizes the ~320 ns fixed per-helper-pass overhead from `compute_block_size`; the op's `L1_BLOCK_BUDGET_BYTES` is the single knob) and buffer depth (`X_DEPTH` / `X_RESIDENT_DEPTH` — note the existing depth heuristic keys off `grid_full`, which the `W`-split invalidates for these shapes, so it must be revisited).
- The op already exposes env overrides for A/B measurement without editing it: `RMS_NORM_BLOCK_BUDGET_KB`, `RMS_NORM_X_RES_DEPTH`, `RMS_NORM_FORCE_STREAMING` (see `test_rms_norm_perf.py`).
- Known dead ends, already measured — do not re-spend the phase on them: `L1_BLOCK_BUDGET_BYTES = 1024 KB` is **0.79×** on the wide prefill shape; running the reader a row-block ahead when the grid is full is **1.04× slower**; holding gamma resident at `NH_core == 1` is **1.12× slower** (a serial prologue).
- The soft precision gate `extras["pcc_threshold"] = 0.9995` still applies at the perf config — a faster kernel that misses it has not passed.

**Done when**: measured device-ns improves on all four flagged decode shapes at their exact pinned config, `(1,1,32,7168)` reaches its `minimum_expected_speedup = 7.0` ceiling (clock-scaled) or the shortfall is reported with the measured bottleneck, each shape's `pcc_threshold = 0.9995` soft gate still holds, the golden suite is green, and there is no regression across a config-spanning guard set (one representative per distinct kernel path × layout × placement: TILE/RM × gamma/no-gamma × resident/streaming × interleaved/sharded).

---

### [ ] Refinement 4 — `HEIGHT_SHARDED` placement (local shard, zero-copy)

**Goal**: add `ttnn.TensorMemoryLayout.HEIGHT_SHARDED` to `SUPPORTED["memory_layout"]` for both the input and the output. This is the Phase-0 row split made physical: the shard grid pins the core assignment, the shard height pins `HT_BLOCK`, and **the reduction stays entirely local** — each core holds full rows, so no cross-core traffic is added.

**Verifier notes**:
- **Knob-turn, not a scheme-change** (op_design Lamp L3): only `cb_input_tiles`' *placement* changes. Back it with `ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_TILES, input_tensor)` — **zero-copy on the core's own L1 shard, no NoC read** — and the reader's input pass disappears entirely. The compute phases are byte-identical, because the resident regime already consumes a whole resident row-strip. Mirror it on the output side.
- **An accessor read of a core's own local shard is not an implementation of this axis.** It would pass the golden cells (each core still holds its full rows, so the accessor reads the right bytes) while the op never actually implemented the placement. Check the dataflow, not the test colour.
- **Why it lands after Refinement 2 despite being easier**: R2 rewrites the core-assignment and CB-placement layer that this refinement then reuses. Landing the hard structural work first on the smaller test surface, then extending it for the easy axis, avoids building this against a structure that changes underneath it. There is no hard dependency in the other direction.
- Unlocks 1 501 xfail cells and the `(1,1,256,512)` HEIGHT_SHARDED loose case. Its `memory_layout` refusal in `validate()` and the sharded-output path should already be generalized by R2 — this refinement adds the value, the shard-height → `HT_BLOCK` derivation, and the local-shard CB placement.
- No implementation skill matches: `memory_layout` / placement is **not** `/memory-layouts` (that is RM ↔ TILE), and `/interleaved-parallel` explicitly excludes sharded tensors. The pattern to follow is the `cb_descriptor_from_sharded_tensor` CB placement named above.

**Done when**: `HEIGHT_SHARDED` is in `SUPPORTED["memory_layout"]`, its golden cells and the `(1,1,256,512)` HEIGHT_SHARDED loose case pass, the input shard is consumed via a sharded CB descriptor (no `TensorAccessor` read of a core's own shard), prior phases still pass, and all three loud verifier categories stay at 0.

---

### [ ] Refinement 5 — Speed up the perf-flagged **prefill** column and the sharded geometries

**Type**: perf

**Goal**: the remaining perf-flagged `LOOSE_CASES` — the four interleaved prefill profiles `(1, 1, 8192, W)` for `W ∈ {1024, 2304, 5120, 7168}` (`achievable_ns` 96 744 / 211 345 / 738 307 / 1 032 281) and the five measured-fastest sharded geometries whose `extras` pin the exact shard: `(32,1024)` WIDTH_SHARDED `[32,128]` on `(8,1)` → 4 110 ns, `(32,2304)` WIDTH_SHARDED `[32,256]` on `(9,1)` → 4 617, `(32,5120)` WIDTH_SHARDED `[32,160]` on `(8,4)` → 5 267, `(32,7168)` WIDTH_SHARDED `[32,256]` on `(7,4)` → 5 481, `(8192,1024)` BLOCK_SHARDED `[1024,128]` on `(8,8)` → 25 640. Same fixed config as Refinement 3 (bf16 / TILE / `fp32_dest_acc_en=False` / HiFi2), same clock-scaling rule. Levers from `ttnn/ttnn/operations/examples/master.md`. No SUPPORTED change.

**Verifier notes**:
- **The prefill column is already close and is READER-bound — size the phase accordingly.** Measured at the default config: `8192×1024` 102 114 ns (ref 96 744, 1.06× away) · `8192×2304` 215 181 (ref 211 345, 1.02×) · `8192×5120` 483 074 (ref 738 307, already ahead) · `8192×7168` 831 973 (ref 1 032 281, already ahead). Per-RISC profiling puts NCRISC at **90–99 %** of kernel time (442 221 of 482 655 ns on `8192×5120`), i.e. these sit at the **interleaved-DRAM read floor**. Only levers that reduce bytes or transactions can move them — batching/overlap levers are already spent (all 110 cores busy, reads coalesced `NW` chunks per barrier when the grid is full). Check `/perf-ceiling-dm` before spending effort here; if the roofline says there is no headroom, say so in the changelog rather than manufacturing a change.
- **The sharded geometries are where the headroom is** (4–6 µs targets vs tens of µs interleaved) and they are cheap to reach *after* R2 + R4: the shard is pre-placed, so this is mostly a placement/geometry tune — match the pinned `shard_shape` + `core_grid` from `extras` rather than letting `auto_shard_config` pick, and keep reads on NoC0 / writes on NoC1 with row-oriented core lines (`noc_placement`: 2.9× for row vs column, 2.5–4.8× for the correct NoC pairing).
- Several ⭐/⭐⭐ levers can share this phase since none is a restructure; the one T3-shaped item (the combine topology) was already spent in R2/R3.
- Do not re-file the dead ends listed under Refinement 3.

**Done when**: measured device-ns improves on the prefill shapes that are still short of their clock-scaled reference and on all five sharded geometries at their pinned shard specs (or the roofline evidence for "no headroom" is recorded), every shape's `pcc_threshold = 0.9995` soft gate holds, the golden suite is green, and there is no regression across the config-spanning guard set (one representative per distinct kernel path × layout × placement).

---

## Coverage check — `TARGET − SUPPORTED` is fully queued

| Axis | Missing value | Covered by |
|---|---|---|
| `dtype` | `bfloat8_b` | Refinement 1 |
| `gamma_dtype` | `bfloat8_b` | Refinement 1 |
| `fp32_dest_acc_en` | `False` (for `bfloat16`) | Refinement 1 |
| `fp32_dest_acc_en` | `False` (for `float32`) | **op-side `EXCLUSIONS`** — permanent refusal, never a refinement |
| `memory_layout` | `WIDTH_SHARDED` | Refinement 2 |
| `memory_layout` | `BLOCK_SHARDED` | Refinement 2 |
| `memory_layout` | `HEIGHT_SHARDED` | Refinement 4 |
| `layout`, `alignment`, `rank`, `gamma_mode`, `gamma_layout` | — | already complete at Phase 0 |

Nothing is omitted. The 5 768 `xfail_expected` cells map one-to-one onto the six queued values (see `verification_report.md` → "The `xfail_expected` bucket is fully accounted for").
