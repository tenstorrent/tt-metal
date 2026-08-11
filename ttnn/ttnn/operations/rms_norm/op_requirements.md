# Operation Requirements: rms_norm

## Definition

- **Formula**: `output[..., h, w] = input[..., h, w] * rsqrt( (1/W) * Σ_{w'=0}^{W-1} input[..., h, w']² + epsilon ) * gamma[w]`
  (the mean is over the **true, unpadded** `W`; `gamma` is optional)
- **PyTorch Reference**:

```python
def torch_rms_norm(x, gamma=None, eps=1e-6):
    """Reference RMSNorm over the last dimension, always computed in fp32."""
    x = x.float()
    out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    if gamma is not None:
        out = out * gamma.float().reshape(-1)
    return out
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
) -> ttnn.Tensor
```

`default_compute_kernel_config()` is exported from the same module and is the single source for the
`compute_kernel_config=None` case (HiFi4 / `fp32_dest_acc_en=True` / `math_approx_mode=False`).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True] (the maxed-out precision corner)
- **SUPPORTED layout**: [TILE, ROW_MAJOR] — both native, no host-side `to_layout` / `pad` / `slice`
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; rank ∈ {2, 3, 4}
- **SUPPORTED op-specific axes**: gamma_mode ∈ {gamma, no_gamma}; gamma_dtype ∈ {float32, bfloat16, "none"}; gamma_layout ∈ {TILE, ROW_MAJOR, "none"}
- **SUPPORTED memory_layout**: [INTERLEAVED]
- **EXCLUSIONS**: `{dtype: float32, fp32_dest_acc_en: False}` (the canonical precision exclusion; currently unreachable because `fp32_dest_acc_en=False` is outside SUPPORTED entirely)
- **Cores**: multi-core from day 1 — the grid is partitioned into `num_row_groups` rectangles; the
  `row` axis is split across groups and the **dependent** `hidden` axis is split *within* a group,
  with partials combined over the NoC (gather-to-root + multicast-back). Both regimes (R1
  `w_group_size == 1`, R2 `w_group_size > 1`) ship.
- **Compute config**: caller-supplied `ttnn.ComputeConfigDescriptor` passed through unchanged;
  `math_fidelity` / `math_approx_mode` ungated, `fp32_dest_acc_en` gated to True.
- **Golden baseline**: 737 / 737 supported cells passing; 6174 xfail_expected; 33900 invalid_skipped;
  0 in every loud category (per `verifier_report.json`).

### [ ] Refinement 1 — Numerical configurability expansion (`fp32_dest_acc_en=False` + `bfloat8_b`)

**Goal**: add `False` to `SUPPORTED["fp32_dest_acc_en"]` and `ttnn.bfloat8_b` to both
`SUPPORTED["dtype"]` and `SUPPORTED["gamma_dtype"]`, keeping `{float32, fp32_dest_acc_en=False}` in
`EXCLUSIONS`. This is the single largest cell unlock in the queue (≈ 1100 interleaved xfail cells:
818 that differ from Phase 0 only by `fp32_dest_acc_en=False`, 100 by `dtype=bfloat8_b`, 140 by
`gamma_dtype=bfloat8_b`, plus their combinations) **and it is the gate on every perf refinement**:
every `group="perf"` loose case in `feature_spec.LOOSE_CASES` runs at `fp32_dest_acc_en=False` +
`math_fidelity=HiFi2`, as does every `resilience` and `pad_poison` loose case. Work items:
the DEST-capacity derivation must follow `DEST_AUTO_LIMIT` in both settings (it doubles when fp32
DEST accumulation is off — the helpers already read it, so verify nothing in the kernel or the L1
solve assumes the halved value); the `Σ x²` accumulator and the whole `cb_stat_*` path must stay
**fp32 in L1 regardless**, because bf16 accumulation error on an all-positive monotonically growing
sum is exactly the failure mode `row_reduce_accumulate` documents; and `bfloat8_b` needs the
intermediate-CB formats and `cb_scaler` left at bf16 (`reduce_helpers_dataflow.inl:185-187`
`static_assert`s anything else). Cells that fail out of the box land in `EXCLUSIONS`, not in their
own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: **must land first** — Refinement 3 (the first perf phase) measures the
`fp32_dest_acc_en=False` + HiFi2 configuration, and a perf number taken at `fp32_dest_acc_en=True`
says nothing about it (different DEST datapath, different DEST capacity, different chosen `R`).
Do **not** proxy it. Two collisions to be deliberate about: (1) `bfloat8_b` × non-tile-aligned is
already parked in `feature_spec.INVALID` (author-scoped), so those cells stay *skipped* — do not
add an `EXCLUSIONS` entry for them expecting cell movement; (2) the golden `pad_poison` cases run at
`fp32_dest_acc_en=False`, so this refinement is the first time the ragged-hidden-tile mask
(`mask_tail_block`) is exercised with a bf16 DEST accumulator — the mask-before-square identity
`(x·mask)² == x²·mask` still holds, but the *poisoned* magnitude (1000.0) squared in a bf16 DEST is
where a lost mask now shows up as a large error rather than a small one. Keep `has_tail`'s stat
column fp32.

### [ ] Refinement 2 — Sharded input/output placements (all three schemes)

**Goal**: add `ttnn.TensorMemoryLayout.HEIGHT_SHARDED`, `WIDTH_SHARDED` and `BLOCK_SHARDED` to
`SUPPORTED["memory_layout"]`, consuming the resident shard **in place** and writing a matching
sharded output (the golden harness requests `memory_config=input.memory_config()` for every sharded
cell). ≈ 1800 interleaved-only xfail cells move (561 HEIGHT / 586 WIDTH / 586 BLOCK at the Phase 0
corner, and ~2000 more once Refinement 1 has landed `fp32_dest_acc_en=False`), plus 3 `_SHARDED`
and 5 `perf` loose cases.

**Verifier notes**: this is one refinement, not three, because `op_design.md`'s lamp S1 already
classifies all three as **placement unlocks of the scheme Phase 0 built** — the cross-core combine
exists, so the shard spec *supplies* the block geometry instead of the selection function choosing
it: HEIGHT ⇒ `w_group_size = 1` and the reduction is local; WIDTH ⇒ `w_group_size` = the shard
grid's core count and `core_w_tiles` = the shard width in tiles; BLOCK ⇒ one grid row of the shard
grid *is* the `Mcast2D` rectangle. Read the extents **off the spec**; do not re-run
`_select_regime` for a sharded input.
Two hard requirements, both native:
(1) point `cb_input_tiles` / `cb_output_tiles` at the resident L1 shard with
`ttnn.cb_descriptor_from_sharded_tensor(...)` (zero-copy) and drop the DRAM leg — re-reading a
core's *own* shard through a `TensorAccessor` is **not** an acceptable slow path: it would mean the
axis value is merely tolerated, not implemented, and this refinement would be sent back. The
accessor keeps owning interleaved I/O only.
(2) the multicast/gather L1 map must stay identical group-wide (`mcast_pipe.hpp:44-45`), which is
why Phase 0 sizes every CB to the group-uniform `C`; a shard grid that does not divide the tensor
evenly (the `_RESILIENCE_SHAPES` block is full of prime tile counts) must keep that invariant.
There is no implementation skill for placement yet — `/memory-layouts` is RM/TILE layout, **not**
placement, so do not invoke it here. Order: after Refinement 1 (so the sharded cells land at both
`fp32_dest_acc_en` values in one pass) and before Refinement 5, which measures the sharded perf
geometries this creates.

### [ ] Refinement 3 — Speed up the perf-flagged wide interleaved decode profile

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` marks `(1, 1, 32, 7168)` at **INTERLEAVED** placement as the
one perf case carrying `minimum_expected_speedup = 7.0` — "expected to expose a decisively better
architecture" — i.e. a goal of `104259 ns × 1350/actual_aiclk_mhz / 7` (≈ 14.9 µs at 1350 MHz),
with its soft `pcc_threshold = 0.9995` still holding. Optimize **that** shape at **its exact
config**: bf16 / TILE / INTERLEAVED / gamma bf16 TILE / `fp32_dest_acc_en=False` /
`math_fidelity=HiFi2`. Take the other three interleaved decode cases (`W` = 1024 / 2304 / 5120,
same config) along as secondary targets. Relevant patterns in
`ttnn/ttnn/operations/examples/master.md`: `width_split` (this is a one-tile-row tensor — the whole
regime), `tensix_all_reduce` (the combine topology: its measured rule of thumb is *grid two-stage
when the grid is busy or the payload is tiny*, and our payload is `block_row_tiles = 1` tile, while
Phase 0 uses flat root-gather + mcast — 1.45–1.60× is claimed there), and `double_buffer` /
`compute_block_size` for the co-tune below. No SUPPORTED change.

**Done when**: measured device-kernel ns improves on `(1,1,32,7168)` at the config above and moves
it toward the clock-scaled `achievable_ns / 7` goal, its `pcc_threshold=0.9995` gate still holds,
the golden suite is still green, and there is no regression across the config-spanning guard set
(one representative per distinct kernel path × layout × placement: TILE-in/TILE-out interleaved,
RM-in/RM-out interleaved, `w_group_size == 1` row-parallel, `w_group_size > 1` cross-core, a
`w_non_aligned` shape, and — once Refinement 2 has landed — one sharded cell).

**Verifier notes** (queue-level facts measured during verification, not levers I am prescribing):
`MAX_W_GROUP_SIZE = 32` is the *only* place the core-assignment knob is turned, and it was measured
**only at `tensor_row_tiles == 1`** (where capping 110 → 22 cores won 1.53–1.59× on these very
shapes). Its consequence for neighbouring shapes is unmeasured and visible in the selection
function: at `tensor_row_tiles = 2` the cap leaves **44 of 110** cores active, at 3 it leaves 66, at
4 it leaves 88 (computed from `_select_regime` on an 11×10 grid) — `(1,1,64,12288)` is exactly such
a `_WIDE` loose case. Both the cap and `MIN_PIPELINE_BLOCKS` live in one host constant each, so
re-tuning them is a knob-turn; replacing flat root-gather with a two-stage grid combine is a
scheme-change and the bigger of the two pieces of work — size the phase accordingly (one T3 lever
is a whole phase).

### [ ] Refinement 4 — Speed up the prefill (bandwidth-bound) profiles

**Type**: perf

**Goal**: the four `group="perf"` interleaved prefill cases — `(1, 1, 8192, W)` for
`W ∈ {1024, 2304, 5120, 7168}` with `achievable_ns` = 96744 / 211345 / 738307 / 1032281 — at their
exact config (bf16 / TILE / INTERLEAVED / gamma bf16 TILE / `fp32_dest_acc_en=False` / HiFi2).
These are aggregate-DRAM-bandwidth cases (the widest moves 33.5 MB), so the relevant
`ttnn/ttnn/operations/examples/master.md` patterns are the ones that raise achieved bandwidth and
reduce per-block overhead rather than add cores: `double_buffer` (bytes in flight per barrier,
buffer depth), `compute_block_size` (coarser blocks amortize init/reconfig; the granularity floor is
whole tiles, and coarser amortizes up to the L1 bound), and `noc_placement`. Co-tune
`block_row_tiles` against `input_cb_depth` / `output_cb_depth` — they trade the same L1 — and check
the tile-row imbalance the selection leaves (256 tile-rows over 110 cores is 3-vs-2). No SUPPORTED
change.

**Done when**: measured device-kernel ns improves on at least the two most-impacted prefill shapes
at the config above, with the golden suite green and no regression across the same config-spanning
guard set as Refinement 3.

**Verifier notes**: two facts to start from rather than re-measure. (1) `MIN_PIPELINE_BLOCKS` was
already swept 1/2/3/4 on these shapes at `fp32_dest_acc_en=True` and came out **flat** (±2%) — the
wall was aggregate DRAM bandwidth (~330 GB/s on the widest case), so a *pure* block-count change is
predicted to do nothing until the DRAM floor moves; re-measure at `fp32_dest_acc_en=False`, where
DEST capacity doubles and the selected `R` changes. (2) The writer's TILE-path drain is one
`noc_async_write_barrier` per **tile-row** (`core_w` tiles), which is inside `double_buffer`'s
measured 4–8-tile sweet spot for the perf shapes (`C` = 11–112) but **below** it for narrow-hidden
shapes (`C` = 2–3 on e.g. `(99991, 64)`, `(1,1,3232,96)`); batching several tile-rows behind one
barrier requires raising `output_cb_depth`, which is one host constant. That is L1-for-overlap, so
measure it against the block-size co-tune rather than stacking both blind.

### [ ] Refinement 5 — Speed up the sharded perf geometries

**Type**: perf

**Goal**: the five `group="perf"` **sharded** loose cases, whose `achievable_ns` are that exact
geometry's measured latencies and are 2–20× tighter than their interleaved twins:
`(1,1,32,1024)` WIDTH_SHARDED `[32,128]` on `(8,1)` → 4110 ns; `(1,1,32,2304)` WIDTH `[32,256]` on
`(9,1)` → 4617 ns; `(1,1,32,5120)` WIDTH `[32,160]` on `(8,4)` → 5267 ns; `(1,1,32,7168)` WIDTH
`[32,256]` on `(7,4)` → 5481 ns; `(1,1,8192,1024)` BLOCK_SHARDED `[1024,128]` on `(8,8)` → 25640 ns.
Same fixed config as the other perf cases (bf16 / TILE / `fp32_dest_acc_en=False` / HiFi2), with the
shard spec pinned by `extras`. The lever set is the zero-copy resident-shard path Refinement 2
builds plus the combine topology (`tensix_all_reduce`: 28–32-core 2-D groups under contention are
exactly the regime where its measured grid-two-stage reducer beats flat root-gather); `noc_placement`
applies because a sharded input removes the DRAM read entirely and leaves the NoC budget to the
combine. No SUPPORTED change.

**Done when**: measured device-kernel ns improves on at least three of the five flagged sharded
shapes at their pinned geometries, the golden suite is green, and no regression across the
config-spanning guard set (which by this phase must include one cell per sharded scheme).

**Verifier notes**: hard dependency on Refinement 2 — these cells cannot even run before it, and the
`achievable_ns` here are config-matched to the *sharded* placement, so they must never be compared
against an interleaved measurement. Expect the binding cost after the DRAM read disappears to be the
per-block combine round (~1.3–1.5 µs measured for a 1-tile payload on a 16-core group in
`tensix_all_reduce`) against a 4–5.5 µs total budget: at these targets the combine *is* the op, which
is why this is a separate phase from Refinements 3 and 4 rather than a continuation of them.
