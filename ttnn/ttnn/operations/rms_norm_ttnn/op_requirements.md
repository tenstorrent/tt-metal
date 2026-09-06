# Operation Requirements: rms_norm_ttnn

## Definition

- **Formula**:

  ```
  t = x + residual_input_tensor                        (optional, BEFORE the statistics)
  m = mean(t^2, dim=-1, keepdim=True)
  y = t / sqrt(m + epsilon)
  y = y * weight                                       (optional, per-channel)
  y = y + bias                                         (optional, per-channel, AFTER the scale)
  ```

  Rank 0 has no reduced dimension: `mean(t^2)` over a one-element row is `t^2`, so the scalar
  case is `t / sqrt(t^2 + epsilon)` and a zero scalar comes out **zero**, never NaN. A
  zero-volume input returns a copy at the requested placement.

- **PyTorch Reference** — exported from the op module as `torch_rms_norm_ttnn`, consuming
  **every** operand:

  ```python
  def torch_rms_norm_ttnn(input_tensor, *, epsilon=1e-12, weight=None, bias=None,
                          residual_input_tensor=None):
      t = input_tensor.to(torch.float32)
      if residual_input_tensor is not None:
          t = t + residual_input_tensor.to(torch.float32)
      if t.numel() == 0:
          return t.to(input_tensor.dtype)
      mean_sq = t * t if t.dim() == 0 else torch.mean(t * t, dim=-1, keepdim=True)
      y = t / torch.sqrt(mean_sq + epsilon)
      if weight is not None:
          y = y * weight.to(torch.float32).reshape(-1)
      if bias is not None:
          y = y + bias.to(torch.float32).reshape(-1)
      return y.to(input_tensor.dtype)
  ```

- **Import Path**: `from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn`

- **Function Signature**:

  ```python
  def rms_norm_ttnn(
      input_tensor: ttnn.Tensor,
      *,
      epsilon: float = 1e-12,
      weight: Optional[ttnn.Tensor] = None,
      bias: Optional[ttnn.Tensor] = None,
      residual_input_tensor: Optional[ttnn.Tensor] = None,
      memory_config: Optional[ttnn.MemoryConfig] = None,
      program_config: Optional[Any] = None,          # DEFAULT or SHARDED variant
      compute_kernel_config: Optional[Any] = None,   # ComputeConfigDescriptor or device CKC
  ) -> ttnn.Tensor
  ```

  Also exported: `default_compute_kernel_config()` (HiFi4 / `math_approx_mode=True` /
  `fp32_dest_acc_en=False` — the single source of truth the golden axis-tagger reads),
  `normalize_compute_kernel_config()`, `torch_rms_norm_ttnn()`, `PROPERTIES`.

---

## Why this queue is all perf

`TARGET - SUPPORTED` is **empty on every axis**, and `EXCLUSIONS` is empty:

| Axis | TARGET | SUPPORTED | Gap |
|------|--------|-----------|-----|
| `dtype` | float32, bfloat16, bfloat8_b | same | — |
| `fp32_dest_acc_en` | True, False | same | — |
| `layout` | TILE, ROW_MAJOR | same | — |
| `alignment` | tile_aligned, w_non_aligned, h_non_aligned | same | — |
| `rank` | 0, 1, 2, 3, 4, 5 | same | — |
| `gamma_mode` | no_gamma, gamma, gamma_bias, bias, residual, gamma_bias_residual | same | — |
| `gamma_dtype` | float32, bfloat16, bfloat8_b, "none" | same | — |
| `gamma_layout` | TILE, ROW_MAJOR, "none" | same | — |
| `memory_layout` | INTERLEAVED, HEIGHT/WIDTH/BLOCK_SHARDED | same | — |

So `xfail_expected` is 0 by construction — there is no xfail bucket to be a queue gap — and the
2:1 generality/perf cadence degenerates. Per the protocol, once generality candidates are
exhausted every remaining phase is a **measured perf refinement**, and the queue keeps going
until the headroom is gone or the roofline says there is none left.

Two categories deliberately produce **no** queue entry, and `verification_report.md` carries
both in full:

- **The 19 red golden cells are not the op's.** 10 `supported_fail` + 3 validation failures are
  two harness defects (`CoreRange.end_coord` on a build that exposes `.start`/`.end`;
  `torch.max()` on a zero-element readback), each independently reproduced outside the op; the
  6 `test_regression.py` numerics cells score the op's own declared default against
  `eval.metrics.DEFAULT_TOLERANCES` instead of the band `helpers.TOLERANCE_OVERRIDES` declares
  for exactly that cell. Three one-line harness fixes, no op change, no refinement.
- **Five `feature_spec.INVALID` entries are misclassified** (`feature_spec.py` labels them
  "author-scoped exclusions ... NOT structural impossibility"; three of them additionally cross
  axes describing *different tensors*). They are not refinements because `SUPPORTED` already
  claims those axis values and `validate()` already accepts them — there is nothing for an
  implementer to add. `test_rms_norm_ttnn_invalid_audit.py` proves all five regions pass, 16/16.

---

## Phases

> **Non-regression rule**: every refinement must pass all tests from prior phases, and
> additionally the **seed** gate —
> `test_rms_norm_ttnn_perf.py::test_program_is_structurally_the_seeds` asserts that every
> operand-free and gamma-only build has the seed's CB set and the seed's kernel args across 14
> scheme-spanning geometries. A change that moves a knob for the operand-free path must keep
> that test green **or** carry a measurement that justifies breaking it (the prompt's rule: "a
> faster program for an operand-free configuration is allowed, but only with a measurement,
> never with an argument").
>
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to
> update SUPPORTED. A **perf** refinement adds nothing to `SUPPORTED`, so `verify_supported`'s
> categories must not move across one — if they do, something other than performance changed.
>
> **Checkbox protocol**: `[x]` complete and all tests pass; `[~]` real work landed but at least
> one named lever is deferred (treated as completed by the queue, surfaced as partial); `[ ]`
> only when nothing usable was produced.
>
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: primary
> refinements are `Refinement N`. A follow-up to a `[~]` partial appends a lowercase letter to
> the parent's number — `Refinement 1b`, `Refinement 1c` — never `Refinement 1.5` or a fresh
> number, and it is ordered immediately after its parent. The parser matches exactly
> `Refinement \d+[a-z]?`.
>
> ### Two things to do at the START of every phase
>
> 1. **Re-rank the targets.** No `LOOSE_CASES` entry carries an `attention` note, so
>    `eval/prompts/perf_refinement_prompt.txt` step 1 falls through to "rank the `perf` group by
>    measured device-ns divided by each case's own `achievable_ns` and take the worst".
>    `ttnn/ttnn/operations/rms_norm_ttnn/perf_target_ranking.py <results_dir> --aiclk <MHz>`
>    computes exactly that from a golden run's own sidecars (`eval_test_runner.sh` captures
>    `device_kernel_ns` per test by default). The ordering below is the Phase-0 ranking, not a
>    permanent assignment — it moves as phases land.
> 2. **Isolate the JIT cache per variant.** This environment sets `TT_METAL_CACHE=<repo>/built`,
>    `TT_METAL_CCACHE_KERNEL_SUPPORT=1` **and** `TT_METAL_JIT_SERVER_ENDPOINT`. A reverted
>    kernel edit was observed being served its *previous* binary — producing silent non-finite
>    output — and `rm -rf built` did **not** clear it. Every refinement below is a kernel A/B.
>    Run each variant under its own `TT_METAL_CACHE=<fresh tmp dir>`, or pass `--no-jit-server`.
>    Without this you will measure the wrong binary and not know it. Full write-up:
>    `verification_report.md` § Harness findings 6.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16, bfloat8_b]
- **SUPPORTED fp32_dest_acc_en**: [True, False] — both, at every dtype
- **SUPPORTED layout**: [TILE, ROW_MAJOR] — both native, no host-side `to_layout` / `tilize`
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned};
  rank ∈ {0, 1, 2, 3, 4, 5}
- **SUPPORTED op-specific axes**: gamma_mode ∈ 6 values (each a distinct compiled program);
  gamma_dtype ∈ {f32, bf16, bf8b, "none"}; gamma_layout ∈ {TILE, ROW_MAJOR, "none"}
- **SUPPORTED memory_layout**: [INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED]
- **EXCLUSIONS**: none
- **Cores**: multi-core, **measured** — `row` split over the full grid with
  `split_work_to_cores(..., row_wise=True)`, plus a cross-core `width` combine (2-level slot
  tree + `mcast_pipe` broadcast) where `row` under-fills the grid and on every WIDTH/BLOCK
  shard. Max measured occupancy **110 = the whole 11x10 grid** over 23 340 cells.
- **Compute config**: exposed; both object types accepted; `math_fidelity` /
  `math_approx_mode` ungated; default HiFi4 / approx / 16-bit DEST
- **Program config**: consumed — both variants validated, `subblock_w` honoured, `inplace`
  returns the input object
- **Golden baseline**: **23 325 `supported_pass`**, 0 `xpass_drift`, 0 `xfail_wrong_mode`,
  0 `xfail_expected`, 10 `supported_fail` (all harness-attributed), over the complete
  121 438-cell collected set
- **Perf baseline**: **16 of 19 perf-group targets already meet their clock-scaled ceiling**
  (measured at 1350 MHz = the reference clock, so scale factor 1.0000)

---

### [x] Refinement 1 — The width-sharded decode combine round

**Type**: perf

**Goal**: the only three perf-group cases that miss their ceiling, and they are one regime —
WIDTH_SHARDED decode at 28–32 cores, where the whole duration is 5–7 µs and a per-round fixed
cost is the entire story:

| ratio | measured | ceiling | cores | case |
|---:|---:|---:|---:|---|
| **1.060** | 5 812 ns | 5 481 ns | 28 | `(1,1,32,7168)` WIDTH_SHARDED `shard=[32,256]` grid `(7,4)`, `gamma`, bf16, HiFi2, `fp32_dest_acc_en=False` |
| **1.050** | 6 882 ns | 6 555 ns | 32 | `(1,1,32,5120)` WIDTH_SHARDED `shard=[32,160]` grid `(8,4)`, **`gamma_bias_residual`**, `fp32_dest_acc_en=True` |
| **1.014** | 5 339 ns | 5 267 ns | 32 | `(1,1,32,5120)` WIDTH_SHARDED `shard=[32,160]` grid `(8,4)`, `gamma`, `fp32_dest_acc_en=False` |

All three are `shard_h = 32` = one tile-row, so `BLOCK_ROWS == 1`: the combine runs its
**identity (non-compact) branch**, one gather + one multicast per tile-row with nothing to
amortize it over. These are also the largest `group_size` values anywhere in the spec, which is
what makes the round cost dominant here and nowhere else. Four levers, all knob-turns on the
already-built combine:

1. **Combine-tree arity, `COMBINE_TREE_F0` (= 4).** At `group_size` 28 and 32 this gives
   `f1 = ceil(G / f0)` = 7 and 8, so the root folds `max(f0, f1)` = 7–8 while each level-1 node
   folds only 4 — an **unbalanced** tree at exactly these sizes. Balanced is `f0 ≈ sqrt(G)` ≈
   5.3 / 5.7. D28 gated the tree on a threshold over *deleted fold tiles* and bracketed that
   threshold; it never swept `f0` itself at `G` in the high 20s/30s. Sweep `f0 ∈ {4, 5, 6, 7, 8}`.
2. **Which NoC carries the gather.** The combine lives entirely on the **writer = NoC1** by
   design, so the reader's NoC0 keeps streaming through pass A. On a decode shape (`Rt = 1`) the
   reader has essentially nothing left to do once pass A is issued, so that trade buys nothing
   here — and `master.md`'s `tensix_all_reduce_ring_transport` (⭐⭐⭐ T3) measures **NoC1 at
   6.07–6.14x slower than NoC0** for forwarding across a rectangular group that spans rows.
   `(7,4)` and `(8,4)` are exactly multi-row rectangles. A/B the gather/multicast direction, or
   the reader/writer split of the combine, on these shapes only.
3. **`GATHER_FACES` (= 2).** Faces shipped per fp32 partial on the `BLOCK_ROWS == 1` branch —
   which is the branch all three cases take. Worth one sweep against `{1, 2, 4}` now that the
   tree changed what the root ingests.
4. **Lamp L-RES-DEPTH**, for the middle case only: `CB_R_DEPTH` is tied to `CB_X_DEPTH`, so two
   double-buffered activation streams cost two extra blocks of L1. Measure `CB_R_DEPTH ∈ {1, 2}`
   at fixed `CB_X_DEPTH = 2` — this is the one of the three that carries a residual, and it is
   also the one at `fp32_dest_acc_en=True`, i.e. the tightest L1.

Relevant catalog patterns: `tensix_all_reduce_ring_transport` (⭐⭐⭐ T3),
`tensix_all_reduce_compute` (⭐⭐ T2), `mcast_topology` (⭐⭐ T2), `noc_placement` (⭐⭐ T2) in
`ttnn/ttnn/operations/examples/master.md`.

No SUPPORTED change.

**Verifier notes**: first because these are the *only* measured misses, and they are cheap —
every lever is an existing constant, no new dataflow and no new CB. Take them in the order
above: (1) is a pure constant sweep, (2) is the one with catalog evidence for a large factor but
touches which kernel owns the combine, (3) and (4) are single constants. **Do not** widen or
re-shape the group: these are WIDTH-sharded inputs, so `group_size` is pinned by the caller's
shard spec, and `WIDTH_SPLIT_MAX_GROUP_CORES` (which governs only the *interleaved* AUTO split)
is not in scope here — the interleaved decode cases already sit at 0.10–0.61 of their ceilings.
Must land before Refinement 2: the finalize's cost is proportional to how many cores wait on it,
and `f0` decides the tree shape that Refinement 2 restructures. Watch precision on every
variant, not just ns: the middle case runs `gamma_bias_residual` at `fp32_dest_acc_en=True` and
its `extras` carry a **soft `pcc_threshold = 0.9995`**.

**Done when**: all three cases measure at or under their clock-scaled ceilings
(`--aiclk` from profiler evidence, not a board nominal), no other perf-group case regresses,
each case's soft `pcc_threshold = 0.9995` still holds, the golden suite is green,
`verify_supported`'s categories are unchanged, and there is no regression on the
config-spanning guard set (one representative per distinct kernel path × layout × placement:
interleaved TILE, interleaved ROW_MAJOR, HEIGHT shard, WIDTH shard, BLOCK shard, RM BAND,
degenerate rank) nor on `test_program_is_structurally_the_seeds`.

**Outcome**: **two of the three land; the perf-group miss count goes 3 → 1.** Harness-native
(`--profile`, AICLK 1350 MHz = the reference clock, scale 1.0000):
`(1,1,32,5120)` `gamma_bias_residual` **6882 → 6420 ns** (ratio 1.050 → **0.979 ✅**);
`(1,1,32,5120)` `gamma` **5339 → 4807 ns** (1.014 → **0.913 ✅**);
`(1,1,32,7168)` `gamma` **5812 → 5766 ns** (1.060 → **1.052**, still a miss).
All four levers landed and none was reverted: (1) `f0` is now DERIVED — the largest divisor of
`GROUP_SIZE` in the measured `[4,10]` band that D28's two gates admit — worth 1.05–1.16x at
`G ∈ {32, 40, 44, 55, 64}` and 32 KB/core of L1 back at `G = 64`; (2) the combine moves to
**NOC_0 (both kernels swapped)** whenever x is a resident shard, gated out on streamed plans
where it measured 0.667x; (3) `GATHER_FACES = 2` re-confirmed, monotone in bytes; (4)
`CB_R_DEPTH` built and parked at its byte-identical default after its premise was falsified on
its own target (that residual is a zero-copy shard alias, so no ring exists to shrink) and
measured null (±0.3%) where a ring does exist. Every interleaved case is 0.999–1.006x — unchanged,
as gated. PCC ≥ 0.999983 everywhere, against the soft 0.9995.

**What still binds the last case, measured**: `(1,1,32,7168)` at 28 cores is **root-serialised,
not knob-limited**. Its permanent per-stage zones put **4130 ns of a 5381 ns kernel (77%) on the
ROOT alone** — `writer_gather_wait` 1040 + `compute_root_fused` 1914 (fold 28 partials, then the
single rsqrt) + `writer_mcast_send` 1176 — with every other core simply waiting. `G = 28` is also
the one group where *no* tree arity pays: forcing the tree on cuts the root chain to 3483 ns but
adds ~1900 ns of level-0 gatherer work in series, so the gate correctly keeps it flat. **What I
would try next, and did not**: (a) **Refinement 2 exactly as filed** — spreading the finalize off
the root is the only lever that attacks `compute_root_fused`, which is the single largest term
here, and it is a scheme-change that this knob-turn phase is explicitly told not to pack in;
(b) eliding the multicast **pre-handshake** when the combine runs exactly one round
(`num_blocks == 1`, which is every `BLOCK_ROWS == 1` decode shape) — `writer_mcast_send` is
866–1176 ns and `mcast_pipe` documents `PRE_HANDSHAKE = false` as a supported fire-and-forget
mode, but it changes a synchronisation contract rather than turning a knob, so it belongs with
Refinement 2's transport work and not here. Not filed as a follow-up: the trailing perf rounds
re-derive the breakdown from scratch, and this record is where the finding lives.

---

### [x] Refinement 2 — Spread the finalize (Lamp L-FIN)

**Type**: perf

**Goal**: on every cross-core combine path the **finalize** — one `rsqrt` per tile-row — is
root-only, and `group_size` cores block on it. It is the single entry in `op_design.md`'s
stall-shadow table marked **not built**, and that table also explains why it cannot be hidden:
nothing is independent of it, because pass B's first operand *is* the finalized stat. The only
remedy is to spread the finalize itself.

D28's tree already spread the **fold** across `f1` level-1 nodes. Each of those nodes could
finalize its own run instead of forwarding a raw sum, turning one core's `BLOCK_ROWS` rsqrts
into `f1` cores' `BLOCK_ROWS / f1` each. This is not a rounding error at these group sizes: D15
established that a per-tile SFPU cost invisible in a tile-op count *dominated* the sharded
geometries, which is why the scoped-rsqrt mitigation exists at all.

Targets, worst-first: the three Refinement-1 cases (28–32 cores — the largest fan-in in the
spec), then the 64-core BLOCK shards `(1,1,8192,1024)` `[1024,128]` on `(8,8)` (ceiling
28 619 ns, currently 24 497) and `(1,1,7168,1024)` `[896,128]` on `(8,8)` with
`gamma_bias_residual` (34 569 ns, currently 34 008 — the tightest margin of any case that
currently passes).

Relevant catalog patterns: `tensix_all_reduce_compute` (⭐⭐ T2), `mcast_topology` (⭐⭐ T2). The
transport half is already `mcast_pipe`.

No SUPPORTED change.

**Verifier notes**: this **stands alone** — it is a scheme-change, not a knob-turn. Moving the
finalize off the root changes which core owns the rsqrt, which changes what the multicast
carries (finalized runs instead of one finalized tile) and therefore the un-permute on the
receive side. One ⭐⭐⭐-class restructure is a whole phase; do not pack knobs into it. Two
invariants the current code depends on and this must not break: `cb_mcast_in` is declared on
**every** core in the multicast box, inactive ones included, so its L1 address is identical
group-wide (the ledger row records that a sharing decision would break that invariant); and
D27's compact transpose caps a combine `BLOCK_ROWS` at `TILE_DIM = 32`, because one tile-row's
stat becomes one *column* of one tile — a 101-row block measured pcc 0.949, silently. Note the
Refinement-1 targets are all `BLOCK_ROWS == 1` (identity branch) while the 64-core BLOCK shards
are compact, so both branches need the change or one of them needs an explicit predicate. If
the level-1 finalize proves not to pay, ship `[~]` **with the measurement** — a recorded
negative result here is worth more than a retry, because this lamp has been open since the seed.

**Done when**: measured device-ns improves on the combine-path cases (with the two 64-core
BLOCK shards not regressed — `(1,1,7168,1024)` has only 1.6% of margin), the golden suite is
green, `verify_supported`'s categories are unchanged, and no regression across the
config-spanning guard set or `test_program_is_structurally_the_seeds`.

**Outcome**: **the named scheme change was built, measured, and LOSES; the phase's win came
from the same round's transport.** All numbers blackhole p150b 1350 MHz, in-process
profiler, median of 5, min over 3 reps, noise floor ±0.3% (calibrated on two cells whose
program is byte-identical across the sweep).

*What I measured.* `COMBINE_FIN_SPREAD` — the last-level fold forwards the RAW group sum,
the multicast carries that, and every core finalizes its own copy in parallel, covering the
identity and compact branches with one predicate — is **correct** (pcc / rel-RMS
bit-identical to the root finalize everywhere) and measures **0.953–1.004x**: a loss at
every combine geometry. It is kept as a live knob at its byte-identical default, not
deleted. The transport half — the pre-handshake elided on a single-round combine, plus a
3 kB identity-path multicast payload — measures **1.035x on `(1,1,32,7168)` W28**
(5713 → 5520 ns, ratio-to-ceiling **1.042 → 1.007**), 1.015x at `(1,1,32,2304)`, 1.013x on
the 64-core ROW_MAJOR BAND, 1.010x at `(1,1,32,5120)`, and 1.004–1.007x on four more, with
no cell below the noise floor. Both 64-core BLOCK shards are multi-round, so the gate keeps
them at the seed's program: 0.997 / 0.999, i.e. flat.

*What the bottleneck actually is now.* Not the finalize, and this phase is the measurement
that says so. The finalize is a **replicated** term, not a divisible one — every core needs
the same value, so relocating the rsqrt moves it along the identical serial chain rather
than dividing it — and two earlier decisions had already banked what there was: D22 fused
the root's rsqrt into the fold's DEST window (no pack at all), and D27 collapsed it from
`BLOCK_ROWS` tile-ops to **one per round**, which is the O(`BLOCK_ROWS`) cost the lamp was
written against. Lamp L-FIN is therefore **closed**, and `op_design.md`'s stall-shadow table
and lamp list now say so with the number. What binds `(1,1,32,7168)` at 28 cores is still
the **root's serial chain** — `writer_gather_wait` + the 28-tile fold + the broadcast — with
27 cores idle behind it; that is a fan-in, not a finalize.

*What I would try next, and why not here.* (a) **Fold in the reduce datapath instead of
pairwise `add_tiles`.** At `BLOCK_ROWS == 1` each sender's 4 kB page carries 32 useful
floats in column 0; if senders wrote into distinct *rows* of one landing tile (a transpose
of D27's compact permute, expressible as `transpose_wh` of the tile `member_pack` already
builds), a group of ≤ 32 would fold in **one `reduce_tile`** instead of `G/2` `add_tiles`,
and the gather's bytes would drop by more than an order of magnitude. That is a
gather-representation change, i.e. a different lamp from this heading's, and it is
⭐⭐⭐-class. (b) **All-reduce the last tree level** — the `f1` level-0 gatherers broadcast
their run sums and every core folds and finalizes locally, deleting the level-1 hop, the
root's l1 gather wait and the single-root broadcast. It needs `f1` concurrent senders on one
receiver rectangle, which `Mcast1D`/`Mcast2D` express only as a *rotating* (per-round)
sender, so it is a mcast-wire change, not a knob. Neither is filed as a follow-up: the
trailing perf rounds re-derive the breakdown from scratch and this record is where the
finding lives.

---

### [x] Refinement 3 — Strip the per-block fixed costs off the interleaved prefill

**Type**: perf

**Goal**: the interleaved prefill cases pass their references but are **not** at the DRAM
roofline, and the gap is compute-side overhead rather than byte count (the byte count is already
minimal — ROW_RESIDENT holds everything and re-reads nothing):

| case | measured | bytes moved | achieved | roofline |
|---|---:|---:|---:|---:|
| `(1,1,8192,1024)` INTERLEAVED | 89 992 ns | ~34 MB | **378 GB/s** | ~450 GB/s |
| `(1,1,8192,7168)` INTERLEAVED | 589 591 ns | ~234 MB | **397 GB/s** | ~450 GB/s |

That is 13–19% of headroom. Re-rank first and take the worst, but as of Phase 0 the narrow
prefill `(1,1,8192,1024)` is the most under-saturated. Four levers, all ⭐⭐ T2, all on the same
code path:

1. **Data-format reconfig elision — take this one first.** Every chain element in the compute
   kernel is spelled `DataFormatReconfig::Enabled` and the reduce runs
   `ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT`. `compute_block_size`'s *second lever* in
   `master.md` measures dropping the reconfig where the format never changes at up to **1.19x**,
   "largest where there are the most transitions" — and this kernel has the most transitions of
   any op in the catalog (tilize → add → square → reduce → finalize → normalize → scale → bias →
   untilize). The descriptor already knows every CB's `data_format`, so whether a given boundary
   is format-constant is a **compile-time fact it can compute and pass down**, not a guess.
   It is emphatically *not* uniformly constant — `cb_scaler` is always bfloat16,
   `cb_row_stat` / `cb_sum_handoff` / `cb_row_final` are always float32, and the two per-channel
   operands carry independent dtypes — so this is one predicate per boundary derived from the
   one format table, never a blanket switch.
2. **Lamp L-RES-FUSE.** `residual_add_block` materializes `cb_x_sum` in **both** passes, but
   pass A only squares `t` — it does not need `t` to survive. A fused
   `BinaryFpu<Add> → Square → PackTile` chain drops one pack and one unpack per tile in pass A at
   **no L1 cost** (`cb_x_sum` is allocated for pass B regardless). Measure on `(1,1,8192,5120)`
   STREAM with `gamma_bias_residual`.
3. **Reader/writer transaction granularity.** `op_design.md`'s block-schedule table states the
   intent as "one NoC barrier per (block, chunk) per stream" on the TILE path; the shipped
   reader issues one barrier per **tile-row** of the chunk (`WT_CHUNK` tiles), and the writer
   mirrors it. `double_buffer` in `master.md` is the pattern. The ring is already
   `DX * BLOCK_ROWS * WT_CHUNK` pages, so a `BLOCK_ROWS * WT_CHUNK` multi-page reserve is legal
   and cannot straddle. This is a genuine trade, not a free win — coarsening the handoff costs
   reader↔compute overlap *inside* a block — which is exactly why it belongs here as a
   measurement. Sweep the unit over `{WT_CHUNK, 2*WT_CHUNK, BLOCK_ROWS*WT_CHUNK}`.
4. **Lamp L-OPERAND-TRIM.** `BIAS_TRIM` copies `GAMMA_TRIM`'s policy. D23's face-row trim was
   measured for a *multiplier*; two trimmed reads per chunk instead of one changes the
   transaction count, which D13 found is what actually mattered. Measure
   `BIAS_TRIM ∈ {0, GAMMA_TRIM}` on the prefill `gamma_bias` cases.

Fold in here, because this phase already re-measures the blocking solve on exactly these
shapes: **`_cb_block_mult` over-prices `cb_x_squared`** at the full chunk width even when D12's
DEST square fold makes that CB `BLOCK_ROWS × 1`. The error is conservative (it can only shrink
`BLOCK_ROWS`, never overflow L1) and only bites at `WT_CHUNK <= 8` where L1 is not the binding
constraint — which is why it is not a fix on its own — but if this phase touches the solve,
correct it and re-measure.

No SUPPORTED change.

**Verifier notes**: third because it is the widest-blast-radius phase — levers 1 and 3 touch
code that **every** configuration runs, including the operand-free path the seed-parity test
pins, so the prompt's rule binds: an operand-free build may only get faster with a measurement
attached. Levers 2 and 4 touch only `HAS_RESIDUAL` / `HAS_BIAS` code and cannot move that test.
Lever 1 has a **correctness edge**: eliding a reconfig is only safe when the two CBs' formats
are *provably* equal for that build, so derive the predicate from the descriptor's own
`data_format` values and `static_assert` it in the kernel — never from "these are usually both
bf16". Lever 3 must be swept *after* Refinement 1, not with it: a coarser transaction unit and a
deeper ring are two halves of the same overlap budget and sweeping both at once confounds them.
Take lever 1 first and report its number even if the phase stops there.

**Done when**: measured device-ns improves on the interleaved prefill cases (moving them toward
the ~450 GB/s roofline), no perf-group case regresses, the golden suite is green,
`verify_supported`'s categories are unchanged, and no regression across the config-spanning
guard set or `test_program_is_structurally_the_seeds` — or, where that test does move, a
recorded measurement justifying it.

**Outcome**: **all four named levers built and measured; three are nulls parked at
byte-identical defaults, the fourth (the trim) is a re-measurement that CONFIRMS the shipped
policy, and the phase's win came from a fifth thing this heading's own title names.**
1.005–1.026x on the interleaved prefill with nothing regressed. All numbers blackhole p150b
1350 MHz, in-process profiler, min over 3 reps of median-of-5.

*The Goal's premise needed correcting first.* The "~450 GB/s roofline" is not reachable on
this box for interleaved DRAM read+write. Measured against the machine's own ceiling — the
same tensors through the cheapest possible kernels — `(1,1,8192,1024)` gives `ttnn.clone`
83 833 ns (400 GB/s) and `ttnn.exp` 88 787 (378) against this op's 88 128 (381); at
`(1,1,8192,7168)` it is `clone` 590 613 (398) and `exp` 624 674 (376) against this op's
576 680 (**407**). A full RMS norm with a weight already **beat a pure DRAM→DRAM copy** at
the wide shape, and the operand-free build matched `clone` to 0.9%. **The interleaved
prefill is DRAM-saturated, not overhead-bound** — so the 13–19% "headroom" the table
reported is mostly the gap between the quoted roofline and the achievable one.

*What I measured.* **Lever 1** is a null and the ablation proves it rather than argues it:
`RMS_ABLATE_RECONFIG` strips EVERY data-format reconfig (an incorrect build, pcc ≈ 0) and
moves the clock 0.989–1.008x over twelve cases — because `eltwise_chain`'s reconfig fold is
**boot-hoisted**, so this kernel pays `stages × num_blocks × NUM_W_CHUNKS` reconfigs per
core (five, on the target shape), not `stages × tiles`; `master.md`'s 1.19x is a
per-tile-reconfig number and does not transfer. **Lever 2**'s four-element chain is
structurally impossible, not merely slow: pack is its own cohort, so an intermediate DEST
value cannot be published and cb_x_sum receives the SQUARE — pcc **0.260**, measured. Its
correct STREAM-only form (`Add → Square → Pack`) is 0.989x; parked live. **Lever 3** moved
both NoC halves together (grouped reserve/barrier/push in the reader, the symmetric
wait/barrier/pop in the writer) and measured flat-to-slightly-negative across the queue's
`{WT_CHUNK, 2·WT_CHUNK, BLOCK_ROWS·WT_CHUNK}` sweep; parked at 1 with `TXN_ROWS | BLOCK_ROWS`
asserted in both kernels as the straddle-free invariant. **Lever 4** refutes its own lamp
with a number: coarser per-channel reads are **0.76–1.00x**, so D23's two-face-row policy
wins and the bias copying it is right. The folded-in `_cb_block_mult` correction is real
(BLOCK_ROWS 20 → 25 on the 64-core BLOCK shard) and measures **0.987x** there, so it ships
parked at the conservative price. **The win**: pass A's `square` was the one chain still at
DEST `block_size 1` while every pass-B chain had taken `PASS_B_BLK` since D21 — giving it the
same (derived, never duplicated) block plus the `PerBlockSize` pack lifecycle it requires is
**1.026x** on `(1,1,8192,2048)`, **1.017x** on the operand-free `(1,1,8192,1024)`, 1.014x with
a bias, 1.013x on the width-split `(1,1,32,7168)`, 1.009x on `(1,1,8192,1024)` `gamma`, and
0.996–1.005x everywhere else. Golden `test_op_loose` 433/443 (the identical prior figure, all
10 harness-attributed) plus a 2 196-cell cartesian slice; unit directory 601 passed.

*What the bottleneck actually is now, and what I would try next.* DRAM bandwidth, at
~400 GB/s for interleaved read+write on this box — the prefill now runs at 384 GB/s at
W=1024 and **409 GB/s at W=7168, 3% faster than `ttnn.clone` of the same bytes**. The only
non-DRAM residue is the per-channel operand: `no_gamma` is 83 087 ns against `gamma`'s
87 372, and a per-stage capture shows all 110 cores issuing their gamma reads at t=0 against
the same few DRAM pages (`reader_read_gamma` spreads 3 800 → 72 000 cycles across cores).
Deleting that would mean **multicasting the per-channel operands** from one reader — one
core reads the row, a `Mcast2D` over the grid distributes it — which is a transport
scheme-change, not a knob, and the byte argument for it was already (correctly) rejected;
the *contention* argument is new and is the reason to revisit it. Not filed as a follow-up:
the trailing perf rounds re-derive the breakdown from scratch and this record is where the
finding lives.

---

### [x] Refinement 4 — Remove the prime-`Wt` granularity cliff (ragged width chunk)

**Type**: perf

**Goal**: `WT_CHUNK` is constrained to a **divisor** of the per-core width (D1), so a prime or
near-prime `Wt` collapses the width chunk to **one tile** — below `master.md`'s granularity
floor (whole tiles at minimum; coarser amortizes), repaying the whole per-phase init / reconfig
/ pipeline fill-and-drain `Wt` times per block instead of `Wt / WT_CHUNK` times.
`op_design.md` carries this as the deferred **RAGGED WIDTH CHUNK** regime, and its reachability
argument is honest about the scope: no `INPUTS` shape and no perf case sits on the cliff; the
resilience group reaches it at `(1,1,32,4064)` and `(1,1,3104,4064)` — `Wt = 127`, prime.

Those two resilience shapes are the target region. They **pass** today, so this is a perf
refinement and not a failure-category move: the ask is a measured device-ns win on prime-`Wt`
shapes with no regression anywhere else. The three mechanisms that force the divisor are all
named in the design's `Mechanism caps` table and all helper-side: `compute_kernel_lib::tilize` /
`untilize` take `block_width_tiles` as a **compile-time** template parameter; `reduce()`'s
`BulkWaitBulkPop` asserts `num_pages(cb_in) % cols == 0`; and a multi-page `cb_reserve_back` +
`get_write_ptr` batch must not straddle the CB ring. A ragged tail needs an answer to each.

Relevant catalog patterns: `compute_block_size` (⭐⭐ T2), `reduce_block` (⭐⭐ T2).

No SUPPORTED change.

**Verifier notes**: last because it serves the narrowest region and because it is the only phase
whose work is mostly *outside* this op — it changes how the op calls three helpers, and possibly
the helpers themselves. Check first whether a **tail instantiation** (`WT_CHUNK` coarse for
`floor(Wt / WT_CHUNK)` chunks plus one compile-time tail of `Wt % WT_CHUNK`) buys the win with
no helper change at all; that is the cheap version and it keeps every existing build
byte-identical. The expensive version (a runtime `cols`) is only worth it if the tail
instantiation's second template blow-up is what costs. **Do not** reach for this to widen the
interleaved width split: `_width_group_cores` is a divisor for a *different and unrelated*
reason — an interleaved core has no pad storage, so a ragged group member would read x tiles it
does not own — and conflating the two is how a correct `gw` constraint gets relaxed by accident.

**Done when**: measured device-ns improves on `(1,1,32,4064)` and `(1,1,3104,4064)` (both
layouts, since the resilience group sweeps placement × layout), those cells stay green, no
perf-group case regresses, `verify_supported`'s categories are unchanged, and no regression
across the config-spanning guard set or `test_program_is_structurally_the_seeds`.

**Outcome**: **the cheap version the verifier notes told me to check first was not merely
cheap enough — it was cheaper than a tail instantiation, because it needs no second
instantiation at all.** `1.34x–10.46x` on the target region, and the compute kernel is
byte-for-byte unchanged.

*What shipped.* Not the deferred regime's "runtime `wt_c`", and not the notes' compile-time
tail either. `_width_chunk` takes the coarsest **balanced** chunk
`ceil(wt_c / ceil(wt_c / cap))` and **pads** the last chunk out to it, so every chunk stays
**uniform** — which satisfies all three `Mechanism caps` rows verbatim (`tilize`/`untilize`
keep their compile-time `block_width_tiles`, the reduce's `num_pages % cols == 0` still
holds, every ring is still a multiple of its push unit) and leaves nothing for a second
template to do. `Wt = 127` at a cap of 32 becomes 4 chunks of 32 with **one** pad tile; the
balanced form bounds the pad at `< NUM_W_CHUNKS`, i.e. under `1/cap` of the width. The pad
tiles are the ragged-**shard** pad this op already carried, moved one axis over: the reader
zeroes them with the device zero API (`publish_native_shard`'s mechanism, and its reason —
a pad tile that is not *exactly* zero inflates `sum(t²)`), the writer skips them with the
`wt < WT` predicate it already had. Only the ROW_MAJOR tail needed new code, and only
because `read_sticks_for_tilize` / `write_sticks_after_untilize` derive the L1 **stride**
from `row_bytes`; that helper gap is recorded at both call sites.

*Measured* (blackhole p150b 1350 MHz, `RAGGED_WIDTH_CHUNK` 0 vs 1, min over 2 reps of
median-of-5). `(1,1,32,4064)` RM `gamma_bias_residual` 590 592 → 56 486 (**10.46x**), RM
`gamma` 376 319 → 37 950 (9.92x), TILE `no_gamma` 101 279 → 16 616 (6.10x), TILE `gamma`
148 493 → 33 278 (4.46x). `(1,1,3104,4064)` RM `gamma_bias_residual` 1 991 000 → 216 590
(9.19x), RM `gamma` 1 145 067 → 138 309 (8.28x), TILE `gamma` 213 058 → 150 576 (1.42x),
TILE `gamma_bias_residual` 329 830 → 246 944 (1.34x). `(1,1,3104,2848)` (`Wt = 89`, the
other prime) RM `gamma` 805 205 → 98 090 (8.21x). And a case that was **not** a target:
`(1,1,1024,16384)` STREAM `gamma_bias_residual` 525 549 → 498 670 (1.05x) — `Wt = 512` is
not prime, but its coarsest *divisor* under the cap (32) is not its coarsest *fitting*
chunk (57), so this pays wherever the two differ. pcc **improves** on every cell that
coarsened (0.99987 → 0.99999 at `Wt = 127`): a coarse chunk clears D7/D8's reduce-datapath
floors. Fourteen guards spanning the interleaved prefill, STREAM, the width-split combine,
the 64-core BLOCK shard and the RM BAND: 0.99–1.05x, nothing outside noise (`WT_PAD == 0`
there ⇒ a byte-identical program). `test_op_loose` 433/443, the identical prior figure with
the identical 10 harness-attributed failures; `resilience`+`perf`+`pad_poison` 384/384; a
2 700-cell strict cartesian slice green; unit directory 1 061 passed.

*What the bottleneck is now, and what I would try next.* The TILE cells are the ones that
moved least (1.34–1.42x) and they are the ones nearest their DRAM roofline:
`(1,1,3104,4064)` moves 50.4 MB and now runs at 335 GB/s against the ~400 GB/s Refinement 3
measured as this box's achievable interleaved read+write ceiling, so there is roughly 1.2x
left there and it is *bandwidth*, not granularity. The RM cells are a different story and
the number to look at is `(1,1,32,4064)` RM at 37 950 ns for 0.52 MB — **13 GB/s**, three
orders off the roofline, because `_width_group_cores` is a **divisor** and a prime `Wt`
therefore runs the whole tensor on **ONE core**. That is the second prime-`Wt` cliff and it
is a much bigger prize than this one was; the verifier notes explicitly and correctly
forbade touching it here (an interleaved width-split member has no pad storage, so a ragged
group member would read x tiles it does not own — a different constraint with a different
answer, probably a per-member `w_real` plus the same zero-fill this refinement just built).
Not filed as a follow-up: the trailing perf rounds re-derive the breakdown from scratch and
this record is where the finding lives.

---

## Not in this queue, and why

| Candidate | Disposition |
|---|---|
| **`GAMMA_MCAST`** — read the per-channel vectors once on an injector and multicast (`shared_input_reuse`, ⭐⭐⭐ T3) | **Not filed.** The design defers it with a positive reason, and the reason is *stronger* than stated. `op_design.md` and `l1_ledger.md` price the reuse-shared traffic at whole tiles (118 MB of gamma on `(1,1,8192,7168)`, 50 MB after ROW_RESIDENT), which ignores D23's trim: a TILE per-channel operand is read as **two face-rows** — `2·TILE_DIM·elem_bytes` = 128 B of a 2048 B tile — so the real figure is ~1/16 of the ledger's, i.e. gamma is ~1% of DRAM bytes, not 18%. A multicast that removes 1% cannot pay for an injector. The ledger's traffic rows are corrected; see `verification_report.md`. |
| **Lamp L-SUBBLOCK** — teach the default blocking about the `subblock_w` × `inplace` interaction | **Not filed — already surpassed.** `feature_spec.py`'s own note calls 25 513 ns reachable on `(1,1,8192,1024)` BLOCK_SHARDED with `subblock_w = block_w` + `inplace`, against 28 619 at `subblock_w = 1`. The op measures **24 497 ns** at its own default blocking, below the figure the lamp treats as the prize. A caller's explicit `subblock_w` is still honoured exactly as the contract requires. |
| **Lamp L-BIAS-INPLACE** — a dedicated `cb_scaled` instead of transforming `cb_normalized` in place | **Not filed.** No perf-group case with a bias misses its ceiling (`(1,1,8192,5120)` gamma_bias_residual sits at 0.527, `(1,1,8192,7168)` at 0.886), so there is no region to point at. Note if it is ever revisited: the implementer's own breadcrumb records that getting the in-place lifecycle pair right and the ring size wrong gives **silently wrong values on the partial final block only** — invisible whenever the block count divides evenly — so any variant must be checked on a row count that does *not* divide `BLOCK_ROWS`. |
| **Lamp L-OVERLAP** — coarsest block vs. one step back with a deeper buffer, at `Rt = 1` | **Not filed as its own phase.** The interleaved decode cases it targets sit at 0.10–0.61 of their ceilings, with the `>=7x` case at 0.612. The one decode regime that *does* miss is the WIDTH-sharded one, where the shard pins the block — so the live part of this lamp is `CB_R_DEPTH`, folded into Refinement 1 lever 4. |
| **`ReduceFp32Mode::Accurate`** for float32 activations | **Not filed — measured and rejected.** At `fp32_dest_acc_en=False` (the op's own default) the SFPU reduce path returns **inf/NaN**; at `True`, where it works, it is identical to `Fast` to five significant figures. Recorded with the A/B table at `D3` in the descriptor's design notes. |
| **`use_welford` single-pass statistics** | The contract requires `use_welford` to raise. Rejected in the design, correctly. |
| **3+ level gather tree** | Measured a loss at 6 of 7 cells in the isolated bench; superseded by the shipped 2-level tree. |
| **Fusing the residual add into pass B's normalize chain** | Inexpressible on the current helper surface: a second `BinaryFpu` reads CBs, not DEST, and `DestReuseBinary` carries no broadcast parameter, so `(x+r) * stat<Col>` cannot share one DEST window. |
| **The five `feature_spec.INVALID` "author-scoped exclusions"** | Not a refinement — `SUPPORTED` already claims those axis values and `validate()` accepts them, so there is nothing for an implementer to add. An INVALID-audit finding and a requested `feature_spec.py` edit; `test_rms_norm_ttnn_invalid_audit.py` proves all five pass. |
| **The 19 red golden cells** | 13 are two harness defects and 6 are a harness scoring gap; none is op-attributed. Three one-line fixes requested in `verification_report.md`. |
| Code-review items, ledger corrections, `math_approx_mode` coverage gap | `verification_report.md`. |


### [ ] Refinement 4b — Remove the prime-`Wt` granularity cliff (ragged width chunk) (debug: fix gate violations)

**Goal**: fix the hard violation from Refinement 4 so the completion gate's three bullets hold.

**Verifier notes** (mechanical, from the harness completion gate):

```
Bullet 3 FAIL: REGRESSION — prior-passing golden cells no longer pass (responsible cells 0/0). A prior-passing cell that failed, hung, or never ran (suite hung before reaching it) is a regression.
```

**Done when**: the gate passes — zero hangs in SUPPORTED, acceptance + refinement tests pass, golden majority with no regression.
