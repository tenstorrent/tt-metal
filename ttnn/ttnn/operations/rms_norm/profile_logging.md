# rms_norm — per-stage profiling & optimization log

Companion to `changelog.md`. Where the changelog records *what shipped*, this file records
the **per-stage device-zone profiling** that motivated each perf change and the **measured
benefit** of each optimization, kept one-by-one so a regression can be bisected.

## Measurement method (DeviceZoneScope)

`DeviceZoneScopedN("name")` (`tt_metal/tools/profiler/kernel_profiler.hpp`) emits a
`ZONE_START`/`ZONE_END` core-cycle timestamp pair per RISC into the device profiler L1 buffer,
dumped to `generated/profiler/.logs/profile_log_device.csv` on `ttnn.ReadDeviceProfiler`. The
macro is a no-op (`(void(name))`) unless the kernel is JIT-built with the profiler env set, so
the zones are **zero-cost in production / golden runs**.

Driver: `.eval/refinements/zone_profile_driver.py`, run with
```
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
  scripts/tt-probe.sh rms_norm < .eval/refinements/zone_profile_driver.py
```
8x8 Wormhole b0, 1000 MHz (1 cycle = 1 ns), bf16, no-gamma, TILE. Shapes are sized to 1 tile-row
per active core (zones fire once/core). **Compute zones reported on the MATH thread (TRISC_1)**
— pack/unpack idle-spin with the next zone open and inflate a cross-TRISC span. Reader/writer
zones are single-RISC = clean wall time.

**Critical caveat:** reader/compute/writer run concurrently (pipelined through CBs), so zones
OVERLAP and do NOT sum to the total. The total is gated by the slowest dependency-chain path.

## Baseline (pre-optimization, per-core mean ns)

### Regime A (row-parallel)
| shape | Wt | RDR-noc | CMP-p1-square | p1-reduce | finalize | pass2 | WR-wait | WR-noc | total |
|-------|----|---------|---------------|-----------|----------|-------|---------|--------|-------|
| (2048,256) | 8  | 4594 | 6121  | 489  | 2434 | 3244 | 9291  | 6047  | 18882 |
| (2048,512) | 16 | 9294 | 11596 | 1032 | 2429 | 9917 | 15645 | 11444 | 32145 |

### Regime B (wide-W cross-core all-reduce)
| shape | cores | RDR-noc | p1-square | ar-wait | ar-xport | combine* | finalize | pass2 | WR-wait | WR-noc | total |
|-------|-------|---------|-----------|---------|----------|----------|----------|-------|---------|--------|-------|
| (32,4096)  | 16 | 1515 | 2991 | 1837 | 3124 | 4634 | 2429 | 2674 | 10930 | 4618 | 17466 |
| (32,8192)  | 32 | 2474 | 3948 | 1821 | 5698 | 8277 | 2436 | 3117 | 15516 | 5238 | 23041 |
| (32,16384) | 32 | 4747 | 6989 | 3110 | 6093 | 8654 | 2468 | 8864 | 19840 | 9697 | 32745 |

\* `CMP-combine` is mostly the MATH thread stalled on `cb_partials_gathered` (the all-reduce barrier).

### Baseline interpretation
- **Not write-bandwidth bound.** `WR-wait` (idle, waiting on compute) > `WR-noc` (actual write)
  everywhere → the writer is starved, not saturated.
- **Latency-bound serial chain** for 1-row/core: `read → square → reduce → finalize → [pass2 ∥ write]`.
- `WR-noc` is **per-tile-barrier latency** (~0.75 ns/byte-ish, ~0.75µs/tile): the TILE writer does
  `noc_async_write_tile` + `noc_async_write_barrier` PER tile (the reader already batches). → Target 1.
- read and square fully serialize (reader pushes the whole row in one `cb_push_back`). → Target 2.
- `finalize` is a flat ~2.4µs for a single tile of rsqrt — almost all init/round-trip, not math. → Target 3.

---

## Optimizations (one at a time)

<!-- each target appended below as it lands: change, helper-mode note, measured before/after, keep/revert -->

### Target 1 — Batch the TILE writer (one `noc_async_write_barrier` per block) — **KEPT**

**Change.** The TILE writer did `noc_async_write_tile` + `noc_async_write_barrier` *per tile*,
serializing the NoC write latencies (~0.75µs/tile). Rewrote it to drain a block of `reduce_block`
tiles, issuing all the async writes then **one** barrier (`rms_norm_writer.cpp`). `cb_output`
grown `2 → 2*reduce_block` in the two TILE descriptors so compute fills block N+1 while the writer
drains block N; write-block passed as writer CT idx 3. RM writer (sticks) was already chunk-batched
— unchanged. (This mirrors the pattern the *reader* already used: issue all reads, one barrier.)

**Measured (per-core mean ns, before → after):**

| shape | WR-noc | CMP-pass2† | total | speedup |
|-------|--------|-----------|-------|---------|
| (2048,256)  A | 6047 → 3973 | 3244 → 771 | 18882 → **17085** | 1.11x |
| (2048,512)  A | 11444 → 9815 | 9917 → 6590 | 32145 → **31301** | 1.03x |
| (32,4096)   B | 4618 → 2285 | 2674 → 756 | 17466 → **14808** | 1.18x |
| (32,8192)   B | 5238 → 3201 | 3117 → 760 | 23041 → **20699** | 1.11x |
| (32,16384)  B | 9697 → 5939 | 8864 → 4501 | 32745 → **28652** | 1.14x |

† **Bonus effect:** `CMP-pass2` also dropped sharply. The old 2-tile `cb_output` back-pressured
pass2's pack (it stalled waiting for the writer to free a slot); the larger CB removes that stall,
so the pass2 *math* thread now shows its true ~0.77µs busy time. So Target 1 fixed two things: the
per-tile write barrier AND an output-CB back-pressure stall.

**Correctness.** `test_rms_norm.py` 20/20 (Regime A + B, ±gamma). No hang.

**Helper note.** None needed — the writer is a hand-written dataflow kernel using the raw
`noc_async_write_*` API, not a `ckl` helper, so this was a pure dataflow restructuring (kept the
helper-free path because it *is* the path). Observation: there is no dataflow-lib "blocked tile
writer" helper; the issue-N-then-one-barrier idiom is re-implemented per op (reader did it inline
too). A `dataflow_kernel_lib::write_tile_block(cb, accessor, page_base, n)` helper would let ops
get this batching for free and is worth adding to the library.

**Verdict: KEEP.** 1.03–1.18x, larger on Regime B (more output tiles per core relative to compute),
correct, low blast radius.

### Target 2 — Overlap reader ↔ PASS-1 square (chunked input push) — **REVERTED (no win)**

**Change (tried).** Reader pushes the resident input per `reduce_block` chunk (not all `Wt` at
once); square's input lifecycle `HeldBulk → HeldCumulative` so it consumes tiles incrementally as
they stream in while still not popping (resident for PASS-2). Goal: overlap read ∥ square instead
of read-then-square. Keeps R5's single-square/single-reduce (only the wait granularity changed).

**Helper note (the important finding).** The helper *already* has the right lifecycle —
`InputLifecycle::HeldCumulative = {Cumulative wait, no pop}` — so **no new mode was needed** for the
streaming. BUT the helper **forbids `TileOffset::Set` + `Cumulative`**
(`eltwise_chain.inl:705`: "OffsetA Set requires Bulk-family or CallerManaged" — iter-dependent wait
counts can't compose with a runtime base offset). The square's base offset is 0, so switching to
`TileOffset::Unset` (front-relative walk) is equivalent and *is* legal with `HeldCumulative`. So it
was expressible with the helper after that one-line adjustment — no helper-free fallback required.

**Measured (T1 → T1+T2, total ns):** (2048,256) 17085→17516 (**+2.5%**), (2048,512) 31301→32226
(**+3%**), (32,4096) 14808→14878 (flat), (32,8192) 20699→21067 (**+1.8%**), (32,16384) 28652→28064
(−2%). Net **neutral-to-negative**; a win on only the widest shape.

**Why it didn't pay off.** The writer is the critical path (`WR-wait` ≫ all). Overlapping read ∥
square shifts when PASS-2 *starts* but does not move the writer-gated total; meanwhile the per-chunk
reader push adds `num_chunks` read barriers (RDR-resv 38→84ns, extra `RDR-noc` barriers) that cost
more than the overlap saves. (A real micro-effect *was* visible — Regime B `RDR-ar-wait` dropped
~1837→1437ns because compute's local Σx² is ready sooner — but it's off the critical path.)

**Verdict: REVERT.** Kept only wins per the brief; this is a wash with two ~3% regressions. The
read→square serialization is real but irrelevant while the writer dominates — would only matter
after the writer stall is removed (i.e. a much larger restructuring).

### Target 3 — Fuse FINALIZE rsqrt into the PASS-1 reduce epilogue — **REVERTED (small regression)**

**Change (tried).** Regime A: fold `rsqrt(Σx²·inv_W + eps)` into the PASS-1 `reduce` via a
`PostReduceOp` lambda (`mul_unary_tile`→`add_unary_tile`→`rsqrt_tile` in DST before pack), writing
recip straight to `cb_recip_rms` and dropping the standalone finalize `eltwise_chain` + its 1-tile
L1 round-trip. Regime B unchanged (must rsqrt the GLOBAL Σx² after the cross-core combine).

**Helper note.** **No new mode needed** — `ckl::reduce` already takes a `PostReduceOp` callable
`(uint32_t dst_idx)` applied in DST (`reduce_helpers_compute.hpp:411`). The fusion was fully
expressible with the helper; the only "raw" part is the lambda body calling the SFPU tile ops
directly, which is exactly the documented `PostReduceOp` contract (see the
`reduction/generic/.../reduce.cpp` `REDUCE_POST_MUL` example), not a helper bypass.

**Measured.** Regime B path is unchanged code but drifted **+1.7–3.2%** across this session →
that is the run-to-run noise floor. Against it, Regime A: (2048,256) 17085→18016 (+5.4%),
(2048,512) 31301→32715 (+4.5%) — a real **regression beyond noise**. No shape improved.

**Why it didn't pay off (the real finding).** The 2.4µs `finalize` was assumed to be an L1
round-trip; the split says otherwise — it's **SFPU init (`binop_with_scalar_tile_init` +
`rsqrt_tile_init`) + the rsqrt compute**. Fusing into the reduce epilogue STILL pays both inits,
but now serialized on the reduce's MATH thread right before pack, losing the separate chain's
unpack/math/pack pipelining across the 3 TRISCs. So it removed the (cheap) round-trip and lost the
(valuable) pipelining → net negative.

**Verdict: REVERT.**

---

## Summary

| Target | Idea | Helper mode needed? | Result | Kept? |
|--------|------|--------------------|--------|-------|
| 1 | Batch TILE writer (1 barrier/block) + bigger `cb_output` | No (raw dataflow); a `write_tile_block` lib helper *would* be nice | **1.03–1.18x**, larger on Regime B; bonus pass2 unstall | **KEEP** |
| 2 | Overlap reader ↔ PASS-1 square (HeldCumulative + chunked push) | No — `HeldCumulative` exists; needed `TileOffset::Unset` (Set+Cumulative is rejected) | neutral / +3% regression on 2 shapes | revert |
| 3 | Fuse finalize rsqrt into reduce epilogue | No — `reduce` has `PostReduceOp` | small regression (loses 3-TRISC pipelining) | revert |

**Net kept: Target 1** (the only measured win). **Overarching lesson:** after the per-tile write
barrier was fixed, this op is gated by the **writer drain + an unavoidable serial prefix**
(read→square→reduce→finalize all process the whole row before PASS-2 can emit a tile). Targets 2
and 3 both attacked that prefix and both were neutral-to-negative because the prefix is masked by
the writer-bound critical path. The next real lever would have to **shorten the prefix itself or
break the write serialization** (e.g. split-DST so PASS-2 can stream output before the full row is
reduced) — a structural rewrite, not a local tweak.

**Helper-library asks surfaced:** (1) a `dataflow_kernel_lib::write_tile_block(...)` batched-writer
helper (Target 1 re-implements it inline; the reader does too); (2) consider allowing
`TileOffset::Set` base 0 with `Cumulative` wait (Target 2 had to use `Unset`). Neither blocked the
work — both optimizations were expressible — but they'd reduce per-op boilerplate.

---

## Round 2 — non-latency-bound shapes + a different all-reduce algorithm

### Profiling non-latency-bound (multi-row / grid-saturated) shapes — findings
The Round-1 shapes were all 1 tile-row/core (worst case for hiding latency). Re-profiled
many-rows-per-core and grid-saturated shapes:

- **Regime A, many rows/core:** `RDR-resv` (reader blocked on `cb_reserve_back`) blows up —
  4.6µs @2 rows/core, 15.5µs @4, 69µs @8 (`(16384,512)`). The resident input CB is single-buffered
  (`Wt`), so the reader cannot prefetch row N+1 until compute frees row N → rows serialize. **Idea 0
  (double-buffer `cb_input_resident`) is validated as the real Regime-A lever** (untested-but-promising).
- **At large sizes it becomes genuinely write-throughput-bound** (`(16384,512)`: `WR-noc` 113µs) —
  so Target 1's write batching matters more there, not less.
- **`(512,8192)` = 163µs pathology:** `nrg·K_min > grid` → falls into the slow single-core-per-row
  Regime A fallback (no transport zones at all). The "oversubscribed Regime B coverage" gap the
  changelog flagged, confirmed live.
- **Regime B stays writer-bound even grid-saturated** — the all-reduce is only ~6-10% of total at
  full grid; it's a meaningful fraction only on single/few-row-group wide-W (e.g. `(32,8192)`:
  `ar-wait + ar-xport ≈ 7.5µs` of 21µs).

### All-reduce algorithm: mode 2 = reduce-then-broadcast — **KEPT (new default)**

**Change.** New Regime-B transport `TRANSPORT_REDUCE_BCAST = 2` (gated by the existing
`transport_mode` CT arg; compute changes too). The root gathers the K partials (same as mode 1),
**reduces them to the single global Σx²**, and mcasts **one tile** instead of K. Peers receive that
tile and **skip the combine reduce entirely** (`run_combine = (transport_mode != 2) || is_root`).
Reuses `cb_partial_sumsq`; plumbed `cb_partial_sumsq` to the reader + `is_root` to compute across
all four descriptor functions. (Implemented by the ttnn-implementer; the expert-debugger fixed a
real cross-thread bug — the reader and compute hold *per-RISC-local* CB pointers, so the reader had
to manually re-sync its `cb_partial_sumsq` read pointer with the compute RISC's pop via
`advance_local_rd_ptr` — captured in audit-trail commits.)

**Measured (mode 1 → mode 2, total ns, bf16):**

| shape | mode 1 | mode 2 | speedup | CMP-combine stall |
|-------|--------|--------|---------|-------------------|
| (32,4096)  | 14693 | 13807 | 1.06x | 4555 → 530 |
| (32,8192)  | 21135 | 18808 | **1.12x** | 8307 → 503 |
| (32,16384) | 28184 | 26753 | 1.05x | 8761 → 494 |
| (64,8192)  | 26269 | 24184 | 1.09x | 5601 → 535 |
| (256,8192) | 56986 | 56630 | 1.01x (grid-saturated → writer-bound) | 4100 → 570 |

The per-peer `CMP-combine` stall collapses **~8µs → 0.5µs**: peers no longer wait for the K-tile
broadcast nor run a reduce. The win is largest on single-row-group wide-W (where the all-reduce is
on the critical path) and tapers to neutral once the shape is grid-saturated and writer-bound.

**Correctness.** With mode 2 forced AND as the default: `test_rms_norm.py` 20/20,
`test_rms_norm_regime_b.py` + `test_rms_norm_rm_regime_b.py` green (99 passed combined, `--dev`,
no hang); `test_rms_norm_transport.py` exercises modes 0/1/2.

**Helper note.** No new helper mode needed — but this exposed a real **dataflow-lib gap**: there is
no safe primitive for a dataflow RISC to read a CB slot that the *compute* RISC produced/consumed,
because CB pointers are per-RISC-local. The fix (`advance_local_rd_ptr`) is hand-rolled pointer
arithmetic against `get_local_cb_interface`. A `dataflow_kernel_lib` helper to "peek a peer-RISC CB
slot by absolute index" would make root-relay/reduce-broadcast transports far less error-prone.

**Verdict: KEEP, made the production default** (`_select_transport → TRANSPORT_REDUCE_BCAST`).
1.05–1.12x on wide-W Regime B, neutral elsewhere, no regression. Mode 0/1 retained behind the CT
arg for the bake-off.

### Idea 0 — double-buffer `cb_input_resident` (multi-row Regime A) — **REVERTED (disabled, net-negative)**

**Change (tried).** Size `cb_input_resident` `Wt → 2*Wt` when a core owns >1 row, so the reader
prefetches row N+1 while compute holds row N. No kernel change (the existing reader loop fills the
spare slot). Behind `_DOUBLE_BUFFER_A` (default now False).

**Measured (controlled A/B, best-of-3 same session, OFF → ON total):**

| shape | rows/core | OFF | ON | speedup | RDR-resv OFF→ON |
|-------|-----------|-----|-----|---------|-----------------|
| (4096,256)  | 2 | 28561 | 26868 | **1.06x** | 4600 → 78 |
| (8192,256)  | 4 | 49482 | 50676 | 0.98x | 14632 → 4051 |
| (16384,256) | 8 | 95073 | 99557 | 0.96x | 37633 → 17717 |
| (16384,512) | 8 | 193093 | 194092 | 1.00x | 84630 → 54000 |

**The instructive part.** Double-buffering ALWAYS collapses `RDR-resv` (the stall it targets), but
the total only improves at 2 rows/core — flat-to-*worse* at 4-8. So **`RDR-resv` was off the
critical path**: an idle reader is slack, not the bottleneck. Compute is serial across rows and the
writer is the long pole, so feeding the input faster can't help; the aggressive prefetch just adds
read/write DRAM contention. This is the same trap as the writer-zone-spanning-total (Round 1) and
Target 2 — a large-looking stall that isn't on the critical path. It also confirms the bandwidth
finding: we're neither DRAM-bound nor reader-bound; the limiter is the serial per-row dependency
chain + writer drain.

**Verdict: DISABLE** (`_DOUBLE_BUFFER_A = False`), implemented + gated off like R7's row-blocking.
The 2-rows/core win is too narrow and shape-specific to justify the 4-8/core regression.

### Bandwidth reality check (settles "is it writer-bound?")
Aggregate DRAM BW achieved (Regime A, bf16, read==write==rows·Wt·2KB):

| shape | read+write | % of ~288 GB/s peak | per-core read vs ~32 GB/s NoC link |
|-------|-----------|---------------------|------------------------------------|
| (2048,256)  | 123 GB/s | 43% | 3.5 / 32 = 11% |
| (8192,256)  | 171 GB/s | 59% | 3.3 / 32 = 10% |
| (16384,512) | 160 GB/s | 55% | 3.5 / 32 = 11% |

**NOT DRAM-bandwidth-bound** — sustained ~55-60% of peak with ~40% headroom; per-core transfers use
only ~10% of the NoC link → transaction/latency-bound per core. The write *burst* (during the
NoC-busy window) reaches ~200-260 GB/s (near peak), but the writer is only busy ~50% of the wall —
the gap is the `WR-wait` stall (writer starved by the serial prefix). So "writer-bound" = writer is
the longest-pole STAGE, but the true limiter is the **serial per-row dependency chain starving a
~50%-duty-cycle writer + per-core transaction latency**, not DRAM bandwidth.

## Deep wall-clock timeline profiling (driver `.eval/refinements/timeline_profile.py`)

Reconstructed per-core absolute-timestamp timelines (all RISCs on one axis, critical core) to find
the true critical path instead of inferring it from durations. Findings **correct earlier claims**:

**Regime A, 1 row/core (2048,256), total ~18000ns — critical path:**
| segment | time | nature |
|---------|------|--------|
| read (compute idles in square-wait) | ~4.8µs | DRAM **read latency** (NoC ~10% util) |
| square + reduce | ~1.1µs | real compute (cheap) |
| finalize (rsqrt SFPU) | ~2.4µs | fixed SFPU init+compute — on critical path |
| pass2 | ~0.8µs | real compute (cheap) |
| **first write block** | ~4–7µs | **ONE-TIME** first-write-burst stall (see below) |
| second write block | ~0.7µs | steady-state write of 4 tiles |

**CORRECTION (raw per-fire writer events, `.eval/refinements/raw_writer_events.py`).** An earlier
draft called this "~7.6µs writer drain / DRAM-write-latency bound." That was WRONG. Splitting the
writer zone into WR-issue (the `noc_async_write_tile` loop) vs WR-bar (the completion barrier), per
block, across runs:
- 1 row/core (2 blocks): block-0 WR-issue = **5218 / 3463 / 3790 ns** across 3 runs; block-1 WR-issue
  = **264 / 264 / 262 ns**. The cost is in *issuing the first block's writes*, is a consistent
  ONE-TIME stall, and the second identical block issues ~20x faster.
- 4 rows/core (8 blocks): WR-issue = [724,1385,1465,758,265,965,294,264] — **no 5µs spike at all**.
So the writer is NOT uniformly slow: there's a one-time first-write-burst latency (likely first-
touch of the DRAM write path / NoC cmd-buffer fill after the writer's long idle) that dominates a
tiny 2-block kernel and **amortizes away** with more blocks/work. "Writer-bound" was an artifact of
measuring 1-row-per-core; in steady state the per-block write is ~1–2.5µs and the writer is balanced
with compute (the 8192,256 timeline shows BRISC ≈ TRISC busy, both ~0 idle).

- **Correction: the "5.9µs square" is ~90% a stall.** `CMP-p1-square` opens at t=0 but the read
  finishes at 5526ns; the zone's first act is `cb_wait_front`, so it's BLOCKED for ~4.8µs and the
  actual squaring is only ~0.66µs. The READ is the prefix cost, not the square.
- **This downgrades Idea 1** (square+reduce fusion): square-compute ~0.66µs + reduce ~0.48µs, so the
  `cb_squared` round-trip is ~0.5µs of L1 (SRAM) traffic — fusing saves little, not multi-µs.
- The two DRAM-latency segments (read ~4.8 + write tail ~7.6) ≈ **69% of the kernel**. The op is
  **DRAM-access-LATENCY bound** (small 2KB transfers, ~10% NoC link), not bandwidth, not compute.
  finalize's 2.4µs SFPU is the only non-trivial compute.

**Regime A, 4 rows/core (8192,256), total 48.7µs:** all RISCs ~0 idle (BRISC 48.1, TRISC 46.3,
NCRISC 41.2µs). Genuine pipeline, **co-limited writer ≈ compute**; the writer's busy is ~half
`WR-noc` / half `WR-wait` (blocked on compute) even when fully pipelined.

**Regime B (32,8192, mode 2), total 18.6µs:** `CMP-combine` now tiny (0.36µs — mode 2 worked), but
`finalize` spans 4837→13090 because the peer blocks on the reader's broadcast; **the transport
(`RDR-ar-xport` ~6µs gather+broadcast) is still on the critical path**, bracketed by read (~3.2µs)
and writer (~4.3µs).

**Bottleneck verdict (revised after the raw-writer drill-down):**
- **1 row/core** is dominated by per-op startup latency: a fixed DRAM **read** front (~4.8µs) + a
  ONE-TIME first-**write**-burst stall (~4–7µs) + the fixed finalize SFPU (~2.4µs). These are
  startup/latency costs that **amortize with more work** — not a throughput wall. NOT
  bandwidth-bound (~55% of peak), NOT compute-bound (cheap except finalize), NOT uniformly
  write-slow (steady-state per-block write ~1µs).
- **Multi-row / steady state** is a genuine pipeline, writer ≈ compute co-limited (~0 idle on both).
- Levers that were tried and explained as NON-levers by the timeline: compute fusion (Idea 1 —
  square-compute is ~0.66µs), input prefetch (Idea 0 — RDR-resv is slack), finalize fusion
  (Target 3 — loses pipelining). Target 1 (write batching) already addresses steady-state per-block
  write overhead. The remaining real costs (read latency, the one-time first-write stall) are
  largely fixed per-op latencies that the op amortizes by running more rows/cores.

### Sub-row Regime B teams (remove full-width-band requirement) — **KEPT**

**Change.** Regime B required `K % gx == 0` (full-width `gh×gx` bands), so the smallest W-split was
K=gx=8. Shapes with `num_row_groups·8 > grid` (e.g. `(512,8192)`, nrg=16 → 128>64) got no valid K →
fell back to slow single-core-per-row Regime A (16 cores, Wt=256 resident, 163µs). Relaxed
`_select_k` to also accept K that **divides** gx (sub-row `1×K` teams, `gx//K` per grid row) and
generalized the geometry in `_regime_b_descriptor` + `_regime_rm_b_descriptor`
(`base_row`/`base_col`/`gw`/`gh`/`teams_per_row`). HOST-SIDE ONLY — kernels unchanged (the transport
already takes a generic rect + K peer coords). `(512,8192)`: K=4, sixteen 1×4 teams tile the full
64-core grid, one row-group per core.

**Measured (bf16):** `(512,8192)` **163.8µs → 100.3µs (1.63x)**; `(1024,8192)` now runs in the fast
path at 191.9µs. Only 1.63x (not 3–5x) because the fallback was single-core-*per-row* (16 active
cores, 48 idle), so reaching all 64 cores via K=4 is ~1.6x — transport/overhead-limited. Existing
full-width shapes UNCHANGED (K=32/16/8 — byte-identical geometry; the proxy still picks the larger
full-width K when feasible).

**Correctness (`--dev`, no hang):** `_select_k(512,8192)`→K=4 sub-row; all-ones maxerr ≤ 0.0005,
PCC = 1.00000 on (512,8192)/(512,4096)/(1024,8192) × {bf16,fp32}; no regression on acceptance(20) +
regime_b + rm_regime_b (99 total).

### Internal padding — Regime B works for ANY Wt (remove "K divides Wt") — **KEPT**

**Problem.** Every valid K is even (divisor/multiple of gx=8) and had to divide Wt exactly. So:
(a) odd Wt above the single-core L1 budget had NO valid K → `NotImplementedError` CRASH (confirmed
Wt=329, 331, 513); (b) even-but-awkward Wt (e.g. 2·prime=334) was stuck at K=2 (half the grid).

**Fix (internal padding).** Regime B now splits `Wt_pad = ceil(Wt/gx)*gx` (rounded up to a multiple
of gx, so K=8+ always divides it). Uniform shard `Wt_s = Wt_pad/K` (stays compile-time). The trailing
tiles beyond the real Wt are PAD: the TILE reader zeros them (contribute 0 to Σx²; `inv_W` uses true
W), and the writer DRAINS all `Wt_s` tiles (so cb_output never backs up → no hang) but WRITES only the
real ones. Compute unchanged. Wt already a multiple of gx → `Wt_pad==Wt`, byte-identical.

**Bonus latent bug fixed (RM Regime B).** The padding probe surfaced that RM Regime B was *already*
silently wrong whenever a shard's `Wt_s % reduce_block != 0` (e.g. Wt=264: maxerr **0.20**): the
reader/writer clamped columns only against global W, so an interior shard's chunk-pad tail (still
< W) over-read the next shard's data into Σx² and over-wrote its output. Fixed by clamping against
`col_limit = min(W, shard_col0 + Wt*TILE_W)` in the RM reader + writer. Wt=264 RM: 0.20 → 0.007.

**Measured (bf16, previously CRASH / stuck):** Wt=334 (W=10688) 22.3µs (was K=2/half-grid);
Wt=329 (W=10528) **34.7µs (was crash → K=24)**; Wt=513 (W=16416) **39.6µs (was crash → K=40)**.
All in the normal Regime-B band (clean ref Wt=256 = 18.3µs), not the slow fallback. So "perf in all
cases" along the width axis: any Wt now splits across (most of) the grid.

**Correctness (`--dev`, no hang):** new widths {329,331,513,334} × {bf16,fp32} × {TILE,RM} × ±gamma,
single+multi-row: all-ones maxerr ≤ 0.001, PCC ≥ 0.99999. No regression: acceptance(20) + regime_b(39)
+ rm_regime_b(40) + layout_matrix(120) green. Clean widths (Wt=256) byte-identical (Wt_pad==Wt).

**Still open (width axis):** `_select_k`'s cost proxy was tuned for clean full-width K; for padded
shapes it may not pick the optimal K (e.g. Wt=329→K=24 vs K=48). Minor — the shapes now work and are
fast; K-tuning for padded widths is a future refinement.

### Still open
- **More-outstanding-transactions levers** (the timeline-indicated direction): batch all `Wt` output tiles behind one barrier (vs `reduce_block`), and/or coalesce reads. Untried; targets the latency-bound read/write directly.
- **`num_row_groups > grid` oversubscription** — multiple row-groups PER core + per-row-group transport loop. The sub-row fix already covers `nrg ≤ grid`; this is the remaining (bigger, hang-prone) corner.
- **Idea 1 — square+reduce fusion**: DOWNGRADED by the timeline (square-compute is ~0.66µs; fusion saves ~0.5µs of L1 traffic). Low priority.
- Ring all-reduce not tried — 1-tile data ⇒ K-1 serial hops (latency-bound), expected to lose to the reduce-broadcast that won.
