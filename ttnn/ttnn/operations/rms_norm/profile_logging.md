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
