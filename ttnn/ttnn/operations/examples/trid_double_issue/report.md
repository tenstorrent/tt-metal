# trid_double_issue — device report

**Box:** `bh-50-special-dstoiljkovic-for-reservation-88042` · **Arch:** Blackhole · **Date:** 2026-09-10 · **Commit:** `dfc0dae18e4`
**Metric:** `DEVICE KERNEL DURATION [ns]`, in-process device profiler, mean of 20 profiled launches after 5 warmups.
**Op:** identity copy of a `[512, 512]` bfloat16 TILE tensor (256 tiles, 2048 B/tile) in interleaved DRAM.
Reader on NCRISC/NoC0, writer on BRISC/NoC1, no compute kernel.

**CB is identical in every cell of a block row:** `total_size = cb_blocks × block × page_bytes` with
`cb_blocks = 6` everywhere, so L1 is never a confound. Numbers are illustrative of the *effect*, not CI bounds.

**Provenance.** Every table below was measured after the final kernel refactor and with the full
`ahead = 1,2,3,4` baseline sweep — an earlier `1,2,4` sweep straddled the baseline's optimum at 3 and
understated it, so any number quoted against a 1/2/4 baseline is superseded. §2/§3 come from one
sweep (5 blocks × 7 methods, 256 tiles); §4 from the block-scaling probe (1024 tiles, `ahead=1..5`);
§5 and §6 from their own sweeps. Run-to-run spread is under 0.5%, so the same quantity may differ in
the last digit between sections — those are separate runs, not disagreements.

---

## 1. The two baselines, and why both are reported

`full_barrier` has its own strength knob, `ahead`: how many blocks it issues before its single
`noc_async_read_barrier()`, pushing them individually afterwards.

- **`ahead=1`** is the idiomatic read-block / barrier / push loop, and what most readers look like.
- **`ahead>1`** spends the spare CB to keep more reads in flight. **It is the strongest a non-trid
  reader can be**, and quoting a win against `ahead=1` alone overstates the result.

A non-trid reader cannot go further, and the reason is mechanical: the only global completion
signal is a **count** (`NIU_MST_RD_RESP_RECEIVED` vs `noc_reads_num_issued`), and read responses
take **dynamically assigned VCs**, so they can land out of order. A count never proves a *specific
earlier* block arrived, so nothing may be pushed until the barrier has drained everything.
`NIU_MST_REQS_OUTSTANDING_ID(trid)` is the only per-group completion signal — exactly the gap
transaction ids fill.

---

## 2. Main table — 1 core, bf16, `cb_blocks = 6` for every cell

`slots` = CB slots the reader reserves at its peak, out of the **6 allocated in every cell** — equal
slots means equal L1 occupancy, so a `trid ×N` row is iso-L1 against the `ahead=N` row. `vs naive` is
against `ahead=1`; `vs best base` is against the best non-trid cell in that row.

**`ahead=3` is the baseline's optimum here**, not 4: with `cb_blocks=6` it splits the CB evenly
between the reader's window and the writer's lag, while `ahead=4` leaves the writer only 2 slots and
regresses. A sweep that skips 3 badly understates the baseline.

| block | method | slots | in-flight | GB/s | vs naive | vs best base |
|---:|---|---:|---:|---:|---:|---:|
| 1 | base ahead=1 | 1 | 1 | 10.1 | 1.00× | 0.42× |
| 1 | base ahead=2 | 2 | 2 | 17.6 | 1.74× | 0.73× |
| 1 | base ahead=3 | 3 | 3 | 24.2 | 2.40× | 1.00× **←best base** |
| 1 | base ahead=4 | 4 | 4 | 23.5 | 2.33× | 0.97× |
| 1 | trid ×2 | 2 | 2 | 19.2 | 1.90× | 0.79× |
| 1 | trid ×3 | 3 | 3 | 29.0 | 2.87× | 1.20× |
| 1 | trid ×4 | 4 | 4 | 37.3 | 3.69× | 1.54× **←best trid** |
| 2 | base ahead=1 | 1 | 2 | 17.9 | 1.00× | 0.45× |
| 2 | base ahead=2 | 2 | 4 | 30.2 | 1.69× | 0.77× |
| 2 | base ahead=3 | 3 | 6 | 39.4 | 2.20× | 1.00× **←best base** |
| 2 | base ahead=4 | 4 | 8 | 35.6 | 1.99× | 0.90× |
| 2 | trid ×2 | 2 | 4 | 34.9 | 1.95× | 0.89× |
| 2 | trid ×3 | 3 | 6 | 51.7 | 2.89× | 1.31× |
| 2 | trid ×4 | 4 | 8 | 64.3 | 3.59× | 1.63× **←best trid** |
| 4 | base ahead=1 | 1 | 4 | 32.7 | 1.00× | 0.53× |
| 4 | base ahead=2 | 2 | 8 | 49.2 | 1.50× | 0.79× |
| 4 | base ahead=3 | 3 | 12 | 62.2 | 1.90× | 1.00× **←best base** |
| 4 | base ahead=4 | 4 | 16 | 53.9 | 1.65× | 0.87× |
| 4 | trid ×2 | 2 | 8 | 62.1 | 1.90× | 1.00× |
| 4 | trid ×3 | 3 | 12 | 84.9 | 2.60× | 1.36× **←best trid** |
| 4 | trid ×4 | 4 | 16 | 84.5 | 2.58× | 1.36× |
| 8 | base ahead=1 | 1 | 8 | 52.1 | 1.00× | 0.61× |
| 8 | base ahead=2 | 2 | 16 | 73.6 | 1.41× | 0.86× |
| 8 | base ahead=3 | 3 | 24 | 85.1 | 1.63× | 1.00× **←best base** |
| 8 | base ahead=4 | 4 | 32 | 68.4 | 1.31× | 0.80× |
| 8 | trid ×2 | 2 | 16 | 97.2 | 1.87× | 1.14× |
| 8 | trid ×3 | 3 | 24 | 99.4 | 1.91× | 1.17× **←best trid** |
| 8 | trid ×4 | 4 | 32 | 97.5 | 1.87× | 1.15× |
| 16 | base ahead=1 | 1 | 16 | 76.9 | 1.00× | 0.79× |
| 16 | base ahead=2 | 2 | 32 | 93.2 | 1.21× | 0.95× |
| 16 | base ahead=3 | 3 | 48 | 97.8 | 1.27× | 1.00× **←best base** |
| 16 | base ahead=4 | 4 | 64 | 77.3 | 1.01× | 0.79× |
| 16 | trid ×2 | 2 | 32 | 108.8 | 1.41× | 1.11× **←best trid** |
| 16 | trid ×3 | 3 | 48 | 104.2 | 1.36× | 1.07× |
| 16 | trid ×4 | 4 | 64 | 100.1 | 1.30× | 1.02× |

---

## 3. The controlled comparison — `ahead=N` vs `trid ×N`

Same CB size, same number of CB slots used, same reads in flight. **Only the barrier differs.**

| block | N=2 base | N=2 trid | gain | N=3 base | N=3 trid | gain | N=4 base | N=4 trid | gain |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.6 | 19.2 | 1.09× | 24.2 | 29.0 | 1.20× | 23.5 | 37.3 | **1.59×** |
| 2 | 30.2 | 34.9 | 1.16× | 39.4 | 51.7 | 1.31× | 35.6 | 64.3 | **1.81×** |
| 4 | 49.2 | 62.1 | 1.26× | 62.2 | 84.9 | 1.36× | 53.9 | 84.5 | **1.57×** |
| 8 | 73.6 | 97.2 | 1.32× | 85.1 | 99.4 | 1.17× | 68.4 | 97.5 | **1.43×** |
| 16 | 93.2 | 108.8 | 1.17× | 97.8 | 104.2 | 1.07× | 77.3 | 100.1 | **1.30×** |

*(GB/s. Best absolute anywhere in the sweep: **108.8 GB/s**, trid ×2 at block 16.)*

**The honest headline is 1.11–1.63×** (each side at its own best config, same allocated CB), or
1.07–1.81× iso-`N`. Not 3.7× — that is only against the naive `ahead=1` loop, and batching alone
recovers most of it. The large iso-`N` gains at N=4 are real but partly the baseline's fault:
`ahead=4` starves the writer, while `trid ×4` holds the same 4 slots without doing so.

**Depth is what pays, not tagging.** At `block=4`, `trid ×2` (62.1) exactly *ties* the tuned baseline
(62.2). A shallow ring over a well-tuned baseline buys nothing; the wins appear at depth 3–4.

**Why the baseline still loses, and why it peaks then degrades.** The baseline can batch reads, but
it must drain to zero at every barrier *and* it pushes its whole batch in a burst afterwards, so
the writer starves through the entire issue-and-drain phase and then gets flooded. Pushing `ahead`
higher trades NoC idle for writer idle, which is why it turns over: at `block=16`, `ahead=2`
reaches 90.7 GB/s but `ahead=4` falls to 75.1 (0.83× of its own best), and at `block=8`, `ahead=4`
(66.7) is already worse than `ahead=2` (70.6). The trid ring has no such turnover in the same range
because it sustains depth *and* hands the writer one block at a time.

**In-flight depth is necessary but not sufficient.** At 16 reads in flight the sweep spans
52.3 → 96.3 GB/s depending on *how* that depth is held. The latency × bandwidth product sets the
ceiling; whether you ever drain to zero decides how close you get.

---

## 4. What trids actually buy: L1, not peak bandwidth

Widening `block` also buys in-flight depth, and it costs L1 (`cb_blocks × block × page_bytes`).
Sweeping block on a 1024-tile bf16 tensor, 1 core, with the CB budget that implies:

| L1 for the CB | block | simplest loop (`ahead=1`) | best baseline | best trid | trid vs simplest | trid vs best base |
|---:|---:|---:|---:|---:|---:|---:|
| 96 KB | 8 | 52.7 | 89.8 (ahead=3) | **107.1** | 2.03× | 1.19× |
| 192 KB | 16 | 79.1 | 111.4 (ahead=3) | **119.5** | 1.51× | 1.07× |
| 384 KB | 32 | 102.9 | 120.0 (ahead=3) | **123.9** | 1.20× | 1.03× |
| 768 KB | 64 | 118.7 | 120.9 (ahead=2) | **122.2** | 1.03× | 1.01× |

*(GB/s. Beyond block=64 the CB no longer fits L1.)*

**Everything converges to ~122 GB/s — the single-core ceiling.** At 768 KB the *simplest possible
reader* (one block, one barrier, no ring, no batching, no ids) is within **3%** of the best number
in the whole study. So neither the batching nor the ids raise the ceiling; they only change how
much L1 you must spend to reach it:

- **trid at 96 KB = 107.1 GB/s.** A tuned baseline needs somewhere between 96 KB (89.8) and 192 KB
  (111.4) to match that — call it **~1.5× less L1**. At the next step up, trid at 192 KB (119.5)
  matches the baseline's 384 KB (120.0): **~2× less L1**. So the saving is **~1.5–2×**, and it
  shrinks as the budget grows.

**This is the honest reason to reach for trids, and the honest reason not to.** If L1 is free,
widen the block and write the trivial loop. Trids are for when the block is capped — by a shard
size, by a wide tensor, by co-resident buffers — and you still want the depth.

It also explains why the two code paths in this example are about the same size (17 vs 23 lines).
The complexity is not the ids; it is keeping N blocks in flight in a circular buffer at all, and
both paths pay it. The trid-specific part is four lines.

---

## 5. Mechanism check — it is a *latency*, not bytes (1 core)

Ratios against the best non-trid baseline, across a 3.8× range of transaction sizes:

| dtype | tile bytes | block | best base (GB/s) | trid ×4 (GB/s) | gain |
|---|---:|---:|---:|---:|---:|
| bfloat8_b | 1088 | 1 | 13.3 (ahead=4) | 20.7 | 1.56× |
| bfloat16 | 2048 | 1 | 24.3 (ahead=3) | 37.3 | 1.54× |
| float32 | 4096 | 1 | 43.3 (ahead=3) | 68.9 | 1.59× |
| bfloat8_b | 1088 | 4 | 34.4 (ahead=3) | 48.6 | 1.41× |
| bfloat16 | 2048 | 4 | 62.2 (ahead=3) | 85.0 | 1.37× |
| float32 | 4096 | 4 | 92.6 (ahead=3) | 112.9 | 1.22× |

**1.54–1.59× at block=1 and 1.22–1.41× at block=4**, across a 3.8× range of tile bytes. Flat enough
within each block that the recovered cost is a fixed round-trip latency, not anything proportional to
payload; the drift at block=4 is float32 reaching the bandwidth knee sooner (92.6 GB/s baseline
already) because each read moves twice the bytes.

---

## 6. Core scaling — compresses, but holds

Best non-trid vs best trid in each cell:

| cores | block | best base (GB/s) | best trid (GB/s) | gain |
|---:|---:|---:|---:|---:|
| 1 | 1 | 24.3 (ahead=3) | 37.4 (×4) | 1.54× |
| 6 | 1 | 132.7 (ahead=4) | 195.8 (×4) | 1.48× |
| 1 | 4 | 62.2 (ahead=3) | 85.0 (×3) | 1.37× |
| 6 | 4 | 253.5 (ahead=3) | 345.6 (×4) | 1.36× |
| 1 | 16 | 97.9 (ahead=3) | 108.9 (×2) | 1.11× |
| 6 | 16 | 281.7 (ahead=2) | 328.4 (×2) | 1.17× |

Each core drains its own NoC, so the effect is per-core and survives to 6 cores at 346 GB/s,
essentially undiminished (1.54× → 1.48× at block=1; 1.37× → 1.36× at block=4) as the aggregate in-flight depth approaches the knee.

**Not measured: the fully saturated grid.** 6 of 110 cores is far from it.

---

## Blackhole p150b — bh-qb-15-special-adamjanovic-for-reservation-139410 (2026-10-19)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid 12×10 = 120 Tensix cores, 8 DRAM banks, AICLK 1350 MHz |
| commit | `f74ee7aad64` + arch-agnostic regimes tests (branch `adamjanovic/ab-tilize-run-scoreboard-catalog`), Release build |
| metric / method | same op and CB as §2: `test_trid_double_issue_device_perf`, `DEVICE KERNEL DURATION [ns]`, mean of 10 profiled launches per cell (the test's default; §2 used 20), 1 Tensix core, `[512, 512]` bf16, `cb_blocks = 6` in every cell |

### Main table (`test_trid_double_issue_device_perf`) — same schema as §2

| block | method | slots | in-flight | ns/op | GB/s | vs naive | vs best base |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | base ahead=1 | 1 | 1 | 103755.3 | 10.1 | 1.00× | 0.42× |
| 1 | base ahead=2 | 2 | 2 | 59468.9 | 17.6 | 1.74× | 0.73× |
| 1 | base ahead=3 | 3 | 3 | 43374.9 | 24.2 | 2.39× | 1.00× |
| 1 | base ahead=4 | 4 | 4 | 43328.1 | 24.2 | 2.39× | 1.00× **←best base** |
| 1 | trid ×2 | 2 | 2 | 54579.6 | 19.2 | 1.90× | 0.79× |
| 1 | trid ×3 | 3 | 3 | 36148.6 | 29.0 | 2.87× | 1.20× |
| 1 | trid ×4 | 4 | 4 | 28087.8 | 37.3 | 3.69× | 1.54× **←best trid** |
| 2 | base ahead=1 | 1 | 2 | 58701.8 | 17.9 | 1.00× | 0.45× |
| 2 | base ahead=2 | 2 | 4 | 34337.4 | 30.5 | 1.71× | 0.77× |
| 2 | base ahead=3 | 3 | 6 | 26438.1 | 39.7 | 2.22× | 1.00× **←best base** |
| 2 | base ahead=4 | 4 | 8 | 29036.7 | 36.1 | 2.02× | 0.91× |
| 2 | trid ×2 | 2 | 4 | 29913.4 | 35.1 | 1.96× | 0.88× |
| 2 | trid ×3 | 3 | 6 | 20180.8 | 52.0 | 2.91× | 1.31× |
| 2 | trid ×4 | 4 | 8 | 16114.0 | 65.1 | 3.64× | 1.64× **←best trid** |
| 4 | base ahead=1 | 1 | 4 | 32067.6 | 32.7 | 1.00× | 0.52× |
| 4 | base ahead=2 | 2 | 8 | 21239.4 | 49.4 | 1.51× | 0.79× |
| 4 | base ahead=3 | 3 | 12 | 16716.7 | 62.7 | 1.92× | 1.00× **←best base** |
| 4 | base ahead=4 | 4 | 16 | 19235.6 | 54.5 | 1.67× | 0.87× |
| 4 | trid ×2 | 2 | 8 | 16841.7 | 62.3 | 1.90× | 0.99× |
| 4 | trid ×3 | 3 | 12 | 12171.3 | 86.2 | 2.63× | 1.37× **←best trid** |
| 4 | trid ×4 | 4 | 16 | 12172.8 | 86.1 | 2.63× | 1.37× |
| 8 | base ahead=1 | 1 | 8 | 20145.6 | 52.0 | 1.00× | 0.61× |
| 8 | base ahead=2 | 2 | 16 | 14124.0 | 74.2 | 1.43× | 0.87× |
| 8 | base ahead=3 | 3 | 24 | 12251.3 | 85.6 | 1.64× | 1.00× **←best base** |
| 8 | base ahead=4 | 4 | 32 | 15213.6 | 68.9 | 1.32× | 0.81× |
| 8 | trid ×2 | 2 | 16 | 10762.7 | 97.4 | 1.87× | 1.14× |
| 8 | trid ×3 | 3 | 24 | 10346.8 | 101.3 | 1.95× | 1.18× **←best trid** |
| 8 | trid ×4 | 4 | 32 | 10544.2 | 99.4 | 1.91× | 1.16× |
| 16 | base ahead=1 | 1 | 16 | 13537.3 | 77.5 | 1.00× | 0.78× |
| 16 | base ahead=2 | 2 | 32 | 11161.9 | 93.9 | 1.21× | 0.95× |
| 16 | base ahead=3 | 3 | 48 | 10615.3 | 98.8 | 1.28× | 1.00× **←best base** |
| 16 | base ahead=4 | 4 | 64 | 13395.5 | 78.3 | 1.01× | 0.79× |
| 16 | trid ×2 | 2 | 32 | 9432.6 | 111.2 | 1.44× | 1.13× **←best trid** |
| 16 | trid ×3 | 3 | 48 | 9862.0 | 106.3 | 1.37× | 1.08× |
| 16 | trid ×4 | 4 | 64 | 10212.1 | 102.7 | 1.33× | 1.04× |

### Block sweep — best base vs best trid, BH p150b vs `bh-50` (§2)

| block (tiles) | BH p150b best base → best trid (GB/s) | gain | `bh-50` best base → best trid (GB/s) | gain |
|---:|---|---:|---|---:|
| 1 | 24.2 → 37.3 | **1.54×** | 24.2 → 37.3 | 1.54× |
| 2 | 39.7 → 65.1 | **1.64×** | 39.4 → 64.3 | 1.63× |
| 4 | 62.7 → 86.2 | **1.37×** | 62.2 → 84.9 | 1.36× |
| 8 | 85.6 → 101.3 | **1.18×** | 85.1 → 99.4 | 1.17× |
| 16 | 98.8 → 111.2 | **1.13×** | 97.8 → 108.8 | 1.11× |
| does not help: block 4, `ahead=3` vs `trid ×2` | 62.7 → 62.3 | **0.99×** | 62.2 → 62.1 | 1.00× |

### Reading it
- **Reproduces `bh-50` within ~1–2% on every cell.** The single-Tensix-core copy does not depend on grid size
  (120 vs 110 Tensix cores), so the p150b numbers confirm the headline: **1.13–1.64×** over a tuned baseline at
  the same allocated CB, gain shrinking with block size.
- **`ahead=3` is again the baseline's optimum** (at block=1 `ahead=4` ties it); `trid ×2` again only ties the tuned
  baseline at block=4 (0.99×), so depth 3–4 is what pays.
- **Not re-measured on p150b:** the L1-budget sweep (§4), the dtype check (§5) and core scaling (§6). WH B0 n150
  has no measurement of this example.
