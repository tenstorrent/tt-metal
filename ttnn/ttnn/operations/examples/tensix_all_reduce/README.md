# Tensix rectangular all-reduce - on-chip collective algorithm bake-off

**Problem type:** cross-core communication  ·  **Concept(s):** Tensix-to-Tensix transfer topology and reduction work distribution
**First profiled on:** `bgd-lab-t3003-special-mstaletovic-for-reservation-40918` · Wormhole B0 · 1000 MHz · 2026-07-10 · `5f0ad060667`
**Measured on:** WH B0 n150 (regimes, 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19) · BH p150b 13×10 grid (`bh-50-special-adamjanovic-for-reservation-143315`, AICLK 1350 MHz, 2026-10-26: flush-before-signal lever, large-payload boundary)

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem

Every Tensix core in a rectangular group owns the same number of tiles. An all-reduce must add
corresponding tiles across the group and leave the complete sum on every member. The communication
pattern becomes the main design choice when the input and output are already sharded in L1.

## What this isolates - and how

- **Concepts:** on-chip collective topology and how reduction work is distributed across cores.
- **Isolation setup:** Tensix-to-Tensix NoC - bf16 inputs and outputs are height-sharded in L1, no
  DRAM is touched by the kernels, and every method produces the same elementwise tile reduction.
- **Why it is kernel-level:** the unicast/multicast schedule, semaphore protocol, root selection,
  and worker assignment are all controlled by the dataflow and compute kernels.

## The methods being compared

| Variant | What it does | Expected mechanism |
|---|---|---|
| `reduce_root_mcast` | Non-root cores unicast to one root; the root reduces all blocks and multicasts the result. | Cuts communication, but serializes all reduction work on the root. |
| `reduce_scatter_mcast` | All `min(num_tiles, group_size)` workers - the root included - gather and reduce disjoint tile indices, write partials to the root, then the root multicasts the assembled result. The tile split need not divide the group. | Keeps the reduced communication volume while parallelizing the root's compute and reads. |
| `reduce_scatter_push` | Same algorithm and gather layout as `reduce_scatter_mcast`, with the transfer reversed: contributors write their tiles into the owning worker's gather buffer instead of each worker reading from every contributor. | Trades the read round trip for a `group_size * num_workers` semaphore handshake that the pull side does not need at all. |
| `reduce_scatter_mcast_flush` | `reduce_scatter_mcast` with one change: before the semaphore increment that publishes a worker's partials to the root, it waits for the writes to be *flushed* (left this Tensix core) instead of *barriered* (acknowledged by the root). | Saves one write-acknowledgement round trip per published signal; the increment still lands after the data because both travel on the same NoC and VC (see *Signal after a flush, not a barrier*). |
| `reduce_scatter_push_flush` | `reduce_scatter_push` with the same change at both of its publishing points (gather writes → each worker's gather semaphore; partial writes → the root's progress semaphore). | Two fenced signals per all-reduce instead of one, so twice the saving of the pull twin. |
| `tree_reduce_mcast` | Hierarchical reduce over the 2-D core grid. Stage 1: within each grid row the `cols` cores gather to their row's leader, which reduces them to a per-row partial. Stage 2: the `rows` row-leaders gather to the group root, which reduces the partials into the group sum. Stage 3: the root multicasts the sum to the whole group. | Splits the reduction across the two grid axes, so each gather stage has a small fan-in (`cols`, then `rows`) instead of one all-to-root fan-in of `rows*cols`, at the cost of one extra communication round. On a 1-D group (one row or one column) there is only one axis to reduce, so it collapses to the single gather-to-root path. |
| `unicast_all_gather` | Every core unicasts its local block to every other core, then every core reduces the gathered blocks. | Simple but creates quadratic unicast traffic. |

### Measured nulls — kept for the record, NOT in the default sweep

Consistently **~4x off** the best on-chip reducer at every size measured (see *Measured result*),
so they sit outside `--variant recommended` (the default) and are never the comparison baseline.
They are not broken — they are the wrong SHAPE for an on-chip group, which is the point of keeping
them: a ring pays a hop per step where the NoC can reach every core directly, and a
rotating-sender all-gather serializes `G` broadcast rounds where one gather-plus-broadcast would
do. Both are the right answer BETWEEN chips, where the topology really is a ring and there is no
all-to-all fabric — that is why they are worth recognizing, and worth not reaching for on-chip.
Re-confirm with `--variant all`.

| Variant | What it does | Why it loses on-chip |
|---|---|---|
| `mcast_all_gather` | The sender rotates through the group; each round multicasts one core's block to every member. | Replaces each all-to-all unicast round with one multicast. |
| `ring_pull` | Each core reads the partial sum from its previous neighbor, adds locally, and exposes the result for the next pull. | Tests remote reads against remote writes with the same reduce-and-forward ring. |
| `ring_push` | A serpentine ring unicasts a partial sum to its next neighbor; the receiver adds its local block before forwarding. Ready and consumed counters protect each hop. | Linear hop count, with communication and compute synchronized at every hop. |


**`reduce_scatter_mcast` and `tree_reduce_mcast` STACK — they partition different things.**
Reduce-scatter partitions the **output**: which core is responsible for reducing which tiles, so the
reduction work itself is shared instead of landing on one root. The tree partitions the **fan-in**:
how contributions travel to whoever is reducing, turning one `rows*cols` gather into `cols` then
`rows`. Because one splits *who does the work* and the other splits *how the data arrives*, they are
orthogonal and compose: run a reduce-scatter within each grid row, then a tree across rows for each
output slice, and you get both a per-core reduction cost of `O(G / workers)` and a fan-in of
`O(cols + rows)` instead of `O(G)`. The variants below measure them SEPARATELY — the combined form is
not one of them.

The naming matters because these are the two independent axes of any on-chip all-reduce, and an op
usually needs to ask about both. If one core is doing reduction work the rest are not, that is a
reduce-scatter question; if the cost is the gather rendezvous itself, that is a tree question. See
`/perf-measure`'s core-balance check for which one your op is hitting.

Every variant uses the same reducer: contributors are paired with FPU `add_tiles`, accumulated
directly in FP32 DST via `acc_to_dest=true`, and packed once after the full reduction. The live
output batch comes from the JIT-derived `DEST_AUTO_LIMIT`. Odd contributor counts seed DST with one
copied block first.

### Ragged tile splits are the normal case

`reduce_scatter_mcast` hands worker `i` the tile indices `i, i+W, i+2W, ...`, so any `num_tiles` that
is not a multiple of the worker count `W` leaves some workers owning one tile more than the rest.
This is the common situation, not an edge case, and it costs nothing measurable - a ragged
20-tiles-over-8-workers point lands on the straight line through its even 16- and 24-tile
neighbours, well inside run-to-run spread (see `report_bh_p150b_1d.md`). Ragged group *widths*
behave the same: 16 tiles over an 11-, 10- or 5-core line all measure and verify cleanly.

Two properties make that hold, and both transfer to any op distributing a ragged split over a CB:

- **Size the CBs at a uniform quantum, not a per-worker one.** The gather and partial CBs are
  allocated at `max_assigned = ceil(num_tiles / W)` and *every* worker pushes and pops exactly that,
  with gather stride `contributor * max_assigned`, whatever its own share is. A CB's capacity must be
  an exact multiple of its push/pop quantum, so a per-worker quantum would wrap at a different offset
  on every core and contributors would land on each other's slots; a uniform one pins each
  contributor's slot to a fixed address. The pad slots a short worker leaves are neither read nor
  written back, so they cost a little L1 and nothing else.
- **Put every core to work, the root included.** `W = min(num_tiles, group_size)` - the root takes a
  worker share and only then multicasts, writing its own partials to itself over NoC loopback so the
  sender's `wait_min` still counts one increment per worker per iteration. Reserving the root would
  idle `1/G` of the group for the whole gather/reduce phase *and* leave `W` coprime with power-of-two
  tile counts, which is the more ragged split rather than the less.

## L1 pressure - what each topology costs per core

The reducers do not just differ in speed; they differ in how much L1 each core has to give up,
and by a much larger factor. With `T` = tiles/core, `G` = group size, `P` = tile bytes (2048 for
bf16), `W = min(T, G)` workers and `A = ceil(T / W)`:

| Variant | CB | Allocated on | Per-core bytes |
|---|---|---|---|
| `reduce_root_mcast` | `CB_GATHER` | **every core in the group** | `G * T * P` |
| `tree_reduce_mcast` | `CB_GATHER` | every core | `cols * T * P` |
| | `CB_PARTIAL` + `CB_STAGE2` | row leaders only | `+ (1 + rows) * T * P` |
| `reduce_scatter_mcast` | `CB_GATHER` + `CB_PARTIAL` | **worker cores only** | `(G + 1) * A * P` |

`CB_OUTPUT` is zero-copy over the sharded output tensor (`cb_descriptor_from_sharded_tensor`), so it
adds nothing; the input and output shards themselves cost `T * P` each out of the same L1.

**Why the gather buffer is group-wide for the root reducers and worker-local for reduce-scatter** -
this is the whole story, and it is a *direction* difference, not an accounting one:

- The root reducers **push**: a contributor writes its block into the root's gather buffer with
  `get_noc_addr(root_x, root_y, gather_addr + my_index * payload_bytes)`, where `gather_addr` is its
  own local `get_write_ptr()`. That only resolves to the right place because the CB is allocated
  identically on every core in the group, so the local address equals the root's address. The
  symmetric `G * T * P` allocation is the addressing mechanism - non-root cores are not wasting it,
  they are relying on it.
- Reduce-scatter **pulls**: a worker reads each contributor's shard straight out of the input tensor
  (already symmetric by virtue of being sharded) into a buffer nobody else addresses. So that buffer
  is purely local, exists only on the `W` worker cores, and holds only the `1/W` slice of tiles that
  worker owns - `G` copies of `A` tiles instead of `G` copies of all `T`.

For `T >= G` that collapses to `(G+1) * (T/G) * P ~= T * P`: **reduce-scatter's per-core footprint is
essentially independent of group size**, where the root reducers' grows linearly with it. Tree reduce
lands in between - `rows + cols` instead of `rows * cols` - and is cheapest on a square group.

| Config | `reduce_root_mcast` | `tree_reduce_mcast` (leader) | `reduce_scatter_mcast` (worker) |
|---|---:|---:|---:|
| `1x8`, T=8 | 128 KiB | = root on 1-D | **18 KiB** |
| `1x8`, T=16 | 256 KiB | = root on 1-D | **36 KiB** |
| `1x8`, T=32 | 512 KiB | = root on 1-D | **72 KiB** |
| `2x8`, T=6 | 192 KiB | 132 KiB | **34 KiB** |
| `4x4`, T=6 | 192 KiB | 108 KiB | **34 KiB** |

### Measured ceiling

Largest `T` that fits before `Statically allocated circular buffers ... clash with L1 buffers`
(Blackhole p150a, 1.5 MB L1/core, bf16, `--num-groups 1`; bracketed by the first size that fails):

| Group | `reduce_root_mcast` | `tree_reduce_mcast` | `reduce_scatter_mcast` |
|---|---|---|---|
| `1x8` (G=8) | 70 (72 fails) | = root on 1-D | **224** (232 fails) |
| `4x4` (G=16) | 36 (40 fails) | 64 (72 fails) | **>= 224** |

So reduce-scatter carries **3.2x** the payload of flat root on an 8-core line and **6.2x** on a
16-core square, and tree reduce **1.8x**. Note what binds at the top end: past roughly 200 tiles/core
reduce-scatter's own CB is no longer the limit - the `2 * T * P` of input and output shards is - which
is why its ceiling stops tracking `G` and flattens out near the same number for both group shapes.

Practical consequence: **L1 headroom, not speed, is usually what rules the root reducers out first.**
If an op needs to keep other buffers resident on the same cores, the gather buffer that scales as
`G * T` is the first thing to go.

## Push or pull the gather?

`reduce_scatter_push` mirrors `reduce_scatter_mcast` with the transfer reversed - contributors write
into the owning worker instead of each worker reading from every contributor. Measured, it is a
second-order choice next to the topology decision above:

- **No L1 difference, and none is required.** The gather buffer's size follows the *work* split, not
  the direction, so both peak at `(G + 1) * A * P` and both top out at 224 tiles/core on `1x8`.
- **It is not contention.** `--collect-noc-traces` replayed through tt-npe puts the congestion cost at
  *exactly 0 cycles* for both directions at both sizes, with average link utilization under 11%.
- **Pull does less work.** A census of those traces shows byte-for-byte identical payload plus 512
  extra `SEMAPHORE_INC` for push (`group_size * num_workers` per group) - the handshake pull never
  needs, because it reads the immutable input tensor and no contributor-ready notification exists.
- **Pull is faster wherever it is well behaved** (2721 vs 3080 ns at 8 tiles/core), but its timing goes
  erratic above ~24 tiles/core - 36 tiles/core runs **1.46x slower than 48**, with noise to 10.4% -
  where push stays monotone at ~247 ns/tile and under 1.5%. That instability, not a push advantage, is
  what can look like a crossover. Its cause is not established here.

Prefer **pull** on groups with 4 or more grid rows; take **push** when predictability matters more than the median. Compare the two only
at `--kernel-iters 1`: push has no consumer back-pressure, so in-kernel repeats let contributors run
ahead of the workers and inflate it for reasons unrelated to steady-state throughput.

The large-payload sweep (BH p150b, 2026-10-26, below) shows where pull's instability starts. It depends
on group shape, not only on tiles per core. On **1-D lines of 10 or more Tensix cores** (`1x13`, `10x1`)
pull goes erratic from **16 tiles per Tensix core**: per-trial spread is 10–40 %, and it loses to root
reduction (0.66–0.97×). On the same lines push's spread stays under 1.5 %, and it keeps a 1.19–1.35×
lead over root. On `1x8` lines the switch comes at 32 tiles per Tensix core (pull 0.94×, push 1.30×),
and on 2-row groups (`2x8`, `2x13`) push is faster from 32 tiles per Tensix core (`2x13` from 8). On groups with 4 or more
rows (`4x4`, `5x6`) pull is the faster form through 32 tiles per Tensix core (`4x4` through 128; `5x6` push
edges ahead at 64). So use **push on lines and 2-row groups past ~16 tiles per Tensix core, and pull on
groups with 4 or more rows**. The cause of pull's spread is still not established;
the sweep only shows it is worst where every worker's reads share one row of links.

## Signal after a flush, not a barrier

Each reduce-scatter worker writes its partial tiles into the root's output shard and then bumps the
root's `progress` semaphore. The push form also does this for its gather: contributor writes, then each
worker's `gathered` semaphore. The barrier twins put `noc_async_write_barrier()` between the two calls.
That waits until every write is **acknowledged**, one full round trip to the destination and back. The
`_flush` twins use `noc_async_writes_flushed()` instead, which waits only until the writes have **left**
this Tensix core, and then issue the increment.

**Why the increment still lands after the data.** The NoC ordering rules
(`docs/source/tt-metalium/tt_metal/apis/kernel_apis/data_movement/ordering.rst`, *Ordering (NOC,
CMD_BUF, VC)*) state: writes on different NoCs have no ordering guarantee; writes on the same NoC but
different VCs have none either; *"if writes are on the same NoC and same VC, they will be ordered based
on program order, regardless whether you use the same CMD_BUF or different ones."* The semaphore
increment is an atomic issued from `AT_CMD_BUF`, and the data goes through `WR_CMD_BUF`. The doc's
"regardless of CMD_BUF" clause covers that difference. Both default to the same static VC:
`noc_async_write` and `Semaphore::up(noc, x, y, n)` (and the `noc_semaphore_inc` it calls) all take
`vc = NOC_UNICAST_WRITE_VC` (VC 1) with `NOC_CMD_VC_STATIC`. So the flush pattern is correct only when
all of the following hold:

1. **Same NoC.** The data and the increment use the same `noc` argument. Here both use the kernel's
   default NoC.
2. **Same VC, statically allocated.** No explicit `vc` override on either call. A **multicast** data
   write rides `NOC_MULTICAST_WRITE_VC` (VC 4), so a multicast write followed by a unicast increment is
   **not** covered.
3. **Same destination Tensix core.** The doc does not state this condition, but it is the only case where "lands
   after" means anything: the increment must target the Tensix core that received the data. Here it
   always does: the partials go to the root and so does its `progress` increment, and the gather pages go to a
   worker and so does that worker's `gathered` increment.
4. **The source buffer can be reused after a flush.** This is what `noc_async_writes_flushed()`
   guarantees ("depart, but will not wait for them to complete"). Popping the partial CB right after it is safe.
5. **Retire the acknowledgements before exit.** A flush is not completion, so the `_flush` kernels call
   `noc_async_write_barrier()` once at kernel end (`retire_writes_at_exit()`). The flush variants pass
   under `--dev` (watcher NoC sanitizer) at 1 and 32 tiles per Tensix core with kernel-iters 10.

**Measured** (BH p150b `bh-50`, AICLK 1350 MHz, kernel-iters 1, interleaved A/B, median of 11). Barrier
twin → flush twin, ns per all-reduce:

| Group (members) | Tiles per Tensix core | pull: barrier → flush | push: barrier → flush |
|---|---:|---|---|
| `1x8` (8), 1 group | 1 | 1 611 → 1 504 ns **1.07×** | 1 493 → 1 260 ns **1.19×** |
| | 6 | 2 307 → 2 201 ns **1.05×** | 2 596 → 2 381 ns **1.09×** |
| | 32 | 11 041 → 11 068 ns **1.00×** | 8 397 → 8 218 ns **1.02×** |
| `1x13` whole row (13), 1 group | 1 | 1 901 → 1 801 ns **1.06×** | 1 748 → 1 506 ns **1.16×** |
| | 6 | 3 273 → 3 173 ns **1.03×** | 3 523 → 3 309 ns **1.07×** |
| | 32 | 17 389 → 17 400 ns **1.00×** | 12 799 → 12 585 ns **1.02×** |
| `2x13` two rows (26), 1 group | 1 | 2 680 → 2 502 ns **1.07×** | 2 577 → 2 255 ns **1.14×** |
| | 6 | 4 055 → 3 893 ns **1.04×** | 4 053 → 3 774 ns **1.07×** |
| | 32 | 27 045 → 27 758 ns **0.97×** (pull noisy) | 13 883 → 13 553 ns **1.02×** |
| `1x4` (4), 1 group | 1 | 1 366 → 1 265 ns **1.08×** | 1 316 → 1 084 ns **1.21×** |

The full table, which also covers `10x1` and every shape packed across the whole grid, is in
[`report.md`](report.md). **Reading:** the saving is a nearly constant amount per fenced signal, about
60–180 ns, larger for longer paths (`2x13` is the largest). The pull form has one fenced signal per
all-reduce and saves ~0.1 µs. The push form has two and saves ~0.2–0.3 µs. Because the saving is a
fixed amount, the ratio depends only on how short the all-reduce is. It helps latency-bound
all-reduces. At 1–6 tiles per Tensix core, isolated groups give pull 1.03–1.08× and push 1.06–1.21×.
Packed across the grid the range is pull 1.02–1.15× and push 1.01–1.18×, with wider spread. Group size
(4–26 members) does not change this. By 32 tiles per Tensix core the lever is a null (0.97–1.05×). The lever has no L1 cost and no
correctness cost under the conditions above. Use it whenever the data write and the signal share a
NoC, VC and destination; expect it to matter only when the all-reduce is a few microseconds.

## Large payloads - where root reduction catches up

The 1-D sweep in [`report_bh_p150b_1d.md`](report_bh_p150b_1d.md) found reduce-scatter's advantage
over root reduction gone by 32 tiles per Tensix core on `1x8` lines. This sweep extends it to longer
lines, 2-D groups and up to 128 tiles per Tensix core (256 does not fit any reduce-scatter
group, because its input and output shards alone take 1 MiB). It runs on BH p150b `bh-50` (13×10 grid,
AICLK 1350 MHz), with an interleaved A/B and the median of 5. Baseline `reduce_root_mcast` → lever
`reduce_scatter_mcast` (pull) at kernel-iters 10. ns per all-reduce; *1 group* = isolated, *full grid*
= the shape packed across all 130 Tensix cores.

| Group (members) | Tiles per Tensix core | 1 group: root → pull | full grid: root → pull |
|---|---:|---|---|
| `1x8` (8) | 8 / 16 | **1.65× / 1.63×** | **1.61× / 1.55×** |
| | 24 / 32 | **1.23× / 1.15×** | **1.16× / 0.97×** |
| | 48 / 64 | **1.05× / 1.21×** | **0.95× / 1.05×** |
| `1x13` whole row (13) | 8 | 4 855 → 3 034 ns **1.60×** | **1.44×** |
| | 16 / 32 | 8 916 → 10 821 ns **0.82×** / 17 238 → 23 011 ns **0.75×** | **0.76× / 0.66×** |
| `10x1` whole column (10) | 8 | **1.63×** | **1.60×** |
| | 16 / 32 / 48 | **0.97× / 0.87× / 0.89×** | **0.94× / 0.77× / 0.81×** |
| `2x8` (16) | 8 / 16 / 32 | **2.30× / 2.88× / 2.15×** | **0.98× / 1.38× / 1.51×** |
| `4x4` (16) | 8 / 16 / 32 | **2.48× / 3.73×** / 19 897 → 4 930 ns **4.04×** | **1.38× / 1.32× / 1.07×** |
| `2x13` two rows (26) | 8 / 16 / 24 | **2.29× / 1.55× / 1.48×** | **1.24× / 1.14× / 1.16×** |
| `5x6` (30) | 8 / 16 | **3.15×** / 18 355 → 3 942 ns **4.66×** | **1.90× / 1.56×** |

Past the last column of each row, root reduction **does not fit in L1**. Its gather buffer on every
Tensix core plus the input and output shards need `(G + 2) · T` bf16 tiles: it fits at 1 344 KiB and
fails at 1 440 KiB, so the limit is about `(G + 2) · T ≤ 670` tiles. Reduce-scatter runs to 128 tiles
per Tensix core on every shape.

At kernel-iters 1 (latency) with the push form added, isolated groups (root → pull | root → push):
`1x8` 32 tiles 0.94× | **1.30×**, 64 tiles 1.40× | 1.34×; `1x13` 16 tiles 0.88× | **1.22×**, 32 tiles
1.04× | **1.35×**; `10x1` 16 tiles 0.78× | **1.19×**; `2x8` 32 tiles 1.36× | **2.21×**; `2x13` 16 tiles
1.58× | **2.14×**; `4x4` 32 tiles **3.67×** | 2.85×; `5x6` 16 tiles **4.13×** | 3.38×. The full
tables are in [`report.md`](report.md).

**Reconciling with "gone by 32 tiles per core":**
- **It holds only for the pull form on 1-D lines.** On `1x8` packed 10 groups to a grid, pull
  reaches parity at 32–48 tiles per Tensix core (0.95–0.97×), which reproduces the earlier claim. Isolated, it is 1.15×
  at 32 and 1.05× at 48. On lines of 10 or more Tensix cores, pull is already gone by **16** tiles per Tensix core
  (0.66–0.97×).
- **It does not hold for 2-D groups.** With 2 or more rows, reduce-scatter wins at every payload where
  root reduction fits: 1.5–4.7× isolated and 1.07–1.9× packed across the grid. The single exception
  is `2x8` packed at 8 tiles (0.98×). The gain does not shrink with tiles on `4x4` (2.5× → 4.0× from
  8 to 32 tiles per Tensix core).
- **It does not hold for the push form on lines.** Push keeps 1.19–1.35× over root on `1x8`,
  `1x13` and `10x1` through 32 tiles per Tensix core (64 on `1x8`).
- **Root reduction has no large-payload regime at all on big groups.** Its time is linear in
  members × tiles, about 40 ns per member-tile (`1x8` 64 tiles: 20.9 µs; `5x6` 16 tiles: 18.4 µs),
  and its L1 runs out at `(G + 2) · T ≈ 670` tiles. A 23-member group at ~200 tiles per Tensix core
  needs ~4 600 tiles (9 MiB) per Tensix core, so a root combine measured there must be a streamed or
  chunked one, which is not this variant. Extrapolating this root's linear cost to 23 × 200 member-tiles
  gives ≈ 184 µs. Reduce-scatter on 2-D groups measured 32–47 µs at 128 tiles per Tensix core (`5x6`
  push, `2x13` push), which is ≈ 50–73 µs scaled to 200 tiles, i.e. ≈ 2.5–3.7×. That is an
  extrapolation, not a measurement, but it is consistent in direction and size with a ~3.3× gain on a
  2-D group (or a push-form line).

## CLI - measure your own shapes and parameters

```bash
python -m ttnn.operations.examples.tensix_all_reduce [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `recommended` \| `all` \| any of `{reduce_root_mcast,reduce_scatter_mcast,reduce_scatter_push,reduce_scatter_mcast_flush,reduce_scatter_push_flush,tree_reduce_mcast,unicast_all_gather,mcast_all_gather,ring_pull,ring_push}` | `recommended` | methods to run. `recommended` skips the measured nulls; `all` includes them. |
| `--group-shape` | `ROWS,COLS` | grid-derived sweep | rectangular shape of each group |
| `--num-groups` | int | `1` | equal groups packed row-major into the worker grid |
| `--num-tiles` | int | `6` | bf16 tiles contributed by every core |
| `--trials` | int | `5` | retained profiler trials after warmup |
| `--kernel-iters` | int | `1` | collectives per kernel launch; `1` measures latency, larger values measure steady state |
| `--report` | path | print only | also write the Markdown report |

```bash
# Compare all methods on four independent two-row groups.
python -m ttnn.operations.examples.tensix_all_reduce \
  --group-shape 2,8 --num-groups 4 --num-tiles 6 --kernel-iters 10

# Compare ring reads and writes on sixteen half-row groups.
python -m ttnn.operations.examples.tensix_all_reduce \
  --variant ring_push ring_pull --group-shape 1,4 --num-groups 16

# Tree reduce against the two root-based reducers on a 4x4 group.
python -m ttnn.operations.examples.tensix_all_reduce \
  --variant reduce_root_mcast reduce_scatter_mcast tree_reduce_mcast \
  --group-shape 4,4 --num-groups 4 --num-tiles 6 --kernel-iters 10
```

## Measured result

Illustrative results from the stamped Wormhole B0 box above. Each entry is the median of five
trials with ten in-kernel all-reduces; times are divided by ten. All placements use 64 cores and
six bf16 tiles per core. `Std / median` values at or above 5% are marked noisy.

| Placement | Group | Groups | Method | Median ns/all-reduce | Std / median | vs best |
|---|---:|---:|---|---:|---:|---:|
| whole rows | 1x8 | 8 | ring push | 16908.2 | 1.6% | 1.00x |
| | | | ring pull | 17629.0 | 0.3% | 0.96x |
| | | | unicast all-gather | 22961.9 | 0.9% | 0.74x |
| | | | multicast all-gather | 8938.1 | 0.0% | 1.89x |
| | | | reduce-root + multicast | 6178.9 | 1.4% | 2.74x |
| | | | reduce-scatter + multicast | **3647.3** | 1.1% | **4.64x** |
| whole columns | 8x1 | 8 | ring push | 17315.6 | 1.0% | 1.00x |
| | | | ring pull | 17699.2 | 0.3% | 0.98x |
| | | | unicast all-gather | 23140.9 | 0.8% | 0.75x |
| | | | multicast all-gather | 9283.9 | 0.4% | 1.87x |
| | | | reduce-root + multicast | 6229.5 | 1.4% | 2.78x |
| | | | reduce-scatter + multicast | **3664.3** | 0.7% | **4.73x** |
| half rows | 1x4 | 16 | ring push | 8575.7 | 1.4% | 1.00x |
| | | | ring pull | 7296.4 | 0.6% | 1.18x |
| | | | unicast all-gather | 9384.7 | 0.2% | 0.91x |
| | | | multicast all-gather | 8139.9 | 19.9% (noisy) | 1.05x |
| | | | reduce-root + multicast | **4004.9** | 0.2% | **2.14x** |
| | | | reduce-scatter + multicast | 5505.0 | 16.8% (noisy) | 1.56x |
| two rows | 2x8 | 4 | ring push | 54182.1 | 3.6% | 1.00x |
| | | | ring pull | 47771.6 | 0.5% | 1.13x |
| | | | unicast all-gather | 58044.7 | 0.9% | 0.93x |
| | | | multicast all-gather | 20547.6 | 2.6% | 2.64x |
| | | | reduce-root + multicast | 12324.6 | 3.2% | 4.40x |
| | | | reduce-scatter + multicast | **8364.3** | 9.8% (noisy) | **6.48x** |

**Reading of the result:** reduce-scatter wins the 8- and 16-core placements because each worker owns
complete tile reductions; the gather is neither replicated on every core nor serialized at one
root. For four-core groups its extra worker/root phase is not amortized, and root reduction wins.
Rotating multicast beats both rings on the 8- and 16-core shapes. The 16-core serpentine ring is
especially costly because its reverse row fights NoC0's preferred direction. Push is slightly
faster on 8-core lines, while pull wins on the 4- and 16-core rectangles.

## Measured result - the 2-D reducers depend on the regime

Blackhole box (`bh-50-special-mstaletovic-for-reservation-48229`, 2026-07-20), median of five
trials, ten in-kernel all-reduces. Comparing the three reducers that make sense on a 2-D grid -
`reduce_root_mcast` (flat), `reduce_scatter_mcast` (tile-index workers), and `tree_reduce_mcast`
(grid-axis hierarchy). There is **no single winner**: it flips with **payload** (tiles per core) and
**NoC contention** (how many groups share the worker grid). `Std / median` >= 5% marked noisy.

**Isolated single group (`--num-groups 1`, 16 cores), 6 tiles/core:**

| Group | `reduce_root_mcast` | `reduce_scatter_mcast` | `tree_reduce_mcast` | best |
|---:|---:|---:|---:|---|
| 2x8 | 4529.2 | **2286.8** | 3344.2 | reduce-scatter, 1.98x vs root |
| 8x2 | 4559.3 | **2335.9** | 3394.2 | reduce-scatter, 1.95x |
| 4x4 | 4555.9 | **2250.7** | 2867.5 | reduce-scatter, 2.02x |

**Same isolated group, 1 tile/core (latency floor):**

| Group | `reduce_root_mcast` | `reduce_scatter_mcast` | `tree_reduce_mcast` | best |
|---:|---:|---:|---:|---|
| 2x8 | 1498.1 | 1981.0 | **1377.3** | tree reduce, 1.09x vs root |
| 8x2 | 1532.0 | 1975.1 | **1361.9** | tree reduce, 1.12x |
| 4x4 | 1547.1 | 1992.4 | **1271.0** | tree reduce, 1.22x |

**Grid-filling (groups packed across the 13x10 grid -> NoC contention), 6 tiles/core:**

| Group | Groups | Cores | `reduce_root_mcast` | `reduce_scatter_mcast` | `tree_reduce_mcast` | best |
|---:|---:|---:|---:|---:|---:|---|
| 2x8 | 5 | 80 | 5338.8 | 6443.0 *(noisy)* | **3641.3** | tree reduce, 1.47x vs root |
| 8x2 | 6 | 96 | 6202.9 *(noisy)* | 6716.7 *(noisy)* | **3877.3** | tree reduce, 1.60x |
| 4x4 | 6 | 96 | 5208.5 | 4896.3 *(noisy)* | **3584.4** | tree reduce, 1.45x |

**Reading of the result - pick the reducer by regime:**
- **`reduce_scatter_mcast` wins an isolated group that has real payload** (16 cores, 6 tiles: ~2x
  over root) because it parallelizes the reduction across tile indices. But it needs tiles to
  parallelize - useless at 1 tile (`min(num_tiles, group_size)` = 1 worker, plus a wasted
  worker->root handoff, so it is the *worst* there) - and it is **contention-sensitive**: from 1
  group to 5 it goes 2286.8 -> 6443.0 ns and its noise jumps from ~1% to 15-28%.
- **`tree_reduce_mcast` is the robust default.** Its per-axis traffic is localized (each gather's
  fan-in is only `cols`, then `rows`, never the whole `rows*cols`), so it is the **steadiest** in
  every regime (<1% noise) and barely moves under contention (3344 -> 3641 ns from 1 -> 5 groups). It
  wins under **grid-filling contention** (1.45-1.60x over root, and clear of the now-inflated
  reduce-scatter) and at the **1-tile latency floor**, and it is never worst. The cost is one extra
  communication round, which is why it does *not* beat tile-index reduce-scatter in an isolated,
  well-fed group.
- **`reduce_root_mcast` is the simple fallback** - one root serializes the whole `rows*cols` gather,
  so it is never fastest but also never blows up; moderate and fairly steady.

So: reach for **tree reduce** when the grid is busy (many concurrent groups) or the payload is
tiny; reach for **tile-index reduce-scatter** for an isolated group with several tiles per core. On a 1-D
group (single row or column) tree reduce has only one axis to reduce and collapses to the root
path.

## Run the predefined sweep

```bash
AR_KERNEL_ITERS=10 AR_TRIALS=5 \
AR_REPORT=ttnn/ttnn/operations/examples/tensix_all_reduce/report.md \
scripts/run_safe_pytest.sh --run-all \
tests/ttnn/unit_tests/operations/examples/test_tensix_all_reduce.py::test_tensix_all_reduce_device_perf
```

Correctness is the only pass/fail condition. Performance is recorded as evidence and is never
asserted.

## Code

The complete host descriptor and inline dataflow/compute kernels are in
`program_descriptor_with_inline_kernels.py`; `__main__.py` provides the CLI.

## When it helps / when it does not
**Rule:** `reduce_scatter_mcast` beats `reduce_root_mcast` when the group has ≥ 8 Tensix cores **and** each
Tensix core contributes ≥ ~4 tiles (enough serial tile-adds on the root to split across workers); groups of
4–6 Tensix cores and 1-tile payloads lose on both boards. Payload crossover on a `1x8` line: between 1 and
6 tiles per core on WH B0 n150 (no points in between), 2–4 tiles per core on BH (`bh-49` sweep).
BH p150b gains less where it helps (1.43× vs 1.68×) and loses more where it does not.
**Upper boundary (BH p150b `bh-50`):** the gain fades with tiles per Tensix core **only for the pull form on
1-D lines**. On `1x8` it reaches parity at 32–48 tiles per Tensix core when packed across the grid, and on
lines of ≥ 10 Tensix cores it loses from 16 tiles per Tensix core (0.66–0.97×). There, switch to
`reduce_scatter_push` (1.19–1.35× over root through 32 tiles per Tensix core). Groups with 2 or more rows keep
1.5–4.7× isolated, and 1.07–1.9× packed across the grid, up to root reduction's L1 limit
`(G + 2) · T ≈ 670` tiles. See *Large payloads - where root reduction catches up*.

Baseline `reduce_root_mcast` → lever `reduce_scatter_mcast`; bf16; "6 tiles" = tiles per Tensix core;
parentheses = min–max over the retained trials.

| Regime (group × groups, tiles per Tensix core) | WH B0 n150 · 8×8 = 64 Tensix cores | BH p150b · 12×10 = 120 Tensix cores | BH P150b `bh-49` · 1-D sweep ([`report_bh_p150b_1d.md`](report_bh_p150b_1d.md)) |
|---|---|---|---|
| **Helps:** 8-Tensix-core line `1x8` × 8, 6 tiles | 6 145 (6 133–6 153) → 3 646 (3 600–3 708) ns **1.68×** | 2 592 (2 584–2 598) → 1 817 (1 798–1 844) ns **1.43×** | 4 tiles: 1 928 → 1 550 ns **1.24×**; 8 tiles: 3 190 → 2 300 ns **1.39×** |
| **Helps:** 8-Tensix-core line `1x8` × 8, 16 tiles | not measured | not measured | 5 682 → 3 527 ns **1.61×** (best ratio measured) |
| **Does not help:** `1x8` × 8, 1 tile (one worker) | 1 614 (1 603–1 623) → 1 994 (1 990–2 000) ns **0.81×** | 990 (988–992) → 1 326 (1 324–1 327) ns **0.75×** | 991 → 1 330 ns **0.75×**; 2 tiles **0.94×** |
| **Does not help:** `1x8` × 8, 32 tiles (bandwidth-bound) | not measured | not measured | 10 689 vs 10 691 ns **1.00×** (all reducers converge) |
| **Does not help:** 4-Tensix-core line `1x4` × 16, 6 tiles | 3 976 (3 950–4 254) → 4 663–5 120 (4 349–5 557) ns **0.78–0.85×** (lever noisy) | 2 086 (1 991–2 739) → 3 812 (3 158–4 390) ns **0.55×** (lever noisy) | not measured |
| **Helps:** full grid, whole rows, 6 tiles | `1x8` × 8: 6 179 → 3 647 ns **1.69×** | `1x12` × 10: 3 684 → 3 087 ns **1.19×** | not measured |
| **Helps:** full grid, whole columns, 6 tiles | `8x1` × 8: 6 230 → 3 664 ns **1.70×** | `10x1` × 12: 3 198 → 2 548 ns **1.26×** | not measured |
| **Does not help:** full grid, half rows, 6 tiles | `1x4` × 16: 4 005 → 5 505 ns **0.73×** (lever noisy) | `1x6` × 20: 2 816 → 2 936 ns **0.96×** | not measured |
| **Helps (vs root):** full grid, two rows, 6 tiles | `2x8` × 4: 12 325 → 8 364 ns **1.47×** (lever noisy) | `2x12` × 5: 9 401 → 6 599 ns **1.42×** (both noisy); `tree_reduce_mcast` 5 171 ns is fastest | not measured |
| **Boundary** | group: 4 → 8 Tensix cores; payload: 1 → 6 tiles | group: 6 → 8 Tensix cores (`1x6` 0.96×); payload: 1 → 6 tiles | payload: 2–4 tiles; gone by 32 tiles (pull form on lines only: ≥ 10-Tensix-core lines from 16 tiles, 2-D groups never, push form never — `bh-50` sweep) |

**Why the boards differ:**
- The root's serial reduction shrinks more on BH p150b than reduce-scatter's fixed worker-to-root handoff
  and second synchronization round: at `1x8`, 6 tiles, `reduce_root_mcast` is 2.4× faster than on WH B0
  n150 (2 592 vs 6 145 ns; AICLK 1350 vs 1000 MHz, 64-byte vs 32-byte NoC flit) but `reduce_scatter_mcast`
  only 2.0× (1 817 vs 3 646 ns). Less serial work left to split → smaller gain, and bigger relative loss
  where there is nothing to split (`1x4`: 0.55× vs 0.78–0.85×).
- Longer lines on the full BH p150b grid (`1x12`, `10x1`) still win, but only 1.19–1.26× (at
  kernel-iters 1, i.e. latency, vs 10 on WH B0).

Reduce-scatter pays for an extra worker-to-root handoff and a second synchronization round, and only earns
that back when the root's serial reduction (`group_size - 1` blocks × tiles per Tensix core) is long enough
to split across `min(tiles per core, group_size)` workers: at 8 Tensix cores and 6 tiles the root does
7 × 6 tile-adds serially and reduce-scatter wins (1.68× WH B0, 1.43× BH p150b); at 4 Tensix cores (3 × 6)
or at 1 tile per core (one worker, so no parallelism to gain) the handoff is pure overhead and root
reduction is 1.2–1.3× faster on WH B0, 1.3–1.8× on BH p150b. The 4-core reduce-scatter times are noisy on
both boards (spread up to ~25% on WH B0, ~30% on BH p150b); the root times are stable. Root reduction wins
4-core groups on both boards; on a busy 2-D grid prefer `tree_reduce_mcast` (see *Measured result - the
2-D reducers depend on the regime* above, and the BH p150b `2x12` row).

*How measured:*
- WH B0 n150: `test_tensix_all_reduce_regimes` (kernel-iters 10, interleaved A/B, median of 11 retained
  trials; three runs, ranges in [`report.md`](report.md)), 2026-10-12. Full-grid rows from the 2026-07-10
  `test_tensix_all_reduce_device_perf` sweep (`bgd-lab-t3003`, kernel-iters 10, N=5).
- BH p150b: the same regimes test (kernel-iters 10, 11 trials) for the first five rows; full-grid
  rows from `test_tensix_all_reduce_device_perf` (kernel-iters 1, N=5);
  `bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19.
- BH P150b `bh-49`: 1-D payload sweep, kernel-iters 10, N=5 (N=2–3 for 24/32 tiles).
- Other boards: `bh-50` 2-D groups (isolated and grid-filling, with `tree_reduce_mcast`) — see *Measured
  result - the 2-D reducers depend on the regime* above.

Reproduce. The first three regimes assert what held on both measured boards; the large-payload and
flush regimes were measured on BH p150b only, and a regime that needs a line longer than the grid's
row is skipped:
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_tensix_all_reduce.py -k regimes
# the sweeps behind the large-payload and flush tables (~100 programs; AR_SWEEP_REPORT=<path> appends the tables)
AR_SWEEPS=1 AR_SWEEP_TRIALS=11 scripts/run_safe_pytest.sh --run-all \
  tests/ttnn/unit_tests/operations/examples/test_tensix_all_reduce.py::test_tensix_all_reduce_flush_sweep
AR_SWEEPS=1 scripts/run_safe_pytest.sh --run-all \
  tests/ttnn/unit_tests/operations/examples/test_tensix_all_reduce.py::test_tensix_all_reduce_boundary_sweep
# latency form with the push lever: AR_BOUNDARY_KERNEL_ITERS=1 AR_BOUNDARY_GROUPINGS=isolated
#   AR_BOUNDARY_LEVERS=reduce_scatter_mcast,reduce_scatter_push AR_BOUNDARY_TILES=8,16,32,64,128
```
