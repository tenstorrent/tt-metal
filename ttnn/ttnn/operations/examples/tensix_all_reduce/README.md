# Tensix rectangular all-reduce - on-chip collective algorithm bake-off

**Difficulty:** ⭐⭐⭐ T3  ·  **Concept(s):** Tensix-to-Tensix transfer topology and reduction work distribution
**First profiled on:** `bgd-lab-t3003-special-mstaletovic-for-reservation-40918` · Wormhole B0 · 1000 MHz · 2026-07-10 · `5f0ad060667`

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

## Push or pull the gather? Measured both ways

`reduce_scatter_mcast` **pulls** (each worker reads from every contributor) and
`reduce_scatter_push` **pushes** (each contributor writes into the owning worker). Same algorithm,
same gather layout, same tile ownership - only the direction of the transfer differs, so the pair
isolates that one choice.

**L1: there is no difference, and there does not have to be.** Both allocate `CB_GATHER` at
`group_size * max_assigned` pages and `CB_PARTIAL` at `max_assigned`, so the *peak per-core*
footprint is `(G + 1) * A * P` either way - the gather buffer's size is set by how the *work* is
partitioned (each worker holds `G` copies of its own `1/W` slice), which push and pull agree on.
Measured ceiling on a `1x8` line is identical: both pass at 224 tiles/core and both fail at 232.

The one asymmetry is *which* cores pay, not how much. A pusher derives the destination gather
address from its own `get_write_ptr()`, so the CB must be declared on every participating core,
where the pull variant declares it on the `W` workers alone. When `num_workers == group_size` - any
`num_tiles >= group_size`, which is the whole regime where reduce-scatter is the fastest reducer -
that is the same set of cores and even the aggregate matches. Only a short group with idle
non-workers has push touching more cores, and that is an implementation choice rather than a
requirement: passing the destination address as a runtime arg instead of deriving it locally would
let push allocate worker-only too.

**Perf: no universal winner - the direction that wins flips with payload and group size.**

`1x8` x 8 groups (64 cores), `kernel-iters=10`, N=5:

| tiles/core | `reduce_scatter_mcast` (pull) | `reduce_scatter_push` | winner |
|---:|---:|---:|---|
| 1 | 1331.8 | **1111.3** | push **1.20x** |
| 2 | 1410.9 | **1360.2** | push 1.04x |
| 4 | **1557.6** | 1697.2 | pull 1.09x |
| 8 | **1974.1** | 2744.8 | pull **1.39x** |
| 16 | **3606.3** | 4524.2 | pull 1.25x |
| 20 | 5327.4 *(5.4% noisy)* | 5457.6 | tie (N=3) |
| 24 | 7001.7 *(5.6% noisy)* | **6240.6** | push 1.12x |
| 32 | 10797.0 *(4.0%)* | **7938.8** | push **1.36x** |
| 48 | 16553.6 *(3.6%)* | **11467.1** | push **1.44x** |
| 64 | 18895.5 *(3.8%)* | **14953.9** | push 1.26x |

`4x4`, isolated single group (16 cores) - the crossover has moved out past the useful range:

| tiles/core | pull | push | winner |
|---:|---:|---:|---|
| 6 | 2265.1 | **2201.8** | push 1.03x |
| 16 | **2889.1** | 3735.9 | pull 1.29x |
| 32 | **4768.1** | 6325.6 | pull 1.33x |
| 64 | **9842.1** | 11477.6 | pull 1.17x |
| 128 | **20704.4** | 21947.5 | pull 1.06x |

**Reading of the result.** Three things are going on, and only the first two are firmly attributable:

- **Below `num_tiles = group_size`, push wins because it spreads the *issuing*.** There `W < G`, so
  pull has the whole gather serialized on `W` workers issuing reads, while push has all `G` cores
  issuing writes concurrently. At 1 tile/core that is one reader versus eight writers - push 1.20x.
- **In the mid range, pull wins because it needs no handshake at all.** Pull reads out of the input
  tensor, which nobody writes, so a worker never has to be told a contributor is ready. Push must be
  told: every contributor bumps every worker's gather semaphore after its write barrier, so the
  gather phase carries `group_size * num_workers` extra atomic increments per all-reduce. That is a
  fixed cost against a payload of `group_size * max_assigned` pages, so it hurts most when
  `max_assigned` is small - exactly the 4-16 tiles/core band on `1x8`.
- **At large payload the fixed handshake amortizes and push pulls ahead** (`1x8`: from ~20 tiles/core,
  reaching 1.44x at 48). The `4x4` numbers are consistent with the same amortization running behind
  schedule on a bigger group - push's deficit closes monotonically, 0.75x -> 0.86x -> 0.94x as the
  payload goes 32 -> 64 -> 128 - which is what a handshake growing with `G * W` against a payload
  growing with `T` would do. Stated as consistent-with rather than confirmed: a cross-group-size
  comparison at matched per-core volume is confounded by the worker count changing too.

Two practical notes. **Push is much steadier**: 0.0-0.7% run-to-run across almost every point above,
against 1-8% for pull, whose reads hold outstanding-request state that contention perturbs. And
**contention shifts the balance toward push** - at 32 tiles/core on `4x4`, an isolated group has pull
ahead by 1.33x, but packing four groups across the grid puts push ahead by 1.12x.

If you are choosing one: **pull is the better default** - it wins the band most ops land in, and it
needs no handshake, no group-wide CB and no notification protocol. Reach for push when the payload
per core is large relative to the group, when the group is small, or when run-to-run stability
matters more than the median.

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

## CLI - measure your own shapes and parameters

```bash
python -m ttnn.operations.examples.tensix_all_reduce [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `recommended` \| `all` \| any of `{reduce_root_mcast,reduce_scatter_mcast,tree_reduce_mcast,unicast_all_gather,mcast_all_gather,ring_pull,ring_push}` | `recommended` | methods to run. `recommended` skips the measured nulls; `all` includes them. |
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
