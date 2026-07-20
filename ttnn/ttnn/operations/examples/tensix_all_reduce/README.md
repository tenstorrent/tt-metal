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
| `ring_push` *(baseline)* | A serpentine ring unicasts a partial sum to its next neighbor; the receiver adds its local block before forwarding. Ready and consumed counters protect each hop. | Linear hop count, with communication and compute synchronized at every hop. |
| `ring_pull` | Each core reads the partial sum from its previous neighbor, adds locally, and exposes the result for the next pull. | Tests remote reads against remote writes with the same reduce-and-forward ring. |
| `unicast_all_gather` | Every core unicasts its local block to every other core, then every core reduces the gathered blocks. | Simple but creates quadratic unicast traffic. |
| `mcast_all_gather` | The sender rotates through the group; each round multicasts one core's block to every member. | Replaces each all-to-all unicast round with one multicast. |
| `reduce_root_mcast` | Non-root cores unicast to one root; the root reduces all blocks and multicasts the result. | Cuts communication, but serializes all reduction work on the root. |
| `two_phase_reduce_mcast` | Up to `min(num_tiles, group_size - 1)` workers gather and reduce disjoint tile indices, write partials to the root, then the root multicasts the assembled result. | Keeps the reduced communication volume while parallelizing the root's compute and reads. |
| `two_stage_grid_reduce` | Hierarchical reduce over the 2-D core grid. Stage 1: within each grid row the `cols` cores gather to their row's leader, which reduces them to a per-row partial. Stage 2: the `rows` row-leaders gather to the group root, which reduces the partials into the group sum. Stage 3: the root multicasts the sum to the whole group. | Splits the reduction across the two grid axes, so each gather stage has a small fan-in (`cols`, then `rows`) instead of one all-to-root fan-in of `rows*cols`, at the cost of one extra communication round. On a 1-D group (one row or one column) there is only one axis to reduce, so it collapses to the single gather-to-root path. |

Every variant uses the same reducer: contributors are paired with FPU `add_tiles`, accumulated
directly in FP32 DST via `acc_to_dest=true`, and packed once after the full reduction. The live
output batch comes from the JIT-derived `DEST_AUTO_LIMIT`. Odd contributor counts seed DST with one
copied block first.

## CLI - measure your own shapes and parameters

```bash
python -m ttnn.operations.examples.tensix_all_reduce [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `{all,ring_push,ring_pull,unicast_all_gather,mcast_all_gather,reduce_root_mcast,two_phase_reduce_mcast,two_stage_grid_reduce}` | `all` | methods to run |
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

# Grid two-stage against the two root-based reducers on a 4x4 group.
python -m ttnn.operations.examples.tensix_all_reduce \
  --variant reduce_root_mcast two_phase_reduce_mcast two_stage_grid_reduce \
  --group-shape 4,4 --num-groups 4 --num-tiles 6 --kernel-iters 10
```

## Measured result

Illustrative results from the stamped Wormhole B0 box above. Each entry is the median of five
trials with ten in-kernel all-reduces; times are divided by ten. All placements use 64 cores and
six bf16 tiles per core. `Std / median` values at or above 5% are marked noisy.

| Placement | Group | Groups | Method | Median ns/all-reduce | Std / median | vs ring push |
|---|---:|---:|---|---:|---:|---:|
| whole rows | 1x8 | 8 | ring push | 16908.2 | 1.6% | 1.00x |
| | | | ring pull | 17629.0 | 0.3% | 0.96x |
| | | | unicast all-gather | 22961.9 | 0.9% | 0.74x |
| | | | multicast all-gather | 8938.1 | 0.0% | 1.89x |
| | | | reduce-root + multicast | 6178.9 | 1.4% | 2.74x |
| | | | two-phase + multicast | **3647.3** | 1.1% | **4.64x** |
| whole columns | 8x1 | 8 | ring push | 17315.6 | 1.0% | 1.00x |
| | | | ring pull | 17699.2 | 0.3% | 0.98x |
| | | | unicast all-gather | 23140.9 | 0.8% | 0.75x |
| | | | multicast all-gather | 9283.9 | 0.4% | 1.87x |
| | | | reduce-root + multicast | 6229.5 | 1.4% | 2.78x |
| | | | two-phase + multicast | **3664.3** | 0.7% | **4.73x** |
| half rows | 1x4 | 16 | ring push | 8575.7 | 1.4% | 1.00x |
| | | | ring pull | 7296.4 | 0.6% | 1.18x |
| | | | unicast all-gather | 9384.7 | 0.2% | 0.91x |
| | | | multicast all-gather | 8139.9 | 19.9% (noisy) | 1.05x |
| | | | reduce-root + multicast | **4004.9** | 0.2% | **2.14x** |
| | | | two-phase + multicast | 5505.0 | 16.8% (noisy) | 1.56x |
| two rows | 2x8 | 4 | ring push | 54182.1 | 3.6% | 1.00x |
| | | | ring pull | 47771.6 | 0.5% | 1.13x |
| | | | unicast all-gather | 58044.7 | 0.9% | 0.93x |
| | | | multicast all-gather | 20547.6 | 2.6% | 2.64x |
| | | | reduce-root + multicast | 12324.6 | 3.2% | 4.40x |
| | | | two-phase + multicast | **8364.3** | 9.8% (noisy) | **6.48x** |

**Reading of the result:** two-phase wins the 8- and 16-core placements because each worker owns
complete tile reductions; the gather is neither replicated on every core nor serialized at one
root. For four-core groups its extra worker/root phase is not amortized, and root reduction wins.
Rotating multicast beats both rings on the 8- and 16-core shapes. The 16-core serpentine ring is
especially costly because its reverse row fights NoC0's preferred direction. Push is slightly
faster on 8-core lines, while pull wins on the 4- and 16-core rectangles.

## Measured result - the 2-D reducers depend on the regime

Blackhole box (`bh-50-special-mstaletovic-for-reservation-48229`, 2026-07-20), median of five
trials, ten in-kernel all-reduces. Comparing the three reducers that make sense on a 2-D grid -
`reduce_root_mcast` (flat), `two_phase_reduce_mcast` (tile-index workers), and `two_stage_grid_reduce`
(grid-axis hierarchy). There is **no single winner**: it flips with **payload** (tiles per core) and
**NoC contention** (how many groups share the worker grid). `Std / median` >= 5% marked noisy.

**Isolated single group (`--num-groups 1`, 16 cores), 6 tiles/core:**

| Group | `reduce_root_mcast` | `two_phase_reduce_mcast` | `two_stage_grid_reduce` | best |
|---:|---:|---:|---:|---|
| 2x8 | 4529.2 | **2286.8** | 3344.2 | two-phase, 1.98x vs root |
| 8x2 | 4559.3 | **2335.9** | 3394.2 | two-phase, 1.95x |
| 4x4 | 4555.9 | **2250.7** | 2867.5 | two-phase, 2.02x |

**Same isolated group, 1 tile/core (latency floor):**

| Group | `reduce_root_mcast` | `two_phase_reduce_mcast` | `two_stage_grid_reduce` | best |
|---:|---:|---:|---:|---|
| 2x8 | 1498.1 | 1981.0 | **1377.3** | grid two-stage, 1.09x vs root |
| 8x2 | 1532.0 | 1975.1 | **1361.9** | grid two-stage, 1.12x |
| 4x4 | 1547.1 | 1992.4 | **1271.0** | grid two-stage, 1.22x |

**Grid-filling (groups packed across the 13x10 grid -> NoC contention), 6 tiles/core:**

| Group | Groups | Cores | `reduce_root_mcast` | `two_phase_reduce_mcast` | `two_stage_grid_reduce` | best |
|---:|---:|---:|---:|---:|---:|---|
| 2x8 | 5 | 80 | 5338.8 | 6443.0 *(noisy)* | **3641.3** | grid two-stage, 1.47x vs root |
| 8x2 | 6 | 96 | 6202.9 *(noisy)* | 6716.7 *(noisy)* | **3877.3** | grid two-stage, 1.60x |
| 4x4 | 6 | 96 | 5208.5 | 4896.3 *(noisy)* | **3584.4** | grid two-stage, 1.45x |

**Reading of the result - pick the reducer by regime:**
- **`two_phase_reduce_mcast` wins an isolated group that has real payload** (16 cores, 6 tiles: ~2x
  over root) because it parallelizes the reduction across tile indices. But it needs tiles to
  parallelize - useless at 1 tile (`min(num_tiles, group_size-1)` = 1 worker, plus a wasted
  worker->root handoff, so it is the *worst* there) - and it is **contention-sensitive**: from 1
  group to 5 it goes 2286.8 -> 6443.0 ns and its noise jumps from ~1% to 15-28%.
- **`two_stage_grid_reduce` is the robust default.** Its per-axis traffic is localized (each gather's
  fan-in is only `cols`, then `rows`, never the whole `rows*cols`), so it is the **steadiest** in
  every regime (<1% noise) and barely moves under contention (3344 -> 3641 ns from 1 -> 5 groups). It
  wins under **grid-filling contention** (1.45-1.60x over root, and clear of the now-inflated
  two-phase) and at the **1-tile latency floor**, and it is never worst. The cost is one extra
  communication round, which is why it does *not* beat tile-index two-phase in an isolated,
  well-fed group.
- **`reduce_root_mcast` is the simple fallback** - one root serializes the whole `rows*cols` gather,
  so it is never fastest but also never blows up; moderate and fairly steady.

So: reach for **grid two-stage** when the grid is busy (many concurrent groups) or the payload is
tiny; reach for **tile-index two-phase** for an isolated group with several tiles per core. On a 1-D
group (single row or column) grid two-stage has only one axis to reduce and collapses to the root
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
