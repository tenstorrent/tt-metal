# Read coalescing

**Difficulty:** T2. **Focus:** replace 32 small row reads with one wide read call
when the source bytes are physically contiguous. The example copies a bf16
ROW_MAJOR tensor from width-sharded DRAM to interleaved DRAM, preserving its
shape, layout, and values. Each input shard spans the full height and lives in
one DRAM bank.

The program has only a reader and writer, connected by one double-buffered
circular buffer. It compares two approaches to the same read bottleneck:
larger transactions through coalescing, and different core-to-bank assignments
through affinity. The four-way A/B measures them separately and together;
their benefits can overlap, compound, or interfere.

## Measured copy result

For bf16 **3328×512**, 128 B per shard row, on Blackhole with 104 active cores:

| Baseline | Coalescing | Coalescing + affinity |
|---:|---:|---:|
| 88.308 µs | 27.256 µs | 22.657 µs |
| 1.00× | 3.24× | 3.90× |

Coalescing is the primary win. Affinity adds **1.20×** here, but makes two longer
cases slower than coalescing alone. Measure both switches independently on your
shape. These are within-timer-cluster spans; [report.md](report.md) explains the
board's timer limitation and lists all four arms across the complete sweep.

## Independent compile-time switches

| Variant | `coalesce_reads` | `bank_affinity` |
|---|---|---|
| `none` | false | false |
| `coalesce` | true | false |
| `affinity` | false | true |
| `both` | true | true |

```python
from ttnn.operations.examples.read_coalescing import read_coalescing

# Width of each DRAM shard in elements, not tiles.
out = read_coalescing(x, shard_width=64, coalesce_reads=True)
out = read_coalescing(x, shard_width=64, coalesce_reads=True, bank_affinity=True)
```

## Concrete storage and work assignment

The headline **3328×512 bf16** input has eight **3328×64** width shards:

```text
DRAM bank 0: columns   0–63, all 3328 rows
DRAM bank 1: columns  64–127, all 3328 rows
...
DRAM bank 7: columns 448–511, all 3328 rows

Within each bank (no gaps):
[row 0: 128 B][row 1: 128 B][row 2: 128 B]...
                     32 rows = one contiguous 4 KiB block
```

The storage is identical for all four variants. With **104 active cores**:

| Assignment | Input work per core | Input bytes per core |
|---|---|---:|
| Affinity off | One 32-row band across all eight banks: 8 banks × 1 block | 32 KiB |
| Affinity on | Eight 32-row bands within one bank: 1 bank × 8 blocks; 13 cores per bank | 32 KiB |

Coalescing changes **each block** from 32 × 128 B read calls to one 4 KiB call.
Affinity changes **which bank successive blocks come from**. Neither requires
the other: a core can coalesce a block from bank 0, then one from bank 1; or it
can stay on bank 0 and issue every row separately. Thus both switches have a
meaningful independent A/B, with identical storage and total work. Their
measured benefits can overlap; this layout enables both but does not guarantee
that both help on every shape.

## Main lesson: coalesce contiguous reads

The reader's `constexpr bool coalesce_reads` combines all 32 consecutive shard
rows in a block into one wide `noc_async_read` call. With the switch off it
issues one call per row. The API splits transfers larger than the hardware
burst limit automatically; one API call does not imply one physical NoC packet.
Both arms reserve one block, issue the reads, barrier once, and publish it to
the writer. Coalescing changes neither barrier count nor buffer capacity.

This is legal because a block covers full shard rows with no alignment gaps,
and every read stays within one shard and one reserved CB slot. Interleaved
input is rejected: consecutive logical pages there are on different banks.
Supported shard widths are 32, 64, 128, 256, or 512 bf16 elements (64–1024 B
per row); H must be a positive multiple of 32.

The writer always sends 32 row slices into the row-major interleaved output.
It uses the same transfer sizes and counts in every arm. Each output byte has
exactly one owner. The copy's overall gain can be limited by this fixed write
stage even when the reader becomes faster.

## Alternative assignment to compare: bank affinity

Without affinity, cores receive contiguous ranges of the row-major
`(32-row block, bank)` work grid. With it, core `k` handles bank `k % banks`,
and cores assigned to that bank split its height. Reader and writer use the
same `constexpr bool bank_affinity` to choose a unit stride of one or `banks`;
the host supplies matching start/count arguments. The tensor's storage does
not change. Bank groups are interspersed across row-major core positions.

Unlike bank staggering, which changes when each core visits each bank,
affinity keeps each core with one source bank for the entire kernel.

## A/B controls

All four variants use the same active cores, CB capacity, bytes copied, output
layout, and NoC selection (reader NoC0, writer NoC1). The active core count is
rounded down to whole bank groups. Default performance shapes give exactly
equal work per core under both assignments. Ragged and short shapes are also
correctness-tested, but their maximum work per core can differ between splits.

The benchmark prints `affinity_ns / both_ns` to show what coalescing adds after
affinity, and `coalesce_ns / both_ns` to show what affinity adds after coalescing.
Values above 1 mean improvement. Do not multiply independent speedups.
See [report.md](report.md) for the measured copy results.

## Run

```bash
python -m ttnn.operations.examples.read_coalescing --trials 7 --launches 10
# H x W x shard_width; W must equal DRAM-bank-count * shard_width
python -m ttnn.operations.examples.read_coalescing --cases 3328x512x64 --report /tmp/coalescing.txt
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_read_coalescing.py
```

Each arm is checked bit-exact, then warmed up. Measurement order rotates between
trials; timings are reported, never asserted as performance thresholds.
`--kernel-iters` repeats the block walk inside a launch (default 1).

Timing uses raw profiler zones and the maximum within-timer-cluster span per
program. With a shared timer base this is the usual first-start to last-end
span. On boards with mismatched bases, it avoids subtracting unrelated clocks
but cannot recover the unknown relative start offset between clusters. The
report labels this limitation and counts affected programs.
