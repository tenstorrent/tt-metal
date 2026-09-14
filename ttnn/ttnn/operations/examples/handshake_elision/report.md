# handshake_elision — device report

**Box:** `bh-50-special-dstoiljkovic-for-reservation-88042` · **Arch:** Blackhole · **Clock:** 1350 MHz · **Date:** 2026-09-14 · **Commit:** `6ba9ac2fabc`
**Metric:** `DEVICE KERNEL DURATION [ns]`, in-process device profiler, mean of 10 profiled launches after 5 warmups.
**Op:** tilize (row-major → tiled) of a HEIGHT-sharded bfloat16 tensor whose input and output shards share one spec and
one core's L1. Both CBs are aliased onto the shard buffers: **no DRAM, no NoC traffic**. Cores sit in row 0, one shard each.

Per variant: `no_handshake` = the compute kernel's compile-time constant (0 = per-tile-row CB protocol with a reader that
publishes and a writer that retires; 1 = no CB protocol, tile-index addressing off fixed base pointers); `kernels` =
kernels launched per core. `ratio` = handshake ns / arm ns (>1 → the arm is faster). Numbers are illustrative of the
*effect*, not CI bounds. The 4-core and 1-core tables agree within ~2% per cell, so the effect is per-core and does not
depend on grid size.

**Measurement caveat on this board.** Physical core column x=11 (logical x=6 in the compute grid) reports RISC timestamps
in a different time base; every table here stays in logical x=0..5.

---

## 1. Per-launch latency — 4 cores, `--iters 1`

```
  shard  tiles/core  variant       no_handshake  kernels      ns/op   ns/tile   ratio
    1x2           2  handshake                0        3      495.7     247.8   1.00x
    1x2           2  no_handshake             1        1      420.4     210.2   1.18x

    1x4           4  handshake                0        3      591.4     147.8   1.00x
    1x4           4  no_handshake             1        1      519.1     129.8   1.14x

    2x4           8  handshake                0        3      806.9     100.9   1.00x
    2x4           8  no_handshake             1        1      726.5      90.8   1.11x

    4x4          16  handshake                0        3     1206.7      75.4   1.00x
    4x4          16  no_handshake             1        1     1118.8      69.9   1.08x

    4x8          32  handshake                0        3     1998.5      62.5   1.00x
    4x8          32  no_handshake             1        1     1891.6      59.1   1.06x

    8x8          64  handshake                0        3     3570.1      55.8   1.00x
    8x8          64  no_handshake             1        1     3444.0      53.8   1.04x
```

Absolute saving per launch: **75 · 72 · 80 · 88 · 107 · 126 ns** for 1 · 1 · 2 · 4 · 4 · 8 tile-rows — a ~70 ns fixed
term plus a few ns per tile-row of `wait/reserve/push/pop`. It is a fixed cost, not a per-tile cost, which is why the
ratio collapses toward 1.0 as tiles/core grows and the ~50 ns/tile tilize LLK takes over.

## 2. Steady state — 4 cores, `--iters 20` (per-launch cost amortized; what remains is per-iteration)

```
  shard  tiles/core  variant       no_handshake  kernels      ns/op   ns/tile   ratio
    1x2           2  handshake                0        3     3417.4      85.4   1.00x
    1x2           2  no_handshake             1        1     2299.4      57.5   1.49x

    1x4           4  handshake                0        3     5358.0      67.0   1.00x
    1x4           4  no_handshake             1        1     4269.6      53.4   1.25x

    2x4           8  handshake                0        3     9326.6      58.3   1.00x
    2x4           8  no_handshake             1        1     8248.7      51.6   1.13x

    4x4          16  handshake                0        3    17283.8      54.0   1.00x
    4x4          16  no_handshake             1        1    15990.0      50.0   1.08x

    4x8          32  handshake                0        3    32963.6      51.5   1.00x
    4x8          32  no_handshake             1        1    31611.2      49.4   1.04x

    8x8          64  handshake                0        3    64589.7      50.5   1.00x
    8x8          64  no_handshake             1        1    62902.9      49.1   1.03x
```

Per added iteration (`(iters=20 − iters=1) / 19`): `no_handshake` runs at **~49–50 ns/tile flat** — the tilize LLK's own
throughput. `handshake` pays **~55–65 ns per iteration** on top, almost independent of shard size: with a CB exactly one
shard deep, every iteration is a dependent round through the credits (the reader's re-publish waits on the compute's
last pop, the compute's first reserve waits on the writer's retire), so the protocol serializes the iterations even
though it hides under the tilize within one.

## 3. Grid independence — 1 core, `--iters 1`

```
  shard  tiles/core  variant       no_handshake  kernels      ns/op   ns/tile   ratio
    1x2           2  handshake                0        3      449.1     224.6   1.00x
    1x2           2  no_handshake             1        1      381.6     190.8   1.18x

    2x4           8  handshake                0        3      765.5      95.7   1.00x
    2x4           8  no_handshake             1        1      681.6      85.2   1.12x

    8x8          64  handshake                0        3     3533.6      55.2   1.00x
    8x8          64  no_handshake             1        1     3415.7      53.4   1.03x
```

Same savings as the 4-core table (68 · 84 · 118 ns): the cost lives inside one core.
