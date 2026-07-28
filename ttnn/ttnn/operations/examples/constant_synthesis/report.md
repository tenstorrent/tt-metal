# constant_synthesis — device perf report

Constant-valued DRAM output: stream the constant from a DRAM-resident tensor
(baseline, real reads) vs. invent one page on-core and replicate it (candidate,
zero reads). Correctness (`value` everywhere, bit-exact) is the only pass/fail;
the numbers below are evidence, not a bound.

Metric: `DEVICE KERNEL DURATION [ns]`, in-process device profiler, averaged over
20 launches after 5 warmup launches. GB/s counts DRAM bytes moved — writes only
for `synthesize`, reads+writes for `stream_from_dram`. Both variants use the same
double-buffered `block`-per-barrier NoC pattern; the only difference is whether
the source bytes are READ or INVENTED.

## Arch: wormhole_b0

- box: `bgd-lab-t3002-special-dnijemcevic-for-reservation-33181`
- arch: wormhole_b0 (WH B0), ~1000 MHz
- date: 2026-07-22
- git: `b32a86c71da`
- params: bf16 row-major DRAM-interleaved output, page = one row, block=8, iters=1

### Headline — 4096 x 1024 (8 MB written), variant x cores

```
cores  variant             ns/op        GB/s     vs baseline
    1  stream_from_dram   539154.8      31.1     (baseline)
    1  synthesize         514569.0      16.3      1.05x
   64  stream_from_dram    87983.9     190.7     (baseline)
   64  synthesize          62097.8     135.1      1.42x
```

### The win grows with output size (64 cores, block-insensitive)

```
output      stream_from_dram (r+w)   synthesize (w only)   ratio
 4 MB       46331 ns / 181 GB/s      35273 ns / 119 GB/s   1.31x
 8 MB       87984 ns / 191 GB/s      62098 ns / 135 GB/s   1.42x
16 MB      170871 ns / 196 GB/s     118165 ns / 142 GB/s   1.45x
```

`synthesize` throughput plateaus at ~142 GB/s for block ∈ {8,16,32} at 16 MB —
it is write-BANDWIDTH-bound, not per-page-latency-bound.

## Reading of the result

The expected direction is CONFIRMED: `synthesize` is substantially faster and it
is because it does ZERO DRAM reads. But the gap is ~1.45x, not the naive 2x, and
the reason is a DRAM-subsystem property, not an implementation flaw:

- The baseline is DRAM-**combined**-bandwidth-bound: reads (NoC0) and writes
  (NoC1) run concurrently and saturate the DRAM controller at ~195 GB/s of
  read+write traffic. Its writes therefore get only ~half the bus (~93 GB/s).
- The candidate moves half the DRAM bytes (writes only) but a pure-write stream
  tops out at ~142 GB/s — below the ~195 GB/s the controller reaches when
  interleaving reads and writes.
- Net: `2 x (142/195) = 1.46x`, matching the measured 1.31–1.45x (approaching the
  ceiling as output size amortizes launch/drain overhead).

At **1 core** the win nearly vanishes (~1.05x): DRAM is not the bottleneck, so the
baseline's reads overlap its writes on the separate NoC for free — removing a
free read saves nothing. The win is a DRAM-bandwidth-contention effect that only
appears once enough cores make the DRAM bus the shared wall.
