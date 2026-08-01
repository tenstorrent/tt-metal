# NoC-trace analysis of moe_fused_swiglu (tt-npe)

## How to capture

The repo already supports the device-side NoC event profiler; `tools/tracy/__main__.py --collect-noc-traces`
just sets the env var, so `run_safe_pytest.sh --profile` works directly:

```bash
export TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000
MOE_SWIGLU_GRID=11x8 MOE_R2_CASES="7168,5120,256,bf16_rm" \
  scripts/run_safe_pytest.sh --profile \
  tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_moe_fused_swiglu_r2_perf.py
```

Output: `$TT_METAL_HOME/generated/profiler/.logs/noc_trace_dev0_ID<n>.json` plus `topology.json`
and `cluster_coordinates.json` (npe needs all three in one directory).

**One cell per run.** `count 512` overflows the device event buffer and then produces *no device log
at all* — the run "passes" and `process_ops_logs` dies on a missing `profile_log_device.csv`. Capture
128 and 256 and extrapolate.

**`num_bytes` saturates.** `KernelProfilerNocEventMetadata::LocalNocEvent::payload_chunks` is a
`uint8_t` of 32 B chunks, so any transfer >= 8160 B is reported as exactly 8160. Our DRAM reads are
576-2304 B (exact); every multicast is saturated and must be re-priced analytically.

Instrumentation overhead: +13 % at count 128, +4.6 % at count 256. Use ratios, not absolutes.

## Scripts

* `noc_summary.py <trace.json> [buckets]` — transfer counts, per-destination bytes, per-NoC and
  per-RISC split, coarse rate timeline.
* `noc_phases.py <trace.json> [buckets]` — the useful one. Zone-occupancy timeline (which stage the
  cores are in) stacked against DRAM-read / L1-read / write rates and multicast issue counts, plus
  per-DRAM-endpoint balance.

tt-npe itself (`npe_analyze_noc_trace_dir.py <dir>/`) gives the congestion model.

## What it measured (88 cores, emb 7168, bf16_rm)

| | count 128 | count 256 |
|---|---|---|
| DRAM bytes | 26.50 MB | 28.04 MB |
| traced wall | 112.6 us | 159.0 us |
| **us with DRAM < 10 GB/s** | **37 (33 %)** | **80 (50 %)** |
| longest contiguous DRAM-idle window | 14 us | 45 us |

Doubling M moves 5.7 % more bytes and costs 41 % more wall; **43 of the 46 extra us are DRAM-idle.**

tt-npe on the count-256 trace: NoC util **7.96 %**, mcast-write util 0.19 %, DRAM BW util 43.65 %,
**congestion impact 0.08 %**. Per-DRAM-channel bytes 3.24-3.37 MB across all 8 (the 16 endpoints are
2 subchannels x 8 channels per `blackhole_140_arch.yaml`; the 2:1 endpoint split is just
NCRISC -> subchannel A, BRISC -> subchannel B). **There is no congestion and no bank imbalance** —
which retires the whole coalescing / bank-striding / routing family for this op, and independently
explains the `WRUN` and dual-NoC nulls.

Phase shape at count 256:

```
t=  0- 59 us   x + W_gate + W_up, 330 GB/s, BOTH RISCs          DRAM busy
t= 59-104 us   gate/up tail + reduce + scatter + h publish      DRAM = 0
t=104-144 us   11 phase-2 rounds                                DRAM 25 % duty
t=144-159 us   down tail + output write                         DRAM = 0
```

Phase 2's DRAM is a square wave, not a stream: ~1.5 us at 700-900 GB/s then ~3 us of silence, x11.
The request path is nowhere near saturated, so the reference op's `BRISC_WEIGHT_K` split buys
nothing here.

## The round-cost model

Correlating each column root's `compute_reduce` END against its own h multicast timestamp:

| round | root | reduce end | h mcast | idle |
|---|---|---|---|---|
| 0 | (1,2) | 101.2 | 102.9 | 1.7 |
| 5 | (6,7) | 100.5 | 124.6 | 24.0 |
| 10 | (15,4) | 88.7 | 145.9 | **57.2 us** |

**Every root is ready by t=101.2; the last broadcast is at t=145.9.** The rounds are not data-gated —
which refutes the Perf-3 "root readiness" reading (that was inferred from a 110-core zone span, not
measured).

Fitting the cadence across both traces (m_eff 4 -> 3.66 us/round, m_eff 8 -> 4.30 us/round):

```
round period = 3.12 us FIXED + 0.147 us per m-tile
```

against ~2.06 us of real work at m_eff 8 (52 224 B of h ingest at ~43 GB/s = 1.21 us + a 144
tile-MAC `down` block at the 8 cycles/tile-MAC LoFi roofline = 0.85 us). So **34.3 us per M-block of
pure rendezvous, independent of M** — and count 512 runs two M-blocks, i.e. **68.6 us**, 27 % of that
cell's wall. This is the same term the ablation floor (42.8 us, every payload stubbed) was measuring.

## Floors

At count 256, 88 cores: DRAM 28.0 MB at the 460 GB/s wall = **61 us**; compute 4272 tile-MACs/core at
the measured 8.1 cycles/tile-MAC = **25.3 us**. `max(61, 25) = 61 us` against a 108 us target and a
152 us actual. **The targets are ~1.8x inside the DRAM floor; the entire gap is serialisation.**

Matmul shapes per `matmul_block` call (88 cores): gate/up `M=m_eff, K=28, N=2` (6 calls/M-block,
1x2 of 8 DEST); `down` `M=m_eff, K=6, N=3` (11 calls, 1x3 of 8 DEST). Small N and a quarter of DEST,
but measured **8.1 cycles/tile-MAC against an 8 cycles/tile-MAC roofline** — the calls are AT
roofline and a custom fused gate+up block would win nothing.
