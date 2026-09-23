# Perf 1 — measured breakdown of the focus shape

Focus: `[1,1,16384,64]` bf16 ROW_MAJOR, DRAM `TensorMemoryLayout::INTERLEAVED` in and out, 64 Tensix
cores (8 tile-rows each, C = 2 tile-columns), bank_coalesced reader. WH B0 n150, AICLK 1000 MHz
(cycles == ns).

Reproduce:
```
python3 ttnn/ttnn/operations/tilize/perf_experiments/breakdown/make_ablations.py
TILIZE_PERF_EXPERIMENTS=1 TILIZE_ABL="full,S,R,C,W,RS,RSC,WC,RSW,RSCW,WCS" \
  TILIZE_ABL_SHAPES="1x1x16384x64,1x1x32768x64,1x1x16384x32" \
  scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/tilize/test_tilize_perf1_ablate.py -v
# zones (add TT_METAL_KERNEL_PERF_ZONES=1), then:
python3 ttnn/ttnn/operations/tilize/perf_experiments/breakdown/zones.py      # per-zone per-RISC summary
python3 ttnn/ttnn/operations/tilize/perf_experiments/breakdown/percore.py <report_dir> BRISC
```
Ablation letters (payload stubbed, synchronization kept): R = bank-coalesced DRAM reads,
S = loopback scatter, C = tilize compute, W = DRAM tile writes. A variant named `RSC` has all three
stubbed.

## DEVICE KERNEL DURATION [ns], one fresh run each

| variant | 16384x64 | 32768x64 | 16384x32 |
|---|---|---|---|
| full | 23573 | 47734 | 14615 |
| -S | 23291 | 46643 | 13134 |
| -R | 18634 | 34151 | 11086 |
| -C | **26721** | 47522 | 15858 |
| -W | 15856 | 26970 | 13421 |
| -RS (writes + compute) | 17687 | 33116 | 9324 |
| -RSC (writes only) | 15662 | 30098 | 8440 |
| -WC (reads + scatter) | 15267 | 25810 | 12952 |
| -WCS (reads only) | 11976 | 22347 | 6392 |
| -RSW (compute only) | 3157 | 4827 | 2922 |
| -RSCW (sync floor) | 2395 | 4038 | 2374 |

Reading (focus): over the 2.4 us floor, writes-only 13.3 us (~158 GB/s for 2 MiB), reads-only
9.6 us (~218 GB/s), the scatter +3.3 us on the reader chain but hidden in the full op (-S saves
0.3 us), compute 0.8 us. Reads and writes nearly add up (9.6 + 13.3 = 22.9 us vs 21.2 us
measured), so they share a resource (DRAM). Stubbing compute is SLOWER (+13 %): unpaced read/write
interleaving contends worse. The whole op moves ~198 GB/s over the floor, at the empirical
64-Tensix-core copy ceiling (190.8 GB/s, `examples/double_buffer`).

## Zones (focus, per-core sums, cycles; p50 / max across the 64 Tensix cores)

| zone | RISC-V | p50 | max |
|---|---|---|---|
| writer_wait (cb_wait_front on compute) | BRISC | 8078 | 15667 |
| writer_issue (4 x 2 KiB writes per quantum; ~190 cycles each = injection back-pressure) | BRISC | 4628 | 11996 |
| writer_flush | BRISC | 3278 | 10517 |
| writer_barrier | BRISC | 346 | 975 |
| reader_issue (per-bank reads) | NCRISC | 3814 | 8212 |
| reader_barrier | NCRISC | 3900 | 7358 |
| reader_scatter (64 loopback reads per unit) | NCRISC | 6661 | 8599 |
| reader_reserve | NCRISC | 163 | 171 |
| compute_tilize (occupancy: includes the helper's waits) | TRISC_0/1/2 | 15632 / 15706 / 16341 | |
| BRISC-KERNEL | BRISC | 19330 | 22871 |
| NCRISC-KERNEL | NCRISC | 15350 | 18751 |

Spatial tail (`percore.py`): bottom-right Tensix cores (NoC0 x 6..8, y 8..10) finish their reads
first (~12-15k cycles) and their writes last (up to 23.5k); top rows finish reads ~17-19k and writes
~19-20k.

## Zone cost on the critical path (why the zones are opt-in)

Same session, profiler on, zones compiled in unconditionally vs HEAD (no zones):
`[1,1,128,64]` 2333 / 2282 -> 2699 / 2611 ns (+14 %), `[1,1,32,2048]` 3328 / 3395 -> 3836 / 3727
(+12 %), `[1,1,2048,64]` 5322 / 5525 -> 5773 / 5939 (+7 %), focus flat. After gating on
`KERNEL_PERF_ZONES`: `[1,1,128,64]` HEAD 2264 / 2265 vs zoned-source 2266 / 2340 (identical within
noise).
