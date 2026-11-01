# dram_stream_bw: residual DRAM read stream + output write twin (BH p100a, base 9daa4c30ecf)

Verdict: NULL. No variant beats the shipped stream on the flagged rows; most regress. Nothing to graduate.

## Layout

- `dsb_op/` - copy of the op descriptor (`descriptor.py`) + kernels, ablation hooks (`LN_ABLATE=compute,write,read,xread`)
  applied. Every variant is a knob in `descriptor.py`, all default to the shipped behaviour (copy == real op).
- `test_dram_stream_bw.py` - perf + guard rows of `test_perf_bench.py` through the real entry point with the copy
  monkeypatched in; `test_bit_exact` runs real op vs copy on the same tensors and requires `torch.equal`.
- `bench.sh <label> [LN_KNOBS] [-k expr] [LN_COPY]` - one `--profile` pass, DEVICE KERNEL DURATION per row -> `logs/<label>.txt`.
  Needs `--import-mode=prepend` (set inside): the repo's importlib mode names this file `ttnn.ttnn.operations...`.
- `landing.py [csv|-] [run|-] [zone]` - per-core grid of the first `<zone>` end (default `writer_residual_block`) + BRISC end.

## Diagnosis (focus1)

- Read-only peel: x L1 reads do not matter (22.85 vs 22.59 us with x reads stubbed). The residual-only DRAM stream
  is ~330 GB/s; focus3's DRAM x stream (reader RISC) reaches ~480 GB/s.
- Landing is position-unfair: per-NoC gradients (NoC1 half: x gradient, NoC0 half: y gradient), every extra byte in
  flight per core makes it worse. Total time = heavy cores' row-0 landing + 2 rows compute, or bottom-left light
  cores' write tail (~6 us for 48 KB), whichever is later; the two are within ~1 us of each other.

## Variants (focus1 device kernel us; baseline 35.3-36.0)

| knob | focus1 | notes |
|---|---|---|
| `RESIDUAL_TRID_DEPTH=2/3` | 36.5 / 39.4 | 8x384 38.2/39.0, 32x32 20.8/21.7 (worse) |
| `RESIDUAL_TRID_DEPTH_HEAVY=2/3` (heavy cores only) | 37.5 / 38.3 | light cores' landing pushed out to ~29 us |
| `WRITER_NOC_PLAN=1..9` (residual+output NoC per core) | 57.6 50.4 40.8 36.2 40.9 37.1 39.1 40.9 36.5 | shipped by-y split is best |
| `WRITER_NOC_SPLIT_Y=3/4/6` | 41.3 / 37.8 / 37.9 | shipped row 5 is best |
| `BANK_NOC_POLICY=1` (residual bank -> NoC, DM_DYNAMIC_NOC) | 35.7, 35.9, 41.5 | right-side cores land at 8 us, left at 15; unstable |
| `BANK_NOC_POLICY=2/3/4` | 40.1 / 46.9 / 44.5 | |
| `BANK_NOC_WRITE_POLICY=1/2/3/4` (output bank -> NoC) | 44.4 / 40.3 / 56.9 / 48.0 | w2 fixes the light write tails, starves the heavy cores' row-1 lone-band residual (15 us for 24 KB) |
| policy 1 + write 1/2 | 47.6 / 49.7 | |
| w2 + light acked writes / + heavy depth 3 | 39.4 / 41.1 | |
| `WRITE_ACKED_CORES=1/2` (ack barrier per flush group) | 35.5 / 35.9 | flat |
| `HEAVY_SIDE=1` (+policy 1), `HEAVY_SIDE=2` + policy 1 | 36.8 (43.2), 39.3 | |
| `WRITE_POSTED=1` | 35.7 | focus2 46.7, focus3 33.75, 8x384 36.0: flat |
| `WRITE_CHUNKS_PER_FLUSH=3`, 3-run medians vs base | 35.50 vs 35.76 | focus2 47.64 vs 47.30, focus3 33.52 vs 33.42, 8x384 36.48 vs 36.83: flat |

Full sweep (one run, perf rows then guards: focus1 focus2 focus3 allL1_res L1_nores 8x384 32x32 affTILE |
f1_bf16 3x500 64x17 2x512):

- base copy: 35.31 47.84 33.75 24.93 20.86 36.42 18.73 48.05 | 35.70 18.71 21.17 13.44
- `BANK_NOC_POLICY=1`: 41.53 51.16 33.32 24.59 20.66 36.01 19.39 52.51 | 37.98 17.63 21.54 13.51
- `BANK_NOC_WRITE_POLICY=2`: 40.89 48.92 35.84 25.07 20.66 37.90 19.11 49.79 | 38.91 19.66 21.81 13.59

Correctness: `test_bit_exact` passes on all 12 rows for `WRITE_CHUNKS_PER_FLUSH=3`, `BANK_NOC_POLICY=1`,
`BANK_NOC_WRITE_POLICY=2`, `WRITE_POSTED=1`.

## Constraints found

- DM_DYNAMIC_NOC (needed for one RISC to read or write on both NoCs) cannot compile the mcast sender:
  `mcast_pipe.inl` calls `async_writes_flushed<TXN_ID>`, which static_asserts `noc_mode != DM_DYNAMIC_NOC`. The bank
  variants keep the affine injector core on DM_DEDICATED_NOC in its own kernel group.
- Posted output writes leave completion unobservable by the kernel; not graduatable without a completion guarantee.
