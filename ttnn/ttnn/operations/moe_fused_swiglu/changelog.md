# moe_fused_swiglu — implementation changelog

Device: **blackhole_p150**, `compute_with_storage_grid_size() = 11 x 10` (the p150a is harvested to
11 columns, not the 13 `op_design.md` assumes). Every core-assignment formula reads the grid, so
`HGROUPS = 11`, `KGROUPS = 10`, and all 110 cores are active in BOTH phases.

## 1. Blocking model as built

| Axis | Split | Knob | Phase-1 value |
|---|---|---|---|
| **Hn** hidden, gate/up output | across grid COLUMNS | `HN_PAD = ceil(HID_T/HGROUPS)` | 6 (col 10 owns 4) |
| **Kg** emb, gate/up contraction | across grid ROWS | `KR_PAD = ceil(EMB_T/KGROUPS)`, per-row `kr(y)` | 23 (emb 7168) / 20 (6144) |
| **Ne** emb, `down` output | across ALL cores | `ec(i)` = `split_work_to_cores(EMB_T, 110)`, stride `EC_MAX` | 2–3 |
| **Kh** hidden, `down` contraction | sequential per core | `HGROUPS` K-blocks of `HN_PAD` | 11 x 6 |
| **M** tokens | sequential outer loop | `M_BLOCK` | 8 |
| **x** over Hn | rotating injector + ROW multicast | — | 1 tile-row per injector |
| **h** over Ne | grid-wide multicast, `HGROUPS` rounds | `DEPTH_H` | 3 |

Buffer depths: `DEPTH_W=2`, `DEPTH_H=3`, `DEPTH_OUT=2`, `DEPTH_XSTAGE=1`, `XSTICK_ROWS=1`.
Read knobs: `WRUN=8`, `WD_AHEAD=1`. Sub-block: `OUT_SUBBLOCK_H=1`, `out_subblock_w = HN_PAD` (gate/up)
/ `ec` (down). L1: **~1.35 MB** of the 1.43 MB budget.

Every CB page count, loop trip count and grid formula derives from those parameters. `count` is the
only runtime value and enters ONLY as `m_blocks` — never as a CB size.

## 2. Deviations from `op_design.md` (and why)

| Design | Built | Reason |
|---|---|---|
| grid 13 x 10 = 130 | 11 x 10 = 110 | the device's real `compute_with_storage_grid_size()`; derived, not hardcoded |
| `M_BLOCK = 16` | `M_BLOCK = 8` | L1. The design's own §5 names `M_BLOCK = 8` as the escape valve. Cost: count 512 needs 2 M-blocks (measured 2x the time), counts 128/256 still read the weights exactly once. Live knob. |
| `KB1 = 6`, `num_k_blocks = Kr/KB1` | `KB1 = KR_PAD`, `num_k_blocks = 1` | `Kr` is 23/20 — not divisible by 6 — and more importantly `matmul_block` **pops in0 per K-block** (`matmul_block_helpers.inl:527`), so K-blocking is incompatible with gate and up sharing one resident `x`. With one K-block, `WaitAndRetainOnLastBlock` pops nothing and the kernel pops `cb_x_tiles` once — the §6 contract, and NOT a second multicast of x. Host-asserted; the documented second-CB fallback is named in the comment. |
| per-sub-block reduce, pipelined against the next sub-block | ONE whole-block reduce | `in1_num_subblocks = 1` with `out_subblock_w = HN_PAD` makes the gate/up block a single sub-block, so there is no `off` loop to pipeline against. Costs the §4.3 overlap. |
| reduce = 2 semaphores, one per level parity | 2 semaphores, `SEM_GO` (parent invites child) + `SEM_DATA` (child signals) | the parity scheme races: a level-`l+2` sender can arrive before the parent consumed level `l`. The invite is the flow control, and both counters are monotone so neither is ever reset. Same mechanism (unicast + counting semaphore), same 2-semaphore budget. |
| `DataReadySignal::Counter` on both mcasts | `Flag` | **Counter HANGS** — see §4. |
| `cb_x_tiles` producer = reader, tilize pushes elsewhere | same, via `cb_x_stage` | the design's table has compute's tilize and the reader's mcast both producing `cb_x_tiles`; `cb_x_stage` splits them so every CB keeps one producer and one consumer. |
| gate/up partial CBs consumed by the writer | `cb_*_acc` (compute->compute, in-place reduce adds) + `cb_*_send` (compute->writer) | the in-place `add` makes compute a consumer, so the writer cannot also consume the same CB. |
| `add_bias_bcast_rows` once over the whole block | once per token tile-row, walked with `bias_offset` | the helper's bias index does not advance with `in0_subblock` (`bias_add_helpers.inl:141`), so an Elementwise bias spanning `M_BLOCK` tile-rows must be walked. Keeps the packer-thread SiLU. |
| `FillScalar` zero pad for the ragged hidden column | `KBlockInnerDimFn` shrinks the phase-2 FMA loop | cheaper: the pad column is never read at all rather than made zero. Same trick retires the `KR_PAD` K padding. |

## 3. Correctness — at the format floor, which is BELOW the gate

All three structural debug tests (`test_moe_fused_swiglu_debug.py`) pass **exactly**: all-ones,
hidden-identity (W_down as a hidden->emb identity, so the output IS `h`) and emb-contraction. That
verifies the Kg row split, the reduce tree, the h all-gather round order, the phase-2 K indexing and
the Ne output split.

Two real bugs were found this way:

1. **`matmul_block` leaves packer L1 accumulation ENABLED** after its last K-block. Neither the
   eltwise chain (`L1Accumulation::Disabled` is a compile-time no-op, `eltwise_chain.inl:1054`) nor
   a `packer_l1_acc=false` matmul resets it, so the `interm -> out` copy ACCUMULATED onto stale L1.
   Fixed with `pack_reconfig_l1_acc(0)` after the phase-2 matmul.
2. The activation `TensorAccessor` was built with the **`cb_x_in` slice size** instead of the
   tensor's own page size, so every bf16 ROW_MAJOR stick read hit the wrong bank offset. Split into
   `X_PAGE` (tensor) and `X_SLICE` (CB).

**MEASURED format floor** — a torch fp32 chain carrying ONLY the bfp4 weight quantization, on the
acceptance test's own fixture (`probes/probe_004.py`):

```
  emb   cap   cnt | floor bfp4w  floor+bfp8 |     LoFi    HiFi2    HiFi4
 7168  1024    32 |     0.97986     0.97983 |  0.97922  0.97937  0.97937
 7168  1024   255 |     0.97972     0.97966 |  0.97909  0.97921  0.97921
 6144  2048   128 |     0.97975     0.97969 |  0.97917  0.97928  0.97928
 7168  2048   512 |     0.97975     0.97969 |  0.97910  0.97923  0.97923
 6144  1024  1024 |     0.97979     0.97973 |  0.97920  0.97932  0.97932
```

The gate is **0.98 — above the floor**. bfp4_b carries a sign + 3 magnitude bits against a shared
exponent, which is ~6.5% per element on `randn` weights and ~22% RMS through three matmuls in
series; the golden harness reports exactly that (`rms=0.223-0.229` on all 44 cells). The kernel sits
within **0.0005** of a ceiling that `eval/golden_tests/moe_fused_swiglu/helpers.py:152-158` itself
describes as one "no correct implementation can beat". HiFi2/HiFi4 buys +0.00012 for 2-4x the FPU
passes and still misses, so LoFi is kept.

Golden suite: **44 failed / 1 passed**, every failure `severity=precision pcc=0.9790-0.9793`, no
hang, no structural failure, no inf/nan anywhere.

## 4. `DataReadySignal::Counter` is a real `mcast_pipe` bug

`op_design.md` §4.2/§4.4 specifies Counter for both collectives. It **hangs**:
`mcast_pipe.inl:153` hands `inc_multicast` the LOOPBACK (INCLUDE-source) fan-out `num_dests_incl_`,
while the atomic multicast itself is EXCLUDE-source (unlike the Flag branch at `:159-165`, which
selects `MCAST_INCL_SRC` explicitly). Its `async_atomic_barrier()` then waits forever for an ack from
a destination that was never addressed. Both families here loopback (`src cb != dst cb`), so both
trip it. `Flag` is used instead, documented at the emission site.

## 5. Performance — measured, and the bound identified

```
  fmt        emb   cap   cnt     MB       ns   util | target ns/util | best-measured
  bf16_rm   7168  5120   128  26.61   222176  0.234 |  91800 / 0.566 | 102000 / 0.509
  bf16_rm   7168  5120   256  28.44   224743  0.247 | 108000 / 0.514 | 120000 / 0.463
  bf16_rm   7168  5120   512  32.11   445388  0.141 | 161820 / 0.388 | 179795 / 0.349
  bf16_rm   7168  1024   256  28.44   226441  0.245 | 108000 / 0.514
  bf16_rm   6144  5120   256  24.38   211047  0.226 | report only
  bf16_rm   7168  5120  5120  98.17  4359899  0.044 | report only
  bfp8_tile 7168  5120   128  25.75   215121  0.234
  bfp8_tile 7168  5120   256  26.72   219090  0.238
  bfp8_tile 7168  5120   512  28.67   428258  0.131
  bfp8_tile 7168  1024   256  26.72   217006  0.241
  bfp8_tile 6144  5120   256  22.91   203480  0.220
  bfp8_tile 7168  5120  5120  63.77  4211298  0.030
```

`capacity 1024` vs `5120` at count 256 is 226441 vs 224743 ns — within noise, so **the allocation
costs nothing**, which is the one cross-capacity requirement the prompt names.

`/perf-measure` ablations, emb 7168 / count 256 / bf16_rm, one fresh-cache run each:

| variant | ns | reading |
|---|---|---|
| baseline | 225 501 | — |
| `WRUN=1` (naive per-tile read) | 231 004 | coalescing worth 3% -> **not transaction-bound** |
| `SKIP_COMPUTE` (matmul LLK elided, all sync kept) | 194 579 | compute worth 13% -> **not compute-bound** |
| no consumer-ready handshake | 213 734 | the 109-ack storm worth 5% -> **not handshake-bound** |
| `WD_AHEAD` 1 / 4 / 11 | 227 767 / 228 688 / 240 115 | deeper prefetch HURTS -> **not DRAM-latency-bound** |

No single payload is saturated, so **~85% of the time is the SERIAL COMPOSITION** of the three
collectives and the weight stream: per M-block the reader runs `M_BLOCK` x-multicast rounds, then the
gate weight block, then up to 4 reduce round trips, then `HGROUPS` h-multicast rounds, each stage
paying its own latency with nothing else in flight.

### Named next steps, in priority order

1. **Software-pipeline the M-block** so M-block `b+1`'s x staging and weight read run under `b`'s
   phase 2. Needs `DEPTH_W`/`cb_x_tiles` double-buffering (`M_BLOCK=8` leaves the L1 for it) and is
   the only fix that attacks the measured bound rather than a component of it. Helps count>=512
   most, where `m_blocks>1` already doubles the time.
2. **Restore the design's per-sub-block reduce** (§4.3): split the gate/up block into
   `ceil(HN_PAD/HN_BLOCK)` sub-blocks so sub-block `off`'s reduce overlaps `off+1`'s matmul. Needs
   `in1_num_subblocks > 1`, which the current `out_subblock_w = HN_PAD` collapses.
3. **Raise `M_BLOCK` to 16** once (1) frees the L1, folding count 512 into one M-block and halving
   its weight traffic — a knob turn, already parameterised.
4. Fix the `mcast_pipe` Counter/loopback fan-out bug upstream, then re-measure Counter: it removes
   the per-round flag-reset round trip that currently serialises all 11 h rounds.
5. `WD_AHEAD` and the W_gate pre-issue are parked at their trivial defaults; both become worth
   turning only after (1)/(2) move the critical path off the collectives.
