# moe_fused_swiglu — post-mrow optimization campaign

Date started: 2026-08-06

This is the live plan and evidence log for the next optimization campaign.  Every experiment is
kept isolated long enough to answer one question.  A change ships only after correctness and
performance are measured; a losing path is reverted or compile-time disabled and its result
remains recorded here.

## Scope and fixed measurement contract

Primary performance shape:

- Blackhole p150, explicit `core_grid=(11, 8)` = 88 cores (11 columns x 8 rows).
- BF16 row-major activations, BFP4 tiled weights, BFP8 output.
- Kimi K2.6 dimensions: emb/K = 7168, hidden/N = 2048, capacity = 5120.
- Preferred/primary weight placement: DRAM ND-sharded.
- Counts used for focused A/B tests: 0, 256, 512, 1024, and 5120.  Counts 32, 64, 128,
  196, 224, 384, and 513 are correctness/regression guards for runtime-M and tail behavior.
- Performance decision: compare medians from the same measurement command and session shape.
  Noise-sized movement is not a win.

Secondary gates:

- GLM 5.1 emb/K = 6144, hidden/N = 2048, because `ec_max=3` currently gives only 72.7%
  output-grid efficiency.
- Interleaved weights, tiled activations, region-fused input/output, and ragged M remain correctness
  coverage even when they are not the performance focus.

Grid scope is fixed: only the 8-row x 11-column worker grid (`core_grid=(11, 8)`) is supported for
this campaign.  The device-full 11x10 geometry is explicitly out of scope and must not be enabled.

### Baseline used at campaign start

The durable baseline table is `perf_table_8x11.txt`; values below are device-kernel medians for
BF16 RM + BFP4 ND-sharded weights:

| M | Kimi 7168/2048 | GLM 6144/2048 |
|---:|---:|---:|
| 0 | 3.556 us | 3.542 us |
| 64 | 79.616 us | 69.666 us |
| 128 | 84.832 us | 76.973 us |
| 256 | 112.866 us | 104.833 us |
| 512 | 190.726 us | 175.508 us |
| 1024 | 342.844 us | 313.991 us |
| 2048 | 642.654 us | 592.540 us |
| 4096 | 1242.881 us | 1148.025 us |
| 5120 | 1542.840 us | 1425.448 us |

The latest focused Tracy run measured 113.407 us at M=256 and 198.127 us at M=512 with profiler
overhead.  Across its three full M-blocks, `compute_down` is approximately 24.3 us/block on
unpack/pack TRISCs and 29.5 us/block on the math TRISC.  The slow-core ideal down math floor is
approximately 18.2 us/block (`8 * 64 * 3` tile-MACs, 16 cycles/tile-MAC, 1.35 GHz).

## Required gates for every retained change

1. `git diff --check` and host-side geometry/unit tests affected by the change.
2. Focused device golden tests for BF16 RM/BFP4 at M=0, small M, 256, 512, and a ragged tail.
3. The broader moe_fused_swiglu golden suite before the campaign is called complete.
4. Determinism stress after any collective, CB-lifecycle, or cross-RISC protocol change.
5. Fresh device-kernel medians using the contract above.  Record the exact command/artifact and
   both absolute time and delta from the immediately preceding retained configuration.
6. Runtime-M audit: host-side CB allocation is static, while `m_eff` and `m_blocks` are device
   runtime values.  A runtime guard must protect protocol, pointer, and layout behavior—not only
   skip arithmetic.

## Execution order and status

| Order | ID | Change | Status | Keep criterion |
|---:|:---:|---|---|---|
| 1 | E | Issue idx and counts page reads before one phase-0 barrier | **RETAINED** | No regression; fixed-cost reduction, especially M=0 |
| 2 | C | Batch hidden rows per W_down matmul call | **REJECTED** | Improve M>=256 slope without small/tail regression |
| 3 | D | Permit the mrow schedule on 11x10 | **OUT OF SCOPE / REVERTED** | 11x10 is not an allowed grid |
| 4 | F | Publish/output-write completed down rows progressively | **RETAINED** | Hide a measurable part of output issue/drain |
| 5 | A | Split full M blocks into two four-row/44-core output groups | **REJECTED / DISABLED** | Material large-M slope win after first-block cost |
| 6 | B | Split whole h rounds across the two NoCs | **REJECTED / DISABLED** | Win after explicit cross-RISC flag/reset ordering |
| 7 | G | Coalesce gate/up scatter completion | **PARTIALLY RETAINED** | Halve atomic fan-in without serializing payloads |

`E` is an independent fixed-cost change. `C` is the first slope experiment.
`A` has the largest ceiling but changes output ownership and therefore comes after the contained
experiments establish the remaining phase-2 budget.  `B` is intentionally after `A`: byte-wise
HSPLIT already lost, and a writer-owned linked flag previously hung on the cross-RISC reset race.

## Experiment E — concurrent phase-0 page reads

### Hypothesis

The reader currently performs:

1. idx page read + barrier;
2. extract global expert id `g`;
3. counts page read + barrier;
4. extract `counts[g]`.

Both transactions fetch page zero into distinct L1 scratch pages.  Only the L1 selection of the
count depends on `g`; the counts page address does not.  Issue both reads and use one barrier/cache
invalidate.  Region-fused `start` cannot join this batch because it deliberately reuses the counts
scratch allocation and must remain a later read.

### Result

**RETAINED.** The full nine-test shared-region suite passed, including both input formats,
preallocated BF16/BFP8 outputs, offset rebasing, zero-count behavior, and bitwise determinism.

Focused M=0 result, BF16 RM/ND-sharded, 31 measured dispatches per binary:

| variant | median | min | max |
|---|---:|---:|---:|
| two serialized barriers (control) | 3.518 us | 3.419 us | 3.679 us |
| concurrent idx+counts, one barrier | **3.376 us** | 3.268 us | 3.676 us |

Delta: **-142 ns / -4.0%** at M=0.  A seven-repetition 0/256/512 guard run showed the longer cells
inside session noise (E-on 113.436/192.613 us versus control 112.201/192.514 us at 256/512; the
E-on 256 cell had a 7.37% spread).  This is retained as a fixed dispatch-cost win, not claimed as a
slope change.

Artifacts:

- `/tmp/moe_round18_e_off31/res_e_off31.json`
- `/tmp/moe_round18_e_on31/res_e_on31.json`
- `/tmp/moe_round18_e_off/res_e_off.json`
- `/tmp/moe_round18_e_on/res_e_on.json`

The sweep parser now represents M=0 explicitly (`tokens/s=0`, undefined `ns/token`) so this fixed
cost remains reproducibly measurable.

## Experiment C — batched-row W_down calls

### Hypothesis and layout

The mrow path currently executes eight `1x3, K=64` matmul calls.  The initial hypothesis was that
a `2x3` output would use six of eight DEST tiles and reuse each unpacked W_down tile across two M
rows:

```text
1x3: 939 unpack B/tile-MAC
2x3: 651 unpack B/tile-MAC
```

`cb_h` must have a capacity divisible by the 128-tile two-row operand.  Under the current phase-CB
alias, `DEPTH_H: 3 -> 4` adds 69,632 B and leaves approximately 36,416 B of modelled L1 headroom.
It also makes eight 64-tile row pushes naturally wrap to the CB base, deleting the payload-free
alignment row.

The fast path remains gated by `wd_mrow && m_eff == M_BLOCK`; smaller/tail paths retain their
existing shape and lifecycle.

### Result

**REJECTED; source restored to the one-row schedule and `DEPTH_H=3`.** Both legal batched shapes
lost decisively even though the full nine-test region/offset/determinism suite passed.

The first `2x3` variant reduced reader phase 2 by approximately 3.1 us/full block, but increased
each compute TRISC by approximately 2.8 us/full block and moved the critical path to compute.  The
reason is LLK orientation: with `ct_dim=3 >= rt_dim=2`, it retains activation reuse rather than
reusing W_down as assumed.

The corrected experiment used a `4x2` head (which does flip to weight reuse) plus a separate `4x1`
tail to preserve the physical three-tile output stride.  Seven-repetition BF16-RM/ND-sharded
medians were:

| M | control 1x3 | 2x3 | 4x2 + 4x1 |
|---:|---:|---:|---:|
| 128 | 85.710 us | 85.524 us | 86.210 us |
| 196 | 108.023 us | 113.093 us | 114.350 us |
| 224 | 107.875 us | 113.173 us | 113.760 us |
| 256 | **113.471 us** | 118.535 us | 117.800 us |
| 512 | **192.683 us** | 199.399 us | 197.710 us |
| 1024 | **342.141 us** | 352.971 us | 355.340 us |
| 5120 | **1540.921 us** | 1583.424 us | 1586.040 us |

The `4x2 + 4x1` split pays a second matmul init and rereads the four activation rows for the tail;
its predicted unpack saving is only about 7.5%, insufficient to cover that overhead.  Row batching
is therefore closed for this three-column shard, rather than being carried into later experiments.

Artifacts:

- `/tmp/moe_round18_c_off/res_c_off.json`
- `/tmp/moe_round18_c_on/res_c_on.json`
- `/tmp/moe_round18_c_4x2_4x1/res_c_4x2_4x1.json`

## Experiment D — 11x10 grid (out of scope)

### Result

**OUT OF SCOPE; fully reverted.** A temporary implementation and A/B established that the idea was
functional, but the user clarified that 11x10 is not an allowed deployment geometry.  No host
predicate, kernel loop, runtime guard, or test change from D remains in the source.  All subsequent
work is restricted to the 88-core 8x11 grid (`core_grid=(11, 8)`).

The discarded measurements are retained only to make the abandoned work auditable:

- `/tmp/moe_round18_d_on/res_d_on.json`
- `/tmp/moe_round18_d_off/res_d_off.json`

## Experiment F — progressive output publication

### Hypothesis and protocol constraint

The compute kernel currently reserves and pushes the entire output block after all mrow calls, so
the writer cannot issue row `r` while compute produces `r+1`.  Publish completed output rows while
pinning the pack base correctly.  The writer may wait for cumulative rows and issue from explicit
offsets, but it must not pop storage until a write barrier proves every issued DMA no longer reads
the CB.  The phase-alias `SEM_PHASE_FREE` edge remains after that barrier.

The `writer_out_issue` zone starts before `cb_wait_front`; its occupancy is not itself output issue
time.  The win must be established end-to-end.

### Result

**RETAINED.** The compute kernel reserves the full output block as before, but the mrow path now
pushes each completed `EC_MAX`-tile row immediately.  The writer waits for cumulative row prefixes
and issues each row as it arrives.  It deliberately does not pop: the existing next-block/epilogue
write barrier still proves that every DMA has stopped reading the CB before the full block is
released, preserving the phase-alias edge.

The full nine-test shared-region suite passed, including both input layouts, preallocated outputs,
offset rebasing, zero count, and determinism. Seven-repetition BF16-RM/BFP4-ND medians on 8x11:

| M | E control | progressive output | delta |
|---:|---:|---:|---:|
| 128 | 85.710 us | 85.260 us | -0.450 us (-0.5%) |
| 196 | 108.023 us | 103.910 us | -4.113 us (-3.8%) |
| 224 | 107.875 us | 103.840 us | -4.035 us (-3.7%) |
| 256 | 113.471 us | **109.320 us** | **-4.151 us (-3.7%)** |
| 512 | 192.683 us | **186.780 us** | **-5.903 us (-3.1%)** |
| 1024 | 342.141 us | **332.650 us** | **-9.491 us (-2.8%)** |
| 5120 | 1540.921 us | **1500.380 us** | **-40.541 us (-2.6%)** |

Artifact: `/tmp/moe_round18_f_on/res_f_progressive.json`.

## Experiment A — two four-row M-groups

### Intended full-block geometry

- Rows 0..3 and 4..7 are two independent 11x4 multicast/output groups.
- Each group partitions output emb across 44 cores: `ec_max=6` for Kimi and 5 for GLM.
- Each core consumes four full hidden rows.  The two rectangles can advance concurrently.
- Phase-2 output storage is `max(8*3, 4*6) = 24` tiles for Kimi, not `8*6 = 48`.
- Resident W_down capacity grows from 114,048 B to 228,096 B/core.  With the safe BF16 phase-scratch
  alias described below, `DEPTH_H=2` funds the change and leaves 27,840 B under the descriptor's
  actual 1,461,376 B budget.  Keeping `DEPTH_H=3` still misses that budget by 41,792 B.

### Runtime-M and traffic constraints

M is device-runtime.  One compiled program must retain the ordinary small-M ownership and select
the grouped layout only after reading `m_blocks`; merely branching the matmul is insufficient.
The W_down read mapping, output `jstart/ec`, CB strides, writer row mapping, and ragged final block
must all agree with the selected mode.

Duplicating resident Kimi W_down adds approximately 8.26 MB of first-block DRAM traffic, a 16 us
ideal DRAM floor at 512 GB/s.  Therefore the initial threshold will be conservative and measured;
streaming the duplicated 16.5 MB every M-block is not the initial design.  A later version may
have one group load each shard and transfer it once to its paired group over NoC.

### Result

**REJECTED and disabled (`WD_MGROUPS=False`).** The complete runtime-M implementation was made
correct first: one resident W_down layout per dispatch, two 11x4 multicast rectangles, four
concurrent rounds per group, group-relative output ownership, and an exact-full-block runtime
guard.  The 9-test region suite and the M=5120 golden test passed; the latter was bit-stable with
PCC 0.984182.

The first profile was a useful false-control: the descriptor compiled `WD_MGROUPS=0` because the
real L1 budget rejected the wider shard.  It measured 186.536/259.636/333.020/625.582/1501.073 us
at M=512/768/1024/2048/5120.  Its compile args and 6-tile physical weight shard were inspected
before it was classified as a control rather than an A result.

The safe funding arm used `DEPTH_H=2`; compile args confirmed `WD_MGROUPS=1`, the down weight shard
was six tiles, and the grouped runtime path activated at M>=1024.  Seven-repetition medians:

| M | control | two groups, DEPTH_H=2 | delta |
|---:|---:|---:|---:|
| 512 | 186.536 us | 194.861 us | +8.325 us (+4.46%) |
| 768 | 259.636 us | 272.775 us | +13.139 us (+5.06%) |
| 1024 | 333.020 us | 342.958 us | +9.938 us (+2.98%) |
| 2048 | 625.582 us | 631.472 us | +5.890 us (+0.94%) |
| 5120 | 1501.073 us | 1497.615 us | -3.458 us (-0.23%) |

The runtime threshold cannot protect small counts from the statically smaller `cb_h`; M remains
device-runtime in one capacity-5120 program.  A 0.23% large-M win does not justify 3-5% regressions.

Two attempts to retain `DEPTH_H=3` by aliasing phase storage were rejected on correctness:

- `gather_up` with `cb_h`: a receiver may accept the next h payload while compute still consumes
  its local gather buffer.  Repeated preallocated-output calls diverged.
- `gate_acc` with `h_local`: a diagonal aggregator can receive a fragment from another column
  before its own column's gate accumulator drains.  BF16 RM happened to pass, but tiled BFP8 input
  exposed PCC 0.3683.

The independent `gate_silu` / `out_interm` BF16 alias is retained: both buffers are compute-owned,
strictly phase-disjoint, and it saves 12,288 B/core.  All alias invariants and the full 9-test region
suite pass after disabling A.

Artifacts:

- `/tmp/moe_round18_a_on/res_a_on.json` (compile-disabled control)
- `/tmp/moe_round18_a_depth2/res_a_depth2.json` (active grouped schedule)

## Experiment B — whole-round dual-NoC h transport

Whole-round ownership preserves a linked payload+VALID chain on one NoC and is better founded than
the losing byte-wise HSPLIT.  It still moves some sends to the writer while the reader clears the
shared per-slot VALID cell.  Before implementation it needs an explicit same-core slot-ready edge
or a private relay flag; the previous off-loop linked-flag sender hung on exactly this reset race.
The theoretical prize is bounded by h transport (roughly 13 us/full block before other work), so
the implementation must not buy ordering with an acked grid-wide barrier.

### Result

**REJECTED and disabled (`H_ROUND_NOC1_MASK=0`).** The implemented ownership is whole-round, not
the failed byte-wise HSPLIT: a diagonal writer waits directly on the existing h-slice arrival and
per-sender H_FREE counters, self-copies its complete row, sends one linked payload+flag chain on
NoC1, flushes and resets its own flag, then publishes same-core completion in the existing mailbox.
Receivers reserve/ack ahead, so writer rounds can launch concurrently with intervening reader
rounds.  No payload, flag or reset crosses a RISC ownership boundary.

The full 9-test region suite passed with the 5/3 split, including both input formats, region offsets,
preallocated outputs and deterministic repetition.  Seven-repetition BF16-RM/BFP4-ND medians:

| M | NoC0-only | 5/3 mask `0x92` | 4/4 mask `0xAA` |
|---:|---:|---:|---:|
| 128 | 85.38 us | 85.47 us | 85.18 us |
| 256 | 107.86 us | 108.85 us | 108.79 us |
| 512 | 186.27 us | 185.75 us | 186.07 us |
| 1024 | 332.51 us | 332.11 us | 332.26 us |
| 5120 | 1500.86 us | 1499.70 us | 1501.31 us |

Both masks are noise-sized overall and regress the important M=256 cell by approximately 0.9 us.
The h transport is therefore no longer the exposed end-to-end critical path after F; moving it is
not a useful trade even with a correct concurrent protocol.

Artifacts:

- `/tmp/moe_round18_b/res_b_off.json`
- `/tmp/moe_round18_b/res_b_5_3.json`
- `/tmp/moe_round18_b44/res_b_4_4.json`

## Experiment G — merged gate/up scatter payload

The current gate and up legs deliberately run concurrently on different RISC-Vs and NoCs.  A merged
12-tile payload halves transaction count only if the new single-owner lifecycle does not serialize
twice the bytes onto one path.  Implement only if profiling after the earlier changes still shows
the scatter transaction rate on the critical path.

### Result

**PARTIALLY RETAINED as the safer one-signal protocol; full payload merging remains deferred.**
The profile justified attacking the rendezvous but not changing the payload layout.  At M=256,
the baseline's slowest `writer_scatter` core spent 41.459 us and the slowest `reader_reduce` core
spent 76.174 us.  A literal merged 12-tile payload would require interleaving two independently
owned accumulator CBs, choosing one lifecycle owner, and either serializing the payload onto one
NoC or adding a raw second-RISC handoff plus a strided/deinterleaving reduce.

The retained implementation leaves the gate and up writes concurrent and byte-identical: writer
owns gate on NoC1, reader owns up on NoC0.  The reader's payload barrier then publishes one
same-core monotone mailbox word.  After its own gate barrier, the writer waits for that word and
emits one completion per destination.  A target now waits for `KGROUPS` signals rather than
`2*KGROUPS`; each signal proves both source payloads have landed.  Thus atomic fan-out is halved
without sharing a CB between RISC-Vs or serializing the data bytes.

The full 9-test region suite passed with the path enabled, covering BF16-RM and tiled input,
BF16/BFP8 output, offsets, preallocated output and deterministic repeated dispatch.  The reverse-
order 11-repetition BF16-RM/BFP4-ND A/B on the fixed 11x8 API grid was:

| M | two signals | one signal | delta |
|---:|---:|---:|---:|
| 128 | 85.38 us | 85.23 us | -0.15 us (-0.18%) |
| 256 | 108.77 us | 107.73 us | -1.04 us (-0.96%) |
| 512 | 186.50 us | 186.01 us | -0.49 us (-0.26%) |
| 1024 | 332.70 us | 331.98 us | -0.72 us (-0.22%) |
| 5120 | 1500.39 us | 1497.74 us | -2.65 us (-0.18%) |

A separate seven-repetition off/on run gave deltas of -0.66/-0.75/+0.07/-0.60/-3.43 us at the
same counts, confirming the M=256 and large-M wins while showing that the sub-microsecond cells are
near measurement noise.  A focused M=256 trace moved whole-op time 107.391 -> 107.128 us and the
slowest `reader_reduce` core 76.174 -> 75.602 us.  `writer_scatter` mean grows because it now owns
the cross-RISC rendezvous, but the target reader is released earlier.  Default is enabled; set
`MOE_SCATTER_ONE_SIGNAL=0` for the control.

Artifacts:

- `/tmp/moe_round18_g/res_g_off.json`, `/tmp/moe_round18_g/res_g_on.json`
- `/tmp/moe_round18_g_reverse/res_g_off.json`, `/tmp/moe_round18_g_reverse/res_g_on.json`
- baseline trace report `generated/profiler/reports/2026_08_06_14_16_56` under shared `TT_METAL_HOME`
- `/tmp/moe_round18_g_profile_on_zones.txt`

## Campaign outcome and final audit

The shipped/default 8-row x 11-column configuration retains four changes from this campaign:

1. phase-0 idx/count reads share one DRAM barrier (E);
2. full-row W_down output is published and issued progressively (F);
3. the phase-disjoint BF16 SiLU/partial buffers share one allocation, saving 12,288 B/core;
4. gate/up scatter retains concurrent two-NoC payloads but uses one completion per contributor (G-lite).

The public grid resolver now caps both implicit and explicit requests at API `core_grid=(11, 8)`;
11x10 cannot be selected.  C was restored, D is out of scope, and the complete A/B experiments
remain compile-time disabled (`WD_MGROUPS=False`, `H_ROUND_NOC1_MASK=0`) so their measured protocol
implementations are available without changing the default path.

Final 11-repetition BF16-RM/BFP4-ND medians for the resolved shipped knobs are the `g_on` values
from the reverse-order run.  M=0 is E's dedicated 31-dispatch measurement; it was measured
separately because longer sweeps obscure a 100-ns fixed-cost change.

| M | final device time |
|---:|---:|
| 0 | 3.376 us |
| 128 | 85.23 us |
| 256 | 107.73 us |
| 512 | 186.01 us |
| 1024 | 331.98 us |
| 5120 | 1497.74 us |

Final evidence on the default configuration:

- `git diff --check`: pass.
- CT-argument, alias, grid, weight-dtype and region-fusion suite: 27 passed.
- Core golden plus every runtime `m_eff` regime: 32 passed.
- Guard determinism matrix, 10 repeats/cell across 12 BF16/tiled shapes plus the negative
  control: 13 passed.
- Runtime-tail golden sweep through M=5120: 19 passed before the following weight-test fixture
  exposed a stale L1 expectation; after retargeting the fixture, the genuine non-resident
  four-M-block path passed independently.
- Preferred/tall/interleaved weight-placement suite: 4 passed.
- Interleaved race stress: 5,000 dispatches across 23 shapes plus 207 zero-count dispatches,
  bitwise stable at every checkpoint.  Its pass line explicitly reported
  `WD_MGROUPS=False`, `H_ROUND_NOC1_MASK=0`, and `SCATTER_ONE_SIGNAL=True`.

### Follow-up: hidden=1024 was a BFP8 phase-alias addressing bug (fixed)

The failing stress cell was deterministic corruption, not a scheduling race and not an input-tail
problem.  It affected both BF16 row-major and BFP8 tiled inputs when the output was BFP8; BF16 output
was clean.  A block-by-block comparison exposed an alternating pattern at hidden=1024: block 0 was
clean, block 1 corrupt, block 2 clean, block 3 corrupt.  Pairwise alias ablations isolated the
necessary pair to `CB_GATHER_GATE + CB_OUT_TILES`; `gather+h` and `h+out` were clean.

The physical cause is the difference between a logical CB cursor and its aliased allocation.  At
hidden=1024 the three logical BFP8 views contain `(24, 6, 48)` pages and share their 48-page LCM.
The reader advances `CB_GATHER_GATE` by its logical 24-page capacity, so successive M-blocks expect
physical offsets 0 and 24.  The writer never reserves/pushes that landing CB and used its own local
`get_write_ptr(CB_GATHER_GATE)` as the destination address on every remote core.  Its independent
CB-interface cursor therefore remained at offset 0.  On odd M-blocks the gate scatter landed in the
wrong half while the receiver consumed the other half.  The values could be finite but wrong (the
second 256-token block measured PCC approximately 0.38), so repeat determinism and finiteness alone
were not sufficient guards.

Two fixes were implemented and measured:

1. restricting the phase alias to `gather+h`, which was correct but gave back 52,224 B/core at the
   graded hidden=2048 shape;
2. retaining the full three-way alias and deriving the remote physical address explicitly as
   `base + ((m_block * gather_pages) % phase_alias_pages) * tile_bytes`.

The second fix is retained.  It preserves the 58,752 B/core three-way BFP8 phase saving (71,040
B/core including the independent BF16 scratch alias) and had no systematic measured performance
cost.  The hidden=512, 768, and 1024 multi-block lattices are clean after the change; hidden=512 and
768 were also potentially vulnerable because their logical gather cycle was smaller than the
physical alias.  Hidden=1792/2048 had equal 48-page logical/physical gather cycles and could not hit
this particular mismatch.

A permanent regression now compares each 256-token BFP8-output block against the same device
computation with BF16 output for both supported input layouts.  Exact-boundary and tail determinism
tests (`M=512/513`, five repetitions each), all region-fusion modes, every weight dtype, and the
non-resident four-M-block W_down path pass with the explicit-address fix.

Final logs:

- `/tmp/moe_round18_final_default_suite.log`
- `/tmp/moe_round18_final_golden_core.log`
- `/tmp/moe_round18_final_determinism.log`
- `/tmp/moe_round18_final_stress_finite.log`
- `/tmp/moe_round18_stress_257_h1024_head.log` (detached-HEAD baseline control)
