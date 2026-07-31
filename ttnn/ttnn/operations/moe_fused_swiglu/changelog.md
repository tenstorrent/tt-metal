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
| **M** tokens | sequential outer loop | `M_BLOCK` (CB sizing) / runtime `m_eff` (work) | 8 / `pow2_ceil(tail M_t)` |
| **x** over Hn | rotating injector + ROW multicast | — | 1 tile-row per injector |
| **h** over Ne | grid-wide multicast, `HGROUPS` rounds | `DEPTH_H` | 3 |

Buffer depths: `DEPTH_W=2`, `DEPTH_H=3`, `DEPTH_OUT=2`, `DEPTH_XSTAGE=1`, `XSTICK_ROWS=1`.
Read knobs: `WRUN=8`, `WD_AHEAD=1`. Sub-block: `OUT_SUBBLOCK_H=1`, `out_subblock_w = HN_PAD` (gate/up)
/ `ec` (down). L1: **~1.35 MB** of the 1.43 MB budget.

Every CB page count, loop trip count and grid formula derives from those parameters. `count` is the
only runtime value and enters ONLY as `m_blocks` and the per-block `m_eff` (Refinement 1) — never as
a CB size.

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
a destination that was never addressed. Both families here loopbacked (`src cb != dst cb`), so both
tripped it. `Flag` is used instead, documented at the emission site.

> **UPDATE (Refinement 1).** Neither family loopbacks any more — both sends are now `src == dst`
> (EXCLUDE-source), because the LOOPBACK shape carries a *second*, independent `mcast_pipe` bug: the
> rotating-sender Flag reset races the sender's own in-flight VALID. See the Refinement 1 entry below
> and the hazard note at `mcast_pipe.hpp`'s `ROTATING_SENDER`. This also removes the loopback fan-out
> that trips the Counter path's atomic barrier, so a future Counter retry (Refinement 2 lever 4) now
> only has to solve the linked-chain/command-buffer half.

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
   > **CORRECTION (verifier pass).** This is NOT reachable by freeing the L1 that (1) needs. The
   > M-scaled CBs total ~855 KB (`cb_x_tiles` 195.5 + `cb_reduce_*` 102 + `cb_h` 153 +
   > `cb_out_tiles` 51 + `cb_*_acc`/`*_send`/`gate_silu`/`h_local` 306 + `cb_out_interm` 48), so
   > doubling `M_BLOCK` costs **+855 KB** against **159 KB** of measured slack. It becomes reachable
   > only after the 310 KB in `cb_w_gate`/`cb_w_up` is broken up, i.e. after `KB1_FRACTION < 1` and
   > therefore the second-CB copy of `x` (`op_design.md` §6). See `op_requirements.md` Refinement 3.
4. Fix the `mcast_pipe` Counter/loopback fan-out bug upstream, then re-measure Counter: it removes
   the per-round flag-reset round trip that currently serialises all 11 h rounds.
5. `WD_AHEAD` and the W_gate pre-issue are parked at their trivial defaults; both become worth
   turning only after (1)/(2) move the critical path off the collectives.

---

# Changelog: moe_fused_swiglu

## Phase 0 — Core Implementation (verified)

- **Date**: 2026-07-31
- **What was done**: Initial implementation via the incremental pipeline (planner -> implementer ->
  verifier). One device program: fused tilize + three bfp4 matmuls + SwiGLU + two Tensix->Tensix
  collectives, with the token count read on device. Sections 1-5 above are the implementer's record;
  this section is the verifier's.
- **SUPPORTED at Phase 0**: `input_format=[bf16_rm, bfp8_tile]`, `weight_dtype=[bfloat4_b]`,
  `emb=[6144, 7168]`, `capacity=[1024, 2048, 5120]`, `fill=[balanced, partial, full, empty]`
  (= TARGET on every axis; `EXCLUSIONS = []`, `INVALID = []`)
- **Accuracy achieved**: PCC = 0.97905-0.97992, max_abs_err = 2.21e5-2.69e5,
  mean_abs_err = 3.46e4-4.06e4, relative RMS = 0.214-0.222, got/true ratio median 1.036-1.045
  (p5 -0.32 / p95 2.40) — measured on 4 shapes x 2 activation formats via
  `test_moe_fused_swiglu_precision_baseline.py`. The measured `bfloat4_b` format floor for the same
  shapes is 0.97972-0.98049, so the **kernel-attributable** shortfall is 5.7e-4-6.8e-4 and the error
  is format-dominated broadband noise (not a scale bug — the ratio spread is enormous around ~1.0).
- **Golden suite at Phase 0**: **1 / 45 cells passing** (`verifier_report.json`) — 44
  `supported_fail`, every one `numerical-precision` at `pcc 0.9789-0.9796` against a `0.98` gate that
  sits ABOVE the measured format floor (0.97967-0.98019 on the suite's own fixture), i.e. unreachable
  by any correct implementation. 0 `xpass_drift`, 0 `xfail_wrong_mode`, 0 `xfail_expected`
  (SUPPORTED == TARGET), 0 hangs, 0 OOM, no inf/NaN. All 45 cells ran on 110/110 cores.
- **Perf at Phase 0** (emb 7168, cap 5120, bf16_rm): count 128 = 221 006 ns / util 0.235
  (target 91 800 / 0.566), count 256 = 227 123 ns / 0.245 (target 108 000 / 0.514),
  count 512 = 439 863 ns / 0.143 (target 161 816 / 0.388). Capacity costs nothing (226 772 /
  227 776 / 227 123 ns at cap 1024 / 2048 / 5120 for count 256). bfp8_tile at count 256 =
  218 430 ns / 0.239. count = capacity = 4 334 803 ns / 0.044 (report only). count = 0 = 6 026 ns.
- **Issues encountered / fixed in the verifier pass**:
  1. `mcast_pipe.inl::signal_ready_` handed the multicast atomic increment the INCLUDE-source
     (loopback) fan-out while `noc_semaphore_inc_multicast` is unconditionally exclude-source, so
     `fence_()`'s non-posted atomic barrier could never complete. **Fixed** (always
     `num_dests_excl_`). Re-testing `DataReadySignal::Counter` on device then exposed a **second,
     independent** blocker — the linked data multicast is only terminated by an unlinked transaction
     on the same command buffer, and the Counter signal goes out on a different one, so the sender
     hangs in `noc_cmd_buf_ready()`. Documented at `mcast_pipe.hpp`'s `DataReadySignal` and at this
     op's emission site; the op ships on `Flag`.
  2. `cb_w_down` was sized to a whole-op dimension (`HGROUPS * HN_PAD == HID_T`, the entire `down`
     contraction extent, 111.4 KB). Replaced with a real depth knob `DEPTH_WD` (+ the FIFO-wrap
     precondition derived host-side, with a documented fallback that keeps the `WD_AHEAD` ablation
     legal). A/B measured over the 9 loose cases: depth 3 = +1.68 %, **depth 5 = +0.15 % (noise)**,
     depth 7 = +0.53 %, depth 11 (old) = baseline. Shipped at 5: **60.8 KB freed for no time**.
  3. 48 KB of unreachable L1 on the `bfp8_tile` path — `cb_x_in` / `cb_x_stage` serve the row-major
     staging path only and are behind `if constexpr (INPUT_FORMAT == 0)` everywhere; now 1 page each
     in that configuration. Per-core L1: **1349 -> 1267.9 KB** (bf16_rm) / **1199.6 KB** (bfp8_tile).
  4. DRY: the bank-run coalescing existed in **six** places (two copies of `remap_n`/`run_len`, four
     copies of the run-enumeration loop) so a `WRUN` turn had to land consistently six times. Now one
     definition in `kernels/moe_fused_swiglu_bank_runs.hpp`. The mailbox word layout was three sets of
     bare `mbox[0..3]` literals; now named once in `kernels/moe_fused_swiglu_common.hpp` (include-free
     so the compute TU can use it). `dest_limit = 8` -> `DEST_AUTO_LIMIT_TILES`; a variable named
     `counter` that held `Flag` -> `data_ready_signal`.
  5. Found and filed, not fixed (needs a device-ns measurement, so it is Refinement 1): the runtime
     `m_tiles` shrink `op_design.md` §3 specifies was never implemented — the op always does
     `M_BLOCK = 8` tile-rows, which is why count 128 and count 256 take the same time.
  6. Corrected the `M_BLOCK = 16` "knob turn" claim in §5 above (it needs +855 KB of L1).
- **Tests added**: `test_moe_fused_swiglu_precision_baseline.py` (PCC / abs / relative-RMS / got-true
  ratio spread + the measured bfp4 format floor, 4 shapes x 2 formats). Pre-existing and re-run:
  `test_moe_fused_swiglu.py` (acceptance), `test_moe_fused_swiglu_debug.py` (3 structural tests, all
  exact), `test_moe_fused_swiglu_perf.py`.
- **Action required of the harness owner**: relax the golden `_PCC_GATE` (0.98) below the measured
  format floor, or quantize the reference weights. Until then no correctness cell can pass and the
  44 red cells say nothing about the kernel. See `verification_report.md`.

## Refinement 1 — Honour the runtime token count (`m_tiles`), instead of always doing `M_BLOCK`

- **Date**: 2026-07-31
- **What was done**: `op_design.md` §3's `m_tiles` is now real. Each M-block works a RUNTIME
  `m_eff = m_tiles_eff(M_t, b, M_BLOCK, M_EFF_MIN)` token tile-rows instead of the constant
  `M_BLOCK = 8`, so `count 128` does 4 tile-rows and `count 32` does 1.
  - **One shared inline**, `moe_fused_swiglu::m_tiles_eff()` in `kernels/moe_fused_swiglu_common.hpp`,
    called from all three kernels. It is a pure function of `(m_t, b, M_BLOCK, M_EFF_MIN)` — all four
    identical on every core and RISC-V — so the reader's multicast round count, compute's
    `MatmulBlockShape` and the writer's CB waits are the same number by construction, which the
    collectives require. `inject_rows()` sits beside it and replaces the host-computed `n_inject`
    runtime arg (that constant was pinned to `M_BLOCK`); compute now takes `my_col` instead.
  - Tail blocks round UP to a **power of two <= `M_BLOCK`** (`m_eff ∈ {1,2,4,8}`), never to a raw
    `M_t`: every M-scaled CB is sized `DEPTH * M_BLOCK * W`, so a power-of-two `m_eff` divides the
    total and no shrunk reserve can straddle a FIFO end. `M_BLOCK` is now host-asserted to be a power
    of two, and `M_EFF_MIN = pow2_ceil(OUT_SUBBLOCK_H)` keeps `m_eff / OUT_SUBBLOCK_H` exact (so
    Refinement 2's plan to split `OUT_SUBBLOCK_H` stays a knob turn).
  - Threaded through: `cb_x_tiles`, `cb_h`, `cb_h_local`, `cb_gate_send`/`cb_up_send`,
    `cb_out_interm`, `cb_out_tiles`, both `MatmulBlockShape`s, the bias-add walk, the x-round loop and
    the output row loop. **One deliberate exception**: `cb_reduce_gate_in`/`cb_reduce_up_in` keep the
    FULL `M_BLOCK * HN_PAD` reserve/push/pop, because the child unicasts to its OWN
    `get_write_ptr(cb_reduce_*_in)` as a proxy for the parent's and that only holds while every push
    is a whole slot (which always wraps back to the CB base, on every core, whatever its child
    count). The child ships only the `m_eff * HN_PAD` live tiles — so the expensive part, the up-to-4
    serial round trips, halves anyway — and compute adds those and drains the tail.
- **A PRE-EXISTING correctness bug this exposed, and fixed** (the bulk of the work):
  `mcast_pipe`'s rotating-sender Flag reset — `data_ready_.set(INVALID)` behind a `fence_()` that is
  `async_writes_flushed()`, i.e. **SENT, not LANDED** — races the sender's own `MCAST_INCL_SRC`
  **loopback** VALID write (`loopback = in_rect_ && src_l1 != dst_l1`). When the late VALID wins, the
  sender's next `receive()` returns on a stale flag, every later round shifts one early, and the
  block's LAST round is consumed before its data lands. Present since Phase 0 on BOTH collectives and
  silent — it was masked by the `(m_eff-1) * KR_PAD` tile-matmuls of cover between the CB push and
  the read, which is exactly the cover `m_eff` removes. Symptom was run-to-run PCC of 0.955-0.979 on
  byte-identical input at `m_eff ∈ {2,4}`, with occasional NaN. Both sends now land their own copy
  locally under `noc_async_read_barrier()` (a real arrival guarantee) and multicast **in place**
  (`src == dst` -> EXCLUDE-source), which is the shape the op's bfp8 x path always had — which is why
  bfp8 was immune to the x half and pinpointed the h half. The hazard is now documented at
  `mcast_pipe.hpp`'s `ROTATING_SENDER`; fixing it inside the pipe (an acked barrier on the loopback
  path) would be better but is a shared-`kernel_lib` change and belongs with Refinement 2's lever 4.
  Both self-copies are then **hidden**, not paid:
  1. all x staging (DRAM stick read + fused tilize + self-copy) moved into a per-injector
     **prologue** before the multicast chain. It has no cross-core ordering, so all `m_eff` injectors
     now stage CONCURRENTLY instead of each one stalling its own round while the rest of the row sits
     in `receive()`. This also takes the DRAM read and the tilize off the multicast chain — a win in
     its own right, worth ~4 % on its own (235 631 -> 227 346 ns at `count 256`).
  2. the h self-copy is ISSUED before the W_down prefetch so ONE `noc_async_read_barrier()` covers
     both; it rides the DRAM read's latency (226 177 vs 232 106 ns at cap 2048). The two `cb_h` /
     `cb_w_down` reserves were swapped to allow this — safe because `cb_w_down` always runs
     `WD_AHEAD` blocks ahead of `cb_h`.
- **Accuracy achieved**: PCC **bit-identical to Phase 0** — all 12 re-measured golden cells match to
  `0.0e+00` (m_eff 1 / 2 / multi-block regimes), acceptance PCC 0.97905-0.97955 unchanged, all 8
  precision-baseline rows unchanged at kernel-attributable dpcc 5.66e-4-6.78e-4, 3 exact structural
  debug tests still exact. `m_eff` only removes work on UNDEFINED rows, so this is the expected
  result, and it is the evidence for it.
- **Perf** (`device_kernel_ns`, graded loose cases, two independent runs agreeing within 0.2 %):

  | cell (emb 7168, cap 5120, bf16_rm) | Phase 0 | now | delta | util |
  |---|---|---|---|---|
  | `count 128` (the target) | 223 496 | **151 620** | **-32 %** | 0.233 -> 0.343 |
  | `count 256` | 226 771 | 227 795 | +0.5 % (noise) | 0.245 -> 0.244 |
  | `count 512` (`m_blocks = 2`) | 442 463 | 439 679 | **-0.6 %** | 0.142 -> 0.143 |
  | `count = capacity` | 4 351 747 | 4 279 071 | **-1.7 %** | 0.044 -> 0.045 |
  | 9-cell sum | 6 132 909 | 5 993 786 | **-2.3 %** | |

  Guard set clean (cap 1024 / 2048 at count 256 within +1.1 %, emb 6144 +0.0 %, `count 0` still
  ~6.1 us and hang-free); the one cell marginally above noise is `bfp8_tile` at +1.1/+1.5 %, the
  residual of the h self-copy on the path that gains least from the prologue. `count 256` being flat
  is the fix landing correctly: `M_t = 8` means `m_eff = M_BLOCK`, i.e. byte-identical work.
  `m_eff` cannot touch the **weight stream** (87 % of read bytes, count-independent), so -32 % is
  near its honest ceiling; the remaining gap to the 91 800 ns target is Refinement 2/3 territory.
- **Issues encountered**: the `mcast_pipe` loopback race above — found with the static analyzer after
  the m_eff shrink turned it from latent to reproducible, then localised by a 4-rep determinism probe
  plus the observation that `bfp8_tile` (no x loopback) was still racy, which isolated the h send.
- **Tests added**: `test_moe_fused_swiglu_m_tiles.py` (16 cases) — asserts **repeat-determinism**
  (3 dispatches, bit-identical output) across every `m_eff` regime x both formats, because the bug
  class here is silent and a single-shot accuracy test passes straight through it; plus `m_eff`
  numerics-invariance under `input_m_tiles` and the `count == 0` no-collective path. Probes
  `probe_008`-`probe_013` document the investigation.

## Refinement 1b — Honour the runtime token count (`m_tiles`), instead of always doing `M_BLOCK` (debug: fix gate violations)

- **Date**: 2026-07-31
- **What was done**: the harness's completion gate failed Refinement 1 on bullet 2 with
  `test_moe_fused_swiglu[bf16_rm-emb=7168-capacity=1024-count=32] AssertionError: 0.979224186218931`.
  The cause is **not a kernel defect** — it is a **stale gate literal**. The acceptance test carried a
  copied `PCC_GATE = 0.98` while its own docstring designates
  `eval/golden_tests/moe_fused_swiglu/feature_spec.py` as the source of that number, and the operator
  relaxed that source to **0.975** on 2026-07-31 (the action Phase 0's changelog asked for under
  "Action required of the harness owner"). The copy never followed. Two independent measurements
  establish that before any code was touched:
  1. **Refinement 1 changed nothing numerically.** Re-running the acceptance file against the Phase-0
     kernels (`git checkout 2c4b563cb0 -- <op code files>`, then restore) reproduces **all 12 failing
     PCCs bit-for-bit** — 0.979224186218931 / 0.9790893717375072 / 0.9791745362399347 / … identical to
     the last digit. The named cell failed identically at Phase 0, so the harness surfaced a
     pre-existing condition rather than a Refinement-1 regression. It also independently re-confirms
     Refinement 1's "PCC bit-identical" claim on a 12-cell fixture the harness picked, not one I did.
  2. **0.98 is unreachable by ANY correct implementation.** On the acceptance file's exact fixture
     (`probes/probe_015.py`, all 5 SHAPES x both formats), the ceiling for a *bit-exact* kernel — the
     torch fp32 chain carrying only the bfp4_b weight quantization, plus the bfp8_b `h` and bfp8_b
     output this op's signature mandates — is **0.97966-0.97983, i.e. below 0.98 on 10/10 cells**:

     ```
       fmt        emb   cap   cnt | floor_w  floor_op |  device | ceil-dev | 0.98 vs ceiling
       bf16_rm   7168  1024    32 | 0.97986   0.97983 | 0.97922 | 6.06e-04 | UNREACHABLE
       bf16_rm   7168  1024   255 | 0.97972   0.97966 | 0.97909 | 5.69e-04 | UNREACHABLE
       bf16_rm   6144  2048   128 | 0.97975   0.97969 | 0.97917 | 5.20e-04 | UNREACHABLE
       bf16_rm   7168  2048   512 | 0.97975   0.97969 | 0.97910 | 5.93e-04 | UNREACHABLE
       bf16_rm   6144  1024  1024 | 0.97979   0.97973 | 0.97920 | 5.27e-04 | UNREACHABLE
       bfp8_tile (same 5 shapes)  | 0.97972.. 0.97966.. 0.97907.. 5.15e-04.. UNREACHABLE (5/5)
     ```

     The kernel-attributable residual is only **5.2e-4-6.2e-4**; the gate was short of its own
     ceiling by 1.7e-4-3.4e-4 *before the kernel ran at all*.
  - **The fix**: both stale copies now **import** the gate instead of duplicating it, so they cannot
    drift again — `test_moe_fused_swiglu.py`'s `PCC_GATE` and
    `test_moe_fused_swiglu_precision_baseline.py`'s `GOLDEN_PCC_GATE` (the latter was making its own
    printout assert the opposite of the truth). Each has a documented literal fallback so the files
    stay standalone-runnable in a checkout without `eval/`. **Zero kernel, host-descriptor or op-file
    lines changed** — the whole diff is two test files plus docs.
  - **Detection power went UP, not down.** The 0.98 gate was ~2e-2 of slack that graded the *bfp4
    weight format*, not the op; the kernel is really held by this suite's
    `floor_pcc - FLOOR_SLACK(0.0015)` against the per-shape **measured** ceiling, ~13x tighter than the
    change here. Two new invariants retire the drift class permanently: the precision baseline now
    asserts the graded gate is **below** the measured ceiling for every shape (a gate above it grades
    quantization, not the kernel — exactly this bug), and `test_pcc_gate_has_one_source_of_truth`
    asserts the three gate references still resolve to one definition.
- **Accuracy achieved**: unchanged and bit-identical — PCC **0.97907-0.97922** on the 10 acceptance
  numerics cells (5 shapes x 2 formats), against a per-shape measured ceiling of 0.97966-0.97983, so
  kernel-attributable `dPCC = 5.2e-4-6.2e-4`. Golden: worst per-cell delta vs the Phase-0 baseline is
  **-4.8e-07** and **0 cells** moved by more than 1e-4. The three structural debug tests
  (all-ones / hidden-identity / emb-contraction) are still exact, `rel < 0.05`.
- **Golden test progress**: **45 / 45 passing, 0 failed, 0 hangs, 0 errors, 0 skipped** (full suite,
  ran to completion in 2 min 07 s; Phase 0 and Refinement 1 were 1/45, the 44 red cells being this
  one unreachable gate). Unit suite **56 / 56 in 49 s**, comfortably inside the gate's 900 s budget.
  110/110 cores on every cell.
- **Perf**: untouched by construction (no kernel/host change), and the golden run re-measures it for
  free: 9-cell `device_kernel_ns` sum **5 991 794** vs Refinement 1's 5 993 786 (**-0.03 %**, noise).
  `count 128` = 152 750 ns (still **-32 %** on Phase 0's 223 496), `count 256` = 228 324,
  `count 512` = 439 407, `count = capacity` = 4 277 071, `count 0` = 6 064 ns and hang-free.
- **Issues encountered**: the one real trap was diagnostic discipline — the reported symptom is a PCC
  assertion, which reads as a numerics bug and invites a precision hunt through the kernel. Two cheap
  checks killed that reading before any code was touched: the failure is **uniform** across all 12
  cells, both formats and every shape (a kernel bug is shape-, format- or `m_eff`-specific), and the
  Phase-0 A/B is **bit-identical**. Worth noting for the queue: the precision refinement removed by
  the operator would NOT have fixed this — it was scoped to close the 6e-4 kernel-attributable gap,
  which lands at 0.97983, still under 0.98.
- **Tests added**: no new file (the coverage already existed — reused it). Two invariants added to
  `test_moe_fused_swiglu_precision_baseline.py`: a per-shape assertion that the graded gate sits below
  the measured format ceiling, and `test_pcc_gate_has_one_source_of_truth` (host-only) pinning the
  three gate references to one definition. Probes `probe_014`/`probe_015` document the ceiling
  measurement.
