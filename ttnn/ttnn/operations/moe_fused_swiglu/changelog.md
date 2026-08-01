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
| **weights** over M | READ ONCE, reused every M-block | `W_RESIDENT` / `WD_RESIDENT` | on / on (Refinement 3) |
| **x** over Hn | rotating injector + ROW multicast | — | 1 tile-row per injector |
| **h** over Ne | grid-wide multicast, `HGROUPS` rounds | `DEPTH_H` | 3 |

Buffer depths: `depth_w=1`, `depth_wd=HGROUPS`, `DEPTH_X=2`, `DEPTH_H=3`, `DEPTH_OUT=2`,
`DEPTH_XSTAGE=1`, `XSTICK_ROWS=1` (Refinement 3: the weight CBs are filled on M-block 0 and REUSED,
so `DEPTH_W`'s second slot became dead and its 155 KB funds the resident-x double buffer).
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

## Refinement 2 — Break the reduce-path serialisation (the measured 85 %)

- **Date**: 2026-07-31
- **What was done**: measured the four named levers, then fixed the thing the measurement actually
  found. Reused: every existing kernel, CB and collective — no new kernel file, no new CB, no
  protocol rewrite. Added: four transport ablations, five perf knobs (three parked), two barrier
  deferrals, one test file.

### 1. The premise, re-measured — the collectives are 18 %, not 85 %

`/perf-measure` ablations, each stubbing ONE payload while keeping every CB reserve/push/pop and
every loop trip count (`MOE_SWIGLU_ABLATE=...`, `+`-separated to peel stages off cumulatively).
bf16_rm, emb 7168, cap 5120, one fresh-cache run each:

| variant | count 256 | vs base | count 128 | vs base |
|---|---|---|---|---|
| baseline | 226 134 | — | 151 387 | — |
| `no_x_xfer` (x row-multicast) | 218 719 | **-3.3 %** | 144 287 | -4.7 % |
| `no_reduce_xfer` (whole reduce tree) | 216 651 | **-4.2 %** | 147 547 | -2.5 % |
| `no_h_xfer` (h all-gather) | 205 849 | **-9.0 %** | 132 181 | -12.7 % |
| `no_w_xfer` (all three bfp4 weight streams) | 184 324 | **-18.5 %** | 103 073 | **-31.9 %** |
| all three collectives off | 185 159 | -18.1 % | 121 664 | -19.6 % |
| + `skip_compute` | 164 933 | -27.1 % | 124 274 | -17.9 % |

Three readings, all of which redirected this refinement:
1. **The collectives are 18 %, not 85 %** — and the three together (-18.1 %) cost barely more than
   the h all-gather plus the reduce alone, i.e. they already overlap each other heavily.
2. **The single biggest term is the weight DRAM stream** (-18.5 % at count 256, -31.9 % at 128). It
   is count-INDEPENDENT and carries a hard floor: 26 MB at 512 GB/s is ~51 us, 23 % of count 256.
3. **With every collective AND the matmul math gone, 165 us of 226 remain.** So no single named
   lever could have delivered a large win; the reduce path in particular is capped at 4.2 %.

### 2. The four named levers — each measured alone, each kept as a live knob

All three built levers are **PARKED at a byte-identical default** with their measurement recorded at
the knob's definition in `moe_fused_swiglu_program_descriptor.py`. None was deleted; none costs L1
at its default (so Refinement 3 keeps the space it was promised).

| lever | knob | count 256 | count 128 | emb 6144 | verdict |
|---|---|---|---|---|---|
| 1 — parallel reduce fan-in | `REDUCE_SLOTS_CAP` 1 -> 2 | **+2.0 %** | -0.9 % | +2.8 % | parked at 1 |
| 2 — per-sub-block gate/up blocking | `HN_BLOCK` 6 -> 3 | +0.8 % | -0.7 % | +0.9 % | parked at HN_PAD |
| 3 — phase-2 DEST occupancy | `OUT_SUBBLOCK_H_DN_MAX` 1 -> 4 | +0.7 % | +0.2 % | +1.4 % | parked at 1 |
| 4 — `DataReadySignal::Counter` | — | not built (see below) | | | analysed |

- **Lever 1** was implemented in full and is fan-in-general: the parent invites its children in
  WAVES of `REDUCE_SLOTS` into disjoint landing slots (child `c` owns slot `c % REDUCE_SLOTS`, its
  own runtime arg) and waits once per wave on the monotone `SEM_DATA`. The whole-CB reserve/push
  granularity is preserved, which is what keeps the child's "my write pointer IS the parent's
  landing address" proxy valid on every core. It **regresses** at count 256 for a measured reason:
  the entire reduce transport is only 4.2 %, roughly half of that is destination-port BANDWIDTH
  (four children write 4 x 102 KB into ONE core's L1, which concurrency cannot speed up), and the
  wave protocol gives up the interleave the one-slot protocol had — child `c`'s transfer used to
  overlap child `c-1`'s in-place `add`. Turn it when the transport becomes latency-bound.
- **Lever 2** is layout-safe at any width (verified from `OutputCBLayout::SubblockMajor`'s
  in0-outer/in1-inner walk: at `out_subblock_h == 1` the order stays `m*HN_PAD + n`, the exact
  `cb_h` in0 order phase 2 reads), including the ragged column's narrowed last sub-block. It is a
  wash because it HALVES the DEST sub-block (6 -> 3 tiles, which `matmul_output_subblock` measures
  as a real slowdown) while the overlap it exists to enable — §4.3's "sub-block `off`'s reduce
  overlaps `off+1`'s matmul" — additionally needs the REDUCE split per sub-block, and lever 1 just
  showed that reduce is bandwidth-bound, not serialisation-bound.
- **Lever 3**'s height is DERIVED (largest power of two whose sub-block fits DEST, so it tracks
  `EC_MAX` across emb widths) and the kernel takes `min(., m_eff)` at runtime, so it never forces a
  larger `m_eff` and Refinement 1 is untouched. `matmul_output_subblock`'s 1.46x is measured on an
  L1-resident compute-bound matmul; this `down` matmul waits on the h all-gather and on a per-round
  DRAM read, so the bigger DEST sub-block buys nothing and its DEST pressure costs.
- **Lever 4 was analysed and deliberately not built.** Counter on its own cannot break the h round
  chain: `PRE_HANDSHAKE` is what serialises it — a receiver acks round `r+1`'s sender only inside
  `receive(r+1)`, which runs after `receive(r)` returned — and that holds under Flag and Counter
  alike. Counter is the PREREQUISITE for the real lever, LOOK-AHEAD acking (reserve `DEPTH_H` cb_h
  slots, pre-ack the next senders so a sender starts the moment it sees the previous round instead
  of after a full 109-ack round trip). That is unsafe under Flag — one shared cell, so a pre-acked
  sender's VALID is lost — which is exactly why Counter is needed. It needs a new ack-only
  `ReceiverPipe` entry point plus the documented unlinked-send + acked-barrier Counter fix in
  shared `kernel_lib`. Recorded as the sharpest remaining idea; not filed as a follow-up, per the
  perf protocol.

### 3. What actually paid: the barriers, not the collectives

The Goal's own words — *"each stage paying its own latency with nothing else in flight"* — turned
out to describe the **read barriers**, not the transports. `noc_async_read_barrier()` drains EVERY
outstanding read, so a prefetch issued a few instructions before one is not a prefetch at all.

1. **Deferred READ barrier (reader, phase 2).** Each of the 11 h rounds used to issue its W_down
   K-block and barrier it immediately, before the collective — paying the full DRAM latency on the
   spot, on all 110 cores, once per round. **This is why `WD_AHEAD` measured neutral at Phase 0**: no
   prefetch depth can help while the barrier that drains it is the next statement. The issue now
   moves AFTER the round's send/receive and its barrier moves to the NEXT round, so the read lands
   underneath a whole grid-wide multicast (`wd_pending` carries the one in-flight block across the
   round boundary; the sender still drains before it broadcasts, one core per round). Measured at
   count 256 / 128 / emb 6144: **222 446 / 144 133 / 208 565** vs 226 009 / 153 883 / 211 220.
   Re-swept `WD_AHEAD` now that it is live: **1 -> 222 446, 2 -> 225 796, 3 -> 233 552**; shipped at
   1, and `depth_wd` now floors at `wd_ahead + 2` for the extra in-flight block.
2. **Deferred WRITE barrier (writer, output).** The writer twin: the output write-back's
   `noc_async_write_barrier()` sat at its issue site, i.e. exactly between M-block `b` and `b+1`.
   It now drains at the top of the next M-block so it rides that block's W_up read (`DEPTH_OUT >= 2`
   makes the extra outstanding block legal). Costs nothing at count <= 256 (one M-block) and pays on
   the multi-block path: **count 512 -1.7 %, count = capacity -1.8 %**.
3. **Tried and reverted, recorded at the site because it is counter-intuitive**: the x-staging
   prologue's own barriers ALSO drain the W_gate prefetch, so the design's "the gate weight block
   hides under the x multicast chain" never happened. Issuing W_gate after the prologue instead
   measured **WORSE** (223 172 / 145 140 / 210 773) — started later, the read no longer overlaps the
   prologue's own stick reads, and that overlap is worth more than the handshake chain's.

### 4. Results

- **Accuracy achieved**: PCC **bit-identical to the pre-refinement baseline — delta exactly
  0.0e+00 on all 8 numeric loose cells**. Everything here is scheduling; no number changes.
  Golden slice PCC 0.97902-0.97955, rtol/atol untouched, no inf/NaN. Shapes: all 9 loose cells
  (emb 6144/7168 x cap 1024/2048/5120 x count 0/128/256/512/5120 x bf16_rm/bfp8_tile) plus the
  14-cell `cap1024` `test_op` slice.
- **Perf** (`device_kernel_ns`, graded loose cases):

  | cell | before | after | delta |
  |---|---|---|---|
  | `count 128` | 150 543 | **143 785** | **-4.49 %** |
  | `count 256` (the target) | 225 932 | **223 062** | **-1.27 %** (-1.6 % on the 5-run median) |
  | `count 512` (`m_blocks = 2`) | 440 390 | **433 160** | **-1.64 %** |
  | `cap 1024`, count 256 | 225 692 | 220 904 | -2.12 % |
  | `cap 2048`, count 256 | 225 810 | 221 847 | -1.76 % |
  | `bfp8_tile`, count 256 | 224 101 | 218 683 | -2.42 % |
  | `emb 6144`, count 256 | 212 899 | 208 793 | -1.93 % |
  | `count = capacity` | 4 274 336 | 4 204 858 | -1.63 % |
  | `count 0` | 6 011 | 6 013 | +0.03 % (noise, hang-free) |
  | **9-cell sum** | **5 985 714** | **5 881 105** | **-1.75 %** |

  Every cell improved; the guard set is inside it (both formats, both emb widths, all three
  capacities, `m_blocks > 1`, and the no-work path). Run-to-run spread on `count 256` measured at
  ~1.5 %, so the single-cell number is reported alongside the 5-run median and the 9-cell sum, where
  8 of 8 cells moving the same way is the real evidence.
- **Golden test progress**: **9 / 9 loose + 14 / 14 `test_op` slice passing, 0 hangs, 0 errors**,
  110/110 cores on every cell, no cell changed category and no PCC moved at all.
- **Issues encountered**: (a) the four named levers were aimed at a bottleneck the ablations showed
  is 4.2 % — the ablations had to come first, and they cost 8 device runs before a line of lever code
  was written; (b) `MOE_SWIGLU_ABLATE=no_handshake`, which Phase 0 recorded at -5 %, now HANGS (the
  consumer-ready ack IS the cb_h flow control) — it is a broken ablation, not a lever, and the
  no-handshake number in §5 above should not be trusted; (c) the `MOE_SWIGLU_ZONES` phase-zone hook
  is committed and compiled out by default, but the one profiled run with it enabled produced an
  EMPTY device CSV — the hook is unproven, so the next reader should debug it before relying on it;
  (d) `count 256`'s ~1.5 % run-to-run spread is wider than the 0.4 % the requirements assume for the
  9-cell sum, so per-cell claims here are medians.
- **Tests added**: `test_moe_fused_swiglu_r2_knobs.py` (20 cases) — every PARKED knob turned to its
  non-default value must produce **bit-identical** output. A parked knob is precisely what rots:
  the shipped path never touches it, and each of these changes a CROSS-CORE protocol
  (`REDUCE_SLOTS >= 2` switches the reduce tree to wave invites with a per-child slot stride;
  `WD_AHEAD >= 2` changes which K-block the deferred barrier carries across a round boundary), where
  "broken" means a hang or run-to-run garbage rather than a compile error. Covers 2 shapes x 2
  formats x 5 knob settings, on a `count 288` case that spans two M-blocks with a shrunk tail so the
  writer's deferred output barrier is exercised too.

## Refinement 3 — Software-pipeline the M-block (the `count >= 512` cliff)

- **Date**: 2026-07-31
- **What was done**: the heading's premise — "nothing of block `b+1`'s x staging, multicast and
  weight stream is hidden under block `b`'s phase-2 compute" — held, but the sharpest reading of it
  was not "hide the weight stream", it was **"don't re-issue it at all"**. Every weight read in the
  three kernels is a pure function of this core's `kstart`/`hstart`/`jstart` with **no M-block index
  in it** (`BR::read(wg_acc, (kstart+k)*HID_T, hstart, …)` and its W_up / W_down twins), so
  `count 512` was reading 26 MB of bfp4 **twice** and `count = capacity` **twenty times**. The
  harness grades `read_bytes` with the weights counted ONCE (`feature_spec.read_bytes`), so every
  re-read was pure loss against the metric as well as against the clock.
  Reused: every kernel, every CB, every collective — no new kernel file, no new CB, no protocol
  change, and **compute is untouched, bit-for-bit**. Added: two residency knobs, one resident-x
  depth knob, one host precondition guard, one test file.

### 1. Cross-M-block weight residency — the lever, and why it needs no compute change

The CB cycle already returns each weight block to the same L1 slot every M-block, so residency is
purely "skip the DRAM read, keep the handshake":

  * `cb_w_gate` / `cb_w_up` hold ONE block (`depth_w == 1`), so reserve/push always lands at the base;
  * `cb_w_down` holds exactly `HGROUPS` K-blocks against exactly `HGROUPS` pushes per M-block, so
    K-block `r` always occupies slot `r`.

`cb_pop_front` only advances a read pointer — it never clears the bytes — and each weight CB has a
single producer, so the block a later M-block re-reserves still holds what block 0 read into it. The
reader/writer therefore keep the **full** reserve/push/barrier cycle and trip counts, and guard only
the `BR::read` loops with `(b == 0) || !RESIDENT`. That is what makes compute byte-identical and the
whole change ~15 lines of kernel code.

`depth_wd == HGROUPS` is **forced, not chosen**: the invariant is `HGROUPS % depth_wd == 0` and
`HGROUPS` is 11 here — prime — so 11 is the only legal depth above `wd_ahead + 2`. That precondition
is now asserted host-side (breaking it would silently matmul against the WRONG weight block on
`b > 0` only: no hang, no compile error, wrong numbers on the multi-M-block path alone).

### 2. Each lever measured alone (one fresh-cache run per variant, 9 graded loose cells)

| lever | knob | `count 512` | `count = capacity` | 9-cell sum | verdict |
|---|---|---|---|---|---|
| gate/up residency (`DEPTH_W` 2 -> 1, **frees 155 KB**) | `W_RESIDENT` | **-6.25 %** | **-12.47 %** | **-9.36 %** | shipped at 1 |
| + W_down residency (`depth_wd` 5 -> 11, costs 60.8 KB) | `WD_RESIDENT` | a further **-1.52 %** | a further **-2.92 %** | a further **-2.04 %** | shipped at 1 |
| + resident-x double buffer (costs 195.5 KB) | `DEPTH_X` | a further -0.47 % | a further **-1.49 %** | a further -1.07 % | shipped at 2, path-gated |

- **Gate/up residency is free on every single-M-block cell** (all within +-0.53 %), exactly as it
  must be — `b == 0` still reads. It also *pays for* the rest of the refinement: collapsing
  `DEPTH_W` to 1 frees 155 KB, which is the budget Refinement 3's verifier notes said `DEPTH_X`
  needed and did not have.
- **`DEPTH_X` moved only where it can act.** It is the heading's named lever, and the five
  single-M-block cells provably never reserve the second slot — yet they moved
  +1.03 / +0.73 / +0.69 / -0.92 / -0.54 %. That is a **free calibration of this op's per-cell noise
  floor at ~1 %**, and it is the yardstick every single-cell claim here is read against.
- **`DEPTH_X` is path-gated** to programs whose *sized* M extent (`ceil(input_m_tiles / M_BLOCK)`)
  can reach a second M-block, so the 195.5 KB slot is never allocated where it is provably dead. The
  runtime count is device-resident and cannot gate a CB size; the sized extent is the tightest
  host-time bound available.

### 3. Results (shipped configuration, confirmation run)

| cell | R2 baseline | shipped | delta |
|---|---|---|---|
| `count 512` (`m_blocks = 2`, **the target**) | 431 030 | **395 080** | **-8.34 %** |
| `count = capacity` (`m_blocks = 20`) | 4 200 686 | **3 512 739** | **-16.38 %** |
| `count 128` | 144 179 | 144 153 | -0.02 % |
| `count 256` | 220 576 | 222 831 | +1.02 % (noise floor — see below) |
| `cap 1024` / `cap 2048`, count 256 | 222 687 / 222 801 | 222 298 / 224 833 | -0.17 % / +0.91 % |
| `bfp8_tile`, count 256 | 217 913 | 217 955 | +0.02 % |
| `emb 6144`, count 256 | 207 901 | 209 399 | +0.72 % |
| `count 0` | 6 058 | 6 024 | -0.56 % (hang-free) |
| **9-cell sum** | **5 873 831** | **5 155 312** | **-12.23 %** |

`count 512` is now **1.77x** `count 256` rather than 1.95x, and the `count >= 512` regime is the
only thing that moved — which is the fix landing exactly where the heading aimed it.

- **The one honest caveat**: across five runs `count 256` read 220 576 / 221 269 / 224 615 / 222 173
  / 222 831 ns with no monotone relation to any knob, and the depth_wd 5 vs 11 configurations
  average +1.03 % apart. So W_down residency *may* cost ~1 % on single-M-block cells (more L1, a
  deeper CB) — it sits right at the independently-measured ~1 % noise floor and cannot be resolved
  further without many more runs. It is kept because the **graded triple is net better with it on**
  (795 785 -> 762 064 = -4.24 %, vs -3.30 % with it off) and it cannot be path-gated: `depth_wd` is a
  CB size, fixed at program build, while `m_blocks` is device-resident.
- **Accuracy achieved**: PCC **0.979044-0.979215** across 29 golden cells, unchanged in category and
  well inside the pre-existing band; residency is scheduling-only and asserted **bit-identical**
  against the re-reading path by the new test (see below). Golden slice **29 / 29 passing, 0 hangs,
  0 errors**, 110/110 cores on every cell. Unit suite **94 / 94 in 67 s**.
- **L1**: 1 533 952 B of 1 572 864 B at emb 7168 (~38 KB free), a net **+101 KB** over Refinement 2
  (+195.5 x, +60.8 wd, -155.2 w). Measured, not estimated — the figure comes from the device's own
  overflow report on the residency-OFF arm.
- **Issues encountered**: (a) turning residency OFF while leaving `DEPTH_X` at 2 asks for both the
  weight double-buffer and the resident-x slot and genuinely does not fit at emb 7168 (1 692 928 B
  against 1 572 864 B) — the exact arithmetic the verifier notes predicted, confirmed on device; both
  knob tests now pair the two, which is also the more meaningful A/B; (b) **a real L1 coupling with
  a prior phase**: Refinement 2's parked lever 1 (`REDUCE_SLOTS_CAP = 2`, +102 KB) can no longer be
  turned *together with* the resident-x slot at emb 7168, so
  `test_moe_fused_swiglu_r2_knobs.py` now exercises it paired with `DEPTH_X = 1` and documents the
  shared budget rather than papering over it. Whoever turns lever 1 for real faces the same choice.
- **Tests added**: `test_moe_fused_swiglu_r3_residency.py` (18 cases) — residency ON vs OFF must be
  **bit-identical**, on shapes spanning 2, 2-with-a-shrunk-tail and **4** M-blocks x both formats,
  plus the same for `DEPTH_X`. Bit-identity is the only assertion sharp enough here: a wrong-weight-
  block bug would move PCC by ~2e-2 while the graded gate has ~5e-3 of slack against a 0.9797 format
  ceiling, so a PCC gate would hide it. Four blocks matter — two proves the slot is re-read, only
  more than two proves the `cb_w_down` cycle **closes** each block rather than drifting a slot per
  block. A host-side guard now asserts that precondition at the single source of truth.

## Perf 1 — Tournament: the bound is 11 cores' ELTWISE, not any transport

- **Date**: 2026-07-31
- **Scope**: perf only. **`SUPPORTED` is byte-identical** (`EXCLUSIONS = []`, `INVALID = []`, every
  axis at TARGET) — a perf tournament moves nothing in the registry. The signal is device-ns.
- **Focus shape**: emb 7168, cap 5120, **count 256**, bf16_rm, bfp4_b weights, DRAM interleaved,
  `default_compute_kernel_config()`. `feature_spec.py`'s `LOOSE_CASES` carries no `attention:` note,
  so the focus was free-selected as the middle GRADED case — the one its own comment calls
  "closest to a real router's count". Every knob it declares is in `SUPPORTED`; no generality gap.

### 0. Instrumentation is now PERMANENT (and the old hook never worked)

`ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` is new: `MaybeDeviceZoneScope("<stage>")`, with
the durability contract written at the top. Every serial stage of all three kernels is now bracketed
**unconditionally** — 8 records/M-block on the reader, 5 on the writer, 6 on compute. With the
profiler off the macro emits **no instructions**, so the shipped kernel is byte-identical to an
unzoned one; the `MOE_SWIGLU_ZONES=1` opt-in is gone.

Refinement 2's note (c) — *"the one profiled run with it enabled produced an EMPTY device CSV; the
hook is unproven"* — is now **root-caused**: the compute TU used `DeviceZoneScopedN` **without
including the profiler header**, and a compute kernel does not get it through `dataflow_api.h`
(it must not see the dataflow API at all). The hook could never have worked as written. That is
precisely why the new header exists — it is the one place that knows the include path, so all three
kernels can spell the same thing. Zone budget is 125 records/core, so per-stage numbers resolve up
to `m_blocks = 15`; above that only the whole-kernel duration is valid.

### 1. Measured breakdown — cumulative peel + per-stage zones

Whole op **222 140 / 223 179 ns** (two fresh-cache runs). Ablations peel stages **cumulatively**,
because stages overlap and a solo removal under-counts:

| cumulative variant | ns | this stage's increment |
|---|---|---|
| baseline | 222 140 | — |
| − h all-gather transport | 209 094 | 14 085 (6.3 %) |
| − matmul math (`skip_compute`) | 187 062 | 22 032 (9.9 %) |
| − all three weight DRAM streams | 136 507 | **50 555 (22.7 %)** |
| − x row-multicast | 129 701 | 6 806 (3.0 %) |
| − reduce transport | **106 625** | 23 076 (10.3 %) |

**48 % of the op survives with every transport, all matmul math and all weight DRAM removed.** The
zones say why. Per-stage mean/max ns across the 110 cores:

| stage | mean | max | | stage | mean | max |
|---|---|---|---|---|---|---|
| `compute_down` | 119 374 | 164 443 | | `reader_phase2` | 143 177 | 185 273 |
| `compute_gateup` | 45 324 | 82 275 | | `reader_reduce` | 22 276 | 94 395 |
| `compute_tilize` | 22 556 | 46 164 | | `reader_xmcast` | 15 209 | 48 954 |
| **`compute_reduce`** | 16 521 | **118 746** | | `reader_xstage` | 15 164 | 41 464 |
| `compute_swiglu` | 5 533 | 5 921 | | `writer_out_issue` | 134 593 | 207 914 |
| `compute_out_pack` | 1 578 | 1 656 | | `writer_reduce_child` | 59 730 | 102 070 |

**The 11 reduce ROOTS are the critical path.** A root's 213 µs TRISC span decomposes cleanly:
`gateup 54 + compute_reduce 116 + swiglu 3 + down 37 + out_pack 2`. The other 99 cores are
**idle-waiting** — their 120–150 µs `compute_down` and 90–177 µs `reader_phase2` are `cb_wait`, not
work. `compute_reduce` on a root is 384 bfp8 tile-adds + 48 SiLU tiles in 116 µs = **~281 cycles per
tile against a ~34–70 cycle roofline, 4–8× off.**

**Roofline gates (`/perf-ceiling-dm`), which killed three idea classes before they were floated:**
- **Weight DRAM stream — AT the roofline.** 24.772 MB of bfp4 at 512 GB/s = 48.4 µs floor vs 50.6 µs
  measured = **1.04×**. No headroom. This retro-explains R2/R3's `WRUN`, `WD_AHEAD` and dual-NoC nulls.
- **h all-gather — cheap.** 14.1 µs for 11 rounds of a 110-core rendezvous + 11 × 52 KB mcast =
  1.3 µs/round. `reader_phase2`'s 143 µs wall is *waiting for the roots*, not transport.
- **Output write — cheap.** 1.95 MB bfp8 = 3.8 µs at peak; `writer_out_issue` is ~99 % `cb_wait`.

So the op was never transport-bound or weight-DRAM-bound. It was bound by **serial eltwise work on
11 of 110 cores** — which is why every prior refinement aimed at a collective returned ≤ 4 %.

### 2. The portfolio — 6 ideas, deliberately overlapping, 4 aimed at the reduce

| # | idea | verdict | measured | predicate |
|---|---|---|---|---|
| 1 | reduce accumulate mechanism (RMW vs `pack_l1_acc` vs DEST accumulation) | **WIN**, not graduated | 13 941 → 11 408 (`pack_l1_acc`, 1.22×) / 10 302 (`dest_acc`, 1.35×) | wins at all 9 cells of fan-in {1,2,4} × tiles {6,24,48}; no inversion |
| 2 | root epilogue fusion | **WIN → GRADUATED** | 53 571 → 50 141 (1.07×) | uniform over tiles {6,24,48} × HN_PAD {4,6} |
| 3 | reduce tree shape | **WIN**, not graduated | 49 406 → **19 804 (2.50×)** two-phase reduce-scatter | unconditional over KGROUPS {4,8,10} × payload {12,48,96} |
| 4 | gate/up ↔ reduce overlap (`op_design` §4.3) | **WIN**, not graduated | 143 001 → 136 275 (−4.7 %) | **S = 2 only**; S ≥ 3 regresses, −41.8 % at HN_PAD 4 / S 4 |
| 5 | gate/up in0 sharing | **NULL** | 28 559 → 28 538 (1.001×) | flat null everywhere swept |
| 6 | x staging coalesce (+ writer twin) | **MIXED**, not graduated | dual-NoC 3 433 → 3 020 (−12 %); whole-page coalescing **2.5× REGRESSION** | dual-NoC uniform over kr {23,22,20,19} × emb {7168,6144} |

Nulls are results, not waste. **#5's null has a mechanism**: `matmul_block_helpers.inl:342-344`
resets `in0_index` at the top of every `in1_subblock` iteration, so merging gate+up into one call
re-walks `in0` regardless — and its `SKIP_COMPUTE` ablation independently confirmed
`compute_gateup` is **89.7 % unpack+math**, i.e. genuinely unpack-bound. **#3's `fanin2` arm was
also a clean null** (+0.02 %): halving the root's own add count does *not* shorten the critical path,
because tree DEPTH is unchanged — root-work relief alone is not the lever. **#6 established that the
32 stick reads cost only ~1.3 µs in isolation** against 15.2/41.5 µs in the op, so that gap is
110-core-vs-8-bank contention, not the read pattern; whole-page coalescing moves 9.7× more bytes for
*zero* transaction reduction and needs 448 KB against ~38 KB free — rejected on measurement.

### 3. What graduated — `ELTWISE_BLK`, the silently-clamped DEST window

`eltwise_convenience`'s `input(cb)` / `output(cb)` default to **per-TILE** wait/pop/reserve/push, and
`eltwise_chain`'s `input_supports_block()` (`eltwise_chain.inl:1511`) admits a block only for
`Upfront` / `Cumulative` / `None+None` / `PerChunk+PerChunk`. A per-tile policy therefore makes
`chain_supports_block_v` false and the chain **clamps `block_size` to 1 at runtime**
(`eltwise_chain.inl:3054`) — silently, with no diagnostic. So every `add<>` / `mul<>` / `copy<>`
written the convenient way ran **one tile per DEST window against a DEST budget of 8**, paying a full
`tile_regs_acquire/commit/wait/release` round trip per tile. The root's epilogue alone ran 104 DEST
windows for 144 tile-ops.

The fast path pairs `PerChunk` lifecycles with `EltwiseShape::tiles(n, ELTWISE_BLK)` at all five
eltwise sites (both reduce adds, both non-root partial copies, the SwiGLU multiply, the output pack).
`OperandKind::Block` is **required**, not decorative: `is_legal_input_policy_for_kind`
(`eltwise_chain.inl:152-172`) admits `PerChunk+PerChunk` only for `Block`, because the default
`Scalar` kind pins the read to tile 0 and leans on the per-tile POP to advance the CB pointer — the
very mechanism a chunked lifecycle removes. Getting it wrong is a compile error, not a wrong answer.

**Predicate: unconditional over the whole SUPPORTED set.** The lever is a per-call CB-lifecycle
policy, not a shape property, so there is no regime to exclude. The ragged tail is safe *by
construction*: numeric `EltwiseShape::tiles(n, blk)` uses `BlockTailSync::ValidTiles`, so the last
window synchronizes only its valid remainder and the per-CB wait/pop/reserve/push **totals are
unchanged** at every `m_eff` (6/12/24/48 tiles at HN_PAD 6) — which is exactly what the cross-core
reduce requires, since the child ships and the parent consumes whole `m_eff * HN_PAD` blocks.
`MOE_SWIGLU_ELTWISE_BLK=1` is the live A/B knob. **No raw LLK**: the graduated path is pure
`kernel_lib`, so there is no helper bypass to justify.

The isolated bench measured 1.07×; **in situ it is worth far more (−8 %)**, because in isolation the
stage is 1/40th of the op while in the op it sits on the roots' critical path.

### 4. Results — every cell faster, none slower

| cell (device_kernel_ns) | Refinement 3 | Perf 1 | delta |
|---|---|---|---|
| `bf16_rm` count 128 | 144 153 | **135 370** | **−6.1 %** |
| `bf16_rm` count 256 (**the focus shape**) | 222 831 | **203 918** | **−8.5 %** |
| `bf16_rm` count 512 (`m_blocks = 2`) | 395 080 | **363 197** | **−8.1 %** |
| `bf16_rm` cap 1024, count 256 | 222 298 | **202 539** | **−8.9 %** |
| `bf16_rm` emb 6144, count 256 | 209 399 | **189 438** | **−9.5 %** |
| `bf16_rm` count = capacity (`m_blocks = 20`) | 3 512 739 | **3 176 187** | **−9.6 %** |
| `bfp8_tile` count 256 | 217 955 | **200 518** | **−8.0 %** |
| `bfp8_tile` count 128 / 512 / cap1024 / emb6144 / cap | — | 132 753 / 349 907 / 200 358 / 189 159 / 3 050 873 | all faster |

Focus shape against my own fresh baselines: 222 140 / 223 179 → 203 918 / 204 801 = **−8.2 %**.
Guard set = **12 / 12 cells faster, 0 regressions** — one representative per distinct kernel path
(bf16_rm tilize vs bfp8_tile no-tilize) × emb width (7168 EC_MAX 3 vs 6144 EC_MAX 2) × M regime
(`m_blocks` 1, 2 and 20, i.e. both sides of the weight-residency path) × all three capacities, plus
`count = 0` via the golden suite. Utilization at the focus shape: 0.247 → **0.270** (target 0.514).

The zones confirm the mechanism rather than just the outcome: `compute_reduce` max
118 746 → **105 540**, `compute_swiglu` mean 5 533 → **3 242** (−41 %), `compute_out_pack` mean
1 578 → **990** (−37 %), while `compute_gateup` (45 324 → 45 614) and `compute_tilize`
(22 556 → 22 829) are unmoved — exactly the stages the change does not touch. Everything downstream
shrank only because the roots finish earlier (`compute_down` mean 119 374 → 107 347, `reader_phase2`
143 177 → 127 481).

- **Accuracy**: **45 / 45 golden passing**, 0 hangs, 0 errors, 110/110 cores on every cell; the three
  exact structural debug tests (all-ones / hidden-identity / emb-contraction) are still exact. The
  change is a DEST-window regrouping of the same FPU ops in the same order at the same formats, so
  no number moves. Nothing in the precision contract was touched: `math_fidelity=LoFi`,
  `math_approx_mode=True`, `fp32_dest_acc_en=False`, `dst_full_sync_en=False`,
  `bfp8_pack_precise=True` and every dtype are as shipped.
- **L1**: unchanged (no CB resized).

### 5. Measured-and-ready, deliberately NOT graduated this round

Each is a real device measurement with a stated gate — not a guess, and not a follow-up to
re-discover:

1. **Two-phase reduce-scatter, 2.50× (idea #3) — the round-2 headline.** Removes the root's add work
   entirely (each of the 10 column cores reduces a disjoint slice of all 10 contributors, then ships
   its finished slice to the root, which does *zero* adds). Wins 2.0–2.9× unconditionally. Blocked
   only on integration: the bench hit a hang on the in-place `add<in(cb),in(cb2),out(cb)>` pattern —
   which the real op uses successfully today, so that is a bench artifact — and worked around it with
   single-use CBs costing 89–714 KB against ~38 KB free. A T3 collective rewrite deserves its own
   round, not the tail of this one.
2. **Accumulate mechanism, 1.22–1.35× (idea #1) — L1-gated.** `pack_l1_acc` is valid **only** with a
   bf16 accumulator: at bfp8_b it measured **PCC 0.412**, because the packer's L1-accumulate register
   does a linear add, which is invalid on a shared-exponent block-float tile. That is not a precision
   *cost*, it is a correctness bug — and it independently confirms `op_design.md` §4.1's
   "`packer_l1_acc` forces ≥ fp16_b for partials". bf16 accumulators cost **+92 KB** on all 110 cores
   (shortfall 53 KB); `dest_acc` needs `REDUCE_SLOTS = fan_in`, i.e. **+306 KB**. The interesting
   round-2 thread: a pack-accumulated CB is never *unpacked* by compute, so compute stops being its
   consumer and `cb_gate_send`/`cb_up_send` (104 KB, plus two 48-tile copies on 99 cores) could be
   deleted — but that makes the reduce's NoC payload bf16, doubling it. A real trade needing its own
   measurement.
3. **Per-sub-block reduce overlap at S = 2, −4.7 % (idea #4).** Measured on an isolated 10-core column
   with no competing traffic; the honest caveat is that it may not transfer. It also changes trip
   counts on all three kernels simultaneously (the hang class this op has been bitten by twice), and
   it composes with — or is superseded by — the tree rework above. Notable sub-finding: the *pure*
   overlap gain is only 0.3–1.2 %; the S = 2 win is the schedule break, not the pipelining. This also
   supersedes R2's lever-2 reading: a two-call split shows **no** DEST-shrink cost at S = 2, unlike
   the one-call `in1_num_subblocks=2` shape lever 2 actually measured.
4. **Dual-NoC split of the x stick read, −12 % (idea #6).** Uniform, zero L1, bit-identical output —
   and still not graduated, for a reason this round's own zones supply: the stage is overlap-hidden at
   count 256, and the lever moves work onto the **writer**, which the post-graduation zones show is
   the **longest** kernel (BRISC 200.6 µs vs NCRISC 193.6 µs vs TRISC 195.8 µs). Loading the critical
   RISC-V to speed up a hidden stage is the wrong trade today; revisit if the reduce rework exposes
   x staging.

**Still hottest after this round**: the same 11 reduce roots — `compute_reduce` is 105.5 µs of a
204.8 µs op. Round 2 re-measures and targets it with (1) and (2), whose L1 gates are the real work.

- **Tests added**: none. Coverage already existed and was reused — the three exact structural debug
  tests are the sharp gate here (a DEST-window regrouping that broke tile ordering would show up as a
  structural failure, not a PCC drift), plus the full 45-cell golden suite and the 12-cell perf guard
  set. Six subagent experiment dirs are committed under `perf_experiments/` as durable artifacts,
  each with its own runnable bake-off.

## Perf 2 — Tournament: the reduce-scatter, and the epilogue was 75 % of the bottleneck

- **Date**: 2026-08-01
- **Scope**: perf only. **`SUPPORTED` is byte-identical** — `git diff --exit-code` on
  `moe_fused_swiglu.py` is clean, `EXCLUSIONS = []`, `INVALID = []`, `PROPERTIES` and
  `INPUT_TAGGERS` unchanged. A perf tournament moves nothing in the registry; the signal is device-ns.
- **Focus shape**: emb 7168, cap 5120, **count 256**, bf16_rm, bfp4_b weights, DRAM interleaved,
  `default_compute_kernel_config()`. `LOOSE_CASES` still carries no `attention:` note, so the focus is
  free-selected as the middle GRADED case — the same one Perf 1 used, for comparability. Every knob it
  declares is in `SUPPORTED`; no generality gap.
- **Headline**: focus shape **204 769 → 150 761 ns (−26.4 %)**, guard set **12/12 faster, sum −34.3 %**,
  golden **45/45** on both paths, the three exact structural tests **bit-identical**, and L1 per core
  **−82 816 B**. Utilization 0.270 → **0.368** (best-measured parity 0.463, stretch target 0.514).

### 1. Measured breakdown — the round-1 graduation moved the path, so it was re-measured

Instrumentation was reused and **extended**: `ABLATE_NO_XSTAGE_XFER` is new, because the activation
DRAM stream had no ablation of its own (`no_x_xfer` drops the row *multicast*, a different stage).
All `MaybeDeviceZoneScope` zones kept.

Cumulative peel on the focus shape (one fresh-cache run each):

| cumulative variant | ns | this stage |
|---|---|---|
| baseline | 204 769 | — |
| − reduce transport | 194 356 | 10 413 (5.1 %) |
| − activation DRAM (solo) | 196 734 | 8 035 (3.9 %) |
| − **ALL** transports | **118 215** | 86 554 total (42 %) |

The 11 reduce roots' TRISC was **100 % busy** — 193 µs of a 195.7 µs span — decomposing as
`tilize 0-45 + gateup 36-53 + reduce 66-103 + swiglu 1.7 + down 30-37 + out_pack 0.9`. The other 99
cores were `cb_wait`. **The single sharpest number of the round**: with the reduce transport ablated,
`compute_reduce` collapses to **58 830 ns UNIFORM on all 11 roots** — that is PURE eltwise, 432
tile-ops at **184 cycles/tile against a ~68 cycle/tile 2-unpack+1-pack roofline (2.7×)**.

**Roofline gates (`/perf-ceiling-dm` + an FPU count), which killed four idea classes before they were
floated** — every one measured, not argued, using the all-transports-ablated run as the pure-work floor:

| stage | pure work | roofline | verdict |
|---|---|---|---|
| `compute_gateup` | 19–27 µs | ~26 µs (2208 tile-MACs, LoFi) | **AT roofline — no ideas** |
| `compute_down` | 16.4 µs | ~18 µs (1536 tile-MACs) | **AT roofline — no ideas** |
| weight DRAM | 50.6 µs | 48.4 µs (24.772 MB bfp4) | **AT roofline, 1.04× (Perf 1, unchanged)** |
| activation DRAM | 8.0 µs exposed | 7.2 µs (3.67 MB) | **AT roofline, 1.1× — bytes gated, only scheduling open** |
| `compute_tilize` | **0.52 µs** | — | **gated** — its 27–45 µs zone is ~99 % WAIT, not work |
| `compute_out_pack` / `compute_swiglu` | 0.9 / 1.7 µs | — | gated |

So the ranked bottleneck was: **(1) the 11 roots' 58 830 ns of eltwise reduce, (2) the 10 413 ns
reduce transport, (3) the activation stream's head-of-line delay.** ~47 µs of a 204.8 µs op was
addressable and everything else was at a wall. The portfolio was sized to that: 1 T3 + 4 T2, four of
the five aimed at the reduce.

**L1, the standing blocker**: 1 422 464 B used of a **1 461 376 B** per-core budget → 38 912 B free.
A **per-core-range role split** was investigated and **rejected on evidence**: this allocator
(`tt_metal/impl/program/program.cpp:1571-1594`) finds no allocator for a NEW sub-range, so
`computed_addr` collapses to `base_cb_address` and the CB silently overlaps CBs 0..25 on those cores.
The safe mechanism is **aliasing format descriptors inside one `CBDescriptor`** (the two role-exclusive
pairs are both 48 × 1088 B and provably never co-live), worth 104 448 B — offered to the subagents as
budget. **In the end nobody needed it**: the winner *returns* L1.

### 2. The portfolio — 5 ideas, deliberately overlapping

| # | idea | verdict | measured | predicate |
|---|---|---|---|---|
| 1 | **reduce-scatter + distributed SwiGLU epilogue** | **WIN → GRADUATED** | 78 011 → 27 853 ns (**2.80×** flat) / 25 273 (3.09× ragged); **3.08× on the real 110-core grid** | **unconditional**: every cell of KGROUPS {2,4,8,10} × m_eff {1,2,4,8} × HN_PAD {4,6}, 1.65–3.11×, zero regressions |
| 2 | root accumulate mechanism | **WIN, superseded** | 12 609 → 6 900 ns (`pack_l1_pair`, **1.83×**); 1.90–2.09× *inside* the scatter | `fan_in >= 2` (0.91–1.01× at fan-in 1) |
| 3 | root epilogue passes | **MIXED, superseded** | `add_silu_chain` 1.016× bit-identical; `sigappx_mul` **3.254×** priced; `single_pass` **0.685×** | `add_silu_chain` unconditional; both shrink ~8× under the scatter |
| 4 | reduce transport shape | **WIN, partly folded in** | 21 201 → 19 642 ns (−7.35 %) = 0.7–1.2 % whole-op | root `fan_in >= 4` (KGROUPS >= 9) |
| 5 | reader head-of-line scheduling | **NULL** | 203 869 → 203 834 ns best (−0.02 %), band 0.86 % | **empty** — null at every regime swept |

Nulls and supersessions are results. Four mechanisms worth more than their verdicts:

- **#3 disproved the coordinator's own hypothesis and reframed the round.** I ranked the 58 830 ns as
  helper-entry overhead. It is not: **~44 100 ns (75 %) is the 48-tile SFPU SiLU**, ~10 700 (18 %) the
  8 plain adds, ~2 700 (4.5 %) up-add + mul, and **~800 ns (1.4 %) is the *entire* per-helper-entry
  cost of the `m_eff`-call bias walk** I had suspected. That is why #1's "keep the epilogue
  distributed" sub-predicate carries ~85 % of its win, and why #2 — which makes the *adds* cheaper —
  could never have been the headline. It also found `silu_tile` **hardcodes the accurate sigmoid and
  ignores the user's `math_approx_mode=True`** (`ckernel_sfpu_silu.h`: `silu_init` calls
  `sigmoid_init<false>()`), while `sigmoid_tile<RC,true>` honours it — worth 3.254× on the epilogue for
  a worst-case −0.00196 PCC. **Recorded as a priced option, deliberately NOT graduated**: it is
  ~40 % of the remaining margin over the 0.975 gate, and under the scatter its critical-path value
  falls from −35 339 to −4 218 ns/core. It is the strongest remaining lever if a later round needs one.
- **#2 invalidated its own round-1 result.** Round 1's 1.22×/1.35× was measured against the *per-tile
  clamped* spelling, which Perf 1 then graduated away; that spelling is **0.55×** of the shipped one,
  so both round-1 "winners" **lose** against the real baseline (`pack_l1_acc` 0.91×, `dest_pair` 0.86×,
  `dest_full` 0.79×). It also fitted an engine-bound model that explains the whole menu: an
  L1-accumulating pack ≈ 2.2 unpacks and `DestReuseBinary` ≈ 3.6, so halving accumulator traffic just
  flips the stage from unpack-bound to pack-bound. Its `pingpong` arm is a clean null (1.00×) —
  **the in-place `add` CB self-dependency costs nothing; do not spend L1 removing it.**
- **#4 root-caused Refinement 2's parked `REDUCE_SLOTS` regression.** It was the **whole-CB WAVE
  PUSH**, not the concurrency: one shared `SEM_DATA` says *how many* children arrived but not *which*,
  so the parent cannot publish a slot until the whole wave lands, destroying the shipped protocol's
  "child c's transfer under child c−1's add" interleave. Reproduced at +12.08 %; **one counter per
  slot** turns it into −1.62 %. It also showed the 1.92× NoC asymmetry is **not link speed**
  (3 176 ns NOC_1 vs 3 270 ns NOC_0 on a contention-free edge) but **torus-wrapped hop count** —
  total payload hops predicts the entire ranking (273 / 197 / 907 / 915) — and that the op's
  `parent = (root_y + r − lowbit(r)) % KGROUPS` orientation was **already** right for NOC_1.
- **#5 is a null with a ceiling, not an effort.** The reader's serial head is at **1.016× its byte
  roofline** (20 185 088 B must land before gate/up can start = 39 424 ns floor vs 40 064 ns measured),
  so the entire winnable budget is **640 ns = 0.31 %**, under the 0.86 % run-to-run band. Transaction-ID
  barriers *are* legal and *do* work — they pull `compute_tilize` 23 172 → 19 669 ns — but bandwidth is
  conserved, so x-early means W-late (+0.9 to +1.0 % on every reorder arm). It also priced the
  ungraduated round-1 dual-NoC x split at **−0.20 %**, closing that thread, and proved
  `cb_x_in`/`cb_x_stage` can never be usefully deepened within an M-block (`inject_rows` gives exactly
  1 tile-row per core at every count, since `m_eff <= M_BLOCK = 8 < HGROUPS = 11`).

### 3. What graduated — the two-phase reduce-scatter with a distributed epilogue

The binary tree down each grid column is replaced by an all-to-all within the column: every core owns
a **disjoint flat tile-index slice** of the `m_eff * HN_PAD` block, every contributor pushes its slice
straight into each worker's landing CB, each worker reduces **only its own slice** over all
contributors, computes `SiLU(gate_slice) * up_slice` on it, and unicasts the finished `h` slice
directly into the root's `cb_h_local` at its tile offset — **the gather IS the assembly**, no copy and
no add at the root. `cb_gate_send` / `cb_up_send` and the two full-block copies on 99 cores are gone;
the writer sends straight out of `cb_gate_acc` / `cb_up_acc`.

- **Predicate: unconditional** on any grid where the slice plan is expressible, which is the whole
  `SUPPORTED` set. The host asserts and **falls back to the byte-identical tree** (printing the reason)
  when it is not: `KGROUPS >= 2`, and — for **every** reachable `m_eff` — the slice size must divide
  both the slice CBs and the landing CBs. That page-count rule is not decoration: violating it makes
  the write pointer wrap mid-block and **silently overrun the next CB** (measured PCC 0.709–0.886).
  Verified expressible for HN_PAD {4,6} × KGROUPS {2,4,8,10}.
- `MOE_SWIGLU_REDUCE=tree|scatter` (default `scatter`) is the live A/B, in this op's established knob
  style. `MOE_SWIGLU_SCATTER_NOC=split|one` likewise.
- **The token-(M)-axis slice was measured and rejected**: 0.79× **regression** at m_eff=1 and 1.25× at
  m_eff=2, because that axis caps the worker count at `m_eff`. The tile-index axis strictly dominates.
  `ragged` (10 workers × 5/4) is 1–10 % faster than `flat` (8 × 6) but forces `lcm(4,5)=20`-page CBs;
  that footgun was deliberately not shipped.
- **#4's finding was applied, partially, and the honest reason is recorded**: the ideal per-*destination*
  NoC split needs both DM RISC-Vs to consume one accumulator CB, and `cb_pop_front` writes the shared
  `tiles_acked` word with the popping RISC-V's own local count — the same single-owner hazard as the
  hang below, one layer down. The shipped split is by **payload** (gate on NOC_1, up on NOC_0), which
  captures the bandwidth half and none of the hop-count half: **−1.45 %** on the guard-set sum
  (`one` 5 596 297 / 5 600 029 vs `split` 5 509 590 / 5 524 843 — the between-group gap is 5× the
  within-group spread). Getting the hop-count half costs +52 KB/core or a private ready-signal.
- **No raw LLK.** The graduated path is `eltwise_convenience` + `bias_add_helpers` + the two
  raw-dataflow deviations already documented at the kernel heads (bank-run coalescing; point-to-point
  scatter edges, because `mcast_pipe`'s `SenderPipe` is a rectangle multicast and its
  `DataReadySignal::Counter` is a documented hang). Nothing new for the verifier's helper-usage pass.
- Zones extended: `writer_scatter`, `writer_hslice`, and the surviving stages keep their **names**
  (`reader_reduce`, `compute_reduce`, `compute_swiglu`) so Perf-1 and Perf-2 numbers are comparable.
  Budget still 8 reader / 5 writer / 6 compute records per M-block, so per-stage time still resolves
  to `m_blocks = 15`. All five `MOE_SWIGLU_ABLATE` hooks work on **both** paths.

**Two silent-wrong-answer hazards were root-caused on the way, and both are general to this op:**

1. **Exactly ONE RISC-V may ever push a given CB.** `cb_push_back` overwrites the shared L1
   `tiles_received` word with the *pushing RISC-V's own local* push count. This is what round 1's
   "the in-place `add<in(cb),in(cb2),out(cb)>` hangs" actually was — seeding an accumulator from the
   reader and then letting the in-place add make PACK a second pusher drove the shared count
   **backwards** (12 → 8) and deadlocked the consumer. **Round 1's 89–714 KB single-use-CB workaround,
   the stated reason this idea did not graduate a round earlier, was unnecessary.**
2. **An `eltwise_chain` containing a `DestReuseBinary` is correct at a DEST window <= 7 and silently
   WRONG at 8 = `DEST_AUTO_LIMIT`** (PCC 0.999828 → 0.972295): `chain_max_block_value()`
   (`eltwise_chain.inl:1482-1493`) subtracts the spare lane only on the `any_dest_accumulation` branch.
   The graduated path uses no dest-reuse, so it does not bite here — checked, not assumed.

**What did NOT transfer from the bench, and it mattered.** `rs_flat_epi` as measured was **not correct
in situ**: with bfp8 slice accumulators the op passed golden 45/45 but **failed `test_emb_contraction`
at max rel 0.0580 against the 0.05 gate** (tree: 0.0204). The bfp8 pack's rounding is a **biased**
half-LSB and every partial here is positive, so the error is **linear in chain length** — the scatter
re-packs a value `KGROUPS` = 10 times where the tree re-packs it `ceil(log2 10)` = 4 times. The bench's
PCC metric (0.999777 vs 0.999823) hid this completely; only the structural test's max-relative-error on
a pathological input exposed it. Fix: the **three slice CBs are `bfloat16`**, not bfp8 — DEST is bf16 at
`fp32_dest_acc_en=False`, so packing DEST → bf16 is *exact*, which deletes the per-step quantisation
rather than merely reducing it. Result **0.0204, bit-identical to the tree**, for +17 280 B/core.
`cb_h_slice` stays bfp8, so the one genuine dtype boundary is still the SwiGLU pack. **Nothing in the
precision contract was touched** — `math_fidelity=LoFi`, `math_approx_mode=True`,
`fp32_dest_acc_en=False`, `dst_full_sync_en=False`, `bfp8_pack_precise=True`, bfp8 partials and bfp8
output are all as shipped; an intermediate CB's format is an implementation choice, not a user knob.

### 4. Results — every cell faster, none slower

| cell (device_kernel_ns) | `tree` | `scatter` | delta |
|---|---|---|---|
| `bf16_rm` count 128 | 139 229 | **109 314** | **−21.5 %** |
| `bf16_rm` count 256 (**the focus shape**) | 205 107 | **150 479** | **−26.6 %** |
| `bf16_rm` count 512 (`m_blocks` 2) | 359 411 | **251 915** | **−29.9 %** |
| `bf16_rm` cap 1024, count 256 | 203 590 | **149 036** | **−26.8 %** |
| `bf16_rm` emb 6144, count 256 | 191 797 | **136 059** | **−29.1 %** |
| `bf16_rm` count = capacity (`m_blocks` 20) | 3 175 661 | **2 037 988** | **−35.8 %** |
| `bfp8_tile` 128 / 256 / 512 | 133 143 / 201 622 / 349 710 | **106 483 / 147 253 / 239 979** | −20.0 / −27.0 / −31.4 % |
| `bfp8_tile` cap 1024 / emb 6144 / count=cap | 201 927 / 187 784 / 3 051 287 | **147 400 / 131 600 / 1 910 670** | −27.0 / −29.9 / −37.4 % |
| **sum** | **8 400 268** | **5 518 176** | **−34.3 % (1.52×)** |

Guard set = **12/12 cells faster, 0 regressions** — one representative per distinct kernel path
(bf16_rm fused-tilize vs bfp8_tile) × emb width (7168 `EC_MAX` 3 vs 6144 `EC_MAX` 2) × M regime
(`m_blocks` 1, 2 and 20, i.e. both sides of the weight-residency path) × all three capacities, plus
`count = 0` via the golden suite. Coordinator's independent fresh-cache re-measure of the focus cell:
**150 761 ns vs the 204 769 ns baseline = −26.4 %** (the subagent's own `tree` run measured 205 107,
0.16 % from mine, so the A/B is on one binary).

**Post-commit confirmation on the SHIPPED tree** (pre-commit's `black` + `clang-format` reformatted the
descriptor and the compute kernel, so every gate was re-run against the committed bytes): golden
**45/45**, the three structural tests still **0.0000 / 0/64 / 0.0204**, and the guard set
**110 657 / 151 498 / 249 383 / 148 739 / 135 481 / 2 042 027** (bf16_rm) and
**109 201 / 146 659 / 239 726 / 146 507 / 131 151 / 1 915 673** (bfp8_tile) ns — sum **5 526 702 vs the
tree's 8 400 268 = −34.2 %**, **12/12 faster, 0 regressions**. The focus cell reads 151 498 here vs
150 761/150 479 above, i.e. this op's ~1 % per-cell run-to-run noise; all three are quoted rather than
the best one.

The zones confirm the **mechanism**, not just the outcome. `compute_reduce` was 66 224–103 332 ns on
11 roots and ~0 on the other 99; it is now **flat at 33 881–41 727 ns on every worker**. The single hot
root is gone. `compute_swiglu` 1 730 → **283–349 ns** (−80 %), and the `m_eff`-call SiLU bias walk
collapsed to one call for free (a slice is <= DEST). `compute_gateup` (45 404 mean, unmoved) and
`compute_tilize` (22 347, unmoved) are exactly the stages the change does not touch.

- **Accuracy**: **45/45 golden** on the default path AND on `MOE_SWIGLU_REDUCE=tree` AND on
  `MOE_SWIGLU_SCATTER_NOC=one`; 0 hangs, 0 errors, 110/110 cores on every cell. The three exact
  structural tests are still exact — `all_ones` max rel **0.0000**, `hidden_identity` **0/64 bad
  hidden tiles**, `emb_contraction` **0.0204, bit-identical to the tree**. The other five op test files
  are 79/79.
- **L1**: **−82 816 B/core** on the reduce/epilogue CB set (417 792 → 334 976). The bench predicted
  −104 448; the +17 280 is the bf16 slice accumulators above and +4 352 is 1-page placeholders for the
  inactive path's CBs. The dual path costs the `tree` configuration +9 536 B of placeholders.

### 5. Still open, with numbers — the bottleneck MOVED

**The roots have stopped being the critical path.** After graduation the hottest stages on the focus
cell are `compute_down` (49 933 mean / **97 664 max**), `writer_out_issue` (87 830 max),
`compute_gateup` (83 471 max, **at its FPU roofline**) and `reader_phase2` (78 339 max) — i.e.
**phase 2, the `h` all-gather and the `down` matmul, at 2–2.5× `compute_reduce`**. 91 % of the measured
stage saving reached the wall clock, and the residual gap to the 3.08× isolated number is Amdahl, not
erosion: the reduce+epilogue was ~33 % of the op, so −69 % of it is bounded at ~1.5×.

Measured-and-ready, deliberately **not** graduated, each with its gate:

1. **Approx-sigmoid SiLU, 3.254× on the epilogue.** `silu_tile` ignores the user's
   `math_approx_mode=True`. Priced at worst −0.00196 PCC (~40 % of the margin over the 0.975 gate) and
   its critical-path value fell from −35 339 to −4 218 ns/core under the scatter. Cheap to revisit if
   phase 2 is ever fixed and compute becomes the bound again.
2. **`pack_l1_pair` accumulate, 1.90–2.09× in the scatter regime** (bf16 accumulator + one
   `BinaryFpu` folding two contributors into DEST, then one L1-accumulating pack; PCC 0.99997, *better*
   than baseline). It now fits (~+102 KB against the freed 82 KB + 38 KB). Not taken because
   `compute_reduce` is now **wait-dominated** — the subagent measured only ~2 µs of actual math in it.
3. **The hop-count half of `dir_noc`** (+52 KB/core or a private ready-signal), and the whole
   10-way slice sum + SiLU as **one** DEST window via `DestReuseBinary` — unmeasured, and it trips the
   DEST <= 7 rule in §3. Both are bake-offs, not integrations.
4. **The CB-aliasing L1 enabler** (104 448 B, root-only `cb_gate_silu`/`cb_h_local` aliased onto
   non-root-only `cb_gate_send`/`cb_up_send`) — designed and verified safe against the address-proxy
   invariant, then **not needed**, because the winner returns L1 instead of consuming it. Kept on
   record as the correct mechanism should a later round need L1, along with the evidence that the
   per-core-range role split is **unsafe** in this allocator.

- **Tests added**: `tests/.../test_moe_fused_swiglu_r2_perf.py` — one env-selected perf harness
  (`MOE_R2_CASES=guard`), because `run_safe_pytest.sh` preserves neither a quoted `-k` expression nor
  `[...]` node-id brackets, so the Perf-1 harness could not be pointed at a single cell. Plus
  `perf_experiments/parse_zones.py`, which turns a `--profile` report into per-stage zone statistics
  and the per-core TRISC work decomposition that located the hot roots in the first place. Five
  subagent experiment dirs are committed under `perf_experiments/` as durable artifacts, each with its
  own runnable bake-off.

---

## Perf 3 — the op is STRICTLY ADDITIVE: a measured decomposition, two graduated levers, and where the wall is

### 1. The measurement that reframes everything

Every ablation stubs ONE payload and keeps every CB reserve/push/pop, every barrier and every loop
trip count, so each diff is that term's EXPOSED cost. Focus cell, emb 7168 / cap 5120 / count 256 /
bf16_rm, fresh kernel cache per arm:

| arm | ns | exposed term |
|---|---|---|
| baseline | 148 742 | — |
| `no_w_xfer` | 110 579 | **weight DRAM 38 163** |
| `skip_compute` | 125 941 | **matmul math 22 801** |
| `no_h_xfer` | 128 513 | **h all-gather 20 229** |
| `no_reduce_xfer` | 138 313 | reduce transport 10 429 |
| `no_x_xfer` / `no_xstage_xfer` | 143 233 / 143 886 | x mcast 5 509 / x DRAM 4 856 |
| all five transports | 71 072 | (transports together 77 670) |
| all five + `skip_compute` | **42 764** | **the pure synchronisation floor** |
| `no_w_xfer` + `skip_compute` | 85 266 | — |

`42 764 + 38 163 + 22 801 + 20 229 + 10 429 + 10 365 = 144 751` against a 148 742 baseline, i.e.
**within 2.7 % of the sum. NOTHING IN THIS OP OVERLAPS ANYTHING ELSE.** That single fact reprices the
whole board, because every individual stage is already at a wall:

- **weight DRAM is at roofline**: 24.772 MB / 50.6 µs = 490 GB/s of a 512 GB/s peak.
- **the matmul is at the FPU roofline** (2208 tile-MACs LoFi, ~16 cycles/tile-MAC).
- **phase 2 is at its ingest roofline**: 512 bfp8 tiles of `h` per core = 557 KB at ~27 GB/s
  achieved against a ~43 GB/s NoC-to-L1 limit, plus a `down` matmul at its own FPU roofline.
- so the entire question is **overlap**, not bytes, not FLOPs, not transactions.

`no_w_xfer + skip_compute` = 85 266 pins the ceiling for overlapping those two terms at **126 µs**,
and `no_w_xfer` = 110 579 pins the ceiling for hiding the ENTIRE weight stream at **111 µs** — which
is still above the 108 000 target. **Reaching the target needs the weight stream hidden AND the
42.8 µs floor cut; no single lever gets there.**

### 2. What graduated

**XPRIO (default 1) — activation-first DRAM priority.** The profile killed the obvious story: the
gate/up matmul does not wait for weights at all (`reader_wg_wait` returns in ~4 µs). It waits for
`cb_x_tiles`, which the x row-multicast publishes at **56 µs**. x is 3.67 MB of phase 1's 20.2 MB,
but EVERY core's matmul needs all of its row's x before it can start, while the weights are consumed
per column — so the small stream is the universally-blocking one and it was arriving last. Each
core's writer now holds its 8.7 MB W_up stream on `SEM_XSTAGED` until that core's reader has staged
x. Intra-core: a plain volatile L1 store the writer polls, no NoC traffic, monotone so no reset, and
never entered at `m_blocks == 0` so the zero-count dispatch still cannot hang.

**GU_CHUNKS (default 2) — N-chunked gate/up weight stream.** The bfp4 block is issued, published and
consumed in 2 chunks of the HIDDEN axis, so the matmul runs on chunk 0 while chunk 1 is in DRAM.

> **Why N and not K, measured rather than argued.** A K-chunk is a PARTIAL SUM: `m_eff * HN_PAD`
> tiles exceed DEST, so each extra K-block costs `m_eff` extra L1-ACCUMULATING packs per matrix, and
> at ~2.2 unpacks per accumulating pack (Perf 2 idea #2's engine model) that is about the whole
> overlap it buys. This is why `op_design.md` forbade `num_k_blocks > 1`, and the reason is now
> written down. An N-chunk is an INDEPENDENT full-K matmul — no partial sums, no extra packs — and
> the block is K-major in the CB, so an N-chunk is contiguous on the DRAM-read side AND the matmul
> in1 side. `matmul_block` gained `out_col_offset` (new, default 0, byte-identical when unused) so
> each chunk packs into its own columns of one shared m-major block instead of producing a
> chunk-major one; it pairs with `caller_owns_pack_target` + `TileRowMajor` + `out_row_width = HN_PAD`.

**THE TWO LEVERS ONLY WORK TOGETHER, and that is the round's most transferable result.** Each is a
null or a regression alone:

| | GU_CHUNKS 1 | GU_CHUNKS 2 |
|---|---|---|
| XPRIO 0 | 149 459 (baseline) | 149 316 (**null**) |
| XPRIO 1 | **154 219 (worse)** | **146 483 (win)** |

Holding W_up back only pays if the matmul can start on a chunk; chunking only pays if x is not the
thing being waited for. Either alone measures nothing, which is exactly the shape that makes a lever
look dead in a one-at-a-time sweep.

### 3. Nulls, with their causes

- **`XSTAGE_FIRST` (issue x before W_gate on the reader) is a NULL** — 150 260 / 151 233 alone, and
  *worse* with XPRIO (152 791 vs 146 945). Profiled cause: re-ordering ONE core's own two issue
  streams cannot move x forward when the queue x is stuck behind belongs to the other 109 cores and
  to the writer's NoC1 twin. Kept as a knob because it is the A/B that proved the diagnosis; default
  0. This also independently reproduces Perf 2 idea #5's finding that phase 1's 20.2 MB head is at
  **1.016× its byte roofline**, so head-of-line re-scheduling has a ~0.3 % ceiling.
- **`WD_AHEAD` deeper is monotonically WORSE**: 1 → 146 945, 2 → 151 390, 4 → 164 193, 11 → 190 720.
  Cause found: the knob pushes all `WD_AHEAD` blocks in ONE `cb_push_back` before the round loop, so
  a deeper prefetch does not pipeline — it makes phase 2 wait for the whole 8.5 MB before its first
  round. `noc_async_read_barrier()` being all-or-nothing is what forces that shape; a per-block
  transaction-ID barrier is the only way to issue early AND publish incrementally.
- **The 2D `down` (contract only over the core's own hidden group, reduce the output along the grid
  row) was priced on paper and REJECTED before implementation.** It replaces 557 KB/core of `h`
  ingest with a row reduce-scatter of a `m_eff x EMB_T/KGROUPS` = 184-tile partial per core — 182 KB
  each way in bfp8, ~2× that in the bf16 the phase-1 scatter's precision lesson demands — plus ~170
  tile-adds. Net ≈ −6 µs before the precision risk, against a restructure of the op's most
  hang-prone path. `h` is 512 tiles and the output is 1792; broadcasting the SMALLER one and keeping
  `down`'s contraction whole is the right way round, and that is now on the record.
- **`MOE_SWIGLU_ABLATE=no_handshake` DEADLOCKS** (26 BRISCs halted, device reset by the harness). It
  is documented as measurement-only, but it does not currently produce a number — so the floor's
  handshake component is still unpriced. Fixing that hook is the cheapest next diagnostic in the op.

### 4. Results — graded cells, and at the core counts the targets were measured on

`MOE_SWIGLU_GRID` is new: the graded `PERF_MEASURED_NS` baselines were taken on **88/99-core** grids,
so the op is now measurable at 11x8 / 11x9 instead of only at this box's 110.

| count | @110 before | **@110 after** | **@99** | **@88** | target | best measured |
|---|---|---|---|---|---|---|
| 128 | 110 654 | **103 498** (−6.5 %) | 104 599 | 106 419 | 91 800 | 102 000 |
| 256 | 148 742 | **145 879** (−1.9 %) | 147 798 | 153 953 | 108 000 | 120 000 |
| 512 | 251 082 | **247 472** (−1.4 %) | 248 363 | 252 448 | 161 816 | 179 795 |

**The op is nearly core-count-insensitive** (99 ≈ 110, 88 only +3–5 %), which is itself a finding: it
confirms from the outside what the ablations say from the inside — this op is bound by serialisation,
not by compute throughput, so adding 25 % more cores buys ~1 %.

Guard set, 12/12 cells, against the Perf-2 shipped tree: bf16_rm
**103 498 / 145 879 / 247 472 / 147 018 / 136 399 / 2 049 576** and bfp8_tile
**101 606 / 142 873 / 235 192 / 141 647 / 130 300 / 1 921 787**. Ten cells faster, two (emb 6144
bf16_rm +0.7 %, count=capacity +0.4 %) inside the op's documented ~1 % run-to-run band. Sum 5 503 247
vs 5 526 702.

- **Accuracy**: golden **45/45**; all 66 tests in the op's six test files pass; 0 hangs. XPRIO moves
  no bytes (a pure ordering constraint) and GU_CHUNKS is the same LLK at the same precision on the
  same formats — no precision knob was touched, and `MOE_SWIGLU_REDUCE=tree` still measures
  200 513 ns, confirming the scatter default and that both paths still build.

### 5. Where the wall is, with numbers

Ranked by measured headroom, for whoever picks this up:

1. **The 42 764 ns zero-payload synchronisation floor — 29 % of the op, and the largest unexplained
   term.** With no payload and no math it is `compute_reduce` ~11 µs, `compute_down` ~13 µs,
   `compute_out_pack` ~14 µs, `writer_out_issue` ~18 µs, `writer_hslice` ~15 µs — i.e. phase 2's
   **11 SERIAL rotating-sender rounds** and the column scatter's invites. The 11 h-broadcast rounds
   are ordered and each is gated on a DIFFERENT column root, so a late root blocks every later round
   (`compute_reduce` spans 43→100 µs while its own math is ~2 µs). Making all 11 roots multicast
   CONCURRENTLY into their own disjoint `cb_h` slot with a per-slot arrival counter is the same
   mechanism that won 3.08× in Perf 2, applied one layer up. Not attempted here: `mcast_pipe`'s
   `DataReadySignal::Counter` is a documented hang and the fallback would have been a revert.
2. **Genuine chunk-through pipelining of the hidden axis** — carry GU_CHUNKS through the reduce,
   SwiGLU and h-broadcast, not just the matmul, so chunk c's collectives run under chunk c+1's weight
   stream. This is the only structure that can put the weight stream under the 57 µs window where
   DRAM currently idles (after the matmul, only W_down's 17 µs is left). Ceiling ≈ 111 µs from
   `no_w_xfer`. Cost: 2× the collective handshakes against a floor that is already 29 % of the op —
   which is why item 1 should land first.
3. **Per-block transaction-ID barriers for W_down**, which is what `WD_AHEAD` actually needed.

### 6. Perf 3 addendum — the h rendezvous is the KEYSTONE, and it is L1-locked

Three further measurements change the priority order in §5, so they are recorded before anything is
built on the old ranking.

**(a) Phase 2 is exactly 11 SERIALISED RENDEZVOUS, and `cb_h`'s depth is not why.** With every payload
ablated phase 2 is 43 µs ≈ 11 × (1.2 µs mcast + 1.7 µs `down` K-block + ~1.5 µs handshake). Sweeping
the new `MOE_SWIGLU_DEPTH_H` knob: 2 → 149 918, **3 → 145 879 (default, best)**, 4 → 147 354,
5 → 147 550 ns. Deeper is not better, so the CB is not the flow-control limit.

The lockstep is STRUCTURAL, in the loop shape rather than in `mcast_pipe`: a core sends its own
column's h at iteration `r == my_col` of the SAME loop in which it receives every other column, so
**the root of column c cannot send until it has received rounds 0..c-1.** Round 10's sender waits on
ten prior receives. That is an 11-long serial chain no CB depth can shorten.

**(b) The obvious fix — absolute per-slot addressing so all 11 roots send concurrently — is
L1-INFEASIBLE, by a measured margin.** It requires `cb_h` to hold all HGROUPS slots at once (the
landing address must not depend on the receiver's progress). Each slot is `M_BLOCK * HN_PAD` bfp8
tiles = **52 224 B**, and the allocator prices the ladder exactly: `DEPTH_H` 7 → 1 660 032 B,
9 → 1 764 480 B, 11 → **1 868 928 B** against a **1 572 864 B** cap. Working back, the op sits at
**1 451 136 B with 121 728 B free** at the default depth, and 11 slots need **+417 792 B** — over by
**296 064 B even after** the 104 448 B CB-aliasing enabler designed in Perf 2. Not close, not a
tuning question.

**(c) Therefore the rendezvous BLOCKS the weight-stream fix, which reverses §5's ranking.** §5 put
chunk-through pipelining of the hidden axis second, on the grounds that it could hide the 38 163 ns
weight stream. Priced against (a) it does not pay: carrying GU_CHUNKS through the reduce and the h
gather **doubles the round count to 22**, and at ~2.95 µs per half-width round that is 65 µs of phase
2 against today's 43 — **+22 µs** versus a weight-overlap gain of at most 34 µs. The same arithmetic
kills any finer chunking. **The per-round rendezvous is the keystone: until it is gone, every scheme
that buys overlap by adding pipeline stages pays for it at the rendezvous and nets out flat.**

Two adjacent knobs were also priced and are dead ends: `HGROUPS` 10 gives `HN_PAD` 7 with a ragged
`hn` of **1**, which clamps GU_CHUNKS to 1, makes the scatter slice plan inexpressible (falls back to
the tree) and overflows L1 at 1 670 464 B; `HGROUPS` 8 gives a perfectly uniform `HN_PAD` 8 and only 8
rounds but overflows at 1 663 104 B and adds 33 % per-core matmul. HGROUPS 11 is a local optimum.

**So the one remaining route is the dual-RISC split** — h SEND on the writer/NoC1, RECEIVE on the
reader/NoC0 — which keeps the 3-slot rolling window (no new L1) and decouples a root's send from its
own receive progress. It is moe_matmul's P7, measured there at 1.13–1.17× on the analogous stage and
deliberately not attempted there either. The design, including the mitigation for the race that makes
it dangerous:

- `mcast_pipe`'s `DataReadySignal` is a VALID/INVALID cell that the receiver RESETS. Split across two
  RISC-Vs, a receiver's `set(INVALID)` for round r-1 can land between the sender's `set(VALID)` and
  its flag multicast for round r, broadcasting INVALID and hanging the grid (moe_matmul P7 found this
  by construction, not by luck).
- **Mitigation: replace the flag with a MONOTONE counting semaphore** — the sender multicasts an
  increment, the receiver `wait_min`s on `b * HGROUPS + r + 1` and never resets anything. One writer
  per round, no reset, race-free by construction, and the same discipline every other semaphore in
  this op already follows.
- The writer cannot call `get_write_ptr(cb_h)` for the landing address (`cb_interface` is per-RISC-V
  local state and the writer is not the pusher); it must derive it arithmetically from the CB base
  captured before the first push plus its own `(b * HGROUPS + r) mod (capacity / block)` counter.
- `consumer_ready` stays as it is: after consuming round r the receiver acks the sender of round
  `r + DEPTH_H`, whose coordinates are static. That cost is already paid today.

### 7. Perf 3 addendum 2 — the `mcast_pipe` Counter path is FIXED, and it disproves the flag-reset hypothesis

§6 named the h rendezvous the keystone. The descriptor's own comment offered a specific culprit inside
it: `DataReadySignal::Flag` is RESET by every receiver, so "the sender of round r+1 [waits] for every
receiver to reset round r's flag, which is part of the measured collective serialisation." The
monotone `Counter` signal removes that reset entirely — and had been **unusable since Phase 0**, filed
as an open helper bug. It is now fixed, and the answer is a clean NULL with a mechanism.

**The bug, fixed in `mcast_pipe.inl`.** `send_data_` issued the data multicast with
`NOC_CMD_VC_LINKED` unconditionally, relying on the following signal to terminate the chain. Only
`Flag` can: it is a multicast WRITE on the SAME command buffer, while `Counter` is a multicast ATOMIC
on `write_at_cmd_buf` — a DIFFERENT one. So a linked data write behind a Counter signal left
`NCRISC_WR_CMD_BUF` linked forever and the next write on it blocked in `noc_cmd_buf_ready`: a
guaranteed hang for every Counter user, previously observed as round-0's h sender stuck in
`noc_async_write_multicast_loopback_src` while all 110 readers sat in their `wait_min`. The fix is the
one the descriptor predicted: **link only for `Flag`**, and order the Counter path's data ahead of its
signal with an **ACKED `async_write_barrier()`** (`async_writes_flushed()` means SENT, not LANDED — a
receiver could see the counter and read a half-written block).

**It works and it is correct**: the Counter path now runs to completion and passes **golden 45/45**,
where it previously hung. That is a real repair to a shared helper, useful to any op that wants a
monotone multicast signal, and it is independent of whether this op ships it.

**And it is SLOWER here — measured, not argued:** `counter` 112 087 / 159 968 / 278 493 ns against
`flag` 105 076 / 145 625 / 247 181 (**+7 to +13 %**). The reason is the interesting part: **the link
that Flag terminates for free is what was ordering data-before-signal.** Unlinked, Counter must buy
that ordering with an acked write barrier plus a non-posted atomic barrier, and that pair costs more
than the flag-reset round trip it deletes. `MOE_SWIGLU_HSIG` keeps both live; `flag` is the default.

**This DISPROVES the flag-reset hypothesis and sharpens §6.** The per-round reset is not what
serialises phase 2. What does is the **LOOP ORDER**: a core sends its own column at iteration
`r == my_col` of the same loop in which it receives every other column, so the sender of round r+1
must first RECEIVE rounds 0..r. No signal change can touch that — only moving the send off the
receiving RISC can, which is §6's dual-RISC split. One fewer candidate, and the remaining one is now
the only one.

### 8. Perf 3 addendum 3 — the dual-RISC split is BUILT, correct, and a measured NULL. Every route is now closed.

§6/§7 left exactly one candidate for phase 2's serialisation: the LOOP ORDER (a core sends its own
column at iteration `r == my_col` of the same loop in which it receives every other column, so the
sender of round r+1 must first receive rounds 0..r). It is now implemented — `MOE_SWIGLU_HSEND=writer`
— and it is a null.

**What was built** (all behind the knob; `reader` remains the default and is byte-identical):
- The h broadcast moved to the WRITER (NoC1), which is not in the receive loop, so a root sends as
  soon as its column's h is assembled and the grid has freed its slot.
- `DEPTH_H` per-slot monotone arrival counters (`SEM_H_RDY_BASE..`), because with senders decoupled
  their increments interleave and a single counter says HOW MANY blocks arrived but not WHICH — the
  same defect Perf 2 #4 root-caused in the reduce. Per-slot is enough: only `DEPTH_H` rounds are ever
  in flight, so it costs no L1.
- A window ack (`SEM_H_FREE`): the receiver acks round r's sender straight after its
  `cb_reserve_back` and BEFORE waiting for data — that order is what makes the split deadlock-free.
- The writer derives the landing address arithmetically (`cb_h_base + slot * block`), because
  `cb_interface` is per-RISC-V local state and the writer is not cb_h's pusher, so `get_write_ptr`
  would never advance there.
- Each core keeps a PRIVATE per-slot expectation, which absorbs the fact that a root is excluded from
  its own multicast (its counter for that slot is one behind the grid's) with no special case.

**Correct**: golden **45/45** on `HSEND=writer`. **And slower**: 110 299 / 152 919 / 264 829 ns
against `reader`'s 104 813 / 146 741 / 246 790 (**+5 to +7 %**).

**The profile says why, and it disproves the hypothesis outright.** Phase 1 actually got FASTER
(`compute_gateup` 2→81 µs vs 2→85, because the send no longer sits on the reader's critical path) —
but `compute_down` stretched 40→144 vs 40→139. So the rounds were never waiting on the loop order.
They wait on **ROOT READINESS**: `compute_reduce` spans 40→96 µs, i.e. the eleven column roots finish
their reduce ~56 µs apart, and that spread is inherited straight from phase 1's DRAM stream. Round r
cannot start before root r's h exists no matter which RISC-V sends it, and decoupling costs an acked
write barrier + a non-posted atomic barrier per round on the sender plus one remote atomic per
receiver per round — more than the ordering it removes.

**Every candidate for the phase-2 serialisation has now been measured and eliminated**: CB depth
(§6a), grid shape (§6), absolute slot addressing (§6b, L1-locked by 296 064 B), the flag reset (§7,
Counter fixed and slower), and now the loop order (built and slower). What remains is not a phase-2
defect at all — it is that the ELEVEN COLUMN ROOTS FINISH PHASE 1 ~56 us APART, and that spread is
the weight stream's, which is already at its 490 GB/s roofline. Closing it means changing the
blocking model so a column's reduce does not wait on the slowest core in its column — a
planner-level scheme change, not a kernel one.

---

## Perf 4 — the NoC trace: nothing is contended, and the rounds are not data-gated

- **Date**: 2026-08-01
- **Scope**: perf only. `SUPPORTED` byte-identical, golden **45/45**, default path byte-identical
  (`MOE_SWIGLU_HACK_AHEAD` defaults to 1 and is additionally pinned to 1 on the shipped
  `HSEND=reader` path). Measured at **88 cores**, the grid the targets were taken on.

### 1. Instrumentation: tt-npe NoC event traces

Built tt-npe and captured device NoC event traces. No repo change was needed —
`TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1` + `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000` in front
of `run_safe_pytest.sh --profile` is all `tools/tracy --collect-noc-traces` does. Capture recipe,
the two saturation/overflow traps, and the analysis scripts are durable under
`perf_experiments/noc_trace/`.

### 2. What it overturns

**(a) There is no congestion and no bank imbalance.** tt-npe on the count-256 trace: NoC link util
**7.96 %**, multicast-write util 0.19 %, **congestion impact 0.08 %**. Per-DRAM-channel bytes are
3.24-3.37 MB across all 8 channels; the 16 endpoints are 2 subchannels x 8 channels
(`blackhole_140_arch.yaml` lists 3 NoC endpoints per channel) and the 2:1 endpoint split is only
NCRISC -> subchannel A / BRISC -> subchannel B. This retires the entire coalescing / bank-striding /
routing family for this op and retro-explains the `WRUN` and dual-NoC nulls — the same reason the
reference op's P1' died once it had two request paths.

**(b) The gap is DRAM-idle time, and it is what scales with M.**

| | count 128 | count 256 |
|---|---|---|
| DRAM bytes | 26.50 MB | 28.04 MB (+5.7 %) |
| traced wall | 112.6 us | 159.0 us (+41 %) |
| **us with DRAM < 10 GB/s** | **37 (33 %)** | **80 (50 %)** |
| longest contiguous idle window | 14 us | **45 us** |

Doubling M costs 46 us of wall and **43 us of it is time the DRAM system does nothing**. That is the
whole M-scaling gap against the unfused reference, which keeps its M-independent weight stream
running and hides the M-dependent work under it (`MCAST_FAMILIES=2` + `INJECT_LOOKAHEAD` on the
activation, Refinement 4's pipelined combine).

**(c) Phase 2 is not DRAM-bound.** Its stream is a square wave — ~1.5 us at 700-900 GB/s then ~3 us
of silence, x11, a 25 % duty cycle on a single RISC. So the reference's `BRISC_WEIGHT_K` split
(1.60x there) is worth nothing here, and neither is deeper `WD_AHEAD` prefetch.

**(d) The phase-2 rounds are NOT gated by root readiness.** Perf 3 concluded they were, from a
110-core zone span. Correlating each root's `compute_reduce` end against its own h multicast:

| round | root | reduce end | h mcast | idle |
|---|---|---|---|---|
| 0 | (1,2) | 101.2 | 102.9 | 1.7 us |
| 5 | (6,7) | 100.5 | 124.6 | 24.0 us |
| 10 | (15,4) | 88.7 | 145.9 | **57.2 us** |

Every root is finished by t = 101.2 us; the last round does not fire until t = 145.9 us on a rigid
4.3 us cadence. The rounds wait for **each other**, not for data.

**(e) The round cost, fitted across both traces** (m_eff 4 -> 3.66 us, m_eff 8 -> 4.30 us):

```
round period = 3.12 us FIXED + 0.147 us per m-tile
```

against ~2.06 us of real work at m_eff 8. **34.3 us per M-block of pure rendezvous, independent of
M** — 68.6 us at count 512 (two M-blocks), which is 27 % of that cell and exactly why 512 is the
worst cell. Same term as the 42.8 us all-payloads-stubbed floor.

**(f) The matmul is at roofline and the fused gate+up block is retired.** Per-call shapes at 88
cores: gate/up `M=m_eff, K=28, N=2` (6 calls/M-block, 1x2 of 8 DEST); `down` `M=m_eff, K=6, N=3`
(11 calls, 1x3 of 8 DEST). Small N and a quarter of DEST — and still **8.1 cycles/tile-MAC against an
8 cycles/tile-MAC LoFi roofline**. Nothing to win there.

**Floors**: count 256 / 88 cores, DRAM 28.0 MB at the 460 GB/s wall = **61 us**, compute
4272 tile-MACs = **25.3 us**, `max(...) = 61 us` against a 108 us target and 152 us measured. The
targets are ~1.8x INSIDE the DRAM floor. The whole remaining gap is serialisation.

### 3. `HACK_AHEAD` — built, measured, a NULL, with the mechanism named

The rendezvous chain is: round r's sender waits for one ack from each of NUM_CORES receivers, and a
receiver only acks after `cb_reserve_back` for round r proves its slot free. `HACK_AHEAD = A`
reserves A blocks at once, so a core acks senders r..r+A-1 together and every ack lands A-1 rounds
early. Clamp is a RUNTIME value (`blocks_cap = DEPTH_H * M_BLOCK / m_eff`, minus one so the reader
is never forced to zero blocks in flight), which is why the clamp lives in the reader.

88 cores, `HSEND=writer` (the per-slot-counter path the lever needs):

| A | count 128 | count 256 | count 512 |
|---|---|---|---|
| 1 (control) | 112 741 | 164 520 | 275 337 |
| **2** | **97 570 / 101 267** | **154 469** | 278 262 |
| max (5 / 2 / 2) | 113 443 | 160 685 | 278 756 |
| shipped (`HSEND=reader`, A=1) | 101 870 | 152 710 | 254 113 |

**The lever works on its own baseline** (-13.5 % / -6.1 % at A=2) **and still loses to the shipped
path**, because `HSEND=writer` starts 11 % behind and A=2 only recovers it. The count-128 "win"
(97 570) did **not** reproduce (101 267 on the same binary) — a 3.6 % spread, so it was noise.
`A > 2` is worse: reserving more blocks throttles the reader harder than the early acks help.

**Why the counter path starts behind, which is the finding worth keeping**: `DataReadySignal::Counter`
pays an **ACKED write barrier every round** — the sender must wait for all 87 destinations to
acknowledge the data multicast before it may bump the counter — where the `Flag` path terminates a
`NOC_CMD_VC_LINKED` chain on the same command buffer and skips it entirely. (That is the constraint
the Perf-3 addendum-2 `mcast_pipe` fix established: an atomic on a different command buffer cannot
terminate the link.) So the Counter path carries a **second** NUM_CORES-way incast per round, and it
eats the one `HACK_AHEAD` removes.

`MOE_SWIGLU_HACK_AHEAD` ships defaulted to 1 (byte-identical) with the table above, because the next
build needs it.

### 4. GRADUATED — per-slot FLAGS (`HSLOT`, default on) + `HACK_AHEAD = 2`

Built after the null above, from its mechanism. `mcast_pipe` offers two data-ready signals and each
gives up something this op needs: **Flag** rides one `NOC_CMD_VC_LINKED` chain with the data, so
ordering is free — but there is ONE cell, so round r+1's sender cannot set VALID until every core has
cleared round r's (mcast_pipe's own documented caveat). **Counter** is monotone per slot so rounds
overlap — but the signal is an atomic on a different command buffer, cannot terminate the link, and
therefore forces an ACKED write barrier every round: a second NUM_CORES-way incast.

**Per-SLOT flags take the union**: the link survives (no barrier) and rounds r and r+1 touch different
cells (no reset chain). Reuses the `DEPTH_H` `SEM_H_RDY_BASE` cells the Counter path already
allocates, so it costs **no semaphore and no L1**. With the flag chain gone, `HACK_AHEAD` becomes
legal on the fast path and the two ship together.

**Every one of the 12 guard cells is faster; none regressed.** 88 cores:

| cell | shared flag | **HSLOT + A2** | ratio |
|---|---|---|---|
| bf16_rm 128 | 99 601 | **98 433** | -1.2 % |
| bf16_rm 256 | 152 521 | **146 648** | **-3.9 %** |
| bf16_rm 512 | 254 279 | **243 754** | **-4.1 %** |
| bf16_rm cap1024 256 | 151 556 | **145 350** | -4.1 % |
| bf16_rm emb6144 256 | 132 361 | **130 086** | -1.7 % |
| bf16_rm 5120 | 2 102 697 | **2 003 063** | **-4.7 %** |
| bfp8 128 | 100 914 | **97 430** | -3.5 % |
| bfp8 256 | 143 630 | **131 081** | **-8.7 %** |
| bfp8 512 | 236 838 | **232 337** | -1.9 % |
| bfp8 cap1024 256 | 144 493 | **138 610** | -4.1 % |
| bfp8 emb6144 256 | 122 739 | **114 476** | **-6.7 %** |
| bfp8 5120 | 1 973 664 | **1 869 101** | -5.3 % |

**Two real bugs were found by the unit suite and are worth recording, because both are invisible to
the 45-cell golden suite (which never runs a multi-M-block count):**

1. **The flag index must be the GLOBAL round, not `r`.** `r % DEPTH_H` restarts every M-block, so at
   `HGROUPS = 11 / DEPTH_H = 3` the flag used by block b's round 9 is reused by block b+1's round 0
   only TWO global rounds later, while the ack only guarantees `DEPTH_H`. A sender then set VALID on
   a cell a slower core had not yet cleared, that core cleared it, and the signal was lost —
   **a hang at count 1024 (4 M-blocks)**. Fixed by indexing both sides with `(b * HGROUPS + r) %
   DEPTH_H`; the Counter path is immune because monotone counters cannot lose an increment, which is
   exactly why this only appeared on the Flag path.
2. **The ack-ahead bound is `min(DEPTH_H, blocks_cap - 1)`, not `blocks_cap - 1`.** `blocks_cap =
   DEPTH_H * M_BLOCK / m_eff` is 6 at m_eff 4 and 24 at m_eff 1, so the CB-slack bound alone lets a
   core ack a sender that will reuse a FLAG the core has not cleared. Only `A <= DEPTH_H` keeps the
   previous user of every cell strictly in this core's past. (This voids the earlier `A = max`
   numbers, which ran unsafely.)
3. `h_free_expected` must live OUTSIDE the M-block loop: `SEM_H_FREE` is monotone across blocks, so a
   per-block expectation is satisfied instantly from block 1 on and the flow control silently dies.

**Deeper is a measured NULL, and this closes the lever.** `A > 2` needs `DEPTH_H >= 4` (at
`m_eff = M_BLOCK`, `blocks_cap = DEPTH_H`), which needs L1. It is available without the Perf-2 §5.4
aliasing enabler: `CB_X_TILES` is `DEPTH_X * M_BLOCK * KR_PAD` bfp8 = **487 424 B**, the largest CB
in the op, and its second buffer only pays ACROSS M-blocks. Trading it (`DEPTH_X=1`) funds
`DEPTH_H=6` with room to spare:

| config | 128 | 256 | 512 | 5120 |
|---|---|---|---|---|
| shipped `DX=2 DH=3 A=2` | **98 433** | **146 648** | **243 754** | **2 003 063** |
| `DX=1 DH=6 A=5` | 99 864 | 146 635 | 248 281 | 2 015 674 |

**Identical at 128/256 and worse at 512/5120** (the latter is Refinement 3's resident-x win being
given back, exactly as priced). Doubling the slot count and more than doubling the ack lookahead
moves the focus cell by 13 ns. **The rounds are no longer slot-limited or ack-limited** — whatever
remains of the ~1.7 us/round fixed cost after this graduation is the multicast and the per-round
barriers themselves, not the rendezvous protocol. Do not spend another round on `DEPTH_H`, and the
CB-aliasing enabler is NOT needed for this.

### 5. Still not enough — what is left

88 cores, bf16_rm, against the targets:

| count | session start | **now** | target | gap |
|---|---|---|---|---|
| 128 | 99 900 | **98 433** | 91 800 | 7.2 % |
| 256 | 152 000 | **146 648** | 108 000 | 35.8 % |
| 512 | 253 700 | **243 754** | 161 816 | 50.6 % |

The rendezvous term is 34.3 us per M-block and this recovers roughly a sixth of it, because
`DEPTH_H = 3` lets only three rounds overlap. Two levers remain, in order:

**(a) ~~`DEPTH_H >= 4`~~ — measured NULL above. Closed.**

**(b) The 45 us DRAM-idle window** (§2b). Even a perfect rendezvous fix leaves count 256 at ~118 us
and 512 at ~185 us, so 256/512 additionally require starting phase-2 rounds during phase 1 — legal,
because round r needs only column r's h and the trace shows every root is ready ~44 us early.

### 6. The original named next build (now done)

**Per-slot FLAGS, not per-slot counters.** Keep the send on the reader and the linked
data+signal multicast (so no acked barrier), but give each `cb_h` slot its own VALID/INVALID flag
instead of the one shared cell. That removes the shared-flag barrier — the only thing forcing
`HACK_AHEAD = 1` on the fast path — while keeping the link that makes the Flag path fast. Round r and
r+DEPTH_H share a flag and the ordering is already safe (a core acks slot s only after clearing it).
This is a `mcast_pipe` change (`SenderPipe`/`ReceiverPipe` take one `DATA_READY_SEM_ID`), and it
must carry the reference op's P7 `ROTATING_SENDER` reset-race mitigation.

Ceiling, stated honestly: 34.3 us/M-block. Even a **perfect** rendezvous fix leaves count 256 at
~118 us (target 108) and count 512 at ~185 us (target 162). **Reaching 256/512 additionally requires
closing the 45 us DRAM-idle window** — i.e. starting phase-2 rounds during phase 1, which is legal
because round r only needs column r's h and every root is ready 44 us early.

- **Accuracy**: unchanged. `HACK_AHEAD` moves the timing of a semaphore increment; the arithmetic is
  bit-identical. Golden 45/45, no precision knob touched.
- **Perf achieved**: **none.** Every measured variant is a null or a regression against the shipped
  path; the deliverable of this round is the measurement, not a number.
- **Tests added**: none — the 45-cell golden suite and the 12-cell guard set already cover both
  settings of the new knob. Durable artifacts under `perf_experiments/noc_trace/`.

---

## Perf 6 — "all weights already in L1": the measurement that reprioritises everything

- **Date**: 2026-08-01
- **Scope**: measurement only, no code change. 88 cores, emb 7168, bf16_rm.

### 1. Two independent ways to get it, and they agree to 0.05 %

**(a) The ablation.** `MOE_SWIGLU_ABLATE=no_w_xfer` stubs every weight DRAM read and keeps every CB
reserve/push/pop, barrier and loop trip count — i.e. exactly "the weights were already sitting in
their CBs".

**(b) The op already does it for real, 19 times.** `W_RESIDENT`/`WD_RESIDENT` mean M-block 0 reads
the weights and blocks 1..19 re-use the same CB slots with no DRAM at all. So at `count = 5120`
(20 M-blocks) the MARGINAL M-block IS the weights-in-L1 case, with no ablation anywhere:

| | with weight DRAM | weights stubbed |
|---|---|---|
| count 5120 | 2 003 063 | 1 964 424 |
| count 256 | 146 648 | 107 088 |
| **marginal M-block** `(5120 - 256)/19` | **97 706 ns** | **97 754 ns** |

**Identical to 0.05 %.** The ablation and the real resident path measure the same thing, which
validates (a) and independently proves the residency actually works.

### 2. The numbers

| count | shipped | **all weights in L1** | target | verdict |
|---|---|---|---|---|
| 128 | 98 433 | **61 773** | 91 800 | **33 % UNDER** |
| 256 | 146 648 | **107 088** | 108 000 | **AT target** |
| 512 | 243 754 | **205 073** | 161 816 | **27 % OVER** |
| 5120 | 2 003 063 | 1 964 424 | — | — |

**The exposed weight-DRAM cost is FLAT: 36 660 / 39 560 / 38 681 / 38 639 ns.** M-independent, as
the blocking model says it must be (24.77 MB of weights, read once, `W_RESIDENT`).

### 3. What it reprioritises

1. **count 128's entire gap IS the weight stream.** Free the weights and it is 61 773 against a
   91 800 target — a third under. Nothing else about that cell is wrong.
2. **count 256's gap is the weight stream almost exactly.** 107 088 vs a 108 000 target.
3. **count 512 is NOT a weight problem.** Even with every weight free it is 205 073 against 161 816.
   Its problem is the MARGINAL M-BLOCK: 97.7 us with all weights resident, against ~40 us of real
   work (16 us gate/up matmul + 12 us `down` + 8 us x DRAM + 4 us output write). A 2.4x
   serialisation tax with nothing to do with weights.

**And the weight stream cannot be hidden by reordering alone.** `WD_AHEAD=11` — prefetch the whole
8.26 MB W_down stream into the phase-1 DRAM-idle window, which Perf 4's trace showed is 45 us of
dead DRAM — re-measured on the current tree: **115 630 / 169 013 / 265 704, a +15 % REGRESSION**
(and it reproduces the pre-Perf-3 result at a completely different baseline). The prefetch competes
with the grid's in-flight W_gate/W_up, and the gate/up matmul is gated on those, and everything
follows the matmul. The idle window is real but it is downstream of the dependency that creates it.

### 4. Peeling the weights-free op further

| arm | 128 | 256 | 512 |
|---|---|---|---|
| weights in L1 | 61 773 | 107 088 | 205 073 |
| + no compute | 44 262 | 72 767 | 136 954 |
| + no h transport | 54 530 | 99 853 | 181 004 |

* **compute exposed** 17 511 / 34 321 / 68 119 — clean 2x per M doubling.
* **h transport exposed** 7 243 / 7 235 / 24 069 — flat 128->256, then 3.3x at 512 (two M-blocks).
* **floor with neither** 44 262 / 72 767 / 136 954 — x + reduce + output + the sync skeleton, and
  still 68 % of the weights-free op at count 256.

**A correction to Perf 4 §2f.** "The matmul is at roofline (8.1 cycles/tile-MAC)" was derived from a
110-core ablation. At **88 cores it is 10.8** — 34 321 ns for 4272 tile-MACs/core — against the same
8 cycles/tile-MAC LoFi roofline, i.e. **~26 % headroom, worth ~9 us at count 256 and ~18 us at 512**.
Per-core work only rises 1.13x from 110 to 88 cores while exposed compute rises 1.51x, so the extra
is not FPU work; the small-N `matmul_block` shape (N=2 gate/up, N=3 `down`, 2-3 of 8 DEST tiles) is
the obvious suspect and is now worth a bench rather than being retired.

### 5. Ranked, with what each is worth

| # | lever | worth | applies to |
|---|---|---|---|
| 1 | hide the 38.6 us weight stream | **-38.6 us, M-independent** | takes 128 and 256 to target |
| 2 | cut the marginal M-block's 58 us serialisation tax | -58 us per block | the only way to 512 / 5120 |
| 3 | the small-N matmul shape (10.8 -> 8 cycles/tile-MAC) | ~9 us @256, ~18 us @512 | all cells |

Lever 1 is NOT a prefetch reorder (measured above). It needs the gate/up matmul to stop being a
barrier — i.e. consume W_gate/W_up chunk-by-chunk as they land, which `GU_CHUNKS` already does for
the READ but not for the round-trip through reduce -> h -> down.

---

## Perf 7 — the peel had a HOLE, and closing it kills two of Perf 6's recommendations

- **Date**: 2026-08-01
- **Scope**: instrumentation + measurement. New ablation `skip_eltwise`. No default-path change.

### 1. The hole

`SKIP_COMPUTE` is a **`matmul_block_helpers` define** and elides only the inner
`ckernel::matmul_block` LLK call. Every `eltwise_chain` in the translation unit keeps running — it
has its own define, `CKL_ELTWISE_CHAIN_SKIP_COMPUTE`, which this op never set. So every
"all payloads stubbed" floor this op has quoted since Perf 3 **silently contained the entire combine
add chain, the SiLU, the SwiGLU multiply and the fused tilize**. Exactly the reference op's `owrite`
trap, where its `down` peel bottomed out in a 47 % "floor" that turned out to hold a 1.95 MB DRAM
stream.

`MOE_SWIGLU_ABLATE=skip_eltwise` closes it.

### 2. The complete peel (88 cores, all transports already stubbed)

| arm | 128 | 256 | 512 |
|---|---|---|---|
| all transports stubbed | 42 379 | 78 976 | 150 123 |
| + `skip_eltwise` | 41 647 | 78 264 | 148 359 |
| + `skip_compute` + `skip_eltwise` | **22 141** | **40 400** | **70 853** |

**The whole eltwise costs 712 ns at count 256.** Combine chain, SiLU, SwiGLU multiply and tilize
together, 0.5 % of the op.

### 3. Two of Perf 6's conclusions are RETRACTED on this measurement

1. **`pack_l1_pair` is dead.** Perf 6 called it "the right next build" on the strength of a
   1.90-2.09x isolated number and the fact that it now fits in L1. It is worth **at most 712 ns**,
   because the stage it speeds up is 0.5 % of the op. The isolated number was real; the stage is not
   material. Same for `dest_acc` (1.35x) and `pack_l1_acc` (1.22x) — all three are now closed on
   size, not on risk.
2. **The matmul is NOT 26 % off shape.** Perf 6 read 10.8 cycles/tile-MAC and blamed the small-N
   `matmul_block` window (N=2, 2 of 8 DEST). Measured directly, with weights stubbed so weight-stream
   pipelining cannot confound it:

   | | 128 | 256 | 512 |
   |---|---|---|---|
   | `GU_CHUNKS=3` (shipped, N=2) | 61 773 | 107 088 | 205 073 |
   | `GU_CHUNKS=2` (N=3) | 61 335 | 106 912 | 204 317 |
   | `GU_CHUNKS=1` (N=6, 6 of 8 DEST) | 60 835 | 105 991 | 202 652 |

   Tripling the DEST width and cutting the call count 3x buys **1.0-1.5 %**. The matmul is not
   shape-limited; the 8 cycles/tile-MAC roofline was an unverified assumption and ~10.5 is very
   likely the real bfp8 x bfp4 LoFi limit (the unpacker decompresses bfp4). **Treat the matmul as
   done** — and note this also re-kills the custom fused gate+up block for the third time, now on a
   direct measurement rather than an inference.

### 4. The corrected budget — count 256, 146 648 ns total

| term | ns | share |
|---|---|---|
| **true sync floor** (CB + semaphore bookkeeping ONLY) | **40 400** | **28 %** |
| weight DRAM | 39 560 | 27 % |
| matmul LLK | 33 135 | 23 % |
| x DRAM + staging | 16 282 | 11 % |
| reduce transport | 6 825 | 5 % |
| h transport | 5 005 | 3 % |
| all eltwise | 712 | 0.5 % |

**The true sync floor is now the largest single term in the op**, and it is not fixed overhead:
22 141 / 40 400 / 70 853 scales as ~M^0.86, i.e. per-block CB ops, not launch cost.

At **count 128 the bookkeeping alone (22 141 ns) is 1.75x the entire matmul roofline (12 658 ns)**
before a single byte moves anywhere. That is the honest reason M=128's math utilisation is ~13 %
shipped and only ~30 % with every transport stubbed.

### 5. `no_owrite` — the LAST hole, and the complete peel

`writer_out_issue` measured **17.4 us** inside what §2 was calling a pure synchronisation floor,
because there was no output-write ablation: the op's 1.95 MB output DRAM stream (count 256) was
still running. This is the *identical* trap the reference op recorded ("without `owrite` the peel
bottomed out in a 47 % floor that silently contained its 1.95 MB DRAM output stream") and it is the
third instrumentation hole this round has closed. `no_owrite` now ships.

**The complete peel, 88 cores:**

| peel step | 128 | 256 | 512 |
|---|---|---|---|
| shipped | 98 433 | 146 648 | 243 754 |
| - weight DRAM | 61 773 | 107 088 | 205 073 |
| - x DRAM + staging | 52 327 | 90 806 | 171 761 |
| - reduce transport | 44 255 | 83 981 | 157 679 |
| - h transport | 42 379 | 78 976 | 150 123 |
| - output write | 41 037 | 74 361 | 145 306 |
| - matmul + eltwise = **TRUE FLOOR** | **20 401** | **34 774** | **65 754** |

**Per-term, and it closes to the total:**

| term | 128 | 256 | 512 | share @256 |
|---|---|---|---|---|
| weight DRAM | 36 660 | 39 560 | 38 681 | **27.0 %** |
| matmul LLK | 20 636 | 39 587 | 79 552 | **27.0 %** |
| sync floor (CB + semaphore) | 20 401 | 34 774 | 65 754 | **23.7 %** |
| x DRAM + staging | 9 446 | 16 282 | 33 312 | 11.1 % |
| reduce transport | 8 072 | 6 825 | 14 082 | 4.7 % |
| h transport | 1 876 | 5 005 | 7 556 | 3.4 % |
| output write | 1 342 | 4 615 | 4 817 | 3.1 % |
| all eltwise | — | 712 | — | 0.5 % |

**Three terms of 27 / 27 / 24 %, none of which overlaps any other.** That is the whole problem stated
in one line. The matmul runs at 12.9 cycles/tile-MAC against a nominal 8 (2136 / 4272 / 8544
tile-MACs per core = 12 658 / 25 316 / 50 631 ns) — but §3 measured that widening its DEST window 3x
buys 1 %, so 12.9 is very likely the real bfp8 x bfp4 LoFi rate and not slack.

### 6. What this leaves

Only two terms are big enough to matter, and neither is a knob:

* **the 40 us sync floor** — CB reserve/wait/push/pop and semaphore ops, counted per block. Reducing
  it means fewer, coarser CB transactions across the reduce-scatter's all-to-all and the phase-2
  round loop, not a faster stage.
* **the 39 us weight stream** — M-independent, and `WD_AHEAD=11` proved it cannot be hidden by
  reordering alone (Perf 6 §3).

Everything else in the op is now measured to be at or near its floor.
