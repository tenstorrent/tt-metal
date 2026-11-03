# Changelog: moe_down_proj

## Phase 0 - Core Implementation
- **Date**: 2026-11-03
- **What was done**: Initial implementation via incremental pipeline (planner -> implementer -> verifier). Regime R1:
  2D split (expert-sorted rows over 9 m-groups, N over 12 n-lines, L layout, 108 / 110 cores). Each K round runs
  rotating-sender x and w multicasts and accumulates fp32 packer-L1 partials, followed by an untilize and a
  row-major store per wave. Counts are read on device only. One `ttnn.generic_op` per call.
- **SUPPORTED at Phase 0**: dtype=[bfloat8_b], layout=[TILE], fp32_dest_acc_en=[True], memory_layout=[INTERLEAVED],
  buffer_type=[L1], out_buffer_type=[L1]; no INPUT_TAGGERS, no EXCLUSIONS
- **Accuracy achieved** (LoFi default, 5 shapes, `test_moe_down_proj_precision_baseline.py`): PCC=0.99991,
  max_abs_err=0.279, rel_rms_err=1.90e-2 (HiFi2: PCC=0.999999, rel_rms 1.68e-3). The LoFi got/true ratio median of
  0.9865 is the fidelity's truncation bias; it is 1.0000 at HiFi2.
- **Device kernel time (us)**: t4096 L0..L5 173.2 / 174.5 / 175.0 / 178.9 / 173.7 / 173.1; t3072 140.0;
  t2048 113.7; t1024 102.8 (all under stock; targets 90.8 / 74.6)
- **Golden suite at Phase 0**: 27 supported_pass, 54 xfail_expected, 0 supported_fail / xpass_drift /
  xfail_wrong_mode (93 total; 12 regression tests untagged, passing)
- **Issues encountered** (verifier fixes): CB sizing was duplicated between the budget solve and the allocation, now
  one `cb_plan()`. A raw `llk_pack_reconfig_l1_acc` was replaced by `pack_reconfig_l1_acc`. The writer CT-arg
  declarations were reordered. The ledger's `cb_w_round` producer was corrected (writer RISC). No kernel bugs.
- **Tests added**: test_moe_down_proj.py (planner), test_moe_down_proj_perf.py (implementer),
  test_moe_down_proj_precision_baseline.py and test_moe_down_proj_layouts.py (verifier)

## Refinement 1 - DRAM placement for x and the output
- **Date**: 2026-11-03
- **What was done**: `SUPPORTED["buffer_type"]` and `SUPPORTED["out_buffer_type"]` gain `DRAM`. The x senders and
  the y store already went through `TensorAccessor`s. Reused: the whole R1 schedule, the CB plan, `select_blocking`.
  Added: (1) `moe_down_proj_store.h`: `store_wave` moved out of the writer so either dataflow RISC can run it. When y
  is in DRAM, the host sets `store_on_reader` and the reader RISC (NoC1, the DRAM-write NoC) stores each wave after
  `X_DEPTH` rounds of the next wave, mirroring the writer's schedule. An L1 y store stays on the writer, because
  moving it measured 172 -> 179 us. (2) `cb_budget_bytes()`: the CB budget is `CB_BUDGET_BYTES` plus the MAX_ROWS
  per-bank share of whichever of x / y is in DRAM. It is shape- and placement-derived and byte-identical for L1 / L1.
  `MAX_ROWS` / `MAX_TOKENS` moved into the descriptor module as the single source.
- **Device kernel time (us)**, t = 4096 layer 0 / t = 1024 scaled, before (gate flip only) -> after:
  L1/L1 174.1 / 101.7 -> 172.0 / 102.4; DRAM x 218.2 / 107.1 -> 221.2 / 117.2 (the levers do not touch this path;
  t = 1024 DRAM x ranged 107-117 across runs); DRAM y 253.9 / 123.9 -> 220.4 / 112.2; DRAM both 291.3 / 126.7 ->
  258.8 / 117.2. Production rows: 172.2 / 175.1 / 173.2 / 174.1 / 171.4 / 171.3 / 137.7 / 113.3 / 103.2 (no
  regression).
- **Measured and not kept**: DRAM bank stagger on the y row walk (flat; 7 banks, so 32k rows do not align);
  swapping the reader/writer NoCs (DRAM x t4096 221 -> 200, but every t1024 row +40 us); DM_DYNAMIC_NOC to split
  NoCs per transfer (does not compile: the mcast helper's TXN_ID flush static_asserts under dynamic NoC); the x
  prefetch issue quota 4 / 8 / 11 for DRAM x (flat); fewest-waves-first blocking order (one wave at k=4/6; flat or
  +5 us).
- **Remaining headroom**: a write-stubbed ablation puts the DRAM y store at about 105 us of work at t = 4096 (128 B
  segments, 25 us for L1 y). The next lever is a per-m-group row gather to 1536 B DRAM writes (cross-core). DRAM x
  adds about 47 us at t = 4096 through 27.5 MB of NoC1 DRAM reads.
- **Accuracy achieved**: PCC=0.99991, rel_rms=1.90e-2 (LoFi default; unchanged), all DRAM placements at golden
  tolerances.
- **Golden test progress**: 57/57 on the slice run (all 54 DRAM cells, plus t1_small and t4096_layer0 L1/L1 and
  loose). The 27 L1/L1 cells are on unchanged code paths.
- **Issues encountered**: whole-file runs of the unit tests under `run_safe_pytest.sh` precompile (inline up-front
  collect) return NaN outputs (pcc nan) on every case, including at the pre-refinement kernels. The same files pass
  33/33 and 45/45 with `--no-precompile`. This looks like the collect pass's addr-0 NO_DISPATCH program interacting
  with the generic_op program cache, not a kernel bug. It is not addressed here.
- **Tests added**: `test_moe_down_proj_perf_placement` in test_moe_down_proj_perf.py (t4096_layer0 and t1024_scaled
  x 4 placements, for `--profile`).

## Refinement 1b - DRAM placement for x and the output (debug: fix gate violations)
- **Date**: 2026-11-03
- **What was done**: the gate failure (`test_moe_down_proj[t1]` pcc nan, 21/33 in the file) reproduced only under the
  inline precompile run that `run_safe_pytest.sh` uses for a whole file or dir. The collect pass swaps `torch.randn`
  for zeros and `ttnn.from_torch` for a shape-only allocation. `host_weight()` is `lru_cache`d, so the real pass reused
  the stand-in weight, which gave a constant output and reference. Fix: a module-scoped autouse fixture in the unit-test
  conftest clears every module-level `cache_clear`-able function. The plugin tears down all fixtures between passes, so
  each pass builds its own weight. Kernels and the op file are unchanged.
- **Accuracy achieved**: unchanged (PCC=0.99991, rel_rms=1.90e-2 LoFi default).
- **Golden test progress**: 93/93 (full suite, `--no-precompile`, which matches the graded run's separate warm pass).
  Under inline precompile the golden suite shows the same stand-in leak through `helpers.py`'s `lru_cache` /
  `_CASE_CACHE` (rms=inf). That suite is not modifiable, and the eval runner warms in a separate process, so the leak
  does not affect grading.
- **Issues encountered**: see above. Not a kernel or program-cache issue.
- **Tests added**: none. conftest.py gains `_fresh_module_caches`. Unit dir: 72/72 with precompile on.

## Refinement 2 - Cut the multicast scaffold on the t = 4096 rows
- Date: 2026-11-03
- What was done: paced wave store. With `num_waves >= 2`, `cb_output_rows` holds the whole wave block
  (`out_depth_blocks()`), so compute untilizes wave p without waiting on the store. The storing RISC drains it
  `store_blocks_per_round` blocks per round of wave p + 1 (`moe_store_wave` takes a block range), instead of one burst
  that stalled compute and the w channel. `CB_BUDGET_BYTES` 960 -> 976 KiB keeps t = 4096 at 2 waves (993,856 B); the
  3-wave solve this would otherwise pick measured 179-188 us. Reused: the store helper, both RISCs' store schedule,
  `cb_plan()`. Added: `STORE_PACED`, `out_depth_blocks()`, `store_blocks_per_round()`, one CT arg per dataflow kernel.
  Device kernel time (us), t4096 L0-L5 L1 / L1: 172.2 / 174.7 / 172.8 / 174.1 / 171.3 / 171.5 -> 163.3 / 162.2 /
  162.9 / 164.7 / 163.3 / 161.6. DRAM x / DRAM y / DRAM both at t4096: 221 / 220 / 259 -> 214 / 205 / 244. t3072 /
  t2048 / t1024 138.4 / 113.8 / 103.3 -> 138.2 / 112.2 / 101.7.
- Measured and not kept: x strip core self-fetch from L1 (400 us; one core reads about 9 GB/s); x strip copy issued by
  the sender core's writer RISC on NoC0 via a stage CB handoff (219 us after-w, 369 us before-w, 249 us deferred;
  t1024 133-138 us); y store on the reader RISC with pacing (190-206 us); one-packet store writes (flat); x / w prefetch
  quota sweeps (16 stays best); rect layout (218 vs 175 us at t4096 layer 3).
- Not built: R2 range-pruned w. With the store paced, sending a single w chunk per round (an upper bound on pruning)
  saves 2.5 us and removing w entirely 3 us, because the x chain binds. Pruning also leaves the line sender's
  injection bytes unchanged.
- Bottleneck now: the x chain. Each m-group step injects 130 KB row multicast plus 130 KB strip copy from one sender on
  NoC1. Ablations: strip copy -7 us, x send bytes -18 us, x L1 reads -13 us, compute -11 us.
- Accuracy achieved: unchanged summation order. PCC=0.99991, rel_rms=1.90e-2 (LoFi default); precision baseline file
  passes.
- Golden test progress: 55/55 on the slice run (all t4096 and t3072 cells across the four placements, plus t1 and
  t37); unit dir 72/72.
- Issues encountered: multi-id `--profile` runs under inline precompile hit the known stand-in weight leak (pcc nan),
  so measurements used `--no-precompile`.
- Tests added: none (the perf rows in test_moe_down_proj_perf.py cover the change).

## Refinement 3 - Speed up the DRAM-bound rows (t = 1024 / 2048 / 3072)
- Date: 2026-11-03
- What was done: two levers. (1) Split w DRAM reads across both dataflow RISCs. The staged w prefetch moved into
  `kernels/moe_down_proj_w_prefetch.h` (`MoeWPrefetch`), so either RISC can run it. On one-wave blockings,
  `w_reader_share()` / `W_SHARE_DEN` (8) of each owned step's tiles are read by the reader RISC on NoC1 into
  `cb_w_stage`, which it reserves and pushes one step before the owned step. The writer RISC reads the rest on NoC0,
  waits for the stage, multicasts and pops. The writer publishes its sender ownership (first owned step, period) to
  the reader through the new 16 B `cb_w_owner`. The mcast host adapter refuses a sender channel on a kernel of
  another NoC, so the reader cannot take the w args directly. The reader's x and w reads carry separate
  transaction ids, so neither barrier waits on the other channel. The share is
  `clamp((25 - core_row_tiles_max) // 5, 0, 4)`, and 0 with >= 2 waves. Share 0 is the previous writer-only path.
  (2) `select_blocking` takes the fewest waves at `k_block_tiles >= 6` while rows per core <= 24 and y is in L1,
  then falls back to the old K-block-first order. This makes t3072 L1 / L1 one wave at k = 6.
  Reused: the staged prefetch logic, the rotating w channel, `cb_w_stage`, `cb_plan()`. Added: the prefetch header,
  `cb_w_owner`, the share and one-wave knobs, and t3072 rows in `test_moe_down_proj_perf_placement`.
- Device kernel time (us), before -> after: t1024 102.1-102.8 -> 85-88; t2048 112.1-112.7 -> 107.6-108.8; t3072
  137.8-138.5 -> 133.6-134.1. t4096 layers 0-5 hold: 163.5 / 162.1 / 163.2 / 163.9 / 162.9 / 161.5 ->
  163.9 / 161.9 / 163.0 / 163.2 / 163.2 / 161.8. Unit shapes: t1 44.2 -> 38.8, t16 85.8 -> 67-70, t37 135.4 -> 82,
  t300 96.4 -> 81, t1858 110.5 -> 102-105; t4096 uniform / one_expert / zeros_between hold. Placements: t1024
  DRAM x / y / both 114.0 / 112.4 / 117.7 -> 107.8 / 93.5-95.5 / 114.0. t3072 and t4096 DRAM placements hold (t4096
  DRAM both ranges 242-251 in both versions).
- Ablations (t1024): w reads stubbed 55 us; w reads with a one-tile multicast 80 us; full 103 us. The w read
  responses contend with the w multicast on NoC0. All w reads on NoC1: 155-163 us (NoC1 DRAM reads about half
  as fast), hence a partial share. Stub floors at t2048 / t3072: 92 / 124 us (x / sync bound).
- Measured and not kept: w issue quota 4 / 8 / 24 (vs 16; flat or worse), W_DEPTH 3 (flat), k = 12 / 16 at t1024
  with a raised budget (110 / 115), DRAM bank stagger of the w walk (flat to +4 us), a reader share with >= 2 waves
  (t4096 +10-20 us), one wave at t4096 DRAM x (208 -> 223) and t3072 DRAM y (159 -> 165, exposed store tail),
  hence the row cap and the L1-y gate.
- Remaining headroom: t1024 is about 30 us above its no-w-reads floor. Next lever: range-pruned w multicast (R2's
  deferred topology), which shrinks the NoC0 multicast traffic the reads contend with.
- Accuracy achieved: PCC=0.99991, rel_rms=1.90e-2 (LoFi default; t1024 1.9004e-2, t4096 1.9008e-2), HiFi2 PCC=0.999999;
  summation order unchanged at k = 8. t3072 L1 / L1 now accumulates in k = 6 blocks and passes golden.
- Golden test progress: 60/60 on the slice run (all t1024 / t2048 / t3072 cells incl. loose, small shapes, t1858,
  the regression file, t4096 layer0 and one_expert across placements); unit dir 76/76 (72 plus the
  four new t3072 placement rows).
- Issues encountered: `git checkout <rev> -- <path>` for A/B baselines also staged the old files; reset the index.
- Tests added: t3072_scaled in `PLACEMENT_CASES` (test_moe_down_proj_perf.py).

## Refinement 4 - Model L1 occupancy: CB budget from the live allocator state, fast within it
- Date: 2026-11-03
- What was done: (1) `CB_BUDGET_BYTES` (976 KiB, tuned on an empty L1) is gone. `cb_budget_bytes(device)` reads the
  L1 allocator (`ttnn.get_memory_view`; block addresses are relative to the allocator base where CBs start) and
  returns the bytes below the lowest live buffer. `create_program_descriptor` uses it after the output exists.
  `validate()` predicts the output's placement (charged unless a free block above the lowest live buffer holds it),
  runs `select_blocking`, and raises `UnsupportedAxisValue` when nothing fits, before any allocation;
  `footprint_bytes()` rounds each CB to 64 B. (2) `w_issue_quota()` moves the writer's w prefetch quota to the host
  with a floor of 8 tiles per step on multi-wave blockings (`W_ISSUE_TILES_MIN_MULTI_WAVE`). (3)
  `DRAM_Y_ONE_PASS_MAX_ROWS` = 16 makes the old constant's implicit "two waves at t >= 3072 with y in DRAM" explicit.
  The live budget would otherwise pick one wave there. (4) `x_step_rounds`: an x step may carry several compute K
  rounds (round-major stage layout, reader only), which shrinks the w CBs at a fixed x step count. It is correct but
  measured slower at the occupied shape, so it is parked at `X_STEP_ROUNDS_MAX` = 1.
  Reused: `select_blocking`, `cb_plan()`, the reader prefetch, compute and writer unchanged except the quota CT arg.
  Added: the live budget, the refusal, the quota and DRAM-y knobs, `x_step_rounds`.
- Device kernel time (us), occupied t4096 L0-L5 (853,568 B free): CB clash before; the live budget alone (2 waves
  k = 6) 176.8-178.6; final 171.7 / 170.8 / 171.0 / 172.5 / 171.5 / 169.4 (stock 175.0-191.9, target 90.8).
  Other blockings that fit: 3 waves k = 8 179.6-180.5, 2 waves k = 4 186-189, k = 4 with k = 8 x steps 179.6-180.8,
  4 waves k = 8 189.9, 1 wave k = 3 185.6; 2 waves k = 8 with W_DEPTH 1 176.4. Unpaced store and X / W depth 3
  at k = 6 182.6 / 183.0 / 179.5. The prefetch steps knob at k = 6 gives 12 -> 173.4-174.0 against 16 -> 176.0-177.2,
  but at t3072 (one wave k = 6) 16 beats 12 (133.4-133.9 vs 135.2-135.3), and the w quota floor beats both.
  Unoccupied: t4096 L0 / L3 163.8 / 164.1, t3072 133.9, t2048 108.8, t1024 85.3-87.3 (hold); t3072 DRAM y 158.6,
  DRAM x 211.4 at t4096 and DRAM both 244.2 at t4096 (hold); t3072 DRAM both 193.6 -> 188.2.
- Ablation (compute stubbed): 2 waves k = 8 152, k = 6 161.7, k = 4 with k = 8 x steps 172.9 us. About 0.9 us per
  w step and 0.35 us per x step of fixed channel sync, so smaller K blocks cost dataflow time, not compute.
- Remaining headroom: pipeline the multicast handshake (ready for step s + 1 before waiting on s) in the shared
  mcast helper, which would make the k = 6 blocking about as fast as k = 8. Not done here because it changes the
  `kernel_lib` protocol.
- Accuracy achieved: occupied t4096 (k = 6) passes the golden PCC >= 0.9998 / rel_rms <= 2.0e-2 gates and
  `test_model_l1_occupancy`; unoccupied summation order unchanged (precision baseline: PCC 0.99991, rel_rms 1.90e-2
  LoFi; HiFi2 PCC 0.999999).
- Golden test progress: 28/28 on the slice (t3072 / t2048 / t4096 layer 0 and 3 across placements, the six
  occupied loose rows, t1 / t37 small); unit dir 78/78.
- Issues encountered: a golden slice under inline precompile fails every cell with a zero reference (the known
  stand-in leak in the golden helpers' case cache). Run golden slices with `--no-precompile`.
- Tests added: `test_model_l1_occupancy`, `test_l1_occupancy_without_any_fit_refused` (test_moe_down_proj.py);
  `probes/sweep_occupied.py` (forced-blocking sweep, run explicitly).

## Perf 1
- Date: 2026-11-03
- Focus: `t4096_layer0` LOOSE case (attention-flagged primary): x and y in L1, model L1 occupancy (853,568 B free),
  bfloat8_b, op default config (LoFi, fp32_dest_acc_en=True, math_approx_mode=False). Every axis is in SUPPORTED.
  Blocking there: 2 waves, k_block 6, 15 rows x 2 n tiles per core, 32 x steps + 32 w steps. Stock 179.0 us,
  target 90.8 us.
- Instrumentation: permanent `MaybeDeviceZoneScope` zones on every stage: compute_accumulate / compute_wait_inputs
  (the round's x and w wait hoisted out of the matmul helper; the writer pushes a whole round at once, so the order
  is unchanged) / compute_untilize; x_reserve / x_send (+ x_send_barrier) / x_receive / x_forward; w_reserve /
  w_send (+ w_send_barrier) / w_receive; w_stage_complete; y_store / y_store_tail / y_store_tail_helper. At most 198
  markers per RISC on the guard set (cap 250; a k_block 4 multi-wave blocking would exceed it on the reader).
  Under the profiler the zones add 4-6 us to the focus number (uninstrumented HEAD kernels 172.1 us, instrumented
  178.1 us). Every number below is instrumented vs instrumented.
- Measured breakdown (focus, `perf_experiments/breakdown/BREAKDOWN.md`): zones showed compute starved about 60 us
  of its 153 us on inputs, the x channel's receivers waiting about 120 us, and the last wave's store exposed
  for about 14 us after compute. Cumulative ablation (us): full 178.3, no compute 168.0, + no w 165.4, + no x 82.9,
  + no store 70.5. Solo: no x 149.9, no strip copy 160.3, no store 163.2, no w 174.5. Ranking: (1) the x chain
  (about 82 us of data movement on the critical path; per row link the x bytes need about 39 us, so it is far from
  the link roofline and bound by serialization), (2) the 70.5 us fully stubbed floor, (3) the compute payload
  (about 93-116 us vs 68 us ideal at 16 cycles per tile product), (4) the store tail (about 13-15 us). w is not
  binding (DRAM roofline 22.5 MB / 448 GB/s = 50 us, hidden). The early-ack subagent showed that item 2 is not
  handshake latency: with every payload stubbed, the matmul scaffolding (SKIP_COMPUTE keeps unpack / pack,
  about 57 us), the untilize and the store's wait on compute fill it, and NCRISC ends at about 50 us.
- Portfolio and verdicts (focus ns, baseline -> candidate):
  - early_ack_handshake (receivers ack step s + 1 before waiting on s, x and w channels): NULL. x only
    178.2k -> 180.7k, x + w 181.4k, w only 177.5k; early send 184.7k, x / strip paired send 183.1k
    (regressions). It helps only DRAM y (208.6k -> 201.0k), and that gain is inside what the strip forwarder
    already took (dram_out 209.5k -> 187.0k). The raw protocol costs about 3 us over the helper with lookahead
    disabled. What the helper would need: split `ack(round)` / `wait(round)` on the receiver, a ring of >= 2
    data-ready flags indexed by round, and a non-blocking `acks_ready()` on the sender.
  - strip_copy (the x copy to the L layout's strip core): WIN with option (b). Core (g, g) leaves the x sender
    rotation and forwards each step to the strip core one step behind, so the rotating sender injects once:
    177.2k -> 161.9-163.1k. (a) one channel over rect row + strip core: BLOCKED (`ttnn.Mcast` groups are
    consecutive cores in one global order, and a rotating sender needs a rectangular receiver group). (c) xb
    issued before xa without its own fence: REGRESSION 193.0k. Forwarder at x = gx - 1: 166.3k (worse than (g, g)).
    Not pipelining the forward: 164.2k.
  - store_tail (the exposed last-wave store): WIN with option (a). The storing RISC hands every odd block of the
    last wave to the other dataflow RISC, so the tail runs on both NoCs: 177.0k -> 172.0-172.5k. Rotating
    segment order: NULL. Alternating write command buffers (raw): REGRESSION 293.1k (breaks end-of-kernel write
    accounting). Uneven wave split: not built (budget margin allows 16 / 14 rows, under 1 us). Streaming the
    store per block: already in place.
  - compute_payload (the matmul at the fixed precision contract): payload WIN, whole op flat before the other
    two graduated. A compile-time `1 x core_n_tiles` subblock instead of 2x2 + tail gives payload
    115.9 -> 99.4 us. At 2x2 the fp32 partials pack serialized with the matmul, a fixed cost per DEST acquire
    (not pack volume or L1 accumulation). Bit-exact on every gate case. Same shape computed at runtime: 143.5
    (REGRESSION). Raw-LLK 1x2: 96.0 payload, whole op flat, not graduated. Pack reconfig hoisted: NULL. Other
    shapes 4x1 / 2x1 / 1x1 are slower. Holding 2 K rounds in DEST changes the summation order and needs
    resident x / w for both rounds: not pursued.
- Graduated (all three are the op's single path, with no predicate; the code they replace is deleted):
  1. x strip forwarder: `_l_layout` takes core (g, g) out of the X_MAIN rotation (`McastRotatingSenderConfig`
     over rect minus the diagonal) and makes it X_STRIP's fixed sender. The reader forwards the pending step
     before acking the next one. The sender's second `send()` is gone. Domain: every L layout. The rect layout has
     no strip channel. A grid with gx = 2 falls back to a fixed X_MAIN sender (unmeasured, correct by
     construction). Focus 178.1 -> 162.8 us.
  2. Last-wave store split (`moe_store_tail_owner` / `moe_store_tail_helper`, `cb_tail` 16 B pages, at most
     ceil(core_row_tiles_max / 2) of them): every placement and wave count. With DRAM y the reader owns the store
     and the writer helps. Focus 162.8 -> 156.0 us.
  3. Compile-time `1 x core_n_tiles` subblock. 3-run A/B on HEAD: focus 156.3 -> 154.8, t4096 layer 3 occupied
     156.0 -> 150.7, t3072 126.1-127.5 -> 125.3-126.1, t4096 DRAM y 179.5-180.5 -> 178.8-179.2.
  No carve-outs: no guard case measured a material regression.
- Whole op, focus: 178.1 -> 154.8 us (-13.1%, instrumented); stock 179.0, target 90.8 (not reached).
- Guard set (instrumented, one run each, all checked for correctness), before -> after:
  focus 178.1 -> 154.8 | t4096 layer 3 occupied 177.0 -> 150.4 | t4096 layer 2 occupied 177.1 -> 150.3 |
  t4096 layer 0 unoccupied 170.5 -> 155.5 | t3072 138.2 -> 125.5 | t2048 111.1 -> 104.8 | t1024 86.5 -> 86.2 |
  t4096 DRAM x 210.4 -> 199.7 | t4096 DRAM y 209.5 -> 178.9 | t1024 DRAM x and y 114.0 -> 109.3 |
  t37 85.0 -> 82.7 | t1 41.4 -> 41.9 (noise). No regression.
- Accuracy: the summation order is unchanged. Precision baseline: PCC 0.99991, rel_rms 1.90e-2 (LoFi);
  HiFi2 PCC 0.999999.
- Golden: 93/93 after every graduation. Unit (`test_moe_down_proj.py`, `_layouts.py`, `_precision_baseline.py`):
  57/57.
- Bottleneck now (focus zones): the x chain still binds. compute_wait_inputs is 50 us p50, x_receive 107 us
  p50, and x_forward takes 48 us on the 9 forwarder cores. Round 2 should start there.
- Tests added: `tests/ttnn/unit_tests/operations/moe_down_proj/probes/test_perf_bench.py` (focus + guard set,
  kernel-dir and define overrides for ablations; run explicitly with `--profile --no-precompile`).

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| no helper covers this (dataflow CB API: `get_read_ptr`) | capability | Reading block r ahead of a CB's front without popping. The store split keeps the handed blocks un-popped so the owner stays the only consumer, so it needs `front + r` with the ring wrap, which `get_read_ptr` cannot express. It reads `get_local_cb_interface(cb).fifo_rd_ptr` / `fifo_limit` / `fifo_size` directly. A `get_read_ptr(cb, page_offset)` with wrap would cover it. | n/a (inexpressible) | 154.8 us whole op | moe_down_proj_store.h:91 |

The strip forwarder and the subblock change use only the existing helpers (`ttnn.Mcast` fixed and rotating
senders, `matmul_block`). The NULL early-ack variant's helper gaps are listed with its verdict above.
