# Changelog: matmul_reduce_scatter

## Phase 0 — Core Implementation
- **Date**: 2026-10-06
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). One
  `ttnn.generic_op` per call: per chip a 2D-multicast matmul on the compute rectangle walks the G scatter blocks in
  transport order (regimes R1 resident invariant operand / R2 streamed), parks each block in L1, and the transport
  row (ports + finals, `4·num_links` cores) runs the line reduce-scatter over Fabric with fp32 adds.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], weight_dtype=[bfloat16, bfloat8_b], layout=[TILE],
  alignment=[tile_aligned], cluster_axis=[0, 1], scatter_dim=[-1, -2], topology=[Linear], num_links=[1, 2];
  EXCLUSIONS=[].
- **Accuracy achieved** (`test_matmul_reduce_scatter_precision_baseline.py`, 6 configs, worst of 8 chips):
  fp32 DEST: PCC ≥ 0.99999, rel-RMS 0.0035–0.0054, max_abs ≤ 0.10. FOCUS at production precision (HiFi2,
  `fp32_dest_acc_en=False`): PCC 0.99986, rel-RMS 0.024 (gate 0.055; unfused production 0.036), max_abs 0.29. It
  carries a K-dependent bf16-accumulation bias (slope 1.016 at K=2048), triaged as rounding and not a structural bug.
- **Perf** (device kernel duration, steady state, max over chips): FOCUS 209.5 us (unfused 394, target 118),
  GLM 241.9 us (409 / 198), MiMo 439.3 us (385 / 215).
- **Golden suite at Phase 0**: 375 supported_pass / 1041 cells, 660 xfail_expected (= TARGET − SUPPORTED),
  0 supported_fail, 0 xpass_drift, 0 xfail_wrong_mode; 6/6 registry-free regression tests pass (per
  `verifier_report.json`).
- **Issues encountered** (verifier fixes):
  - Persistent per-plan L1 `handoff_l1` shard leaked across plan keys and clashed with later plans' CB regions
    (~600 golden failures). Now allocated per call; generic_op re-points the CB on a cache hit.
  - Per-plan global semaphores leaked the same way. Now one set per `(mesh, cluster_axis, num_links)` over the whole
    worker grid.
  - L1 CB budget used `get_max_worker_l1_unreserved_size()` (includes the kernel-config ring) and over-budgeted by
    70.6 KB, so R1 `scatter_dim=-2` plans overflowed into the hand-off shard. Now uses the allocator's L1 bank size.
  - DRY: accumulator dtype / tile size and DEST capacity each defined once.
- **Tests added**: `test_matmul_reduce_scatter.py` (acceptance, planner), `test_matmul_reduce_scatter_debug.py`,
  `test_matmul_reduce_scatter_perf.py` (implementer), `test_matmul_reduce_scatter_precision_baseline.py` (verifier).

## Refinement 1 — Ring topology (both ring directions)
- Date: 2026-10-07
- What was done: `Topology.Ring` added to `SUPPORTED["topology"]` (design regime R3). Each block's reduction chain
  is a line centred on its owner: fwd list `p+hf .. p+1`, bwd list `p-hb .. p-1` (mod G), `hf = ceil((G-1)/2)`,
  `hb = G-1-hf`, plus the wrap hop. Both ring directions carry data; the busiest direction carries `hf` blocks
  instead of `G-1`.
  - Host: `_groups(..., ring)` emits wrap neighbours; `_links` then covers the wrap hop; `_schedule_mmrs(p, G, ring)`
    returns per-entry `has_upstream` flags; landing slot is `G` for the receiver's own block arriving backward;
    consumer kind comes from list membership; finals' `has_fa/has_fb` derive from the neighbours' lists; the plan
    cache key includes the topology; validation #5 (`_check_ring_fabric`): the fabric config must wrap
    `cluster_axis` (TORUS_X: 1, TORUS_Y: 0, TORUS_XY / 1D_RING: both), else `ValueError`. A 2-device ring runs the
    line schedule.
  - Kernels: the port reader takes a 4th entry field `has_upstream`. An upstream-less entry pushes only the gather;
    arrival CB reserve, push and position are tracked separately. The add kernel copies the upstream-less entries
    through (`CopyTile` → `PackTile`, so `cb_xport_sum` keeps one producer), then adds the relay entries.
  - Two hangs found and fixed in the shared transport (both latent at Phase 0 and reachable with Linear too):
    1. The sender deferred arrival increments across block boundaries (every 8 segments of the whole relay stream).
       In a ring nothing bottoms out that wait, so on short entries (`full < 8`) it deadlocked. Now increments
       happen every 8 segments of a block plus on each block's last segment, and the reader expects
       `ceil(full/8)` per upstream block.
    2. The transport add was 1D-blocked with `add_block=4` over 7-tile segments (production payload), so it held a
       segment's tail until the next segment arrived. That lookahead closed a wait cycle through the compute
       hand-off cursor (Linear G=8, 256×256 hung on the Phase-0 code too). The add now walks
       `grid(segments, seg_tiles)` row-blocked, with `add_block = min(4, seg_tiles)`.
  - Reused: port/final/sender/reader kernels, CBs, semaphores, scratch layout. Added: no new kernel, CB or
    descriptor branch.
- Accuracy achieved: ring-mock FOCUS 640×2048×7168 (`scatter_dim=-1`) worst-chip PCC 0.99997; 256×256
  (`scatter_dim=-2`) PCC 0.99999; MiMo 2048×2048×4096 (`scatter_dim=-2`) PCC 0.99993. Golden tolerances hold on
  every Ring cell. Ring output was bit-exact across 4 repeated calls interleaved with Linear calls on the same mesh
  (shared global semaphores), per probe_031.
- Perf: Ring is functional only on this board (the ring mock does not measure perf). Linear, FABRIC_2D, steady
  state, max over chips: FOCUS 210.8–212.8 µs vs 210.1–211.9 µs before; MiMo 438–442 µs vs 440–444 µs before.
  Both are within noise, so no regression from the add-walk change.
- Golden test progress: `test_ring_mock.py` 72/72 (36 Ring cells, previously xfail; 36 Linear cells under the
  wrapping configs). `test_fabric_configs.py` + `test_regression.py` 30/30. A `test_golden.py` slice of 13 Linear
  cells passes. The real per-axis Ring cells of `test_golden.py` are pruned on this 2×4 LoudBox (no wrap links);
  they need a Galaxy.
- Issues encountered: the two transport deadlocks above (triaged from hang reports: port readers stuck at the
  arrival wait, compute NCRISCs at the hand-off drain).
- Tests added: none in the unit directory (the multi-device ring needs a wrapped fabric config per mesh, which the
  golden ring-mock fixture provides). Probes 026–031 cover repeated-call determinism and the G=8 tiny-shape hang.

## Refinement 2 — Numerical configurability: bfloat8_b activations + bfloat4_b weights
- Date: 2026-10-07
- What was done: `ttnn.bfloat8_b` added to `SUPPORTED["dtype"]` and `ttnn.bfloat4_b` to `SUPPORTED["weight_dtype"]`.
  It was a SUPPORTED-only change with no kernel or descriptor edits, because the existing paths already followed
  the input dtypes:
  `cb_act_operand` / `cb_weight_operand` take `a.dtype` / `w.dtype` as page format with `_tile_bytes(dtype)` page
  sizes (Bfp8_b 1088 B, Bfp4_b 576 B); the injectors read with the `a_tile_bytes` / `w_tile_bytes` CT args; the
  compute kernel is `matmul_block` behind `compute_kernel_hw_startup(act, weight, handoff)` (per-operand unpack
  formats, no hard-coded formats). Output stays bf16, transport stays bf16 on the wire with fp32 adds, and
  `cb_partial_accum` still follows `fp32_dest_acc_en`. `compute_kernel_config` is unchanged (HiFi2 / fp32 DEST
  default).
  - Residency predicate: confirmed it exploits the smaller tiles (host-only planner run, approx. BH compute grid,
    fp32 DEST). 2048x4096x4096 `scatter_dim=-1`: A bf16/W bf8 → R2 k=8; A bf8 → R1 k=32; A bf8/W bf4 → R1 k=64.
    `scatter_dim=-2`: W bf4 flips R2 → R1 (k=32, k=64 with A bf8). 640x8448x7168 `scatter_dim=-1`: R1 k 12 → 24
    (A bf8) → 44 (A bf8 + W bf4).
  - Reused: every CB, kernel and planner path. Added: two SUPPORTED entries, plus a precision-baseline axis.
- Accuracy achieved (`test_matmul_reduce_scatter_precision_baseline.py`, worst of 8 chips, reference on the
  device-quantized operands, HiFi2):
  - FOCUS 640x2048x7168, A bf8 / W bf8, fp32 DEST: PCC 0.999996, rel-RMS 0.0034
  - FOCUS, A bf16 / W bf4, fp32 DEST: PCC 0.999991, rel-RMS 0.0054
  - FOCUS, A bf8 / W bf4, bf16 DEST: PCC 0.99981, rel-RMS 0.039, median ratio 1.024. This is the same
    bf16-accumulation bias as the Phase-0 `focus_bf16acc` case (rel-RMS 0.024, 1.016).
  - MiMo rows 2048x2048x4096, A bf8 / W bf4: PCC 0.999996, rel-RMS 0.0035
  - large K 640x8448x7168 (axis 0), A bf8 / W bf4: PCC 0.999997, rel-RMS 0.0024
  - All golden tolerances (bfloat4_b: PCC 0.99, rel-RMS 0.08) hold with margin.
- Golden test progress: targeted `test_golden.py` slices, 300/300 pass. Shapes: 640x2048x7168 (2-D and 4-D A),
  640x8448x7168, 640x1536x32, 640x1536x128, 1x1x2048x4096x4096, 1x2048x2048x4096 (3-D A), 640x4608x7168. Each slice
  covered every new-dtype cell (both cluster axes, both scatter dims, 1/2 links), and a few prior cells were
  caught by the `-k` filter. No EXCLUSIONS needed (every TARGET shape is tile-aligned).
- Issues encountered: None.
- Tests added: `test_matmul_reduce_scatter_precision_baseline.py` gained an activation-dtype axis and 5
  bfloat8_b / bfloat4_b cases (11/11 pass).

## Refinement 3 — Speed up the PERF FOCUS case: transport send rate
- Date: 2026-10-07
- What was done: lamp L5 transport placement, built per chip.
  - `_plan_placement(mesh, L, groups, links)` puts each port (direction, link) on the transport-row core with the
    fewest NoC1 hops (the sender's NoC) to that connection's Ethernet core. Finals take the remaining row cores.
  - The Ethernet channel is the first word of `ttnn.setup_fabric_connection(...)`. The physical Ethernet and worker-column
    maps come from the cluster descriptor and the SoC yaml (fabric_all_gather's host helpers, without its probe
    dispatch). The op still makes one dispatch per call.
  - `Placement` port/final lists are now keyed by mesh coord. Each sender addresses its peer chip's cores (ready
    fence → peer's opposite port, relay arrivals → peer's same-direction port, own-block arrivals → peer's finals).
    The compute cores' ack consumers come from their own chip's placement.
  - Global semaphore sets are now keyed by `(mesh, cluster_axis, num_links, ring)`. A line-end chip's ports move
    with the wrap hop, so Linear and Ring no longer share a set (still bounded: 8 per mesh).
  - `XPORT_PLACEMENT` (env `MMRS_XPORT_PLACEMENT`) = "eth" (default) / "simple" (old fixed layout) is the A/B knob.
    If the Ethernet layout is unknown (emulator, missing descriptor entry), placement falls back to "simple".
  - noc_placement checked: port reader (NCRISC, NoC0 reads), sender and final writer (BRISC, NoC1 writes) already
    hold, so no change.
  - split_reader (L4) not built: profiler zones show the port senders are fabric-slot bound, not issue-bound.
  - Perf harness: opens the mesh with the production router config (14 KiB + 64 B payload), per feature_spec's
    operator note; `MMRS_PAYLOAD` overrides it. Added `smallk` / `r2` cases and an `MMRS_LINKS` override.
  - Reused: every kernel, CB and descriptor path. Added: placement planner + peer-placement lookup, with no kernel
    change.
- Perf (device kernel ns, steady state, max over 8 chips, 2×4 LoudBox FABRIC_2D, 14 KiB packets, before → after):
  - num_links=2:

    | Case | Before | After |
    |---|---|---|
    | FOCUS | 208.5 us | 166.6 us |
    | GLM | 278 us | 219 us |
    | MiMo | 394 us | 357 us |
    | small-K 640×512×7168 | 151 us | 133 us |
    | R2 2048×4096×4096 | 444 us | ~440 us (compute-bound) |

  - num_links=1: FOCUS 252 → 240 us; small-K 223 → 215 us.
  - At the old 4352 B payload: FOCUS 210 → 208 us max, 200 → 191 us median; MiMo 444 → 432 us.
  - Roofline: end-chip steady-state send ≈ 36 GB/s per link, which is the box's best-seen link rate (feature_spec:
    36.4). The link floor for FOCUS at 14 KiB is 3.44 MB / 36.4 GB/s ≈ 95 us + ~17 us fill, i.e. ≈ 112 us. Achieved
    167 us max / 143 us median.
  - The remaining gap: first-block starvation (~36 us; the matmul is ~28–34 us per scatter block on ~70 cores),
    relay-chain latency, and cross-chip launch skew at the ready fence. Next levers: Refinement 4 (pipeline fill)
    and matmul core utilization.
- Accuracy achieved: unchanged kernels, so numerics are unchanged. FOCUS loose case passes its rel-RMS 0.055 gate.
  Probe 033: worst-chip PCC 0.999993 (FOCUS), 0.999991 (axis 0, 256×512×256, 1 link), 0.999976 (MiMo).
- Golden test progress: test_fabric_configs + test_regression + test_ring_mock 102/102. test_golden slice
  (3 loose cases + 640×512×7168 + 640×1536×32) 75/75. Unit acceptance 53 passed, 1 skipped.
- Issues encountered: the unit perf harness measured at the router-default 4352 B payload, where FOCUS is
  packet-overhead bound and placement shows only ~5% (median). Under the production payload the win is 20%.
- Tests added: perf harness cases `smallk`, `r2`; env knobs `MMRS_PAYLOAD`, `MMRS_LINKS`. Probes 032 (Ethernet
  channel ↔ core map) and 033 (placement dump + PCC).

## Refinement 4 — Speed up the PERF FOCUS case: pipeline fill (sub-block sends)
- Date: 2026-10-07
- What was done:
  - **Fill measured first** (verifier gate): a zone on the port senders from kernel start to the first sendable
    segment. End chips send their first packet at ~35 us of a ~135 us kernel. Interior chips start at 45–68 us of
    148–166 us; that is upstream data plus ~20 us of cross-chip launch skew. The fill is not hidden, so headroom
    exists.
  - **R4 waves built** (`waves_per_block`, live knob, parked at 1). Each wave is one compute unit with its own grid
    factorization, hand-off slot, ready/ack round, and transport entry.
    - Waves split the **scatter axis**: rows for `-2`, whole-segment column ranges for `-1`. Each wave streams a
      disjoint slice and replays the resident one, so DRAM traffic is unchanged.
    - The design's row split for `-1` was built first and measured: wave 0 stays W-stream bound (20 of 33 us), since
      every wave re-reads the block's W. Replaced.
    - Transport entries carry a window (rows × segment columns, `Window` / `next_wave_seg`). The sender counts
      final-core increments per wave and finds the final half from `seg / L`.
    - The final writer walks the same windows. `handoff_slots = HANDOFF_DEPTH·waves` keeps hand-off slack at
      `HANDOFF_DEPTH` blocks.
    - The plan cache keys on the blocking knobs.
  - **Measured, not shipped on by default** (FOCUS, max of chips):
    - Column waves: 167 → 192 us with 2 slots, 183 us with `2·waves` slots. End-chip fill drops 36 → 25 us, but a
      20×28 wave factorizes to 2×4 tiles/core on the 11×9 grid, i.e. 16 vs 14 tiles/core per block. End chips turn
      compute-bound, because their own block, computed last, gates the finals.
    - MiMo row waves: 361 → 406 us.
    - The planner takes waves only with no per-core work growth (`WAVES_WORK_SLACK = 0`), and `WAVES_MAX = 1`.
      `MMRS_WAVES` pins them for A/B.
  - **Shipped: what binds the fill.**
    - Zones showed the line injectors (W on BRISC, A on NCRISC) are **issue-bound**: 3.1 us to issue a 112-tile
      K-block, then a 1.5 us barrier and a 1.8 us multicast, about compute-balanced at ~27 us/block.
    - `read_pages_strided` (injector header): pages `num_banks` apart share a bank, so a run is one stateful
      `noc_async_read_one_packet_set_state` plus `_with_state` reads. W reads column by column (rows `Nt` pages
      apart, one bank when `Nt % banks == 0`); A reads by residue mod banks. Issue time 3.1 → 1.8 us per K-block;
      FOCUS end chips 135 → 131–134 us.
    - `K_BLOCKS_MIN = 4` (lamp L2, pipeline fill at K-block granularity): every pass has ≥ 4 K-blocks. MiMo's
      K-block had been all of K, so block 0 waited for the whole 835 KB resident W slice (~115 us of fill).
  - **Parked**: injector read-ahead (`INJECT_READ_AHEAD`, issue K-block s+1 before multicasting s). It cost +3 us on
    FOCUS: it delays the first multicast and never engages in steady state (no free slot).
  - Shared injector walk (`matmul_reduce_scatter_injector.hpp`) for the A and W line kernels. Per-unit A row / W
    column origins and `fresh` flags come from the host (replaces the CT `a_resident` / `w_resident`).
  - Reused: every CB, kernel and descriptor branch. Added: two small kernel headers (injector walk, wave-window
    walk), no new kernel or CB.
- Perf (device kernel duration, steady state, max over 8 chips, 2×4 LoudBox, FABRIC_2D, 14 KiB packets, same-session
  before → after):

  | Case | Before | After |
  |---|---|---|
  | FOCUS (median of 3 runs) | 168.6 us | 167.4 us |
  | FOCUS end chips | 135 us | 131–134 us |
  | GLM | 219.0 us | 204.6 us |
  | MiMo | 365.5 us | 339.6 us |
  | small-K | 130.5 us | 131.9 us |
  | R2 | 435 / 449 us | 449 / 445 us (noise; end chips 394–416 → 384–404) |
  | `num_links=1` FOCUS | 241.4 us | 240.1 us |
  | `num_links=1` small-K | 215.4 us | 215.3 us |

  FOCUS max-of-chips is an interior chip: end-chip fill (~33 us) + the link-bound send of 3 blocks (~96 us) + relay
  and final tail + ~20 us of launch skew.
- Accuracy achieved (precision baseline, worst of 8 chips):
  - FOCUS production (HiFi2, bf16 DEST): PCC 0.99986, rel-RMS 0.0241 (gate 0.055).
  - fp32 DEST: PCC ≥ 0.99999, rel-RMS 0.0034–0.0054.
  - MiMo: PCC 0.999991, rel-RMS 0.0054.
  - A bf8 / W bf4 / bf16 DEST: rel-RMS 0.026.
- Golden test progress:
  - `test_ring_mock` + `test_fabric_configs` + `test_regression`: 102/102.
  - `test_golden` slice (3 loose cases incl. the FOCUS rel-RMS gate, 640×512×7168, 640×1536×32, every
    2048×2048×4096 cell): 171/171.
  - Unit: acceptance + debug 56 passed / 1 skipped (default and `MMRS_WAVES=2`); precision baseline 11/11.
- Issues encountered: the first column-window version computed a window's last segment column with a floor. When a
  block row ends in a ragged segment (G=8 MiMo-rows `-1`: 16 tiles = 7 + 7 + 2), the unwaved window dropped that
  segment (ring-mock PCC 0.93). Fixed with a ceiling, and pinned in the new test.
- Tests added: `test_matmul_reduce_scatter_waves.py`. It pins waves=2 for column waves (`-1`) and row waves (`-2`),
  asserts the plan took them, and covers the unwaved ragged-segment window (3/3).

## Perf 1 — perf tournament round 1: transport add throughput (raw-LLK fp32-DEST add walk)

All numbers: device kernel duration, steady-state calls (calls 3–5 of a profiled run), max over the 8 chips of a 2×4
Blackhole LoudBox, FABRIC_2D, 14 KiB + 64 B packets, FOCUS = `640×2048×7168`, A bf16 / W bf8b, HiFi2,
`fp32_dest_acc_en=False`, `cluster_axis=1`, `scatter_dim=-1`, Linear, 2 links. The focus config is in SUPPORTED, so no
generality gap.

### Instrumentation (permanent)
- Restored `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`, which was missing on this branch.
  `MaybeDeviceZoneScope` is opt-in: it needs `--profile` **and** the `KERNEL_PERF_ZONES` define.
- Host switches:
  - `MMRS_PERF_ZONES=1` adds `KERNEL_PERF_ZONES` to every kernel.
  - `MMRS_ABLATE=MATMUL,OPERANDS,XREADS,LINK` stubs those stages' payloads and keeps their synchronization. Perf
    only; the results are wrong.
- Zones per kernel:
  - Compute: `compute_matmul_block`, one per block.
  - Injectors and receivers, per K-block: `inj_reserve`, `inj_read`, `inj_mcast`, `recv_reserve`, `recv_mcast`.
  - Compute NCRISC: `handoff_drain`.
  - Transport reader, per entry: `xr_wait_ready`, `xr_entry`.
  - Transport add: `xadd_copy`, `xadd_add`.
  - Port sender: `snd_fence`, `snd_entry` (per entry), `snd_tail`, `snd_drain_in`.
  - Final writer: `fw_store`, `fw_rearm`.
- Budget: FOCUS uses at most ~100 markers per RISC. That is under the 250 cap, and the zones cover each kernel's
  span.

### Measured breakdown (before)
- **Whole op:** 166.9 µs median, same-session A/B (170–174 µs in the first run of the day). Per chip:
  171 / 131 / 150 / 133 / 161 / 131 / 150 / 132. The interior chips are the long ones.
- **Cumulative ablation peel (max over chips):**

  | Payloads stubbed | Max over chips |
  |---|---|
  | none | 170 µs |
  | LINK | 164 µs |
  | LINK, XREADS | 140 µs |
  | LINK, XREADS, MATMUL | 101 µs |
  | all | 81 µs |
  | MATMUL only | 128 µs |
  | MATMUL, OPERANDS | 119.5 µs |

  Compute side and transport side are **balanced**: removing either alone leaves the other holding the wall. The
  links themselves are not binding (removing them saves 6 µs).
- **Zones, slowest chip (dev 0, an interior chip):**
  - The matmul runs ~25.6 µs per block when unstalled.
  - The W line injector is serial: about 3.6 µs read plus 1.9 µs multicast per K-block.
  - Block 3, the own block, is stalled about 30 µs in pack. With `handoff_depth = 2` it waits for block 0's
    hand-off slot. A relay port holds that slot for ~73 µs while it waits for upstream arrivals.
  - The finals start when the own block finishes (135 µs) and take ~27 µs.
- **All-stubbed zones:** the relay ports are **add-bound**. One port adds its 280 tiles per block in 21–28 µs. The
  link would carry those 280 bf16 tiles in about 16 µs at 36 GB/s. The 3-input final add takes ~33 µs.
- **Ranked by headroom:**
  1. Transport fp32-DEST add on one TRISC per port or final. It sits under the DM roof (the link needs 16 µs per
     280 tiles), so it is not at a ceiling.
  2. Hand-off back-pressure on interior chips (`handoff_depth = 2`).
  3. Serial read-then-multicast in the W injector.
  4. Links (not binding).

### Portfolio floated (round cap: 1 experiment)
- **A, selected:** faster transport add, both the 2-input relay add and the 3-input final add, at the same fp32-DEST
  precision.
- **B:** `handoff_depth` up to G when L1 fits. FOCUS would need +56 KB per compute core.
- **C:** a fusion: do the own-block final add on the compute grid. This would supersede A for the tail.
- **D:** overlap the W injector's read and multicast (read-ahead with `operand_depth = 3`).
- **E:** spread the relay add over idle transport-row cores.

B to E were not tested this round because of the cap. They are carried to round 2.

### Verdicts
- **A — `transport_add_throughput`: WIN.**
  - Isolated bench (single core, inputs resident in L1, the op's CB cadence), 280 tiles with 7-tile segments:

    | Variant | Baseline (`eltwise_chain`) | Raw LLK |
    |---|---|---|
    | 2-input | 9.83 µs | 6.14 µs |
    | 3-input | 27.96 µs | 10.69 µs |
    | Line-end 2-input | 10.95 µs | 6.16 µs |

  - Sweep over segment sizes 1–12, long runs, and ring copy-through followed by the add: faster everywhere, except the
    2-input add with 1-tile segments, which is flat. Flat is not an exception.
  - Domain: everywhere. There are no exceptions.
  - Precision: 2-input output is bit-identical to the baseline. 3-input output is *more* accurate: the running sum stays
    in fp32 DEST, where the helper moved it DEST→srcA. Mismatches against the RNE fp32 sum went from 15007 to 4238
    out of 57344 values.
  - Artifacts: `perf_experiments/transport_add_throughput/` (bench, all variants, `results.jsonl`).

### Graduated
- `matmul_reduce_scatter_xport_add.cpp` replaced. The new add walk is the op's **only** path.
  - The `eltwise_chain` BinaryFpu / DestReuseBinary / PackTile add code is deleted.
  - The ring copy-through still uses the helper.
  - No predicate, no carve-out.
- How the new walk works:
  - Each segment is waited for, added, pushed and popped on its own. That is the same no-lookahead contract as before.
  - It uses one hoisted ELWADD (`acc_to_dest` for 3 inputs). The 3-input form is `dest += partial + A`, then
    `dest += B + 0`, with srcB taken from the unpacker's zero filler.
  - Per DEST block: one unpack config context, ELWADD MOP runs back to back, and one pack MOP.
- The kernel-head comment states the measured bypass justification.
- The `xadd_copy` / `xadd_add` zones carry over to the new path.

### Whole op (same session, old kernel vs new, calls 3–5, max over chips)

| Case | Before | After |
|---|---|---|
| FOCUS | 166.5–167.2 (median 166.9) | 160.4–164.2 (median 164.0) — **−2.9 µs median, −1.7 %** |
| GLM `640×4096×6144` | 203.6–204.6 | 199.3–203.3 |
| MiMo `2048×2048×4096` rows | 342.6–346.5 | 334.5–338.3 (**−10 µs**) |
| small-K `640×512×7168` | 129.8–130.2 | 126.8–127.6 |
| R2 `2048×4096×4096` | 410–446 | 410–422 (noise band, as in Refinement 4) |

- The isolated stage win (final add −17 µs) only partly reaches the whole op.
- Zones after the change: the FOCUS final on dev 0 still takes ~28 µs (132.5 → 160.3). It is now bound by its reads
  and arrivals (gather plus two DRAM arrival reads per segment on NCRISC), not by the add.
- The relay port's entry is still gated by upstream arrivals.
- Block 3's hand-off stall is still there: compute ends at 128.5 µs where it would end at ~104 µs unstalled.
- That stall and the final's read path are what round 2 should target first (B, C).

### Correctness
- `eval/golden_tests/matmul_reduce_scatter/`: `test_golden` 939/939 (run in 23 node-id slices) and
  ring_mock + regression + fabric_configs 102/102.
- Unit tests: acceptance + precision baseline, 64 passed / 1 skipped. They run on the router's default 4352 B payload
  (2-tile segments), so they cover a second payload.
- FOCUS production precision is unchanged: PCC 0.99986, rel-RMS 0.0241 against the 0.055 gate.

### Guard set (one per kernel path × placement)
- Covered: FOCUS (`-1`, R1), GLM (`-1`, R1), MiMo (`-2`, fp32 DEST), small-K (link-bound), R2 (streamed operands).
- Ring and the line-end-only finals are covered by the golden suite.
- No case regressed beyond noise.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `eltwise_chain` `BinaryFpu<Add>` + `DestReuseBinary<Add, DEST_TO_SRCA>` + `PackTile` (3-input final add) | capability | no 3-operand add that accumulates in fp32 DEST without moving DEST→srcA (the reuse path re-runs `add_init` + `add_reuse_dest_init` every DEST block and truncates the running sum to srcA precision); no zero-srcB operand for `dest += B + 0`; `add_block` / `pack_block` are per-tile loops, with no one-context unpack, back-to-back MOP or single multi-tile pack MOP issue | 27956 | 10690 | `matmul_reduce_scatter_xport_add.cpp:145-186` |
| `eltwise_chain` `BinaryFpu<Add>` + `PackTile` (2-input relay / line-end add) | capability | per-tile unpack/math/pack issue inside a DEST block (one context and one MOP per tile instead of per block); CB handshakes per DEST block rather than per caller-defined unit (segment) | 9827 | 6142 | `matmul_reduce_scatter_xport_add.cpp:145-186` |

All ideas measured: 1 graduated, 0 null. 4 floated ideas (B to E) are untested because of the round cap. FOCUS is
faster by 2.9 µs median (−1.7 %) and MiMo by 10 µs, with no regression in the guard set.
