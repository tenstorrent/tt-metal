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
