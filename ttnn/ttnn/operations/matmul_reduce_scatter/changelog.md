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
