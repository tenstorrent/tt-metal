# Changelog: point_to_point

## Phase 0 — Core Implementation
- **Date**: 2026-07-02
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier). Self-contained Python CCL op built on
  `ttnn.generic_op` + a two-entry `ttnn.MeshProgramDescriptor` (SEND program on the
  sender coord, RECEIVE program on the receiver coord), with newly-authored fabric
  dataflow kernels. Pure identity byte-copy over the Tenstorrent fabric: after the op the
  receiver device's output shard equals the sender device's input shard bit-for-bit; every
  other device's shard is unchanged. Fabric egress uses
  `dataflow_kernel_lib::ccl::FabricStreamSender`; the cross-device handshake uses one
  op-internal cached `GlobalSemaphore` (sender resets before its outgoing inc, receiver
  after its wait — the cache-reuse re-arm rule).
- **SUPPORTED at Phase 0**: dtype=[bfloat16, float32, bfloat8_b], layout=[TILE, ROW_MAJOR],
  topology=[Linear, Ring], alignment=[tile_aligned, non_tile_aligned]. EXCLUSIONS=[].
- **Accuracy achieved**: PCC=1.000000, max_abs_err=0.0, mean_abs_err=0.0, rel_rms_err=0.0
  (bit-exact identity copy, measured on 4 shapes × {bf16, f32} via
  `test_point_to_point_precision_baseline.py` — 8/8 bit-exact).
- **Golden suite at Phase 0**: curated multi-device sim run (topology `bh_8xP150_p2p`,
  `(2,4)`/FABRIC_1D) = 15 supported_pass / 16 xfail_expected / 3 invalid_skipped;
  all loud categories 0 (`supported_fail=0`, `xpass_drift=0`, `xfail_wrong_mode=0`,
  `supported_marked_xfail=0`) — per `verifier_report.json`. The only non-passing region is
  the integer-dtype xfail band (`dtype ∈ {uint16, int32, uint32}`), queued as Refinement 1.
  (The full TARGET×INPUTS matrix is 432 cells; `mesh_device` is function-scoped so the
  8-chip sim re-inits fabric per test — a curated selection covering every category was run
  instead of the full matrix.)
- **Issues encountered**:
  - Fixed a test bug: `test_point_to_point_precision_baseline.py` opened a `(1,2)` mesh,
    which mismatches the only sim topology that lists this op (`bh_8xP150_p2p`, fixed to
    `(2,4)`) and would hang fabric init. Changed to `(2,4)`. Now 8/8 green.
  - Applied a kernel cleanup: `point_to_point_receiver_reader.cpp` declared the packet
    scratch L1 address as `uint64_t`; it is a 32-bit L1 address (narrowed on every use).
    Changed to `uint32_t` to match the sender. Re-verified 7/7 across shapes — no
    behavioural change.
  - No drift fixes to SUPPORTED were needed (zero xpass_drift). INVALID audit passed
    (canonical `bf8b + ROW_MAJOR`, well-formed).
- **Tests added / updated**:
  - `test_point_to_point.py` (acceptance, pre-existing — 59 cases; smoke + program-cache +
    output_tensor + nonparticipating slices run green).
  - `test_point_to_point_precision_baseline.py` (mesh shape fixed to `(2,4)`; 8/8 bit-exact).
  - `test_point_to_point_extended.py` (NEW — multi-hop unicast `num_hops = 2, 3`; 4/4 green).
