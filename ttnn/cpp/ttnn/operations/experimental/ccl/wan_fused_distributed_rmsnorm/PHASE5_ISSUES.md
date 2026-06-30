# Phase 5 wiring — issues log (hangs / determinism)

## FINAL STATUS — ALL GREEN
- DistributedLayerNorm (fused Welford LN, WAN_USE_FUSED_LAYERNORM): module test 5/5 —
  wan_tp2/tp4/tp8 line, ltx_tp4 line, wan_tp4 RING. fused pcc >= composite (both ~100%
  vs fp32-PyTorch), det=True across 3 launches.
- DistributedRMSNorm (fused, WAN_USE_FUSED_RMSNORM): module test 5/5 same configs —
  fused 100.0006% >= composite 99.9996%, det=True.
- Issues found: 1 (ISSUE 1, a test-only two-CCLManager ring hang). RESOLVED. No
  determinism issues (every config bit-exact across launches). No perf regression
  (fused paths are the recip-LUT/single-program wins; the manager change is test-only).


Tracking issues found while wiring the fused Welford LayerNorm (and validating the
fused RMSNorm) into `models/tt_dit/layers/normalization.py` and testing on the 4x8
galaxy: submesh lines, 4x8 line, 4x8 ring. Each issue: symptom → root cause → fix.

## Test matrix
`test_distributed_rmsnorm_fused.py::test_layernorm_module_corr` — DistributedLayerNorm
(adaLN, batch=1) composite vs fused vs fp32-PyTorch, asserting fused pcc >= composite.
Configs: wan_tp2_line, wan_tp4_line, ltx_tp4_line (1xN submesh, Linear); wan_tp4_ring
(4x8, Ring, tp_axis=0); wan_tp8_line (4x8 full, Linear, tp_axis=1).

---

## ISSUE 1 — `wan_tp4_ring` (4x8 Ring) module test HANGS — RESOLVED (test-only)
**Root cause:** test artifact, NOT an op bug. fused-only on ring PASSES and composite-only
on ring PASSES; only composite-THEN-fused in one process hung, and only on Ring (Linear
was fine). The test built a SECOND `CCLManager` per method (a second set of fabric
global-semaphores) and ran the composite all-gather then the fused fabric-forwarder
all-gather back-to-back — two managers' fabric resources collide on a closed Ring.
**Fix:** use ONE `CCLManager` + ONE module for both methods (also matches production,
where a model holds one manager and selects composite OR fused via the env flag, never
both). All 5 configs then pass: wan_tp2/tp4/tp8 line, ltx_tp4 line, **wan_tp4 ring** —
fused pcc >= composite, det=True across 3 launches.
(historical detail below)
### original triage

**Symptom:** the 3 line/submesh configs pass (composite + fused both det=OK, fused pcc
>= composite). The 4th config `wan_tp4_ring` (tp_axis=0, Ring topology, full 4x8 mesh)
hangs in dispatch — pytest-timeout (300s) fires a callstack at
`dispatch_kernel_initializer.cpp:264`. No LNMOD result line is emitted for it.

**Context / leads:**
- The op-level `test_layernorm_corr` did NOT cover Ring (all its params were Linear),
  so this is the first fused-LN-on-ring exercise.
- The fused RMS `test_corr_det` DID cover `wan_tp4_ring` and passed (12/12), so the
  fabric ring + fused-RMS path itself works. Difference: LN gathers 2 stats (256 B
  sticks) vs RMS 1 (128 B), uses the new recip LUT + LN-sized stats buffer, AND the
  module test runs the COMPOSITE dit_layernorm on ring first (possibly itself untested
  on a 4x8 ring — the existing unit test only does small Linear submeshes).
- Memory: ring/full-mesh paths are fragile on this galaxy.

**Isolation (LNMOD_METHODS knob):**
- composite-only on `wan_tp4_ring`: **PASSES** (pcc 100.0006%, det=True). So the composite
  dit_layernorm + all_gather on the 4x8 ring is fine.
- => the hang is the **FUSED LN path on Ring**. The fused RMS works on the identical ring
  config (test_corr_det wan_tp4_ring), and fused LN works on the LINE configs (wan_tp2/tp4
  line, both AG/use_mux). So it's RING + LayerNorm specific. LN-vs-RMS deltas on that path:
  256 B sticks (2 stats mean+var) vs 128 B (1), the recip-LUT reader read, the 2x stats
  buffer, the LN compute (welford + merge). Linear-vs-Ring delta: fabric wrap-link routing.

**Next:** --dev watcher run to get the stuck-core waypoints (forwarder vs worker vs compute).
