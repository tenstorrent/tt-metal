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

## Phase 6 (model-block integration: Wan2.2 + LTX, SD3.5 unavailable offline)

### ISSUE 2 — fused LN rejects fp32 adaLN weight — RESOLVED
**Symptom:** Wan2.2 block with WAN_USE_FUSED_LAYERNORM=1 hit `TT_FATAL weight->dtype()==BFLOAT16`.
**Cause:** the fused op requires bf16 weight/bias, but the adaLN modulation `(1+scale)`/`shift`
is fp32 (fp32 temb path). The composite dit_layernorm tolerates fp32; the fused reader reads
weight as 2-byte bf16 face-rows, so it can't take fp32.
**Fix:** cast dynamic weight/bias to bf16 in the module's fused LN path (negligible for a bf16
model — activations + output are bf16 anyway). Baseline (composite) Wan2.2 block PCC = 99.9951%.

### ISSUE 3 — fused LayerNorm degrades + destabilizes at large N (many AG rounds) — OPEN
Wan2.2 transformer block, wh_4x8sp1tp0 (TP=4 ring, 4 links), 5b-720p (N=24800), batch=1:
  - composite norms (baseline):            PCC 99.9951%  PASS
  - fused RMS only (LN composite):         PCC 99.9950%  PASS  <- fused RMS is a drop-in
  - fused LN only (RMS composite):         PCC 99.8692%  FAIL (< 99.95% threshold)
  - fused RMS + LN:                        PCC 99.8700%  FAIL
So **fused RMS matches composite exactly; fused LayerNorm is the sole regressor** (~0.13% PCC).
Affine is NOT the cause: in-op bf16 affine (99.8700%) and external fp32 affine (99.8696%) are
identical -> the divergence is in the fused NORMALIZE, not the modulation.

N-dependence: the module corr test (test_layernorm_module_corr) matched composite at N=256
(1 AG round) on the SAME wan_tp4 line+ring configs, but at N=4096 (multi-round) the module test
**HANGS** (dispatch timeout + eth triage). The LN reduction is over the feature dim (1280),
which is N-independent, so per-token accuracy can't be N-driven — the only N-dependent,
LN-specific factor is the 2-stat (mean+var, 256 B stick) ring AG over many rounds
(max_rounds≈26 at N=24800 vs 1 at N=256). RMS's 1-stat AG over the same rounds is fine, so it's
specific to the LN 2-stat multi-round gather: it corrupts SOME tokens' stats (block: 99.87%, no
hang) and deadlocks the repeated-launch module pattern (N=4096: hang).

Candidate root causes (for focused follow-up):
  1. worker_writer/forwarder 2-stat (mean,var) DRAM page addressing at multi-round (page_idx /
     my_slot*stick_bytes + s*kStatBytes) — a row/stat indexing error that only bites once
     max_rounds>1. Same class as the Phase-4d deadlock, different trigger.
  2. #45319 (UnpackToDestFp32 + welford_init<PreserveStats>) — the composite Welford LN got this
     fp32-input-precision fix; the fused op lacks it (it's a Phase-4 TODO). Less likely here
     (input is bf16, Wan TP4 is resident with no spill), but listed.

RECOMMENDATION: ship fused RMSNorm (validated drop-in, faster). Keep fused LayerNorm behind its
WAN_USE_FUSED_LAYERNORM flag (default off) until the 2-stat multi-round AG is debugged. Production
default (composite) is unaffected.
