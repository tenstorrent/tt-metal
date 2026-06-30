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

### ISSUE 3 — TRIAGE RESULTS (focused unit test: test_layernorm_module_corr + LNMOD_ROWS/MEAN0/LAUNCHES knobs)

**The "hang" is NOT a fused-op bug — it's a test artifact.** Isolated on wan_tp4 line AND ring:
  fused-only, 3 launches, N=4096, mean0:  PASS, det=True, pcc 99.9902%  (no hang, deterministic)
The 16-min hang was the BOTH-methods run (composite chain THEN fused op, back-to-back in one
process) — the ISSUE-1 class composite/fused-transition, not the fused op. Real models run ONE
method (env flag) and space norms apart, so the fused op does not hang in a model (the Wan block
ran). => No fused-op hang to fix.

**The correctness gap is a small, N-dependent, LN-specific (2-stat) AG error.** Single-launch
op-level pcc vs composite (vs fp32 torch), wan_tp4 line:
  N=256  mean4:  fused 100.0007 / comp 100.0006   gap ~0
  N=256  mean0:  fused 100.0004 / comp 100.0007   gap ~0.0003%   (single AG round)
  N=4096 mean0:  fused  99.9901 / comp  99.9977   gap ~0.008%    (multi AG round)
=> the gap GROWS with N (= AG rounds), is ~0 at single-round, and is mean-INDEPENDENT.
In the Wan block (N=24800, ~26 rounds) this op gap amplifies ~13x (vs composite's ~1.9x) to the
0.13% block drop — the fused error is STRUCTURED (per-row, correlated), not uniform precision,
which is why it amplifies through attention/FFN far more than composite's near-random error.

RULED OUT:
  - hang: fused op is deterministic + hang-free at multi-launch (line+ring).
  - dropped/aliased packets: the error is small (~0.01%), not garbage; single-launch is clean.
  - #45319 UnpackToDestFp32 (Welford input precision): that error is N-INDEPENDENT (LN reduces
    over the feature dim, 40 tiles/token, regardless of N) — contradicts the N-growing gap.

REMAINING SUSPECT: the 2-stat (mean+var) ring all-gather introduces a small structured per-row
error that grows with round count, that RMS's 1-stat AG does not. Addressing in worker_writer /
forwarder is structurally consistent on inspection (slot=i*256, stat s at +s*128, page per round);
the error is subtle (not a gross misindex). Pinning it needs device-level debug (DPRINT won't
compile in the forwarder, triage callstacks broken in-container -> code bisection), a focused
multi-iteration effort. Fused LN stays gated; fused RMS ships.

### ISSUE 3 — "are we doing different math?" — ANSWERED + partial fix
Compared the fused Welford LN vs the composite dit_layernorm step-by-step:
- mean/variance: IDENTICAL — same per-shard Welford (same SrcA path for bf16, same 1/(i+1)
  reciprocal LUT, same reduce width/order) and the EXACT SAME shared combine_welford_partials
  merge code (so the merged (mean,var) is bit-identical given identical per-shard inputs).
- ONE real difference FOUND + FIXED: the rsqrt approximation. Composite uses rsqrt_tile<true>
  (legacy); fused used the non-legacy default (in combine_welford's RSqrtPolicy path AND the
  is_tp_1 path). Added a `legacy` field to RSqrtPolicy (default false; other callers unchanged)
  and set legacy=true in the fused LN. => fused now uses the same rsqrt as composite. But this
  moved the Wan block PCC only 99.8692 -> 99.8698 — real different math, NOT the dominant cause.
- RULED OUT: UnpackToDestFp32/#45319 — the composite enables it ONLY for fp32 input
  (welford_unpack_fp32_active = in==Float32 && fp32_dest_acc; pre factory L120); for bf16 BOTH use
  the SrcA path, so it's not the difference. Also ruled out the merge (identical code), the recip
  LUT (identical values), and eps (scalar-add vs tile-add of the same eps).

CONCLUSION: the residual ~0.008% op gap (-> 0.13% in-block via ~13x amplification) is NOT a single
logic bug — it's fp operation-ordering/rounding between two INDEPENDENT Welford PRE kernels
(different tile_regs/reconfig/DEST sequencing), amplified on low-variance rows. They are not
bit-exact because they are different kernels. Making them match would require the fused PRE to
mirror the composite's exact op sequence (guided by a per-element fused-vs-composite device diff).
The rsqrt fix is kept (correct alignment, 5/5 module corr still pass). Fused RMS unaffected/ships.

### ISSUE 3 — DEFINITIVE per-element device diff (supersedes the "fp op-ordering" guess above)
Test: `test_layernorm_fused_vs_composite_diff` (models/tt_dit/tests/test_distributed_rmsnorm_fused.py).
No-affine (norm_elementwise_affine=False -> no weight/bias confounder), SAME seed-fixed bf16 mean-0
input to BOTH paths, fp32 output, one shared DistributedLayerNorm+CCLManager, flag toggled. Gathers
both [N,dim] fp32 outputs and, per row, fits out = a*x + b to RECOVER the (mean, invstd) each path
actually applied (output is linear in the host-known input x). N=4096. Composite is the accurate
reference: pcc(composite : fp32-torch) = 99.9997%+ at every TP; max|comp-ref| ~ 0.013. Composite is
NOT the problem — fused is.

THERE ARE TWO SEPARATE FUSED FAILURES, by per-device shard width (= num_tile_cols, the wide-shard
threshold for block-major POST is num_tile_cols >= 56 with use_recip_lut && use_mux, factory L587):

ISSUE 3A (SEVERE, wide-shard / block-major POST path). WAN TP=2 -> 2560/dev = 80 tiles >= 56 ->
force_recip_stream -> streaming + block-major POST. Result on a FRESHLY RESET galaxy:
  - 960 / 4096 rows (23%) come out DEGENERATE = exactly CONSTANT (affine-fit slope 0, resid 0.0).
  - NON-DETERMINISTIC run-to-run (max|run1-run2| ~ 0.015).
  - pcc(fused:composite) = 87.5%, max err 3.46 (= the row's true max |normalized x|: the token is
    left un-normalized / collapsed).
TP=4 (40 tiles) and TP=8 (20 tiles) take the RESIDENT path and are CLEAN: deterministic, 0
degenerate rows. => the bug is the recently-added block-major-POST + force_recip_stream wide-shard
path (num_tile_cols>=56), NOT the AG round count. The earlier "grows with N / AG rounds" framing was
wrong; it tracks SHARD WIDTH (which path), not N.

ISSUE 3B (MILD, narrow-shard path = the real ~0.013% in-block regressor). TP=4 and TP=8:
deterministic, no degenerate rows, but fused variance is ~4x noisier than composite:
  - per-row recovered std error: composite max 0.0037, never > 0.5%; fused mean 0.0066, p99 ~0.035,
    ~25% of rows > 1% std error, 90+ rows > 3%. SIGNED mean ~ 0 (symmetric, zero-mean spread).
  - UNCORRELATED with row variance (corr(row_err, 1/std) ~ 0) -> NOT the rsqrt approximation.
  - UNIFORM across token-index blocks -> NOT a multi-round AG addressing bug.
  - Each affected row is still a CLEAN affine of x (tiny resid) -> the op applies a consistent but
    slightly-wrong (mean, var) per token; it's a stats-precision difference, not normalize noise.
  PRE kernel (welford), recip LUT values (1/(i+1) fp32, identical), combine_welford merge (shared
  code), and stats transport (all fp32 CBs / RawUInt32 packets) are all confirmed equivalent to
  composite -> the residual is the only-not-shared component: the fabric ring-AG of the 2-stat
  partial vs composite's all_gather_persistent_buffer. Small, zero-mean, deterministic.

WARM-GALAXY CONTAMINATION: tp4/tp8 showed degenerate rows ONLY on a galaxy warmed by MANY prior
(broken tp2) runs; after `tt-smi -glx_reset` they are clean, and a single tp2->tp4->tp8 sequence on
a fresh galaxy does NOT corrupt the later configs. So broken (block-major) runs leave device-
persistent fabric state that accumulates and eventually degrades otherwise-clean narrow-shard runs.
This explains earlier unstable model-level numbers. ALWAYS reset before correctness runs.

FIX PRIORITIES: (A) the block-major-POST/force_recip_stream wide-shard path (token collapse + non-
determinism) is the real blocker — gate it off or fix the streaming POST token addressing; (B) the
narrow-shard ~0.66%-std variance noise is minor (the 0.013% regressor) and lives in the fused 2-stat
fabric AG. Fused RMS (1-stat) is unaffected and ships.

### ISSUE 3A — MITIGATED at model level (kernel fix still open)
DistributedLayerNorm (`models/tt_dit/layers/normalization.py`) now gates the fused LN OFF for wide
per-device shards via `_FUSED_LN_MAX_RESIDENT_TILE_COLS = 40`: if width_per_device//32 > 40 the
module falls back to the composite Welford chain (correct at any width). 40 tiles = WAN TP=4, the
widest config proven clean on the resident POST path; everything wider (FLUX tp2=48, LTX tp2=64,
WAN tp2=80, all tp1) takes the broken block-major/streaming POST path and is excluded. Validated:
test_layernorm_fused_vs_composite_diff[wan_tp2] now fused==composite (pcc 100%, 0 degenerate rows,
deterministic); [wan_tp4] unchanged (still fused, 99.987%); test_layernorm_module_corr 5/5 pass.
The underlying block-major POST kernel race (token collapse + non-determinism) is NOT fixed — it
stays open (the model just never exercises it now). Fix it to re-enable fused LN on wide low-TP
shards.

### ISSUE 3A — DEEP TRIAGE (root narrowed, kernel fix still OPEN; model still gated)
Reproduced precisely with test_layernorm_fused_vs_composite_diff (knobs LNMOD_DIM/LNMOD_DIFF_MEAN
+ degenerate-row + collapse-value diagnostics) and DPRINT in the worker writer (use the NEW
`DPRINT("fmt {}\n", ...)` macro from "api/debug/dprint.h" — the old `DPRINT << ...` is now a hard
static_assert; old-style is why DPRINT "didn't compile" before). Findings:

- The trigger is EXACTLY num_tile_cols >= 56 = the force_recip_stream threshold (52 tiles clean,
  56 broken), i.e. the block-major + streaming POST path. INDEPENDENT of ring size (tp2 ring2 AND
  a synthetic tp4 80-tile ring4 both break; tp2 at <56 tiles is clean). The earlier "odd tile-rows"
  was a 2-rounds-per-worker artifact: the real pattern is WARM rows (round >= 1) break, round 0
  (cold) is clean.
- The collapse is invstd->0 from inf VARIANCE: the local per-shard Welford (c_16, BEFORE any
  transport) already has var=inf / running-mean ~4.98e36 for specific token lanes on warm rows.
  Confirmed via DPRINT of the local stat: r=0 mean/var fine, r=1 mean(t17)=4.98e36 var(t17)=inf.
  So it is NOT the AG/forwarder/combine transport (gathered stats just carry the bad local value),
  NOT a fabric race — it is the streaming Welford PRE producing garbage on warm rows.
- The garbage value (4.98e36) is DATA-INDEPENDENT (bit-identical on both devices) -> a stale
  accumulator/register left from the prior row's compute, surfacing only on the streaming path
  (cb_wait/pop between welford_updates) when a per-row `combine` ran the prior row. is_tp_1 wide
  (same streaming PRE + block-major POST, NO combine) is clean; non-streaming TP>1 (same combine,
  no cb-ops-mid-welford) is clean -> it's the (combine state) x (streaming welford) interaction.
- RULED OUT by direct on-device tests (each rebuilt + re-run): packet double-buffer (forced single
  slot - still broke); per-tile welford replay re-arm (transpose_wh_init_short +
  welford_init<PreserveStats>); clearing var/mean DST before finalize; moving welford_init after the
  first cb_wait_front; SFPU lane re-enable in the welford clear (TTI_SFPENCC(3,0,0,10) = enable-all)
  - NONE removed the inf. So it's not the replay buffer, not the DST stale, not a CC lane disable
  in the clear.
- NOT yet pinned: whether the bad value is in input_cb (an L1/reader interaction at >=56 tiles) or
  in the Welford LREG4/5 accumulator that survives welford_init's clear. The fill_tile(0)-before-
  transpose test was inconclusive (transpose overwrites). Next step: DPRINT input_cb raw uint16 in
  the COMPUTE for a warm row (token 17 ~= input_cb byte 1056) to settle input-vs-accumulator; if
  input is clean it's an LLK welford-state issue needing LLK-team input.

MODEL STATUS: still gated (DistributedLayerNorm._FUSED_LN_MAX_RESIDENT_TILE_COLS=40). Narrow shards
(WAN tp4/tp8, the common model configs) keep fused LN and are correct; wide shards use composite.

### ISSUE 3A — standard-kernel review + fix direction (still OPEN; model still gated)
Reviewed the non-distributed streaming Welford layernorm (layernorm/device/kernels/compute/
layernorm_large_tensor_welford.cpp) per request. Two paths, and the contrast is the key:
- **no_fuse path**: accumulates the running mean/M2 in LREG4/5 across streamed tiles, resetting
  per row with a plain `welford_init()`. Works across many (warm) rows ONLY because there is NO
  clobbering SFPU op between rows.
- **fuse_pre_add path**: has a clobbering pre-add BETWEEN welford updates, and therefore NEVER
  trusts LREG to survive it. It SPILLS the welford state to CBs (cb_ex/cb_ex2) with
  `welford_save_state(mean_dst)` and RELOADS it with `copy_tile(cb -> mean_dst/var_dst)` +
  `welford_restore_state(mean_dst)`. `copy_tile` is an L1->DST read (UNPREDICATED), so it
  reliably writes every lane — unlike `welford_init`'s SFPLOADI clear (and `fill_tile`), which
  only write CC-enabled lanes.
It also re-arms per tile after an fp32 transpose (`transpose_wh_init_short` +
`welford_init<WelfordInitMode::PreserveStats>()`), and handles the padded last tile with
`welford_update_rows<W>` (we omit this — fine only because our W is tile-aligned).

OUR BUG maps exactly to the fuse_pre_add situation: the per-row `combine` (TP>1 only) is a
clobbering SFPU op, and our streaming PRE resets with a plain `welford_init()` like the no_fuse
path — which is INSUFFICIENT after a clobber. Confirmed on device: welford INPUT is clean
(in[tok]=~0.96) but the welford-produced running mean for some token lanes is ~4.98e36 on warm
rows (bit-identical across devices = a stale accumulator lane, not data). The SFPLOADI-based clear
(and fill_tile) do not reset those lanes.

Tried + FAILED on device: per-tile welford replay re-arm; fill_tile(0)+welford_restore_state;
SFPENCC(3,0,0,10) lane-enable in the clear; legacy<->non-legacy rsqrt in the combine. None reset
the lane. The one robust pattern NOT yet tried is the fuse_pre_add path's exact mechanism: capture
a zeroed welford state into a CB ONCE at cold start (CC clean), then each row reload it via
`copy_tile(zero_cb)` + `welford_restore_state` (UNPREDICATED L1 path) instead of relying on the
predicated clear. `fill_tile` failed because it is ALSO SFPLOADI-predicated; `copy_tile` is not.

FIX DIRECTION (implementable, mirrors the shipping fuse_pre_add kernel): add a small zeroed-state
CB, pack zeros once at kernel start, and replace the per-row reset with copy_tile(zero)+restore.
Model stays gated (_FUSED_LN_MAX_RESIDENT_TILE_COLS=40) until this lands.

### ISSUE 3A + 3B — FIXED (welford warm-row accumulator reset via unpredicated reload)
ROOT CAUSE: on the TP>1 path the per-row `combine` (legacy SFPU rsqrt + binary/square ops with
data-dependent predication) leaves the SFPU condition code predicated, so the NEXT row's
`welford_init()` SFPLOADI accumulator clear skips the disabled token lanes. Those lanes keep a
stale ~1e36 running mean -> M2 overflow -> inf variance -> invstd 0 -> the token's output row
collapses to a constant. WIDE/block-major shards collapse ~half the tokens of every warm row
(catastrophic, non-deterministic) = ISSUE 3A; NARROW shards show the same bug as a small zero-mean
variance perturbation = ISSUE 3B (the ~0.013% in-block regressor). Same root.

FIX (mirrors layernorm_large_tensor_welford fuse_pre_add): capture a zeroed welford state into a
dedicated CB (welford_zero_cb, c_21, 2 fp32 tiles) ONCE at cold start while the SFPU is clean
(welford_init + welford_save_state + pack); then each row reset the accumulator via
copy_tile(welford_zero_cb -> mean_dst/var_dst) + welford_restore_state — an UNPREDICATED L1->DST
path that resets every lane regardless of the leaked CC, unlike the SFPLOADI clear (and fill_tile,
which is also SFPLOADI-predicated). welford_init() is still called per row to program the replay
buffer; only its clear is bypassed.

VALIDATED (fresh galaxy, _ttnncpp.so rebuilt + synced to lib/):
  - wan_tp2 80t WIDE: pcc(f:c) 87.5% -> 100.0005%, 960 -> 0 degenerate, deterministic.
  - wan_tp4 80t WIDE ring4: 87.6% -> 99.9996%, 0 degenerate, deterministic.
  - wan_tp4 40t / wan_tp8 20t NARROW: 99.987%/99.9997% -> 99.9999%/99.9997% (3B fixed too).
  - test_layernorm_module_corr 5/5 fused=100.00%; test_rmsnorm_module_corr 5/5 (RMS unaffected by
    the shared CB/CT-arg addition; welford_zero_cb is a 1-tile stub for RMS).
Module width gate (_FUSED_LN_MAX_RESIDENT_TILE_COLS) REMOVED — fused LN is correct at any width.
CAVEAT: welford_zero_cb adds 8 KB resident L1 (LN only); not yet folded into
decide_streaming_low_l1/decide_block_major_post accounting (task #10) — fits at <=80 tiles
(validated); very wide is_tp_1 (160t) would fail loudly via the allocator, not silently.
