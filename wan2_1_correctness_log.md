# Wan2.1 correctness sweep — log

Goal: find & fix configs that produce blurry/artifacty images (suspected: traced paths).

## Metrics
- **CLIP score**: open_clip ViT-B-32 (openai) cosine(image, prompt) × 100. No reference; low = bad/blurry/misaligned image. Primary "is this image good" signal.
- **PSNR (dB)**: vs the baseline image for the same shape/seed. Baseline = 4x4 sp0tp1, non-CFG, **untraced**. High = matches baseline; low = the config diverged (bug).

## Method
- Device tests save each generated image to `outputs/correctness/<run_tag>/<shape>.png` (lossless uint8 RGB), seed=42, fixed prompt, fixed steps. Toggles: `WAN_RUN_TAG`, `WAN_TRACED=0/1`, `WAN_SWEEP_SHAPES`.
- Offline `collate_correctness.py` computes CLIP + PSNR(vs baseline) from the PNGs → tables below.

## Run matrix
| run_tag | config | CFG | traced |
|---|---|---|---|
| 4x4_sp0tp1_untraced | 4x4 sp0tp1 | no | no (BASELINE) |
| 4x4_sp0tp1_traced | 4x4 sp0tp1 | no | yes |
| 4x4_sp0tp1_cfg_untraced | 4x4 sp0tp1 | yes | no |
| 4x4_sp0tp1_cfg_traced | 4x4 sp0tp1 | yes | yes |
| 2x4_sp0tp1_untraced | 2x4 sp0tp1 | no | no |
| 2x4_sp0tp1_traced | 2x4 sp0tp1 | no | yes |
| 2x4_sp0tp1_cfg_untraced | 2x4 sp0tp1 | yes | no |
| 2x4_sp0tp1_cfg_traced | 2x4 sp0tp1 | yes | yes |

## Progress

### Baseline: 4x4_sp0tp1_untraced — CLEAN ✓
Visually inspected (Read tool): all sharp, on-prompt (cats boxing, spotlit stage). No outliers.
CLIP per shape: 768x1024=38.8, 1024x1024≈40, 1280x720≈40, 1536x2048=41.8, 2048x1152=41.0. **mean 40.2.**
=> "good image" CLIP band ≈ 38-42. Flag threshold 22 is safely below.
Shape subset for sweep: 1x768x1024, 1x1024x1024, 1x1280x720, 1x1536x2048, 1x2048x1152.

### Remaining runs — BLOCKED on hardware (2026-06-02)
Instrumentation validated (baseline ran clean via WAN_RUN_TAG/WAN_TRACED). But after the
baseline, board 17 fell off the PCIe bus (`Read 0xffffffff over PCIe ID 17`) and
`/dev/tenstorrent/1` disappeared; `tt-smi -glx_reset` now fails (`[Errno 6] No such device
... /dev/tenstorrent/1`) and can't recover it. Needs a host reboot / IPMI power-cycle (or
`tt-smi -r`, which is otherwise discouraged on this box). No new device data until recovered.

Status of remaining runs: 4x4_sp0tp1_traced; 4x4_sp0tp1_cfg_{untraced,traced};
2x4_sp0tp1_{untraced,traced}; 2x4_sp0tp1_cfg_{untraced,traced} — all pending hardware.

### Env recovery after host power-cycle (2026-06-02)
The power-cycle reset the root FS to a base image, wiping (a) the uv-managed cpython the
venv pointed at (`/usr/local/share/uv/...`) and (b) the apt-installed ULFM OpenMPI
(`/opt/openmpi-v5.0.7-ulfm`). `/localdev` (venv + repo) persisted. Fixes (no sudo):
- Repointed `python_env/bin/python` -> `/usr/bin/python3.10` (3.10.12; ABI-compatible with
  the venv's 3.10.19 wheels) and set `pyvenv.cfg home=/usr/bin`.
- Re-downloaded `openmpi-ulfm_5.0.7-1_amd64.deb`, `dpkg-deb -x` to
  `/localdev/cglagovich/ompi_ulfm`, appended its lib to `LD_LIBRARY_PATH` in setup_env.sh
  (fixes `libtt_metal.so: undefined symbol MPIX_Comm_shrink`).
ttnn imports; `tt-smi -glx_reset` re-inits all 32 boards. Resuming the sweep.

## RESULTS

### 4x4_sp0tp1_traced (non-CFG) — BUG: corrupts after the first resolution
| shape (sweep order) | CLIP | PSNR vs base | verdict |
|---|---|---|---|
| 1536x2048 (1st) | 41.82 | **inf (byte-identical to baseline)** | PERFECT |
| 2048x1152 (2nd) | 15.51 | 11.3 | GARBAGE |
| 768x1024 (3rd) | 17.61 | 4.8 | GARBAGE (tiled noise, verified visually) |
| 1024x1024 (4th) | 20.75 | 6.1 | GARBAGE |
| 1280x720 (5th) | 18.21 | 6.5 | GARBAGE |

=> Tracing is **bitwise-correct for the first resolution**, then every subsequent
resolution is garbage (blocky tile noise). Strong signal: **cross-resolution trace-state
corruption** — the trace/buffers from shape 1 aren't fully reset before shape 2 captures.
### TWO DISTINCT TRACING BUGS identified
- **4x4_sp0tp1_cfg_untraced: PERFECT** (every shape PSNR=inf, byte-identical to baseline).
  => untraced (CFG and non-CFG) is numerically correct; bugs are tracing-only.
- **Bug #1 — non-CFG traced: cross-resolution corruption.** 1st swept shape perfect
  (byte-identical), every later shape garbage. In the committed traced path.
- **Bug #2 — CFG-parallel traced: garbage on ALL shapes incl. the first** (CLIP ~16-21,
  PSNR 5-8). In the new threaded CFG denoise. The perf sweep only measured speed, never
  checked image correctness, so this was hidden.

Visual (Read tool): bug #1 non-CFG-traced = **blocky tile noise** (wrong-shape/sharding
assembly — looks like a stale-resolution trace replayed on the new shape). bug #2
CFG-traced = **fine random static** (denoise produced pure noise — replay computed nothing
useful, consistent with the concurrent-thread trace replay corrupting per-step state).

### 2x4 results refine the picture
- 2x4_sp0tp1_untraced: GOOD all shapes (CLIP ~40, PSNR 21-25 vs 4x4 baseline — high, just
  not bit-exact due to different sharding).
- **2x4_sp0tp1_traced**: only `1024x1024` GARBAGE; `2048x1152`, `768x1024`, `1280x720` GOOD.
  Note `768x1024` was GARBAGE in the 4x4 run but GOOD here. => bug #1 is **order/accumulation
  dependent**, NOT intrinsic-per-shape and NOT simply first-good-rest-bad. Points to
  trace-region / per-resolution state that corrupts depending on capture history.
  (4x4 run order: 1536x2048 ok, then 2048x1152/768x1024/1024x1024/1280x720 all bad.
   2x4 run order: 2048x1152 ok, 768x1024 ok, 1024x1024 bad, 1280x720 ok.)

### DIAGNOSTICS (single shape 768x1024, 4x4 sp0tp1)
- **diagB non-CFG traced single shape: PERFECT (PSNR inf == baseline)** — but same shape was
  GARBAGE in the multi-shape sweep. => **Bug #1 = cross-resolution state corruption**
  (one resolution traces fine; running multiple resolutions corrupts later ones).
- **diagA1 (CFG traced, threaded) == diagA2 (CFG traced, NOTHREAD): BYTE-IDENTICAL garbage**
  (CLIP 21.85 both). => **Bug #2 is NOT threading** and NOT cross-resolution (bad even
  single-shape). It's deterministic in the CFG traced path. Threading is safe to keep.

### Root-cause hypotheses + fixes
- Bug #2 (CFG traced): CFG-specific vs working non-CFG traced = per-step fresh `from_torch`
  DEVICE spatial input (triggers "unsafe allocation during active trace") + gather_output=True.
  Fix attempt 1: build the traced spatial input as a HOST sharded tensor (on_host=traced) so
  the Tracer manages one persistent device buffer (like the timestep) — no per-step device alloc.
- Bug #1 (non-CFG traced): cross-resolution state not reset under tracing. TBD after bug #2.

### Bug #2 investigation (CFG traced garbage) — ruled out
- Threading: A1(threaded)==A2(NOTHREAD) BYTE-IDENTICAL garbage → deterministic, not a race.
- Per-step device alloc of spatial input (on_host=traced): no effect (byte-identical).
- gather_output inside trace (moved SP gather outside, untraced): no effect (~20 CLIP).
- cfg_dit[0] IS self.transformer (same model that traces fine in non-CFG diagB w/ do_cfg=True).
- Remaining suspect: do_classifier_free_guidance=False path (CFG uses False; the working
  non-CFG traced uses True). do_cfg=True discriminator could NOT run — device ETH core
  e3-6 (ASIC ...836) heartbeat failures started, blocking ALL mesh opens (CFG and non-CFG).
- do_cfg=True discriminator: ALSO garbage (20.24) => do_cfg flag is NOT the bug either.
- **ROOT CAUSE (hypothesis):** non-CFG passes the SAME persistent `self.latent_buffer`
  object to the traced combined_step every step, so the Tracer NO-OPs the spatial input
  (update is an external ttnn.copy into that buffer). CFG passes a FRESH spatial each step,
  forcing the Tracer's `_update_input` to copy a SHARDED tensor into the captured buffer.
  The Tracer's input-update is only validated on the REPLICATED timestep; it mishandles the
  SHARDED spatial → trace keeps using step-1 latent → constant/garbage. **Fix: persistent
  per-submesh spatial buffer + external ttnn.copy, pass the same object (Tracer no-ops).**

### SUMMARY OF FINDINGS (full 8-config matrix)
| config | untraced | traced |
|---|---|---|
| 4x4 sp0tp1 (non-CFG) | CLEAN (baseline) | Bug #1: 1st shape OK, later shapes garbage (blocky) |
| 4x4 sp0tp1 CFG-parallel | CLEAN (==baseline) | Bug #2: ALL shapes garbage (static) |
| 2x4 sp0tp1 (non-CFG) | CLEAN | Bug #1: order-dependent (e.g. only 1024x1024 bad) |
| 2x4 sp0tp1 CFG-parallel | CLEAN | Bug #2: ALL shapes garbage |
=> **Every UNTRACED config is correct; every bug is TRACING-only.**

### Bug #1 (non-CFG traced, cross-resolution) — confirmed, not yet bisected
- Single shape alone = PERFECT (diagB). Multiple resolutions => later ones corrupt
  (order/accumulation dependent; not intrinsic-per-shape). Root cause undiagnosed
  (suspects: trace-region fragmentation/overflow across resolutions, or some shape-keyed
  state not reset under tracing). Needs device bisection (bigger trace region; same-shape-
  twice; release-order) — BLOCKED by hardware.

### Bug #2 (CFG-parallel traced) — investigated, likely ttnn-level
- Ruled out: threading (A1==A2 byte-identical), per-step spatial alloc (on_host), gather
  inside trace (external gather), do_cfg flag (do_cfg=True also garbage), spatial-update
  mechanism (non-CFG uses fresh sharded spatial too via get_model_input typecast, and works).
- Remaining distinguishing factor: combined_step is traced on a CFG submesh that has a
  SIBLING submesh (both children of the parent). A LONE submesh (non-CFG create_submesh)
  traces fine. Both CFG configs (4x4 auto-tiled, 2x4 explicit-offset) are garbage.
  => appears to be a ttnn limitation: tracing on tiled/sibling submeshes of a parent mesh.
  Likely needs a ttnn fix or an untraced fallback for CFG (correct but loses the ~2x).

### Bug #1 refined analysis (device-free, 2026-06-02)
Confirmed (not theory): single resolution traces bitwise-perfect; multi-resolution corrupts
later ones; pattern is NON-monotonic & mesh-dependent (4x4: pos2+ bad; 2x4: 2048x1152 ok,
768x1024 ok, 1024x1024 bad, 1280x720 ok). 768x1024 is bad at pos3 (4x4) but good at pos2
(2x4) => NOT intrinsic-per-shape, NOT clean accumulation. The non-monotonic + "blocky
reads-wrong-memory" character points to **trace buffer aliasing across capture/release
cycles**: after `_clear_per_resolution_state` frees+reallocs the persistent input/CCL
buffers and a new trace is captured, the new trace's baked activation addresses can overlap
reused DRAM (fragmentation-dependent => non-deterministic). Manager-level reset is adequate
(untraced is clean); the hazard is at the ttnn trace/allocator level.

Candidate experiments to run when HW returns (in priority order):
1. Same shape twice back-to-back (need a knob to bypass WAN_SWEEP_SHAPES dedup): if the 2nd
   identical-resolution traced run is ALSO bad => it's "any 2nd capture/release cycle", not
   resolution-change => strongly implicates ttnn trace-region reuse.
2. Big trace region (WAN_TRACE_REGION_SIZE_4X4=300MB): if later shapes become good =>
   region fragmentation/overflow; fix by sizing region or fully resetting it per resolution.
3. Skip `_clear_per_resolution_state` (keep buffers across shapes): isolates whether the
   free+realloc is what enables the aliasing.
Likely fixes: a stronger per-resolution trace/allocator reset (a ttnn trace-region clear if
one exists), or pin the persistent input buffers so their addresses don't get reused by a
subsequent trace's activations.

### HARDWARE (2026-06-02) — degraded, needs power-cycle
ETH-core heartbeat checks fail intermittently at mesh open on DIFFERENT cores each attempt
(e3-6, e6-0, e9-6) and an IOMMU/sysmem "undefined behavior" (silicon_sysmem_manager.cpp:386).
Persists through many glx_reset + shm cleanup + 60s settle. Device degraded from this
session's repeated crashes/resets. Needs a host power-cycle (+ cooldown), not glx_reset.
Blocks all further bisection/fix validation.

## BUG #2 FIXED (persistent prompt + spatial buffers)
Per-step mean logging revealed: in the MEASURE call, submesh-1 pred std COLLAPSES to ~0.056
(degenerate) while submesh-0 goes erratic — but the WARMUP replay was normal. The only thing
that changes warmup->measure is the prompt buffer is rebuilt fresh each pipeline() call,
forcing the Tracer to _update_input a TP-SHARDED prompt tensor (mishandled). Non-CFG keeps a
PERSISTENT prompt buffer (Tracer no-ops). Fix: persistent per-submesh prompt buffers
(self.cfg_prompt_bufs) + persistent spatial buffers (self.cfg_spatial_buffers), updated via
external ttnn.copy, passed as the SAME object each step.
=> fix2e: CFG 4x4 sp0tp1 traced single-shape 768x1024 CLIP **38.84, PSNR inf** (byte-identical
to baseline). FIXED.

## RESUMED ON NEW 4x8 GALAXY (2026-06-03) — Bug #1 reproduced & narrowed
Env healthy (32 devices, ULFM OpenMPI system-installed, no ETH heartbeat issues). Build OK.

- **Decisive WAN_REPEAT=2 result (exp_repeat):** same shape (768x1024) run twice back-to-back
  — each rep is a full capture/release cycle (release_traces() in the per-shape finally), but
  NO `_clear_per_resolution_state` (shape unchanged). **BOTH reps CLEAN.** => recapture-after-
  release is fine; the bug is NOT "any 2nd capture cycle". It REQUIRES a resolution change.
- **5-shape reproduction (exp5_traced, order 1536x2048,2048x1152,768x1024,1024x1024,1280x720):**
  1st shape (1536x2048) CLEAN (sharp boxing cats); **every** subsequent shape GARBAGE (blocky
  tile noise; 1280x720 = uniform dead noise). Deterministic: first resolution CHANGE triggers
  corruption that persists for all later shapes.
- **Code narrowing:** the ONLY differing path between the clean WAN_REPEAT case and the corrupt
  shape-change case is `_clear_per_resolution_state()` (frees+reallocs latent/condition/
  cfg_spatial/CCL ping-pong buffers + solver, resets ping-pong indices; does NOT reset global
  semaphores, does NOT free prompt buffers). Trace INPUTS are robust to addr change
  (Tracer._update_input copies new->baked when buffer_address differs, tracing.py:239). So the
  hazard is the documented caveat (tracing.py:33): a buffer allocated AFTER capture landing on
  a baked trace-activation address => clobbered on replay. The trace is RE-captured fresh per
  shape, so within-shape it should be self-consistent — pointing at allocator-layout/region
  state left by the prior shape's free+realloc.
- **exp_bigregion (WAN_TRACE_REGION_SIZE_4X4=300MB, [1536x2048,768x1024]):** 768x1024 STILL
  bad but corruption CHANGED from blocky tile noise -> near-uniform dark/dead image. => NOT
  region exhaustion; it's allocator-LAYOUT-dependent address aliasing (region carved from DRAM
  top shifts the layout -> different aliasing).
- **exp_skipclear (WAN_SKIP_CLEAR=1, skip _clear_per_resolution_state, [1536x2048,768x1024]):**
  768x1024 STILL blocky garbage. => the free+realloc is NOT the trigger. Trigger = capturing a
  trace at a NEW shape (different activation layout) on a device that already ran a trace.

## BUG #1 ROOT CAUSE (2026-06-03) — rope re-copy clobber
The non-CFG denoise loop builds rope features (rope_cos_1HND/rope_sin_1HND/trans_mat) ONCE on
iter 0 (`if permuted_latent_tt is None`) as FRESH from_torch device tensors, then passes the
SAME objects to combined_step EVERY step. The Tracer re-copies every input into its baked
buffer on each replay (tracing.py _update_input, since the fresh addr != baked addr). Per the
Tracer caveat (tracing.py:33), a tensor allocated AFTER capture can occupy a freed
trace-activation address; iter-0's replay then WRITES that activation, CLOBBERING the rope
buffer. Iters 1-19 copy the corrupted rope -> garbage output.
  - Why first shape clean / all later corrupt: the FIRST capture runs on a fresh allocator
    layout where the post-capture rope alloc happens not to alias an activation; later shapes
    have a different activation layout (new seqlen) that does alias. Deterministic per shape.
  - Why spatial_1BNI is SAFE: it's recreated each step (get_model_input typecasts the float32
    latent_buffer -> fresh bf16) and read (copied into baked) BEFORE that step's replay, then
    discarded — clobber-after-replay is harmless. timestep is SAFE (host tensor when traced).
  - Rope is uniquely unsafe: a CONSTANT buffer built once and re-read across ALL replays.
FIX (ATTEMPTED): persistent rope buffers ... — **WRONG, REVERTED.**

## ROPE HYPOTHESIS REFUTED (2026-06-03)
get_rope_features ALREADY caches rope tensors keyed by hidden_states.shape
(self.cached_rope_features in transformer_wan.py:390). So rope is already persistent/stable
per resolution, computed on the FIRST call for a shape = eager warmup = BEFORE capture, live
during capture. It is NOT freshly allocated after capture, so it was never clobbered. The rope
fix not only addressed a non-bug, it CRASHED: on the 2nd call `fresh = get_rope_features()`
returns the SAME cached object as the adopted buffer, so `ttnn.copy(buf,buf)` +
`ttnn.deallocate(buf)` deallocated the model's cached rope -> "Tensor is not allocated" at
trace capture (storage.cpp:162), index 3/4/5 = the three rope inputs. Reverted all 3 edits.
=> Bug #1 root cause is still OPEN. Inputs are all safe under the Tracer copy-before-replay
discipline (rope cached, spatial recreated per-step & read-before-replay, timestep host,
latent/condition/prompt persistent). So the offender is likely NOT a trace input.

## NEXT HYPOTHESIS: kernel-binary residency / "first-capture privilege"
A prior session left an env-gated experiment in _clear_per_resolution_state
(WAN_CLEAR_PROGCACHE=1 -> mesh_device.clear_program_cache() on resolution change). Theory:
ttnn mesh-trace gives the FIRST captured trace some residency privilege; a 2nd trace at a new
shape reuses program-cache kernel binaries whose placement conflicts -> corrupt replay. Fits
"first shape clean, all later corrupt". Running exp_progcache (2-shape, clear progcache on
shape change): if 768x1024 turns CLEAN => kernel-binary residency confirmed.

## exp_progcache RESULT (2026-06-03): HANG, inconclusive
Shape 1 (1536x2048) clean as always. On resolution change, clear_program_cache() ran, then
the 768x1024 TRACED WARMUP started and HUNG (>15 min silence; a full 20-step measure is ~36s).
=> mesh_device.clear_program_cache() wedges the subsequent trace capture. NOT a viable fix and
inconclusive for residency (the hang shows trace capture hard-depends on program-cache state).
Had to SIGINT the wedged process. WAN_CLEAR_PROGCACHE path is a dead end.
NEXT: env-gated per-step latent norm logging (the technique that cracked Bug #2) on a 2-shape
traced measure run — compare shape A (clean) vs shape B (corrupt) to localize whether
combined_step's FIRST replay is already garbage (capture/replay broken) or it accumulates.

## DEVICE DEGRADED (2026-06-03 ~15:45) — needs power-cycle
The WAN_CLEAR_PROGCACHE hang had to be terminated mid-trace-capture (SIGINT then SIGTERM; NOT
kill -9). The process exited but WITHOUT graceful device shutdown. A lightweight probe
(ttnn.GetNumAvailableDevices) opened/closed all 32 chips clean, BUT the next real mesh-open
(test_resolution_sweep) failed instantly:
  RuntimeError: ETH core heartbeat check failed ... ETH core e9-6 (NOC0), post code: 2040000
This is the documented degradation symptom -> needs a HOST POWER-CYCLE (not glx_reset).
Instrumentation in tree: WAN_STEP_DEBUG=1 per-step pred mean/std/absmax+addr logging in the
non-CFG denoise loop (env-gated, harmless). WAN_CLEAR_PROGCACHE experiment toggle still in
_clear_per_resolution_state (env-gated, OFF) — proven a dead end (wedges trace capture); remove
when convenient. Rope fix fully reverted. RESUME after power-cycle by running exp_stepdbg:
  WAN_STEP_DEBUG=1 WAN_STEPS=10 WAN_RUN_TAG=exp_stepdbg WAN_TRACED=1 \
  WAN_SWEEP_SHAPES=1x1536x2048,1x768x1024 pytest ...-k wh_4x4_sp0tp1 --timeout=100000000 -s

## DEVICE RECOVERED via glx_reset (2026-06-03 ~16:00). Instrumentation findings:
WAN_STEP_DEBUG added to non-CFG loop: per-step pred per-shard std + post-loop PRE_GATHER/
POST_GATHER/POST_PROCESS latent stats (across ALL 16 device shards via get_device_tensors;
local_device_to_torch only reads device 0 so it MISSED the corruption initially).

KEY RESULTS (2-shape 1536x2048 then 768x1024, traced, seed=42, 10 steps):
- Per-step noise PRED is healthy on ALL 16 shards every step for BOTH shapes (std ~1.0-1.4,
  nbad=0). So combined_step trace replay does NOT explode/NaN.
- BUT the final LATENT for the traced 2nd shape (768) is DEGENERATE: POST_PROCESS std=0.377,
  absmax=0.97 vs shape-A 0.728 and 768-EAGER-warmup 0.796. Low-variance latent -> the dark/dead
  image. The traced WARMUP of 768 was even worse: nbad=16, all shards std=0.19 (collapsed);
  structured: shards d0-d11 absmax~0.6, d12-d15 absmax=2.0 (last SP rank differs).
=> Corruption is NOT magnitude-explosion; it's SUBTLE WRONG VALUES (plausible magnitude) in the
  traced 2nd-shape denoise that integrate to a degenerate latent. Trace replay produces
  plausible-but-incorrect predictions for the 2nd captured shape. Confirms a ttnn-level
  "2nd-trace-capture" bug (program-cache/trace-allocator), NOT a model buffer bug (untraced is
  byte-perfect; all model inputs safe under copy-before-replay).
## DEFINITIVE: 2nd captured trace replays WRONG from step 0 (2026-06-03)
Compared traced vs untraced, both seed=42, step 0 (identical input):
  shape A (1536, 1ST trace): traced step0 pred std[1.208,1.343] == untraced [1.207,1.341]. MATCH.
                             final latent 0.728 (both). CORRECT.
  shape B (768, 2ND trace):  traced step0 pred std[1.056,1.061] (UNIFORM, low) vs untraced
                             [1.280,1.543] (varied). MISMATCH from step 0. traced final latent
                             0.377 (degenerate) vs untraced 0.724. WRONG.
=> The FIRST trace captured on the mesh replays correctly; the SECOND (different shape, after
   release+recapture) replays with wrong values FROM THE FIRST REPLAY STEP. Not accumulation.
   Tracer is recreated fresh per shape (_needs_new_tracer True after release_trace) so NO
   Tracer-level stale state -> bug is in ttnn begin_trace_capture/execute_trace for the 2nd
   trace. Model is correct (untraced byte-perfect).
SIGNATURE CLUE: broken replay has UNIFORM std across all 16 shards (1.056-1.061) vs varied in
correct runs -> smells like CCL (TP all-gather/reduce-scatter) producing averaged/replicated
data. NOTE: CCLManager.reset_global_semaphores() EXISTS but is NEVER called; _clear_per_
resolution_state resets ping-pong indices (clear_persistent_buffers) but NOT global semaphores,
which accumulate -> candidate desync across resolution change.
- exp_resetsema (WAN_RESET_SEMA=1): 768 traced latent std=0.377 (UNCHANGED), warmup still
  nbad=16 std 0.19. Semaphore desync REFUTED.

## BUG #1 — CONCLUSION: ttnn multi-shape mesh-trace bug (model-level fixes exhausted)
The 2nd mesh-trace captured on the device (after release+recapture at a DIFFERENT shape) replays
INCORRECTLY from the first replay step: predictions are uniform-across-all-16-shards and
low-variance (CCL-degenerate signature), integrating to a degenerate (std ~0.38 vs ~0.73)
latent -> dark/blocky image. The 1st trace is always correct (traced==untraced byte-level).
RULED OUT (all refuted by experiment): rope (cached), free/realloc fragmentation (skipclear),
trace region size (bigregion), program-cache clear (hung+degraded device, inconclusive), CCL
global semaphore reset (resetsema), Tracer stale state (recreated fresh per shape). Model +
untraced path are correct. => This is below the model layer: ttnn begin_trace_capture/
execute_trace for a 2nd trace on a mesh with CCL collectives.
Env toggles left in tree (all env-gated OFF, for follow-up; remove before final commit):
WAN_CLEAR_PROGCACHE, WAN_RESET_SEMA, WAN_STEP_DEBUG (per-step/post-loop shard-norm logging).
UNTESTED next candidates: (a) per-(model,shape) PERSISTENT traces, never release between shapes
(WAN_REPEAT shows same-shape release+recapture is fine -> maybe keeping traces live avoids it);
(b) minimal non-Wan ttnn repro (capture shapeA, release, capture shapeB, replay vs eager) to
file/fix upstream.

## BUG #1 ROOT CAUSE FOUND + FIXED (2026-06-03) — unreleased ENCODER trace
NOT ttnn. The T5/UMT5 text encoder forward is @traced_function (model_t5.py:108;
UMT5Encoder = T5Encoder). It captures a trace on the first traced run. But pipeline
release_traces() ONLY released the combined_step (transformer) tracers via
WanTransformer3DModel.combined_step._tracers -> the ENCODER trace was NEVER released. Its
captured trace bakes references to the encoder's CCL persistent buffers, and
encoder_ccl_manager IS vae_ccl_manager (pipeline_wan.py:262). _clear_per_resolution_state
frees + reallocs those buffers on every resolution change, so the stale (never-released)
encoder trace REPLAYS into freed/reused memory on shapes 2..N -> corrupt prompt embeds ->
broken cross-attention -> degenerate denoise FROM STEP 0. Explains: shape 1 always clean
(encoder trace fresh), all later shapes corrupt; latent degenerate/uniform-across-shards (not
NaN) because embeds are plausible-magnitude but wrong; traced-only (untraced never captures).
FIX (pipeline_wan.py release_traces): also release the encoder tracer:
    enc = self.tt_umt5_encoder
    for t in list(type(enc).forward._tracers.values()): t.release_trace()
so the encoder recaptures fresh per resolution against valid buffers.
VALIDATION (fix_encrelease, 2 shapes): traced 768 POST_PROCESS latent std 0.377 -> 0.7238
(byte-matches untraced 0.7238), nbad=0 on all 16 shards, image = sharp boxing cats. FIXED.
fix_full5 (all 5 shapes traced, 4x4 sp0tp1): ALL CLEAN. Visual = sharp boxing cats every
resolution. collate_correctness: fix_full5 avg CLIP 37.92 (good ~38-42); NONE of the 5 shapes
flagged (all >22; pre-fix broken runs were ~17-21). Bug #1 CONFIRMED FIXED on 4x4 sp0tp1.
Now confirming on 2x4 (fix_2x4) + CFG multi-shape next.

## 2x4 OOMs at MODEL LOAD (unrelated to Bug #1 fix) — 2026-06-03
fix_2x4 (wh_2x4_sp0tp1) FAILED loading token_embeddings: "Out of Memory: Not enough space to
allocate 4200595456 B DRAM ... each bank needs 350052352 B, free 325366208 B" — ~25 MB short
PER BANK, at LOAD time, before any inference. This is a pre-existing 2x4 DRAM-tightness issue
(8-device submesh shards weights across fewer devices; the 72MB trace region carved from DRAM
adds pressure). NOT related to the encoder-release fix (which only runs post-inference in
release_traces). Can't simply shrink the trace region: a single capture is ~58MB so the region
must stay >=~58MB, while the shortfall is ~25MB. Needs a separate memory fix (smaller region +
encoder sharding, or different submesh placement) — out of scope for the Bug #1 correctness fix.
Bug #1 fix stands on 4x4 sp0tp1 (validated). Proceeding to CFG multi-shape on 4x4.

## CFG MULTI-SHAPE VALIDATED (2026-06-03) — fix_cfg
cfg2_4x4_sp0tp1, 2-shape traced (1536x2048, 768x1024): BOTH sharp/clean. The encoder-release
fix covers CFG (same encode_prompt / tt_umt5_encoder path); combined with the prior Bug #2
single-shape fix (fix2e), CFG multi-shape traced is now correct.

## SUMMARY / STATUS
- Bug #1 (non-CFG cross-resolution traced corruption): ROOT-CAUSED + FIXED (release the
  never-released text-encoder trace in release_traces). Validated 4x4 sp0tp1 5-shape (CLIP
  37.92, all sharp) + 4x4 CFG 2-shape.
- Bug #2 (CFG single-shape): previously fixed (fix2e); CFG multi-shape now also clean.
- 2x4: model-load OOM (pre-existing memory tightness, unrelated to the fix).
- Working tree: the keeper change is the single release_traces hunk in pipeline_wan.py
  (manager.py reverted; all debug instrumentation removed). Test file + this log have
  instrumentation/notes from the debug sessions. NOT committed (awaiting user go).
TODO before commit: remove debug instrumentation (WAN_STEP_DEBUG per-step + post-loop shard
logging, WAN_TRACE_DIFF block in pipeline_wan.py; WAN_CLEAR_PROGCACHE block in _clear). The
encoder-release fix is the keeper. Likely also fixes/helps Bug #2 (CFG) which shares CCL+trace.

## BUG #1 — ROPE HYPOTHESIS DISPROVEN; ROOT CAUSE IS ttnn TRACE-LEVEL (2026-06-03, resumed)
The rope-clobber fix did NOT work (fix_rope2 768x1024 still garbage) and was REVERTED. Every
persistent/re-read buffer (rope, solver state, CCL ping-pong, prompt) is allocated during the
eager warmup (pre-capture) and is live during capture, so the post-capture clobber model can't
explain the corruption.

### Decisive order-vs-size experiments (4x4 sp0tp1, traced, 20 steps)
- **exp_recover  [1536x2048, 768x1024, 1536x2048]:** pos0=CLEAN, 768=GARBAGE(near-black),
  pos2=CLEAN. => Corruption is RECOVERABLE, not cumulative device degradation.
- **exp_first768 [768x1024, 1536x2048, 768x1024]:** pos0=CLEAN, 1536=GARBAGE(block noise),
  pos2=CLEAN. => Refutes the "large-always-clean / small-after-large" (size) theory.

### BEHAVIORAL LAW (rock-solid, reproducible)
A trace replays correctly IFF its spatial shape MATCHES THE FIRST trace captured on the mesh
device (since process start). Any later DIFFERENT shape is corrupt; returning to the first
shape recovers. ORDER-determined, not size-determined.
- Model caches verified shape-correct (rope keyed by shape; CCL ping-pong shape-keyed+cleared;
  latent/solver cleared per resolution; prompt shape-independent). Fresh-correct inputs STILL
  produce garbage => ttnn/tt-metal mesh-trace internal, NOT a python/model bug.
- ttnn.release_trace does NOT reset the privilege (shape2 after release is still corrupt).
- Pipeline state mgmt verified correct: _clear_per_resolution_state fires on shape change
  (pipeline_wan.py:1050); eager warmup re-runs per shape.
- Signature matches a KERNEL-BINARY / PROGRAM-RESIDENCY issue across multi-shape traces.
- IN FLIGHT: exp_progcache — WAN_CLEAR_PROGCACHE=1 clears mesh_device program cache on
  resolution change. If 768 (2nd shape) becomes CLEAN, the model-level fix is per-resolution
  program-cache clear.
