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
