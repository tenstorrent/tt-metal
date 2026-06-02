# Wan2.1 correctness debugging — instructions to self (resume on new device)

## THE GOAL (from the user)
Some traced Wan2.1 outputs are blurry/artifacty. (1) Instrument the resolution sweep with
PSNR + CLIP. (2) Collect a 4x4 sp0tp1 non-CFG **untraced** baseline (debug any outliers).
Then test traced+untraced for: 4x4 sp0tp1, 4x4 sp0tp1 CFG-parallel, 2x4 sp0tp1, 2x4 sp0tp1
CFG-parallel. Identify which configs produce bad images and **debug until all correctness
bugs that cause blurry/artifacty images are fixed.** Keep a markdown log; commit when needed.

## STATUS (what's DONE)
- ✅ Instrumentation committed: `collate_correctness.py` (offline CLIP via open_clip ViT-B-32
  + PSNR vs baseline from saved PNGs) and test toggles. (commit "Instrument Wan2.1 sweep…")
- ✅ Baseline `4x4_sp0tp1_untraced` collected & verified visually clean (CLIP ~40, sharp).
- ✅ Full 8-config matrix collected. **KEY RESULT: every UNTRACED config is byte-perfect vs
  baseline; every TRACED config is corrupt.** Two bugs:
  - **Bug #1 — non-CFG traced:** cross-resolution corruption. A single resolution traces
    bitwise-perfect; running MULTIPLE resolutions corrupts later ones. Non-monotonic,
    mesh-dependent (768x1024 bad at pos3 on 4x4, good at pos2 on 2x4) => trace-buffer
    aliasing across capture/release cycles is the leading hypothesis.
  - **Bug #2 — CFG-parallel traced:** garbage on ALL shapes incl. the first (deterministic;
    threaded≡serial byte-identical). Ruled out: threading, per-step alloc, gather-in-trace,
    do_cfg flag, spatial-update mechanism. Only distinguisher left: traced `combined_step`
    runs on a CFG submesh that has a SIBLING (a lone submesh traces fine in non-CFG).
    Suspected ttnn-level limitation tracing tiled/sibling submeshes.
- Full detail + every experiment in `wan2_1_correctness_log.md`.

## ENV SETUP on the new device (do FIRST)
```
cd /localdev/cglagovich/tt-metal   # or wherever the repo is on the new box
source ../setup_env.sh && source python_env/bin/activate   # adjust path
python -c "import ttnn; print(ttnn.GetNumAvailableDevices())"   # expect 32 (or device count)
```
If `import ttnn` fails with `undefined symbol: MPIX_Comm_shrink`: the ULFM OpenMPI lib isn't
on LD_LIBRARY_PATH. On the OLD box I extracted it to /localdev/cglagovich/ompi_ulfm and added
`export LD_LIBRARY_PATH=/localdev/cglagovich/ompi_ulfm/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH`
to setup_env.sh. The new box likely has openmpi-ulfm installed system-wide (/opt/openmpi-v5.0.7-ulfm/lib)
— add that to LD_LIBRARY_PATH if needed.
If `python_env/bin/python` is broken: repoint to system python3.10 (ABI-compatible) and set
`home=/usr/bin` in `python_env/pyvenv.cfg`. (These were old-box reboot artifacts; the new box
may be fine.)

## HOW TO RUN (env-gated correctness sweep)
Tests live in `models/tt_dit/tests/models/wan2_1/test_performance_wan2_1.py`.
- `test_resolution_sweep` -k ids: `wh_4x4_sp1tp0`, `wh_4x4_sp0tp1`, `wh_2x4_sp0tp1`, `wh_2x4_sp1tp0`
- `test_cfg_parallel` -k ids: `cfg2_4x4_sp1tp0`, `cfg2_4x4_sp0tp1`, `cfg2_2x4_sp0tp1`
- Env: `WAN_RUN_TAG=<tag>` (saves images to outputs/correctness/<tag>/<BxHxW>.png),
  `WAN_TRACED=0/1`, `WAN_SWEEP_SHAPES="1x768x1024,..."`, `WAN_STEPS=N`,
  `WAN_REPEAT=N` (resolution sweep only: run each shape N× back-to-back, saves _rep0/_rep1).
- Use `--timeout=100000000` (default pytest-timeout 300s kills sweeps and wedges fabric).
- Shape subset used so far: `1x768x1024,1x1024x1024,1x1280x720,1x1536x2048,1x2048x1152`.
- After runs: `python collate_correctness.py` → `wan2_1_correctness_metrics.md` (CLIP+PSNR,
  flags CLIP<22 or PSNR<20). Good image CLIP ≈ 38-42; garbage ≈ 16-22.
- **Visually inspect** flagged PNGs with the Read tool (it renders images).

## NEXT STEPS (priority order)

### Bug #1 (non-CFG traced) — IN PROGRESS, this is the tractable/important one
1. **DECISIVE first experiment (queued):**
   `WAN_REPEAT=2 WAN_RUN_TAG=exp_repeat WAN_TRACED=1 WAN_SWEEP_SHAPES=1x768x1024`
   on `-k wh_4x4_sp0tp1`. Saves `0x768x1024_rep0.png` and `_rep1.png`.
   - rep1 GARBAGE (rep0 fine) => bug is "any 2nd capture/release cycle" (NO `_clear` runs for
     same shape) => ttnn trace-region reuse across capture/release.
   - rep1 CLEAN => bug needs resolution CHANGE (`_clear_per_resolution_state` + re-capture).
2. If resolution-change-dependent: run 2 shapes `WAN_SWEEP_SHAPES=1x768x1024,1x1024x1024
   WAN_TRACED=1`; confirm shape2 corrupt. Then:
   - Big trace region: `WAN_TRACE_REGION_SIZE_4X4=300000000` → if shape2 clean, it's
     region size/fragmentation; fix by sizing or resetting the region per resolution.
   - The pipeline already does `release_traces()` (test finally) + `_clear_per_resolution_state`
     (pipeline __call__ on shape change). Check ORDER and whether ttnn fully frees the trace
     region. Look for a ttnn API to reset/clear all traces on a device.
3. Implement the fix the data points to (most likely: stronger per-resolution trace/region
   reset, or pin persistent input buffers so a later trace's activations don't alias them).
   Validate: re-run the full 5-shape traced sweep for wh_4x4_sp0tp1 → all CLIP ~40, PSNR high.
4. Confirm fix also resolves 2x4 traced. Commit.

### Bug #2 (CFG-parallel traced) — likely ttnn-level
- Confirm the sibling-submesh hypothesis: read ttnn mesh trace capture/replay
  (`tt_metal/distributed/mesh_trace*`, `ttnn.begin_trace_capture/execute_trace`) for any
  per-parent-mesh / shared-context state that breaks when a parent has 2 child submeshes
  both tracing.
- If bug #1's fix is a general trace-state reset, RE-TEST bug #2 (it might share the cause).
- If genuinely ttnn-level & unfixable in-model: the CORRECTNESS fix is to make CFG-parallel
  run UNTRACED (proven byte-perfect) — e.g., force traced=False in `_cfg_parallel_denoise`
  with a logged warning — accepting loss of the ~2x until a ttnn fix lands. CONFIRM with the
  user before shipping that perf tradeoff. Then validate cfg2_4x4_sp0tp1 + cfg2_2x4_sp0tp1
  traced→clean and commit.

## WIP code state (committed alongside this file)
- `test_performance_wan2_1.py`: instrumentation + WAN_REPEAT knob.
- `pipeline_wan.py`: bug-#2 experiment changes (persistent per-submesh spatial buffer
  `cfg_spatial_buffers`, external SP gather, `WAN_CFG_NOTHREAD` debug toggle). NONE fixed
  bug #2 — they're neutral/cleaner but UNPROVEN. Consider reverting to the committed CFG path
  (gather_output=True inside trace) if not pursuing them, OR keep as the base for bug-#2 work.

## HARDWARE NOTE
The OLD box (wh-glx 6U) degraded badly this session: board fell off PCIe → ETH-core heartbeat
failures at mesh-open (different core each try) → device enumeration failed (arch NOTSET).
glx_reset/shm-cleanup/settle could not recover it. Being moved to a new box. If the new box
degrades similarly, runs error at setup with "ETH core heartbeat" / "undefined behavior"
(IOMMU sysmem) / "Read 0xffffffff over PCIe" — needs a power-cycle, not glx_reset.
NEVER `kill -9` a tt-metal process mid-device-op (knocks boards off the bus).
