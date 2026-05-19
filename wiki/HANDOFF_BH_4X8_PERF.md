# Handoff: WAN 2.2 S2V perf on BH Galaxy 4×8

You are taking over a working S2V port. The pipeline produces reference-exact
output on BH-LB **2×4** (8 chips). Your job is to make it run — then run well —
on BH Galaxy **4×8** (32 chips).

## Where things stand (BH-LB 2×4)

Last verified run (4-clip × 5-step, warm_weird_22s.mp3 + girl-hoodie-glasses.png,
LAT_TARGET=20, AdaIN active): **TOTAL ≈ 304 s**, output `(1, 317, 480, 832, 3)`.

Read these first (in order):
- `wiki/WAN_S2V_BRINGUP.md` — bringup decisions, constraints, `_prepare_torch_state` contract violations.
- `wiki/WAN_S2V_PERF_CLEANUP.md` — the perf queue, what's landed, what's left. **The priority order in there applies to 4×8 too**, just with different shapes.

## Step 0 — Does `create_pipeline` accept (4, 8) yet?

```bash
grep -n "device_configs" models/tt_dit/pipelines/wan/pipeline_wan_s2v.py
```

The dict only has `(2, 4)` last time anyone checked. Add a `(4, 8)` entry mirroring
the 2×4 one (sp_factor=8, tp_factor=4) with `sp_axis=1, tp_axis=0`. If wrong,
``WanPipelineS2V.create_pipeline`` raises ``NotImplementedError`` immediately.

## Step 1 — Baseline run on 4×8

```bash
tt-smi -r 0,1,...,31
source python_env/bin/activate && export PYTHONPATH=$(pwd)
export S2V_REF_IMAGE=girl-hoodie-glasses.png S2V_AUDIO=warm_weird_22s.mp3 S2V_CLIPS=4 S2V_STEPS=5
pytest 'models/tt_dit/tests/models/wan2_2/s2v/test_pipeline_wan_s2v.py::test_pipeline_inference[blackhole-resolution_480p-bh_4x8sp1tp0]' -s --timeout=2400
```

Add a `bh_4x8sp1tp0` parametrize row to that test if it isn't there. Then capture
the per-stage timings from `wiki/WAN_S2V_PERF_CLEANUP.md`'s "Baseline timeline"
table format — measure `prepare_latents`, denoise, vae_decode, vae_motion_encode.

## Step 2 — Encoder conv3d sweep at (h=4, w=8)

Production VAE encoder uses `encoder_t_chunk_size=4`. The 2×4 entries in
`models/tt_dit/utils/conv3d.py` are for h=2, w=4 — they will fall back to
DEFAULT or NONE at 4×8. Same 13 unique shapes as the 2×4 sweep.

Mechanical steps:
1. Add `_SWEEP_LAYERS_H4W8_480P_ENC_T4` to `models/tt_dit/tests/models/wan2_2/bruteforce_conv3d_sweep.py` — copy the 2×4 list (search `_SWEEP_LAYERS_H2W4_480P_ENC_T4`) and change every `h=2, w=4` to `h=4, w=8`. Spatial H/W per-stage also halve along w (W: 832/8 = 104 vs 832/4 = 208). Verify the layer shapes against `compute_encoder_dims(480, 832, 4, 8, 4, temperal_downsample=[T,T,F])`.
2. Add a `test_bruteforce_sweep_h4w8_480p_enc_t4` pytest function.
3. Run one-at-a-time with reset between (the fixture re-init constraint that bit us at 2×4 is the same):
   ```bash
   /tmp/run_enc_sweep_h4w8.sh   # write a 2×4-style loop; ~10 min device time
   ```
4. Land the 12-13 winners as a new section in `conv3d.py`'s `_BLOCKINGS`. Expect ~30-50 % VAE encode speedup (the 2×4 sweep gave 33 s → 4 s on wav2vec2; encoder gains will be smaller in absolute time but proportional).

## Step 3 — DiT denoise matmul sweep (the big one — Item 1 in PERF_CLEANUP)

Denoise dominates total runtime (~70 % on 2×4). At 4×8 the per-device M shrinks
by 2× (more sp), so the matmul shapes change. Run a 4-clip 5-step
test with `models.tt_dit.utils.matmul:get_matmul_config` warnings enabled —
collect every `(M, K, N)` shape that prints "No known best blocking".

Then sweep each via `models/tt_dit/utils/matmul.py` (the registry path). The expected
(M, K, N) on 4×8 BH 480p:
- noisy per-dev: `ceil(31200 / (8 × 32)) × 32 = 31488 / 8 = 3936`
- const-with-motion per-dev: `ceil(3874 / (8 × 32)) × 32 / 8 = ?` — verify via Step-1 warnings.
- Total per-dev: `noisy_per_dev + const_per_dev`. Likely two clip variants like 2×4 (with motion / without).
- Self-attn `(M, 5120, 7680)`, output proj `(M, 5120, 1280)` (TP=4 at 4×8 vs TP=2 at 2×4), FFN `(M, 5120, 3456)` / `(M, 3456, 5120)`.

Expected savings: 2-4× on block stack → **100-150 s saved per 4-clip run**.

## Step 4 — Things that DON'T need re-tuning

- **AdaIN padding fix** — already production. Will Just Work at 4×8.
- **wav2vec2 feature-extractor chunking** — audio cost is small and mostly
  host-bound; carries over unchanged.
- **VAE motion-encode trace** — Item 2 in PERF_CLEANUP, deferred. Skip
  unless 4×8 perf reveals it as a hotspot.

## Step 5 — Verify reference-exact frame count

Output shape must be `(1, 317, 480, 832, 3)` for 4-clip on warm_weird_22s.mp3.
Anything else means a frame-count constant (`_INFER_FRAMES_PIXEL=80`,
`_LAT_TARGET_FRAMES=20`, `_S2V_VAE_CLIP0_TRIM=3` in `pipeline_wan_s2v.py:598-618`)
got perturbed during the (4, 8) port. Don't ship that.

## Out of scope

- `wiki/WAN_S2V_PERF_CLEANUP.md` Items 4b (motioner conv3d patchify), 6 (LANDED),
  7 (doc-only) — leave alone.
- Anything in `solvers/` — already matches main.
- Test files outside `tests/models/wan2_2/s2v/` — not yours.
