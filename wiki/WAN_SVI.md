# WAN SVI

## Goal
Reproduce Stable-Video-Infinity (SVI 2.0 Pro) output for the WAN 2.2-I2V-A14B
model on Blackhole hardware, matching upstream's Python reference
(`inference_svi_2.0_pro.py`) in this PR and matching the ComfyUI workflow in
a follow-up.

## Decisions
| Decision | Reason | Rejected Alternative |
|---|---|---|
| **LoRA stays in `experimental/`** (utils + pipeline + tests + docs all live under `experimental/`) | First attempt promoted LoRA to `pipelines/wan/`, but the multi-LoRA stacking + regex key parsing + `LoRASpec` API added surface area to the canonical pipeline that the user judged "not clean enough". Keeping it experimental keeps the iteration area contained. | Promote to first-class (rejected by user during bringup). |
| SVI itself stays in `experimental/` | Regime/scheduler bit-exactness story is unresolved (see below). | Make SVI first-class right away; risks locking down API before the ComfyUI path is proven. |
| **ComfyUI regime uses UniPC at `flow_shift=8`, NOT `dpm++_sde`** | tt-metal's device-side solver layer only has `UniPCSolver` + `EulerSolver`. `dpm++_sde + fixed` is stochastic with Karras sigmas and needs a new `Solver` subclass plus per-step `generator` plumbing. UniPC at the same step count / CFG / shift / LoRA stack produces flow-matching output that is functionally equivalent (deterministic instead of stochastic) — visually close but not bit-identical to ComfyUI. | Block ComfyUI enablement on `DPMSolverSDESolver` (delays the headline feature); approximate with UniPC at default shift (loses the flow-matching parameter). |
| Keep `WanPipeline.create_pipeline` `**extra_kwargs` passthrough + `boundary_ratio` param | Small generic fix that any subclass benefits from (SVI passes `svi_high`, `regime`, etc.). Not LoRA-specific; not what the user objected to. | Revert to the original signature (would force SVI's kwargs through env vars, same env-var dance the old LoRA pipeline had). |
| Anchor in pixel space, pre-fill `video_condition` before VAE encode | Reuses base TT VAE encoder unchanged; avoids running the encoder a second time on a single-frame anchor. | Encode anchor separately and tile in latent space (closer to upstream's `WanVideoUnit_ImageEmbedderVAE`, but doubles encoder calls and complicates parallel/sharding plumbing). |
| Static LoRA (CPU-side fusion) for SVI | Each SVI clip uses one fixed LoRA stack throughout. No mid-generation switching needed; zero per-step device cost. | Runtime device-side `LoRALinear` (needed for mid-clip switching or interactive scale tuning) — out of scope for SVI itself. |
| SVI uses `__call__` wrapper + `self._svi_*` instance state to thread `anchor_image` / `prev_last_latent` through base `__call__` | Base `__call__` is a 200-line method with no `**kwargs` forwarding; re-declaring would duplicate too much. | Modify base `__call__` to accept arbitrary `prepare_latents_kwargs` (cleaner but touches the base contract for SVI's sake). |

## Constraints & Workarounds
- **Hardware:** BH Loud Box 8-chip mesh (`bh_2x4sp1tp0` topology). BH Galaxy `bh_4x8sp1tp0_ring` also parameterized but not yet exercised.
- **Dtype:** bf16 throughout DiT; VAE in bf16/fp32 mix as per base I2V.
- **Compute regimes:** Both `python` (50-step UniPC, shift=5) and `comfyui` (6-step UniPC at `flow_shift=8`, LightX2V+SVI stack) implemented and verified end-to-end. Permanent fix for `comfyui` scheduler: build `DPMSolverSDESolver` in `models/tt_dit/solvers/`.
- **Anchor splicing is pixel-space, not latent-space.** Upstream tiles the anchor latent across all latent frames; we pre-fill pixel frames 1..N-1 and let the temporal VAE produce the conditioning. Equivalent semantically but not bit-equivalent.
- **Per-clip CPU fusion cost** (~25 s for 2-LoRA ComfyUI stack per expert × 2 experts; ~10 s for single-LoRA Python stack). One-time per pipeline instance; subsequent clips reuse the TT cache.
- **LoRA stack cache namespace** is SHA1 over ordered `(resolved_path, scale)` per expert. Switching stacks ⇒ cache miss ⇒ full transformer reupload.

## Surprises & Discoveries
- **One `_BLOCKINGS` entry can cause a localized per-clip artifact.** The chunk=7 `conv_out` blocking on BH 2x4 had `T_out_block=4` with `T_res=30`, leaving a 2-frame T-leftover. Combined with `C_out=3` padded to `C_out_block=32` it produced visible distortion at pixel frame 24-25 of every clip. Bisection via `test_wan_decoder_chunked_consistency` against chunk=1 baseline isolated the entry; switching to `T_out_block=1` fixed it. Lesson: the chunked-decode consistency test needs to be exercised against the actual production blockings, not the default fallback — the original test built `WanDecoder` with `height=width=0` which never hit `_BLOCKINGS`.
- **Upstream SVI Pro Python uses a plain Euler step on a flow-match schedule.** `FlowMatchScheduler.step` in diffsynth is `sample + model_output * (sigma_next - sigma_curr)` — identical to tt-metal's `EulerSolver`. We were running UniPC (solver-order-2 correction). Fixed by dispatching the solver on scheduler type in `pipeline_wan.py:300`.
- **tt-metal has its own device-side solver layer.** `_solver.step(step, latent, velocity_pred)` is *not* a diffusers `scheduler.step` call — it's a custom on-device stepper that only reads sigmas/alphas from the diffusers scheduler. Adding new schedulers (e.g. `DPMSolverSDEScheduler`) requires a matching `Solver` subclass in `models/tt_dit/solvers/`.
- **Stochastic schedulers need explicit `generator` plumbing.** Current `__call__` uses `seed: Optional[int]` and a single `torch.manual_seed(seed)` before `prepare_latents`. For stochastic SDE solvers (k-diffusion `dpm++_sde`) we'd need a per-step `torch.randn(generator=...)` plumbed through `solver.step`.
- **`WanPipeline.create_pipeline` hardcoded `boundary_ratio=0.875`** and dropped unknown kwargs, forcing the experimental LoRA pipeline to use env vars. Fixed in this PR by adding `**extra_kwargs` and making `boundary_ratio` a proper parameter.
- **`WanPipelineOutput` is a non-frozen dataclass** — we can `setattr` `last_latent` on it without subclassing.
- **`fuse_lora_state_dict` had a latent fp32-aliasing bug.** `base_weight.to(torch.float32).add_(delta)` returns the same tensor when `base_weight` is already fp32, mutating the source dict on the next stack pass. Fixed by switching to `base_weight.float() + delta` (always allocates).

## Open Questions
- [ ] Per-clip latency at 480p / 720p on BH 4x8 — needs measurement.
- [ ] PCC vs upstream Python reference — needs side-by-side run with identical seed/prompt/LoRA.
- [ ] Does the pixel-space anchor pre-fill produce visually equivalent output to upstream's latent-space tile? Subjective; needs side-by-side.
- [ ] Should `prev_last_latent` shape be validated more strictly? Currently we warn and skip on spatial mismatch.

## State
- [x] `WanPipeline.create_pipeline` `**extra_kwargs` passthrough + `boundary_ratio` parameter
- [x] `WanPipeline.__call__` `return_last_latent` kwarg
- [x] `experimental/utils/lora.py` (fusion utilities, `LoRASpec`, `fuse_lora_stack`, PEFT-style adapter key support)
- [x] `experimental/pipelines/pipeline_wan_lora.py` (`WanPipelineI2VLora` with multi-LoRA stacking, kwarg-based API)
- [x] `experimental/models/Wan2_2_LoRA.md`
- [x] `experimental/tests/test_pipeline_lora.py`
- [x] `experimental/pipelines/pipeline_wan_svi.py` (`WanPipelineSVI`, both regimes + `generate_long_video`)
- [x] `experimental/tests/test_pipeline_wan_svi.py`
- [x] `experimental/models/Wan2_2_SVI.md`
- [x] Smoke run on BH 2x4 with real SVI Pro LoRA — ComfyUI regime 2-clip generation 7:02 wall clock (~42 sec/clip)
- [x] **Twitches reported in first end-to-end output diagnosed:** distortion at pixel frame 24-25 in *every* clip (latent frame 6→7 boundary) traced to the `conv_out` blocking entry for chunk=7 on BH 2x4 (`T_out_block=4`, which left a 2-frame T-leftover interacting buggily with `C_out=3` padded to `C_out_block=32`). Fixed in-place by setting `T_out_block=1`. The other 6 stage blockings in the chunk=7 table were briefly disabled during bisection but had no visible impact and have been restored.
- [ ] **Switch SVI Python regime to Euler scheduler:** upstream's `FlowMatchScheduler.step` is a plain Euler step (`prev_sample = sample + model_output * (sigma_next - sigma_curr)`). We were running UniPC, which has solver-order-2 correction. The pipeline_wan.py dispatch now selects EulerSolver when given FlowMatchEulerDiscreteScheduler, and the SVI Python regime passes that scheduler with sigma_shift=5 (matches upstream `FlowMatchScheduler("Wan")`). Not yet tested end-to-end, but the dispatch is in.
- [ ] Bit-exact match vs ComfyUI: requires `DPMSolverSDESolver` follow-up.
- [ ] **Follow-up PR:** `DPMSolverSDESolver` (k-diffusion `sample_dpmpp_sde` + Karras `get_sigmas`); plumb per-step `generator` into `solver.step`; flip the ComfyUI scheduler from UniPC to dpm++_sde for true upstream match.

## Key Measurements

### BH-LB (2x4) Linear, 480p, prompt_image.png, "golden retriever" prompt, seed=0

| Regime | Steps | Per-clip latency | 2-clip wall (incl. setup) | Output |
|---|---|---|---|---|
| `python` | 50 | ~3:50 | 17:45 (first run, cold cache) | `wan_svi_python_2clips_832x480.{mp4,pt}` |
| `comfyui` (UniPC@shift=8) | 6 | ~42 sec | 7:02 (warm base cache; LoRA cold) | `wan_svi_comfyui_2clips_832x480.{mp4,pt}` |

Concat shape: `(158, 3, 480, 832)` = 2 × 81 − 1 × 4 (overlap). No NaN, value range `[0.002, 1.0]`, per-frame std ~0.2.

LoRA fusion (one-time, both experts):
- Python regime stack (SVI only): ~10 s/expert.
- ComfyUI regime stack (LightX2V + SVI): ~25 s/expert (loads + fuses 405 low-rank + 649 direct deltas + 400 low-rank).

Reproduce with:
```bash
export TT_DIT_CACHE_DIR=/home/kevinmi/.cache
export TT_DIT_ALLOW_HF_DOWNLOAD=1
export SVI_HIGH_PATH=/home/kevinmi/.cache/svi/version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors
export SVI_LOW_PATH=/home/kevinmi/.cache/svi/version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors
export LIGHTX2V_HIGH_PATH=/home/kevinmi/.cache/lightx2v/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors
export LIGHTX2V_LOW_PATH=/home/kevinmi/.cache/lightx2v/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors
export PROMPT_IMAGE=$TT_METAL_HOME/prompt_image.png

# ComfyUI regime (fast):
pytest models/tt_dit/experimental/tests/test_pipeline_wan_svi.py \
  -v -k "bh_2x4sp1tp0 and resolution_480p and comfyui" --timeout 3600 -s

# Python regime (slow):
pytest models/tt_dit/experimental/tests/test_pipeline_wan_svi.py \
  -v -k "bh_2x4sp1tp0 and resolution_480p and python" --timeout 7200 -s
```
