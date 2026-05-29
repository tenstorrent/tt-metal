# LTX-2 Audio Decoder — On-Device Port Plan

The audio decode path in `pipeline_ltx.py:decode_audio_reference` (lines
1706-1770) currently runs **two CPU models** in fp32 on every generation,
chained through the reference `AudioDecoder` block:

1. **Mel-VAE decoder** (`AudioDecoder` in `ltx_core.model.audio_vae.audio_vae`):
   audio latent `(B, 8, T_frames, mel_bins=64)` → mel-spectrogram-like
   features `(B, 2, T, mel_bins)`.
2. **Vocoder + BWE** (`VocoderWithBWE` in `ltx_core.model.audio_vae.vocoder`):
   mel `(B, 2, T, mel_bins)` → waveform `(B, 2, T_audio)` at 24-48 kHz, via
   BigVGAN-v2 with bandwidth-extension.

End-to-end on CPU fp32 this is about **2-5 s per generation** (single shot,
post-denoise). The pipeline currently has to free the transformer before
calling it (see the `audio_block = AudioDecoder(...)` instantiation at
`pipeline_ltx.py:1741`).

This doc maps the full audio-decode chain to TT primitives, calls out the
**three** real gaps in `models/tt_dit`, and proposes a **staged port** with
clear go/no-go points. Unlike `LATENT_UPSAMPLER_PORT.md`, this is *not* a
trivially reused VAE — the vocoder is `Conv1d`-dominated and `models/tt_dit`
has no `Conv1d` primitive today.

## TL;DR

| Component | On-device today? | Effort | Priority |
|---|---|---|---|
| Mel-VAE decoder | No — runs CPU | **Medium** — looks like an LTX VAE but in 2D and with self-attention | **Stage A** |
| Vocoder (BigVGAN v2 / AMP1) | No — runs CPU | **High** — ~100 sequential `Conv1d` ops, no `Conv1d` primitive in tt_dit | **Stage B** |
| BWE (mel-STFT + 2nd vocoder + sinc resampler) | No — runs CPU | **Medium** — Conv1d again, STFT bases are stored as weights | **Stage C** |
| Snake / SnakeBeta + anti-aliased Activation1d | N/A | **Medium-low** — three elementwise ops + two fixed-kernel `Conv1d`s | covered in B/C |

End-to-end ROI is **lower** than the latent upsampler: both are one-shot
per generation (the upsampler runs once between stage 1 and stage 2 in
`pipeline_ltx_fast.py:_upsample_latent_reference`, not per inner step), but
the upsampler costs ~6-12 s *plus* forces a transformer free/reload to make
host RAM available — total ~15-25 s of wall time on the critical path. The
audio decode is a flat ~2-5 s with no free/reload tax (the VAE is already
resident at that point). **Recommendation: do this *after*
`LATENT_UPSAMPLER_PORT.md` lands, and only the mel-VAE half if you want a
quick partial win** — leaving the vocoder on CPU only costs ~1-2 s.

## Reference architecture

The reference path from `pipeline_ltx.py:decode_audio_reference`:

```text
audio_latent (1, audio_N, 128)
  ↓ reshape (1, audio_N, 128) → (1, 8, audio_N, 16)   # unpatchify
  ↓ permute (B, 8, T, 16)                              # to 2D CHW-style
  ↓ AudioDecoder(...).forward                          # mel-VAE decoder
  │   AudioPatchifier.patchify + per_channel_statistics.un_normalize
  │   conv_in (CausalConv2d 8→512, k=3)
  │   mid: ResnetBlock × 2 with vanilla self-attention between
  │   up_blocks: 3 levels (ch_mult=(1,2,4), so 512→256→128)
  │     each level: 3× ResnetBlock + optional self-attn (curr_res ∈ {8,16,32})
  │                 + Upsample (nearest-2x + CausalConv2d, drops 1 frame)
  │   norm_out (GroupNorm / PixelNorm) → SiLU → conv_out (CausalConv2d → 2)
  │   AudioPatchifier.unpatchify with adjust_output_shape
  ↓ mel_spec (1, 2, T_mel, 64)
  ↓ VocoderWithBWE(mel_spec).forward                   # vocoder + BWE
  │   # All in autocast(dtype=fp32) — bf16 degrades spectral metrics 40-90%
  │   waveform_24k = self.vocoder(mel_spec.float()):
  │     conv_pre (Conv1d 128→1024, k=7)
  │     5× upsample stages:
  │       leaky_relu / Snake → ConvTranspose1d → mean(3× parallel ResBlock1)
  │     act_post (Activation1d(SnakeBeta) if AMP) → conv_post (Conv1d → 2, k=7)
  │     tanh
  │   mel = self._compute_mel(waveform_24k):
  │     MelSTFT.mel_spectrogram → causal-STFT (Conv1d with DFT-Hann basis)
  │     → matmul(mel_basis, magnitude) → log
  │   residual_48k = self.bwe_generator(mel)            # 2nd full vocoder
  │   skip_48k = self.resampler(waveform_24k)           # sinc-kernel Conv1d
  │   clamp(residual_48k + skip_48k, -1, 1)
  ↓ Audio(waveform, sampling_rate=48000)
```

There are **three** distinct sub-models with different building blocks:

- **Mel-VAE decoder** is 2D-conv-based. Looks structurally like
  `vae_ltx.py:LTXVideoDecoder` but in 2D, and with vanilla self-attention
  inside the mid-block and at certain resolutions.
- **Vocoder** is `Conv1d`-dominated. ~108 sequential `Conv1d` ops in the
  combined vocoder+BWE chain. Activations include Snake/SnakeBeta wrapped in
  the BigVGAN `Activation1d` (anti-aliased: `UpSample1d → snake →
  DownSample1d`, both with kaiser-sinc kernels).
- **MelSTFT** is a `Conv1d` with a precomputed DFT × Hann window basis, plus
  one matmul against a mel filterbank. Trivial once `Conv1d` exists.

## Component mapping

| Reference module | tt_dit primitive |
|---|---|
| `make_conv2d(...)` (regular Conv2d, non-causal) | **WAN Conv2d-via-Conv3d** pattern (`vae_wan2_1.py:WanConv2d`). Conv3d with kernel `(1, k_h, k_w)`. |
| `CausalConv2d` (asymmetric pad on causality axis) | Extend `vae_ltx.py:LTXCausalConv3d` to take a 2-axis padding spec — same `ttnn.concat`-based padding trick it uses for temporal, applied to whichever spatial axis is causal. |
| `GroupNorm(num_groups, C)` | `models/tt_dit/layers/normalization.py:GroupNorm` (existing). Needs a 4D `(B, H, W, C)` → 4D wrapper, same as `LATENT_UPSAMPLER_PORT.md` describes. |
| `PixelNorm` (per-(B,T,F)-RMS over channel) | `vae_ltx.py:LTXPixelNorm` (existing) — works on any rank tensor with channel as last dim. |
| `torch.nn.SiLU` | `ttnn.silu` |
| `AttnBlock` (vanilla self-attention via 1×1 Conv2d Q/K/V + reshape softmax) | Reduce to a single `ttnn.transformer.scaled_dot_product_attention` call after reshape + the Q/K/V `1×1` convs (just three matmuls). Pattern matches `models/tt_dit/encoders/clip/model_clip.py:160-487`. |
| `ResnetBlock` (GroupNorm/PixelNorm → SiLU → conv → ...) | **New** `LTXAudioResnetBlock` — similar to `LTXResnetBlock3D` but 2D and with either GroupNorm or PixelNorm. ≈60 lines. |
| `Upsample` (nearest-2x + CausalConv2d + 1-frame drop) | `ttnn.upsample(scale_factor=2, mode="nearest")` exists. Wrap with the causal-conv post-step. |
| `AudioPatchifier.{patchify,unpatchify}` + per-channel stats | Pure reshape + multiply/add — ~10 lines. Per-channel stats stored as `(1, C)` `Parameter` (same pattern as `vae_ltx.py:LTXVideoDecoder.per_channel_mean/std`). |
| `_adjust_output_shape` (crop + zero-pad) | Host post-processing — keep on host, return the device tensor and slice/pad there. |
| `torch.nn.Conv1d` (vocoder) | **No tt_dit primitive.** Cleanest: build a `Conv1dViaConv2d` wrapper around `ttnn.experimental.conv3d` with kernel `(1, k, 1)`. Same trick `WanConv2d` uses to dodge a missing native primitive. |
| `torch.nn.ConvTranspose1d` (vocoder upsample) | **No tt_dit primitive, no `conv_transpose3d` either.** Implement as `ttnn.upsample(scale_factor=stride, mode="nearest") + Conv1dViaConv2d(stride=1)`. Math differs from `ConvTranspose1d` but matches the BigVGAN-v2 design rationale ("upsample then smooth"); verify PCC. If PCC is poor, fall back to `interleave-zero + Conv1d` (exact equivalent, 2× weight memory). |
| `Snake` / `SnakeBeta` | Pure elementwise: `x + (1/(α+ε)) · sin(α·x)²`. ~5 ttnn ops. |
| `Activation1d` (`UpSample1d → activation → DownSample1d` w/ kaiser-sinc kernels) | `UpSample1d` and `DownSample1d` are `Conv_transpose1d` / depthwise `Conv1d` with a *fixed* kaiser-sinc kernel. Bake the kernels as device constants once at load time; reuse `Conv1dViaConv2d`. |
| `MelSTFT` (Conv1d with DFT-Hann basis + matmul against mel filterbank + log-clamp) | All trivial once `Conv1d` exists: one `Conv1dViaConv2d` + one `ttnn.matmul` + `ttnn.clamp` + `ttnn.log`. The DFT basis is **stored in the checkpoint as bf16** (per `MelSTFT` docstring); load it as a `Parameter`. |
| `UpSample1d` (resampler — kaiser/hann sinc Conv-transpose) | Same as the vocoder ConvTranspose1d replacement — `ttnn.upsample(mode="nearest", scale_factor=ratio) + Conv1dViaConv2d` with the precomputed sinc kernel. |
| `torch.tanh`, `torch.clamp` | `ttnn.tanh`, `ttnn.clamp` |
| `torch.bmm` (in attention) | `ttnn.matmul` |
| `torch.nn.functional.softmax` | `ttnn.softmax` |

The bold rows are the genuinely new primitives we need.

## What needs new code

### 1. `Conv1dViaConv2d` (or `Conv1dViaConv3d`) wrapper

Lives in `models/tt_dit/layers/audio_ops.py` (new). Wraps
`ttnn.experimental.conv3d` with kernel `(1, k, 1)` and stride `(1, s, 1)`,
operating on a tensor laid out as `(B, 1, T, 1, C)`. Padding (zero or
asymmetric for causal) handled the same way `LTXCausalConv3d` does for the
temporal axis: external `ttnn.concat` pads.

```python
class Conv1dViaConv2d(Module):
    """
    1D convolution implemented via ttnn.experimental.conv3d with degenerate axes.
    Input/output layout: (B, 1, T, 1, C) ROW_MAJOR.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding_mode: str = "zeros",      # "zeros" | "causal" | "replicate"
        causal_left_only: bool = False,    # True for causal: pad only at front
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,  # vocoder needs fp32 — HiFi4 path
    ) -> None:
        ...
```

Why `(1, k, 1)` over `(k, 1, 1)`: the existing `LTXCausalConv3d` already
treats the *first* spatial dim (T) the same way; reusing that pattern keeps
the blocking configs (`get_conv3d_config`) sensible. Whichever orientation is
ultimately picked, the *concept* — "abuse 3D conv with a degenerate axis" —
is identical to `WanConv2d`.

### 2. `LTXCausalConv2d` (or extend `LTXCausalConv3d` to 2-axis causal)

The mel-VAE decoder needs `CausalConv2d` with asymmetric padding on **one of
two** axes (`causality_axis="height"` per the LTX checkpoint). The cleanest
fit is to add a `causality_axis` arg to `LTXCausalConv3d` that says "which of
the 3 spatial dims gets asymmetric front-pad and which get symmetric." For
the audio case we'd run a Conv3d with kernel `(1, k_h, k_w)` where the height
axis is the causal one — wrap externally with `ttnn.concat` for the asymmetric
pad and pass `internal_padding=(0, 0, k_w // 2)` for the symmetric one.

### 3. `LTXAudioResnetBlock` + `LTXAudioAttnBlock`

In `models/tt_dit/models/audio_vae/audio_decoder_ltx.py` (new).

`LTXAudioResnetBlock` mirrors the reference (`audio_vae/resnet.py:ResnetBlock`):

```text
x → norm1 → SiLU → conv1 → norm2 → SiLU → dropout → conv2 → + shortcut
shortcut = x if in==out else nin_shortcut(x)  # 1×1 CausalConv2d
```

Differences from existing `LTXResnetBlock3D`: 2D not 3D, `GroupNorm` or
`PixelNorm` parameterized by `norm_type`, optional `nin_shortcut` (1×1 conv)
instead of layernorm-then-conv shortcut.

`LTXAudioAttnBlock` is the four-conv vanilla self-attention from
`audio_vae/attention.py:AttnBlock`:

```text
x → norm → q,k,v = 1×1 conv(x) → softmax(q·kᵀ/√C) → v·attn → 1×1 conv_out → + x
```

Implemented as: three `1×1 Conv2dViaConv3d`, then one
`ttnn.transformer.scaled_dot_product_attention` call with shape
`(B, 1, H·W, C)`. Pattern already proven in
`models/tt_dit/encoders/clip/model_clip.py:CLIPAttention`.

### 4. Snake / SnakeBeta + `Activation1d`

```python
class Snake(Module):
    """y = x + (1 / (α + ε)) · sin(α · x)²,  α: (C,) learned, log-scale optional."""

class SnakeBeta(Module):
    """y = x + (1 / (β + ε)) · sin(α · x)²,  α, β: (C,) learned."""

class Activation1d(Module):
    """
    Anti-aliased: UpSample1d(ratio=2) → activation → DownSample1d(ratio=2).
    The up/down filters are non-learnable kaiser-sinc kernels baked in at __init__.
    """
```

All ~50 lines of straightforward elementwise + fixed-kernel Conv1d.

### 5. Vocoder weight handling

The vocoder uses `weight_norm` on every Conv1d in the reference, but at
inference time the parametrization is collapsed (the saved checkpoint stores
`weight_g` × `weight_v / ||weight_v||`). Either:

- **(Recommended)** Apply the `weight_norm`-collapse on the host once during
  `_prepare_torch_state`, store the materialized weight as a regular
  `Parameter`. No runtime cost.
- Implement on-device `weight_norm` — overkill; the vocoder runs once per
  generation, and the savings vs upfront collapse are zero.

Same for the `Snake`/`SnakeBeta` `alpha_logscale` flag — collapse
`α = exp(α_log)` at load time and store `α` directly.

## Files & placement

```
models/tt_dit/layers/audio_ops.py                          # NEW: Conv1dViaConv2d, Snake/SnakeBeta, Activation1d
models/tt_dit/models/audio_vae/__init__.py                 # NEW
models/tt_dit/models/audio_vae/audio_decoder_ltx.py        # NEW: mel-VAE decoder (Stage A)
models/tt_dit/models/audio_vae/vocoder_ltx.py              # NEW: Vocoder + AMPBlock1 (Stage B)
models/tt_dit/models/audio_vae/bwe_ltx.py                  # NEW: VocoderWithBWE + MelSTFT + resampler (Stage C)
models/tt_dit/models/vae/vae_ltx.py                        # EDIT: add causality_axis arg to LTXCausalConv3d
models/tt_dit/pipelines/ltx/pipeline_ltx.py                # EDIT: decode_audio_reference → decode_audio_device
models/tt_dit/tests/models/ltx/test_audio_decoder.py       # NEW: PCC vs reference
wiki/AUDIO_DECODER_PORT.md                                 # this file
```

Rationale for `models/audio_vae/` rather than nesting under `models/vae/`:
the mel-VAE alone *is* a VAE-shaped network, but the chain is mel-VAE +
vocoder + BWE — three different model families. A sibling directory mirrors
the reference (`ltx_core/model/audio_vae/`) and keeps the WAN / SD3.5 / LTX
video VAEs in their existing `models/vae/` home.

## Staged rollout

The audio decode is **one-shot per generation** (post-denoise), so each
stage is independently shippable — the pipeline can call the on-device
version of one stage and the reference for the rest. Stage gating is a
single `if` in `decode_audio_reference`.

### Stage A — Mel-VAE decoder only (~1.5 s saved)

Smallest-scope, highest signal-to-noise: structurally similar to the existing
LTX video VAE decoder, just in 2D and with a self-attention block. Mostly
exercises the new `Conv1dViaConv2d` indirectly (via the 1×1 Conv2d in
attention) and the `LTXAudioResnetBlock` / `LTXAudioAttnBlock` primitives.

Decode endpoint changes from:

```python
audio_block = AudioDecoder(checkpoint_path=..., device=cpu, dtype=fp32)
mel_features = audio_block.decoder(audio_spatial)  # cpu fp32
```

to:

```python
mel_features_device = self.tt_audio_decoder(audio_latent_device)
mel_features = ttnn.to_torch(ttnn.get_device_tensors(mel_features_device)[0])
# Then call reference vocoder on host as before
```

Validation: PCC ≥ 0.998 on mel output vs reference fp32. Anything ≥ 0.99 is
likely inaudible after the vocoder washes spectral noise into waveform.

### Stage B — Vocoder (AMP1/SnakeBeta) (~3 s saved cumulative)

This is the **bulk** of the work. Highlights:

- Implement `Conv1dViaConv2d` and verify it on `vocoder.conv_pre`
  (Conv1d(128→1024, k=7)) before wiring up anything else.
- Implement `ConvTranspose1d` substitute via
  `ttnn.upsample(scale_factor=stride, mode="nearest") + Conv1dViaConv2d(k, s=1)`.
  Confirm PCC on a single layer before stacking 5 of them.
- `AMPBlock1` (`Activation1d(SnakeBeta) → Conv1d → Activation1d → Conv1d`) ×
  3 parallel branches × 5 stages, results mean-pooled. The mean is the
  parallelism: each branch is independent, so the natural mapping is
  per-branch on the mesh — but the activation peak is small enough that
  full replication is the simpler call.
- **Run in fp32 throughout.** Per the reference's comment at
  `vocoder.py:563`, "bfloat16 accumulation errors compound through 108
  sequential convolutions and degrade spectral metrics by 40-90%." Use
  `dtype=ttnn.float32` on every Conv1d Parameter; the HiFi4 path is
  automatic via `MATH_FIDELITY` in `layers/linear.py`.

Validation: PCC ≥ 0.99 on raw vocoder waveform (24 kHz). Mel L1 ≤ 0.5 dB.
MRSTFT ≤ 1.0 (matches the reference's "fp32 baseline ± 5%" target).

### Stage C — BWE + MelSTFT + resampler (~0.5-1 s saved cumulative)

The smallest piece, but the most numerically fragile (the entire reason
`VocoderWithBWE.forward` is wrapped in `autocast(fp32)`).

- `MelSTFT`: `Conv1dViaConv2d` (DFT×Hann basis) + `ttnn.matmul`
  (mel basis) + `ttnn.log(ttnn.clamp(...))`. The DFT basis is stored as bf16
  in the checkpoint (`vocoder.mel_stft.stft_fn.forward_basis`); load it
  directly into a `Parameter`. **Don't** recompute the DFT basis at
  init-time — that would lose the bit-exact training-time bases the BWE was
  trained against.
- `resampler` (`UpSample1d` with hann window): same nearest-then-Conv1d
  pattern as the vocoder, but with the hann sinc kernel baked in.
- `bwe_generator`: a second full `Vocoder` instance (reuse Stage B code).
- Forward wrapping: no autocast needed on device — every Conv1d is already
  fp32 weights + HiFi4 + fp32 dest acc.

Validation: end-to-end PSNR of 48 kHz waveform vs reference fp32 ≥ 30 dB
(same bar as the VAE-decode swap).

## Performance considerations

### Memory

| Component | Weight size (fp32) | Activation peak (fp32) |
|---|---|---|
| Mel-VAE decoder | ~40 MB | ~20 MB at 256-ch 64×256 |
| Vocoder | ~120 MB | ~80 MB at 32-ch 24000 samples |
| BWE generator (2nd vocoder) | ~120 MB | ~80 MB at 32-ch 48000 samples |
| MelSTFT + resampler | <1 MB | ~10 MB |
| **Total fp32 resident** | **~280 MB** | **~80 MB** |

At bf16 (matmul wins): ~140 MB resident — comfortable on BH-LB even
co-resident with the transformer. **No need to free the transformer**
before audio decode (the current pipeline does this purely because the CPU
torch fp32 path needs ~600 MB of host RAM, not because of device pressure).

### Parallelism

The audio decode is **one-shot per generation** and operates on tiny
tensors compared to the transformer. *Replicate weights across the mesh,
run the entire chain on one chip, broadcast the result.* CCL overhead would
dominate any TP/SP gain.

If we wanted to get fancy: the 3 parallel `AMPBlock1` branches per vocoder
stage are independently evaluable and could be mesh-sharded — but a single
BH chip swallows the whole branch trivially, so this is over-engineering.

### Math fidelity

**fp32 mandatory** end-to-end through the vocoder + BWE chain. The reference's
40-90% spectral-metric degradation under bf16 (see `VocoderWithBWE.forward`
docstring at `vocoder.py:553-573`) is *not* a parameter we can tune. The
prescription: every `Conv1dViaConv2d`, every elementwise op (Snake, sinc
filters, mel matmul) runs at `dtype=ttnn.float32` weights + HiFi4 +
`fp32_dest_acc_en=True` + `packer_l1_acc=True`. This is the exact same recipe
the gate fix uses (`wiki/ONDEVICE_GATE_FIX.md`).

For Stage A (mel-VAE decoder), bf16 is probably fine — the spectral
degradation is downstream in the vocoder, and the mel-VAE output is
relatively forgiving. Start bf16 to keep memory low; if PCC drops below
0.998, upgrade to fp32.

## Trade-offs

### Why this might *not* be worth doing right now

- **One-shot cost.** 2-5 s out of 60-90 s is a single-digit percentage win.
  The latent upsampler is also one-shot but a bigger lever (6-12 s of compute
  plus a forced transformer free/reload to free host RAM).
- **Numerical fragility.** The fp32 requirement means any sloppy
  intermediate cast nukes audio quality in ways that aren't visible until
  end-to-end PSNR / mel-L1 testing. High risk of "ship it, it sounds fine"
  → user reports buzzing on minute 12.
- **No `Conv1d` primitive in tt_dit today.** Stage B fundamentally requires
  building one (via Conv3d), with its own tuning to avoid regressing on the
  ~100 sequential convs.
- **`weight_norm` collapse** at load time means we need a custom
  `_prepare_torch_state` chain — slight cache-key churn.

### Why it might be worth doing later

- Removes the `LTX-2/packages/ltx-pipelines/` and `LTX-2/packages/ltx-core/`
  Python imports entirely from the runtime hot path of the AV pipeline
  (`LATENT_UPSAMPLER_PORT.md` removes them for the upsampler too — together
  they fully sever the runtime dependency).
- Removes the transformer-free / VAE-reload dance that complicates the
  pipeline.
- Unblocks future streaming audio modes (the vocoder is causal-aware).
- `Conv1dViaConv2d` is reusable across other audio-flavored models (Wan
  audio, future T2A models, etc.).

### Recommendation

1. **Land `LATENT_UPSAMPLER_PORT.md` first** — bigger win, smaller scope.
2. **Then ship Stage A (mel-VAE) by itself**, falling back to the host
   vocoder. ~1.5 s saved, validates the new `Conv2dViaConv3d` patterns end
   to end, low risk.
3. Stage B + C only if user-visible end-to-end latency is the next priority
   *and* the team is OK with ~1-2 engineer-weeks. Otherwise, leaving the
   vocoder on host indefinitely is a defensible call — it's a one-shot CPU
   cost and the audio reference path is rock solid today.

## Validation harness

`models/tt_dit/tests/models/ltx/test_audio_decoder.py` should follow the same
shape as the LTX-2 video VAE test:

1. Load the reference `AudioDecoder` + `Vocoder` on CPU fp32. Run on a fixed
   synthetic audio latent `(1, audio_N=2048, 128)`.
2. Construct the TT modules from the same state dict.
3. Run TT on device, gather to host, compare:
   - Mel features after AudioDecoder: PCC ≥ 0.998
   - Waveform after vocoder (Stage B): PCC ≥ 0.99, mel L1 ≤ 0.5 dB
   - Waveform after BWE (Stage C): PSNR ≥ 30 dB vs reference fp32

End-to-end test: a real generation, with audio PSNR ≥ 28 dB and no audible
artifacts in a 12 s clip.

## What gets simpler downstream

- `pipeline_ltx.py:decode_audio_reference` drops the `sys.path.insert` +
  `import torch.cuda.synchronize = lambda...` hack + the AudioDecoder block
  instantiation. ~50 lines deleted.
- The `Audio` reference type still comes from `ltx_core.types` but only at
  import time, not in any hot path.
- The transformer-free / VAE-reload sequencing in `pipeline_ltx_fast.py` and
  `pipeline_ltx_two_stages.py` can drop the "free transformer before audio
  decode" branch (the device upsampler already lets us keep the transformer
  alive; audio decode just needs the VAE alive, which it always was).

## Rollback

Each stage is gated by a single attribute in the pipeline (e.g.
`self.tt_audio_decoder is not None`). Stage failure falls back to the next
stage's CPU reference path:

- Stage A fail → use reference `AudioDecoder` + reference vocoder.
- Stage B fail → use TT `AudioDecoder` + reference vocoder.
- Stage C fail → use TT `AudioDecoder` + TT vocoder + reference BWE.

The reference path stays in the codebase indefinitely as the "known good"
fallback, the same way `LTXVideoDecoderTorch` is kept alongside the on-device
`LTXVideoDecoder` in `vae_ltx.py`.
