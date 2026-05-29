# LTX-2 Audio Decoder — T-Sharding Implementation Plan

Mirrors the `LTXCausalConv3d` H/W-sharding pattern from `vae_ltx.py` onto the
audio decoder + vocoder primitives so the audio decode chain actually uses the
mesh instead of replicating work on every chip.

## Motivation (data from 2026-05-28)

- `test_ltx_vocoder` (1×1 mesh, full warm forward) = **162s per single vocoder**.
- E2E pipeline `bh_8x4sp0tp1_linear` with `LTX_ON_DEVICE_AUDIO=1` (warm):
  `Audio decode: 404.2s` = `62s build + ~324s forward (2 vocoders × 162s)`.
- Per-chip cost is identical on 1×1 and 32-chip mesh → every chip does the
  same full work redundantly. T-sharding amortizes per-chip compute across the
  mesh (1/32 work per chip when fully sharded along axis 0+1, 1/8 when along
  axis 0 only) while halo exchanges replace what internal conv padding does
  today.
- Host audio baseline: 14.2s. Realistic target after full T-sharding: ≤ 20s
  on-device for the audio chain.

## Approach

Same pattern as `models/tt_dit/models/vae/vae_ltx.py:LTXCausalConv3d`:
- Drop internal conv padding on the sharded axis.
- Halo via `ccl_manager.neighbor_pad_persistent_buffer` with `dims=[1]`
  (T is dim 1 in BTC).
- Pad sizes: `(k-1, 0)` for causal, `(k//2, k//2)` for symmetric "zeros".
- Boundary chips get zero pad; interior chips exchange context with neighbors.
- Pure elementwise ops (Snake, SnakeBeta, mul, add) are shard-transparent.

## Primitives that need T-sharding

| Primitive | File | Status |
|---|---|---|
| `Conv1dViaConv3d` | `layers/audio_ops.py` | **Done** (this commit) |
| `_AlignedOutConv1d` | `models/audio_vae/vocoder_ltx.py` | TODO |
| `LTXDilatedConv1d` | `models/audio_vae/vocoder_ltx.py` | TODO |
| `LTXLowPassFilter1d` | `models/audio_vae/vocoder_ltx.py` | TODO |
| `LTXUpSample1d` | `models/audio_vae/vocoder_ltx.py` | TODO (slice/mul/accumulate loop — halo trickier) |
| `LTXDownSample1d` | `models/audio_vae/vocoder_ltx.py` | TODO |
| `LTXConvTranspose1d` | `models/audio_vae/vocoder_ltx.py` | TODO |
| `LTXHannUpSample1d` | `models/audio_vae/bwe_ltx.py` | TODO |
| `LTX_STFTFn` (mel STFT) | `models/audio_vae/bwe_ltx.py` | Stay replicated (host unfold + matmul); add gather at boundary. |

## Composition layers that need `parallel_config` plumbing

- `LTXVocoderActivation1d` (uses UpSample1d + activation + DownSample1d)
- `LTXAMPBlock1` (alternating SnakeBeta + Conv1d × 3 parallel branches)
- `LTXVocoder.__init__` (wires all of the above)
- `LTXVocoderWithBWE.__init__` (wires two vocoders + mel STFT + resampler)
- `LTXAudioDecoder` (Stage A mel-VAE) — uses `Conv2dViaConv3d`; can adopt the
  existing vae_ltx H/W sharding pattern directly. Out of scope for first cut
  (only ~30-50s of the 342s, fix vocoder first).

## Pipeline wiring (`models/tt_dit/pipelines/ltx/pipeline_ltx.py`)

`_build_tt_audio_decoder`: construct TT modules with `parallel_config` =
`ParallelFactor(factor=8, mesh_axis=0)` for an 8×4 Galaxy mesh (axis 0 has
size 8 — the SP axis). This shards T 8 ways, leaving the 4-wide TP axis
replicated. Reasonable first cut. Can refine to full 32-way (combined axes)
later if needed.

`decode_audio_device`:
1. Upload `audio_latent` to device as ROW_MAJOR (replicated).
2. `ttnn.mesh_partition` along T on `parallel_config.mesh_axis`.
3. Call `tt_audio_decoder(latent_sharded)` → returns mel sharded.
4. Call `tt_vocoder_with_bwe(mel_sharded)` → returns waveform sharded.
5. `ttnn.all_gather` along T on `parallel_config.mesh_axis` to reassemble.
6. `to_torch` from chip 0 → `Audio(waveform, sampling_rate=48000)`.

## Order of work (suggested)

1. [Done] Conv1dViaConv3d + `_t_neighbor_pad` helper.
2. Add a unit test for sharded Conv1dViaConv3d on an N-axis mesh (e.g., 1×8).
   PCC vs the unsharded reference output.
3. T-shard the resampling primitives (Up/Down/ConvTranspose/LowPass/Hann).
   Each has slice/mul/accumulate loops that need halo around the access window.
4. Thread `parallel_config` through `LTXVocoderActivation1d`,
   `LTXAMPBlock1`, `LTXVocoder`, `LTXVocoderWithBWE`.
5. Update `_build_tt_audio_decoder` + `decode_audio_device` to shard at the
   boundary.
6. Re-run unit tests on sharded mesh, then e2e on 8×4.

## Edge cases / gotchas

- **Stride > 1 + sharded = chip-position-aware crop.** Reference
  `LTXUpSample1d` (and `Down`/`ConvTranspose`/`HannUpSample`) chains
  `replicate-pad → zero-stuff → zero-pad → conv → crop`. The `crop` removes
  boundary effects ONLY at the global input boundary (left edge on rank 0,
  right edge on last rank). When sharded, interior chips' output already
  carries data that aligns with their neighbors' outputs — uniformly cropping
  on every chip would discard that data and leave gaps. Solving this requires
  either (a) per-chip-rank crop logic via a position tensor sharded across the
  mesh, (b) a different algorithm that avoids the global crop, or (c) the
  AllGather-then-uniform-crop approach (wasteful but correct). Default for
  this PR: `LTXLowPassFilter1d` T-shard supports `stride=1` only (the
  `Activation1d` path), and the resampling primitives stay unsharded for now.
  Need to revisit before the chain runs end-to-end sharded.
- Initial vocoder input mel: `(B, 128, T_in=64)`. T_in=64 split 8-way = 8 per
  chip. With conv_pre kernel=7, halo=6, per-chip work = 14 samples. Acceptable
  but the halo dominates compute at the early stages — the BIG wins are in the
  late upsample stages (T=10240 / 8 = 1280 per chip, halo negligible).
- `LTXAudioPatchifier` (Stage A mel-VAE input prep) is pure host-side reshape.
  No sharding work needed.
- `mel STFT` does a host-side `unfold` then a device matmul. Easiest to keep
  unsharded: `all_gather` the input waveform before mel STFT, `mesh_partition`
  the output back to sharded before feeding to bwe_generator.
- Snake / SnakeBeta α/β are `(1, 1, C)` per-channel — shard-transparent.
- For sharded outputs that need a host `to_torch`, AllGather on T first.

## Validation targets

- Per-primitive: PCC ≥ 0.998 vs the unsharded reference (random weights).
- Full vocoder PCC ≥ 0.99 on production config (matches current unsharded
  test_ltx_vocoder bar).
- End-to-end audio: PSNR ≥ 28 dB on the e2e pipeline output vs the host
  reference path (same bar as `AUDIO_DECODER_PORT.md` Stage C).
- Perf target: total `Audio decode` ≤ 30s warm on `bh_8x4sp0tp1_linear`.
