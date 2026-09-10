# Wan2.2 5B vs 14B — single BH Galaxy comparison (temporary)

> Scratch doc for the 5B bring-up vs 14B reference. Safe to delete later.
> All numbers measured on **one BH Galaxy** (`4x8` = 32 chips, `bh_4x8_ring`),
> **warm/steady-state traced** (2nd traced iteration), seed 42, non-distilled checkpoints.
> Prompt: *"Two anthropomorphic cats in comfy boxing gear and bright gloves fight
> intensely on a spotlighted stage."*

## Caveats (read first)
- **Frame counts differ right now:** 5B measured at **121 frames** (5 s @ 24 fps);
  14B at **81 frames** (5 s @ 16 fps). **Going forward we run 5B at 81 frames too**
  (matches 14B for apples-to-apples) until the VAE is optimized.
- **Mesh differs from the customer's 14B reference:** our 14B numbers are on a **single**
  Galaxy. The customer's ~38 s @ 720p/40-step figure was on a **quad** Galaxy (~4x this mesh).
- **`720p` resolution:** 5B uses `1280x704` (heights must be a multiple of 32); 14B uses `1280x720`.
- `denoise/step` = per-step transformer time (device). `VAE decode` = full decode for the clip.
  `E2E` = text-encode + denoise + VAE decode + host glue (host is <0.1%).

## Wan2.2-TI2V-5B (T2V) — 121 frames @ 24 fps — **VAE-bound**
| Resolution | Steps | E2E (warm) | Denoise/step | VAE decode |
|---|---|---|---|---|
| 480p (832x480) | 20 | 17.3 s | 196 ms | 13.3 s |
| 480p (832x480) | 40 | 21.2 s | 196 ms | 13.3 s |
| 720p-class (1280x704) | 20 | 34.9 s | 430 ms | 26.0 s |
| 720p-class (1280x704) | 40 | 43.3 s | 430 ms | 26.0 s |

## Wan2.2-T2V-A14B (14B, two experts) — 81 frames @ 16 fps — **denoise-bound**
| Resolution | Steps | E2E (warm) | Denoise/step | VAE decode |
|---|---|---|---|---|
| 480p (832x480) | 20 | ~21.8 s (derived) | 1.06 s | 0.27 s |
| 480p (832x480) | 40 | 42.95 s (measured) | 1.06 s | 0.27 s |
| 720p (1280x720) | 20 | ~73.9 s (derived) | 3.64 s | 0.60 s |
| 720p (1280x720) | 40 | 146.7 s (measured) | 3.64 s | 0.60 s |

*(14B 20-step rows derived from the measured, constant denoise/step; E2E = fixed(VAE+encode) + steps x denoise/step.)*

## Why the two models look opposite
- **5B is VAE-bound, 14B is denoise-bound.**
  - 5B: small transformer + high-compression Wan2.2-VAE (4x16x16, 48-ch latent) -> cheap denoise,
    **expensive VAE** (must upsample 16x in H/W).
  - 14B: two 14B transformers -> **expensive denoise**; mature Wan2.1-VAE (4x8x8) that is
    **H/W-sharded across all 32 chips** -> ~0.3-0.6 s decode.
- **The 5B VAE is slow because it is not parallelized yet** (biggest factor):
  - 14B builds the VAE with `VaeHWParallelConfig` (height/width sharded over the mesh).
  - 5B currently runs the VAE **serially + temporally chunked** (`vae_t_chunk_size=7`, to fit DRAM),
    with make-it-fit workarounds (ROW_MAJOR pixel-shuffle, `C_in_block<=256`, SDPA `k_chunk=128`)
    and **un-swept** conv3d/matmul blocking. All correct (chunk-vs-full-T PCC = 1.0), just un-tuned.

## Projected 5B after VAE optimization (VAE ~1 s, denoise unchanged)
| Resolution | Steps | Projected E2E | Today |
|---|---|---|---|
| 480p | 20 | ~5 s | 17.3 s |
| 480p | 40 | ~9 s | 21.2 s |
| 720p | 20 | ~10 s | 34.9 s |
| 720p | 40 | ~18-19 s | 43.3 s |

Optimizing the VAE flips 5B to **denoise-bound**; 720p/40 floors at the ~17 s denoise, and 480p
(the customer's actual target) drops to **~5-9 s** — well under their 2-5 min ask. VAE target is
low-single-digit seconds (0.5 s is an aggressive stretch given the heavier VAE + more frames).

## Optimization backlog (tracked)
1. **VAE OPT #1 (biggest win):** wire `VaeHWParallelConfig` for the 5B VAE (shard H/W across 32 chips).
2. **VAE OPT #2:** sweep/tune conv3d + matmul blocking tables for 5B VAE shapes.
3. **VAE OPT #3:** relax the make-it-fit workarounds once memory is parallel-split.
4. **VAE OPT #4:** re-benchmark E2E (at **81 frames**), target VAE decode <= ~1-2 s.

*(Separately: I2V wiring for 5B is still pending — per-token timestep AdaLN + VAE encoder.)*
