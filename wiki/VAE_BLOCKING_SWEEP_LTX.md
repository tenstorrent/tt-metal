# LTX-2 VAE conv3d blocking sweep

How to find optimal `(C_in_block, C_out_block, T_out_block, H_out_block, W_out_block)`
values for every conv3d layer in the LTX-2 VAE decoder, and how to install them
into `models/tt_dit/utils/conv3d.py:_BLOCKINGS` so they hit the exact-match path
(no fallback warnings, no leftover perf on the table).

The current commit ships hand-picked starter blockings derived from nearest-shape
Wan analogues. This doc is how to refine them.

## Goal

Replace every `[fallback]` warning emitted by `get_conv3d_config` on a VAE decode
run with an `[exact]` debug line. For LTX-2.3 22B at 121×512×768 there are
**10 unique conv3d shape classes**:

| C_in→C_out | kernel | T | H | W | Call count |
|---|---|---|---|---|---|
| 128 → 1024 | (3,3,3) | 16 | 16 | 24 | 1 |
| 1024 → 1024 | (3,3,3) | 16 | 16 | 24 | 4 |
| 1024 → 4096 | (3,3,3) | 16 | 16 | 24 | 1 |
| 512 → 512 | (3,3,3) | 31 | 32 | 48 | 4 |
| 512 → 4096 | (3,3,3) | 31 | 32 | 48 | 1 |
| 512 → 512 | (3,3,3) | 61 | 64 | 96 | 9 |
| 256 → 256 | (3,3,3) | 121 | 64 | 96 | 12 |
| 256 → 512 | (3,3,3) | 121 | 64 | 96 | 1 |
| 128 → 128 | (3,3,3) | 121 | 128 | 192 | 8 |
| 128 → 48 | (3,3,3) | 121 | 128 | 192 | 1 |

These come from walking `decoder_blocks` (reversed) and tracking BTHWC shape
through `conv_in → up_blocks → conv_out`, assuming `causal=False` (matches
LTX-2.3 22B JSON metadata: `causal_decoder=False`). Verify by grepping
`conv3d blocking [fallback]` in a recent run log — counts and shapes must match.

## Sweep tool

Use `models/tt_dit/tests/models/wan2_2/bruteforce_conv3d_sweep.py`. It already
implements the safety rules that prevented BH hangs in earlier Wan sweeps:

- `hw_product=32`: only sweep `H_block * W_block == 32` (e.g. (4,8), (8,4),
  (16,2), (2,16), (32,1)). Other products hung BH at 480p.
- `max_t_block=8`: `T_block ≥ 9` triggered hangs on large spatial tiles.
- L1 budget = `1,572,864 − 200 KB`, matches conv3d_program_factory CB sizing.

Run one shape at a time on a single chip (the LTX VAE is single-device, so the
1×1 mesh is the correct sweep target — don't sweep on (2,4) or (4,8)):

```bash
source python_env/bin/activate
export PYTHONPATH=$(pwd)
mkdir -p sweep_results_ltx_vae
pytest models/tt_dit/tests/models/wan2_2/bruteforce_conv3d_sweep.py \
    -k "1x1 and ltx_<layer_name>" -s --timeout=0 \
    | tee sweep_results_ltx_vae/<layer_name>.log
```

The script's `build_all_blockings` already enumerates the validity-filtered
combos and sorts them by L1-OOM-safe priority. Each survivor runs `WARMUP=2`
warmup + `RUNS=3` timed calls. The fastest combo per layer is your blocking.

The Wan sweep test is parametrized by layer name. To sweep an LTX shape, add a
parametrize entry in the test that constructs the right `(C_in, C_out, kernel,
T, H, W, h_factor=1, w_factor=1)` tuple — copy the pattern from the existing
Wan layer entries.

## Updating `_BLOCKINGS`

For each layer, append/replace the entry in `models/tt_dit/utils/conv3d.py`
under the LTX section:

```python
# ===================================================================
# LTX-2.3 22B VAE decoder (single device, h=1 w=1), output 121×512×768
# Sweep date YYYY-MM-DD, results in sweep_results_ltx_vae/.
# ===================================================================
(1, 1, 128, 1024, (3, 3, 3), 16, 16, 24): (Cin, Cout, T, H, W),  # conv_in — Xus
...
```

Include the kernel duration in the trailing comment so future debugging can
spot regressions.

## Cache invalidation

Cached VAE weights depend on `C_in_block` (used by `prepare_conv3d_weights` at
load time). When blockings change, the existing cache is wrong. Two ways to
handle:

1. **Preferred:** Use `conv3d_blocking_hash(self.vae_decoder)` in the VAE cache
   subfolder name, mirroring `pipeline_wan.py:512`:
   ```python
   subfolder = f"vae_{conv3d_blocking_hash(self.vae_decoder)}"
   ```
   When the hash changes the cache rebuilds automatically.

2. **Manual:** Delete the existing cache:
   ```bash
   rm -rf $TT_DIT_CACHE_DIR/<checkpoint>/vae
   ```

LTX-Fast already uses (1) post-this-PR.

## Validation

After installing swept blockings:

```bash
NO_PROMPT=1 SEED=46 pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_fast_av.py \
    -k "bh_2x4sp1tp0" -s --timeout 7200 2>&1 | grep "conv3d blocking"
```

You should see **only `[exact]` debug lines**, zero `[fallback]` warnings.
VAE decode portion of the e2e log should drop substantially.

## Cost vs. value

Each layer sweep takes ~5–15 minutes of device time depending on how many
combos survive the L1 + parallelism filters. 10 layers ≈ 1–3 hours on one BH
chip, fully unattended.

The current hand-picked blockings recover most of the win (~3-5× over the
fallback). The sweep typically finds another ~20-40% on top of that. Worth
running before shipping LTX-Fast to production; not worth running for
exploratory work.
