# Fix: fused scale-mask softmax tile-padding leakage at non-32 widths

## Root cause

The fused scale-mask softmax kernel processes the attention scores in 32x32 tiles.
When the sequence width `W` is not a multiple of 32, the final tile is zero-padded
to 32 columns. The mask is applied as `scores + mask` before the row-max reduction.

The bug: the pad columns are filled with `0.0` in the scores buffer, but the mask
buffer for those columns is also `0.0` (additive identity) rather than `-inf`.
So the padded columns contribute `exp(0 - max) = exp(-max)` to the denominator of
the softmax instead of `0`. At small `max` (early rows, or heavily masked rows)
this leaks a non-trivial probability mass into the padding, corrupting every
output column of that row.

At `W` a multiple of 32 the problem is invisible because there are no pad columns.

## Reproduction

```python
import torch, ttnn
x = torch.randn(1, 1, 1, 40)          # width 40 -> 1 full tile + 8 real + 24 pad
mask = torch.zeros(1, 1, 1, 40)
ref = torch.softmax(x, dim=-1)
out = ttnn.transformer.fused_scale_mask_softmax(x, mask, scale=1.0)
assert torch.allclose(out[..., :40], ref, atol=1e-5)   # FAILS: rows drift by ~1e-2
```

## Fix

In `fused_scale_mask_softmax.cpp`, the pad region of the mask must be set to
`-inf` (or the dtype minimum) so `exp(-inf - max) == 0` and the pad columns
contribute exactly zero to the denominator. Two changes:

1. **Mask fill:** when materializing the per-tile mask, write `-inf` into every
   column index `>= W` instead of `0.0`. This matches the semantics already used
   by the non-fused `ttnn.softmax` path, which masks before reduction.

2. **Denominator guard:** after the row-sum reduction, clamp the denominator with
   `max(sum, FLT_MIN)` so a fully-masked row (all `-inf`) yields `0` rather than
   `NaN` from `0/0`. This preserves the existing all-masked-row contract.

```cpp
// fused_scale_mask_softmax.cpp — mask materialization
for (uint32_t col = 0; col < TILE_WIDTH; ++col) {
    const uint32_t global_col = tile_col * TILE_WIDTH + col;
    mask_tile[col] = (global_col < W) ? mask_in[global_col]
                                      : -std::numeric_limits<float>::infinity();
}

// denominator guard after row-sum
float denom = row_sum;
denom = denom > FLT_MIN ? denom : FLT_MIN;   // fully-masked row -> 0 output, no NaN
```

## Verification

- `pytest tests/ttnn/unit_tests/operations/transformer/test_fused_scale_mask_softmax.py`
  with widths `{33, 40, 63, 65, 96, 127}` — all now match the PyTorch reference
  to `atol=1e-5`.
- Width `32` and all multiples of 32 unchanged (no pad columns -> no behavior change).
- Fully-masked rows still return all zeros (no `NaN` regression).

## Files changed

- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/fused_scale_mask_softmax.cpp`
- `tests/ttnn/unit_tests/operations/transformer/test_fused_scale_mask_softmax.py`
