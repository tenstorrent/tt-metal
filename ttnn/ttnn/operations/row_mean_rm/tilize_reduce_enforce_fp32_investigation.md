# Tilize + Reduce enforce_fp32_accumulation Bug (WH B0)

## Summary

On Wormhole B0, calling `tilize` followed by `reduce` with `enforce_fp32_accumulation=true` in the same compute kernel produces incorrect results. The tilize operation corrupts hardware state that the reduce's D2B/B2D Hi/Lo16 transpose path depends on.

## Reproduction

The `row_mean_rm` operation does per tile-row: tilize RM→tiled, then reduce-row with `enforce_fp32_accumulation=true`.

```
compute_kernel_hw_startup(cb_input_rm, cb_scaler, cb_mean);
for each row:
    tilize(cb_input_rm → cb_input_tiled)       // tilize_init + tilize_block + tilize_uninit
    reduce<SUM, REDUCE_ROW>(cb_input_tiled → cb_mean)  // reduce_init + reduce_tile + reduce_uninit
```

Row 4 gets `0.0649` instead of `-0.1114` (and other rows have smaller errors).

## Isolation tests

| Configuration | Result |
|---|---|
| Tiled input → reduce (no tilize, no untilize) | PASS |
| Tiled input → reduce → untilize | PASS |
| Tilize → reduce (no untilize) | **FAIL** |
| Tilize → reduce → untilize | **FAIL** |
| Tilize → reduce with `enforce_fp32_accumulation=false` | PASS |
| `ttnn.mean` (PoolType::AVG, no tilize in kernel) | PASS |
| `ttnn.sum` (PoolType::SUM, no tilize in kernel) | PASS |

**Conclusion:** Tilize before reduce is the sole trigger, and only when `enforce_fp32_accumulation=true`.

## Root cause

`tilize_uninit()` on WH only cleans up UNPACK state (`llk_unpack_tilize_uninit`). It does **not** restore MATH or PACK state. The subsequent `reduce_init` (called inside the reduce helper) does reconfigure MOP, addrmod, and SrcA ALU format, but this is insufficient — three pieces of hardware state remain dirty.

The `enforce_fp32_accumulation` path in `_llk_math_reduce_` uses a delicate D2B/B2D Hi/Lo16 transpose that splits 32-bit dest values into two 16-bit halves, moves them to SrcB, transposes, and moves back. This path is extremely sensitive to:
- Correct unpack configuration (for transpose at unpack level)
- Correct dest sync state (to read/write the right dest half)
- Correct pack dest configuration (to read final 32-bit results)

Without `enforce_fp32_accumulation`, the reduce uses a simpler single-pass D2B/TRNSPSRCB/ELWADD path that tolerates the stale state.

## Fix: HW state restore after tilize

Adding three calls between tilize and reduce fixes the issue:

```cpp
// After tilize, before reduce
UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(cb_input, cb_scaler)));
MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>(cb_output)));
```

### Why each call is needed

| Call | Thread | Purpose |
|---|---|---|
| `llk_unpack_hw_configure` | UNPACK | Restores `THCON_SEC0_REG2` config register. `tilize_uninit` writes it from scratch zeroing fields beyond `out_data_format` and `throttle_mode`. Full hw_configure restores the complete register so reduce's haloize-mode transpose works correctly. |
| `llk_math_pack_sync_init` | MATH | Resets MATH↔PACK dest register synchronization. Tilize advances sync counters via `dest_section_done` calls per tile. After tilize, the sync state may point at the wrong dest half. With 32-bit dest (`DST_ACCUM_MODE=true`), the D2B/B2D transpose reads from a specific dest location — wrong sync state means wrong data. |
| `llk_pack_dest_init` | PACK | Resets packer dest access. Tilize uses `llk_pack<..., true/*tilize*/>` which configures tilize-mode layout. Reduce's `pack_tile` needs standard dest access to read 32-bit accumulated values after D2B/B2D. |

### Bisect results (remove one at a time from all 6 hw_startup calls)

| Removed call | Result | Verdict |
|---|---|---|
| `llk_unpack_hw_configure` | FAIL | **Required** |
| `llk_math_pack_sync_init` | FAIL | **Required** |
| `llk_math_hw_configure` | PASS | Not needed |
| `llk_pack_init` | PASS | Not needed |
| `llk_pack_hw_configure` | PASS | Not needed |
| `llk_pack_dest_init` | FAIL | **Required** |

## Broader implications

Any compute kernel that interleaves tilize with reduce (or likely any operation using `enforce_fp32_accumulation`) on WH needs to restore hardware state between them. This is a gap in `tilize_uninit` — it should clean up all three threads (UNPACK, MATH, PACK), not just UNPACK.

The proper long-term fix would be either:
1. Make `tilize_uninit` fully restore MATH and PACK state on WH
2. Document that `tilize_uninit` requires manual HW state restore before operations that use `enforce_fp32_accumulation`
3. Add a helper (e.g., `tilize_uninit_full`) that wraps the 3 required restore calls

## Files involved

- Compute kernel: `ttnn/ttnn/operations/row_mean_rm/kernels/row_mean_rm_compute.cpp`
- Tilize helper: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` / `.inl`
- Reduce helper: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` / `.inl`
- LLK tilize: `tt_metal/hw/inc/api/compute/tilize.h`
- LLK reduce: `tt_metal/hw/inc/api/compute/reduce.h`
- LLK math reduce (WH): `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_reduce.h`
- LLK unpack tilize (WH): `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_tilize.h`
