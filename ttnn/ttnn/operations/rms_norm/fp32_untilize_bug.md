# fp32 Untilize Bug Investigation

## Symptom
rms_norm fails for fp32 + ROW_MAJOR + Wt>1. TILE layout and bfloat16 always pass.

## Root Cause (narrowed)
`fast_tilize_uninit` does not fully restore HW state when `compute_kernel_hw_startup` was called with a **bfloat16 srcB** CB. The leftover state corrupts `pack_untilize_block` for fp32+Wt>1, producing garbage in faces 2/3 (rows 16-31).

**All three conditions required:**
1. fp32 data format
2. Wt >= 2 (multi-tile width)
3. `compute_kernel_hw_startup(srcA=fp32, srcB=bfloat16, ocb)` — mixed-format init

## Error Pattern
- Rows 0-15 (faces 0,1): correct (max err ~1e-3)
- Rows 16-31 (faces 2,3): garbage (err up to 1e38)

## What Works
- `fast_tilize → pack_untilize` with srcB=fp32 (same format): **PASS**
- Regular `tilize → pack_untilize` with srcB=bfloat16: **PASS**
- `fast_tilize → pack_untilize` with bf16 data: **PASS**
- `fast_tilize → pack_untilize` with fp32 Wt=1: **PASS**
- `ttnn.untilize` standalone (separate kernel): **PASS**
- fp32 TILE path (no tilize/untilize): **PASS**

## Minimal Reproducer
- Kernel: `kernels/repro_fast_tilize_untilize.cpp` — toggle `USE_FAST_TILIZE`
- Test: `tests/ttnn/unit_tests/operations/rms_norm/test_fast_tilize_repro.py`
- Run: `scripts/tt-test.sh tests/ttnn/unit_tests/operations/rms_norm/test_fast_tilize_repro.py -v -s`

## Likely Suspect
`_llk_pack_fast_tilize_uninit_` restores `PCK_DEST_RD_CTRL_Read_32b_data` and calls `_llk_init_packer_dest_offset_registers_` + `_llk_pack_init_`. But when the initial HW config was set up with a bfloat16 srcB by `compute_kernel_hw_startup`, some register (dest offsets, ADCXX, or pack format config) is not properly restored for the fp32 pack_untilize path.

## Workaround
Use regular `tilize_init`/`tilize_block`/`tilize_uninit` instead of `fast_tilize_*` for the RM→TILE conversion. This avoids the HW state issue entirely.
