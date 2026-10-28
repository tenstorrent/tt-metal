# ckv_bfp4_ttnn changelog

Source: none (new op), deepseek_v41_flash_d_p 4-bit compressed KV (owner 2026-10-28, FIX.8 developed on the 2x2).

## Changes

### Initial op
- What: bf16 TILE rows -> 288 B bfloat4_b row records (SFPU round to nearest even before the pack), optional bf16
  rounded output.
- Why: the attention table of the compressed KV in 4 bits at the reference's width (F208); the device packer alone
  truncates bfp4 in effect (row error 0.20 vs 0.11).
- Needed by: deepseek_v41_flash_d_p FIX.8
- Files: ckv_bfp4.py, kernels/*, tests/*
