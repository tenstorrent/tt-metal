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

### The packer writes the row records (320 B, exponents twice)
- What: records grow to 320 B (`[32 exponents | the same 32 | 256 nibble bytes]`) and the packer writes them straight
  from DEST (two bfloat4_b streams per record, strided DEST reads, BH DEST remap); the writer's byte scatter is gone.
  Full-sync DEST (a band's 16 tiles at once), bfp8_pack_precise (the packer reads DEST as bf16).
- Why: owner rule, the data-movement RISCs do no byte work; the second exponent copy makes a gathered block of rows a
  row-major bfloat4_b tensor for the reader's unpacker (BH DRAM reads need 64 B congruence; one copy at 32 B failed).
  Byte-exact vs the host rule (tests/unit, 5 pass; test_bfp_packer_rounding sfpu cases, 4 pass).
- Gotcha: without `llk_math_reconfig_remap(true)` the strided read's second interface reads DEST 8 rows early on BH.
- Needed by: deepseek_v41_flash_d_p FIX.8
- Files: ckv_bfp4.py, kernels/ckv_bfp4_compute.cpp, kernels/ckv_bfp4_writer.cpp, tests/*
