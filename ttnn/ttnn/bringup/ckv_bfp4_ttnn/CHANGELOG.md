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

### Option prepared: the program built once, row0 patched (2026-10-29)
- What: `ckv_bfp4(..., prepared=True)`: one PreparedProgram per (mesh id, n, rounded, memory configs) (LRU of 64),
  row0 the writer's value slot (prepared_program option value_slots), ckv / rows / out addresses patched; default False
  keeps the per-call descriptor through generic_op.
- Why: DS41-135: 1.33 ms host a call on the 8x4 (4 calls a chunk) in a host-bound model.
- Needed by: deepseek_v41_flash_d_p P.7 (CKV_BFP4_PREPARED)
- Files: ckv_bfp4.py, tests/unit/test_ckv_bfp4.py, README.md
- Tests: tests/unit 8 passed (5 before + 3 prepared); sdpa tests/unit/test_sparse_sdpa_row_bfp4.py passed.

### bfloat8_b records, bfp_round, a shorter SFPU round
- What: `ckv_bfp4(dtype=ttnn.bfloat8_b)` (default bfloat4_b, unchanged program apart from the round below) writes
  576 B bfloat8_b records; new `ttnn.bringup.bfp_round` (bf16 TILE -> bfloat4_b / bfloat8_b TILE, new or in place at a
  tile row); the SFPU round templated on the magnitude bits and shortened to an add / subtract of 2^23 x the group's
  LSB (round to nearest even by the SFPU adder) instead of integer shifts.
- Why: deepseek_v41_flash_d_p lowbit (window KV at fp8 width, index keys / query at fp4 width; owner 2026-10-29).
  The shorter round: bfp_round of 512 keys 32 -> 2.1 us (also every core), an index query 39 -> 16-28 us, the window
  records 38 -> 28 us, ckv_bfp4 ~34 -> 24-27 us a call.
- Exactness: byte / value equal to the host rule (tests/unit 23 pass; test_bfp_packer_rounding sfpu cases 4 pass).
- Needed by: deepseek_v41_flash_d_p DS41-150 (lowbit phase 1)
- Files: ckv_bfp4.py, __init__.py, kernels/ckv_round_bfp4.hpp, kernels/ckv_bfp4_compute.cpp, kernels/bfp_round_*.cpp,
  tests/reference.py, tests/unit/*

### ckv_bfp4 blocks=True: bfloat8_b window blocks of 32 rows a page
- What: `ckv_bfp4(..., dtype=bfloat8_b, blocks=True)` writes [1, 1, R / 32, 18496] pages: a band's 32 rows'
  exponents at 0, again at 1056, their values at 2112 (sparse_sdpa's window blocks). The packer writes a band as one
  stream (two passes, as the records); the writer one NoC write a band. Default (blocks=False) unchanged.
- Why: sparse_sdpa reads a consecutive window in two reads a block instead of two a row (see the sdpa CHANGELOG).
- Exactness: byte-equal to the host rule on the written bytes (tests/unit/test_ckv_bfp4.py::test_ckv_bfp8_blocks),
  other pages untouched, deterministic. Time: 28 us for 2176 rows, as the records.
- Needed by: deepseek_v41_flash_d_p DS41-150 (lowbit)
- Files: ckv_bfp4.py, kernels/ckv_bfp4_{compute,writer}.cpp, tests/reference.py, tests/unit/test_ckv_bfp4.py

### bfp_round prepared=True
- What: `bfp_round(..., prepared=True)` builds the round program once per (mesh, tiles, dtype, memory configs) and
  dispatches it by ttnn.bringup.run_prepared, the table's first page a value slot (as ckv_bfp4's row0). The writer
  takes the core's first tile and the table's first page as two runtime args (was one sum), so the page is one value
  for every core. Default (prepared=False) the same program apart from that argument split; same bytes.
- Why: deepseek_v41_flash_d_p lowbit phase 2 rounds the index keys and query every chunk (INDEX_K_FORMAT /
  INDEX_Q_FORMAT=bfp4): host time per call (DS41-135, P.7's pattern).
- Exactness: tests/unit/test_bfp_round.py::test_bfp_round_prepared (new outputs and in-place writes at other rows /
  tables equal generic_op byte for byte, prepared once); tests/unit 19 pass.
- Needed by: deepseek_v41_flash_d_p DS41-150 (lowbit phase 2)
- Files: ckv_bfp4.py, kernels/bfp_round_writer.cpp, tests/unit/test_bfp_round.py
