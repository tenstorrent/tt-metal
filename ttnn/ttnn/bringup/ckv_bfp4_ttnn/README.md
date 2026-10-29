# ttnn.bringup.ckv_bfp4 (ckv_bfp4_ttnn)

## What it does
New op (Python ProgramDescriptor, no TTNN source). bf16 TILE rows `[1, 1, n, 512]` -> 320-byte bfloat4_b row records
written in place into a row table `[1, 1, R, 320]` uint8 ROW_MAJOR at rows `[row0, row0 + n)`; optionally also the
rounded values as a new bf16 TILE tensor (`rounded=True`).
- Rounding: tt-metal's host packer rule (round to nearest even on the significand shifted to the group's max exponent,
  clamp at 7), bit-exact. The device packer alone rounds half away to bfloat8_b and then drops 4 bits (truncation in
  effect, row error 0.20 vs 0.11 on DeepSeek-V4.1's compressed KV), so an SFPU pass rounds onto the bfp4 grid first
  (`kernels/ckv_round_bfp4.hpp`) and the pack only copies bits.
- Row record: 32 shared exponents (one per 16 values), the same 32 again, then 256 bytes of sign-magnitude nibbles
  (value 2i in the low nibble of byte i). The second exponent copy lets a reader's 64 B-aligned NoC read put a row's
  exponents at either half of a 64 B line, so a gathered block of rows is a row-major bfloat4_b tensor the unpacker
  reads (sparse_sdpa ROW_BFP4).
- Program: bands of 32 rows split over the core grid; reader -> compute -> writer. Compute: a band's 16 tiles in a
  full-sync DEST, copy, SFPU round [, pack bf16], then the packer writes the band's 32 records itself: per record, two
  bfloat4_b pack streams over the tile row (strided DEST read as pack untilize does, with BH's DEST remap on), stream A
  at +0 with a 64 B exponent section, stream B at +32 with a 32 B section, both putting the values at +64. The writer
  only NoC-writes the 32 records (and the rounded tiles); no data-movement RISC touches the bytes.

## Options
- `rounded` (default False): also return the rounded values as a new bf16 TILE tensor `[1, 1, n, 512]` (else None).
- `memory_config` (default DRAM): the memory config of the `rounded` output.
- `prepared` (default False: the ProgramDescriptor rebuilt in Python and dispatched through generic_op every
  call, ~1.3 ms host on a 32-chip mesh): the program is built once per (mesh, n, rounded, memory configs) and
  dispatched with ttnn.bringup.run_prepared (prepared_program), row0 a value slot and every tensor address patched; the
  same program and bytes. Falls back to the generic_op path if a frozen runtime arg collides with an io address (the
  patch count says so).

## Measured
- One p150b: records byte-exact vs the host rule (tests/unit). 2x2 deepseek_v41_flash_d_p rung `last` (chunk 2048):
  0.04 ms a chunk vs 0.006-0.011 ms for the bf16 slice_write it replaces.
- Galaxy 8x4, deepseek_v41_flash_d_p FIX.8 (chunk 5120, 4 KV sources): device 0.209 ms a chunk (4 calls) vs 0.066 ms
  for the bf16 slice_write; HOST 1.33 ms a call (5.3 ms a chunk, of which 1.4 ms the Python `program()` descriptor
  build, the rest generic_op's dispatch of a fresh descriptor) vs 0.12 ms for slice_write. The model is host-bound, so
  this is the op's real cost there (DS41-135).
- prepared=True (P.7, 8x4, chunk 5120): host 1.43 / 1.31 -> 0.50 / 0.51 ms a call in the model (cProfile, two pairs;
  3.4-3.7 ms a chunk); isolated n=1280 0.514 -> 0.133 ms, n=64 0.168 -> 0.094 ms. Same device program.

## Consider it instead of
- `ttnn.typecast(x, ttnn.bfloat4_b)` / any op with a bfloat4_b output where round-to-nearest matters (they truncate in
  effect), followed by a tile -> row-record reshuffle.

## Limits
- A 16-value group whose max is below 2^-124 comes out 0 (the host rule would keep it).
- ckv must be bf16 TILE `[1, 1, n, 512]` with n % 32 == 0; rows uint8 ROW_MAJOR `[1, 1, R, 320]`, row0 + n <= R. On a
  mesh the same program runs on every chip (row0 is the same everywhere).
- `prepared=True`: the program cache is keyed on (mesh.id(), n, rounded, memory configs), LRU 64 entries; any other
  change (a new shape) builds and prepares a new program (~ms, once).

## How to call
```python
rows = ttnn.zeros([1, 1, R, 320], dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
rounded = ttnn.bringup.ckv_bfp4(ckv_bf16_tile, rows, row0, rounded=True)  # rows written in place
rounded = ttnn.bringup.ckv_bfp4(ckv_bf16_tile, rows, row0, rounded=True, prepared=True)  # built once, then patched
```

## Tests
`tests/unit/test_ckv_bfp4.py` (one device, ~10 s): records == `tests/reference.py` (host rule) byte for byte at n = 32 /
64 / 2560 / 5120, other rows untouched, rounded == the records' values, deterministic (A, B, A x5); prepared=True
over 5 calls (other row0, inputs, tables) byte-equal to the generic_op path, prepared once.
The device rounding facts: `models/demos/deepseek_v41_flash_d_p/tests/test_bfp_packer_rounding.py`.

## Used by
deepseek_v41_flash_d_p (CKV_FORMAT=bfp4_rows: the compressed-KV attention table; prepared=True under
CKV_BFP4_PREPARED=on, P.7).
