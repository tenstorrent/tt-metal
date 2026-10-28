# ttnn.bringup.ckv_bfp4 (ckv_bfp4_ttnn)

## What it does
New op (Python ProgramDescriptor, no TTNN source). bf16 TILE rows `[1, 1, n, 512]` -> 288-byte bfloat4_b row records
written in place into a row table `[1, 1, R, 288]` uint8 ROW_MAJOR at rows `[row0, row0 + n)`; optionally also the
rounded values as a new bf16 TILE tensor (`rounded=True`).
- Rounding: tt-metal's host packer rule (round to nearest even on the significand shifted to the group's max exponent,
  clamp at 7), bit-exact. The device packer alone rounds half away to bfloat8_b and then drops 4 bits (truncation in
  effect, row error 0.20 vs 0.11 on DeepSeek-V4.1's compressed KV), so an SFPU pass rounds onto the bfp4 grid first
  (`kernels/ckv_round_bfp4.hpp`) and the pack only copies bits.
- Row record: 32 shared exponents (one per 16 values), then 256 bytes of sign-magnitude nibbles (value 2i in the low
  nibble of byte i) = the bytes of a bfloat4_b tile row; 32 records turn back into 16 bfloat4_b tiles by a byte shuffle.
- Program: bands of 32 rows split over the core grid; reader -> compute (copy, SFPU round, pack bfp4 [+ bf16]) ->
  writer (each tile's bytes scattered into the band's 32 records in L1, then 32 page writes).
- Limit: a 16-value group whose max is below 2^-124 comes out 0 (the host rule would keep it).

## Consider it instead of
- `ttnn.typecast(x, ttnn.bfloat4_b)` / any op with a bfloat4_b output where round-to-nearest matters (they truncate in
  effect), followed by a tile -> row-record reshuffle.

## Sample
```python
rows = ttnn.zeros([1, 1, R, 288], dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
rounded = ttnn.bringup.ckv_bfp4(ckv_bf16_tile, rows, row0, rounded=True)  # rows written in place
```

## Tests
`tests/unit/test_ckv_bfp4.py` (one device, ~10 s): records == `tests/reference.py` (host rule) byte for byte at n = 32 /
64 / 2560 / 5120, other rows untouched, rounded == the records' values, deterministic (A, B, A x5).
The device rounding facts: `models/demos/deepseek_v41_flash_d_p/tests/test_bfp_packer_rounding.py`.

## Used by
deepseek_v41_flash_d_p (CKV_FORMAT=bfp4_rows: the compressed-KV attention table).
