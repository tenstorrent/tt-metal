# g6_zeros changelog

## Phase 0 (implementer)

Built the design's single regime (bank-run zero write): one writer kernel (RISCV_0, NoC1), one
`cb_zero_source` of `Z` bytes zeroed once per core by NoC loopback, each core's share of one DRAM
bank written as single-packet writes from that source, one write barrier. No reader or compute
kernel (the design's kernel set). SUPPORTED: `dtype` float32, `layout` TILE.

### Deviation from the design's Phase 0 knob values (measured)

The design's Phase 0 values were `num_worker_cores` = full grid (110) with round-robin `c mod B`
placement and `Z = min(burst, largest run)`. It listed core count, placement and `Z` as perf lamps
to measure. Measured on Blackhole (8 DRAM banks, 11x10 grid); the defaults now take the winners.
All three stay live knobs (`Knobs` in `g6_zeros_program_descriptor.py`).

Device kernel duration, ns (one fresh run each; the p1 and p4 rows that decided `Z` were repeated
3x, spread < 1%):

| shape | W=8 dram_adjacent Z<=4K (default) | W=8 dram_adjacent Z<=16K | W=8 row_major | W=16 row_major | W=110 row_major (design Phase 0) |
|---|---|---|---|---|---|
| p1 (32831,1,32,256) bfp8, 285.8 MB | 559,491 (510.8 GB/s) | 559,603 | 902,311 | 959,613 | 1,106,365 |
| p2 (16,12,128,128) fp32, 12.6 MB | 25,384 | 25,693 | 40,706 | 43,405 | 47,645 |
| p3 (8,12,128,128) fp32, 6.3 MB | 13,099 | 13,421 | 20,830 | 22,319 | 24,129 |
| p4 (1,4,256,128) bf16, 0.26 MB | 1,425 | 1,721 | 1,853 | 2,165 | 1,707 |

Unprofiled host time per call (steady repetition, one sync; RB-053) agrees: p1 563 us at W=8
dram_adjacent vs 1112 us at W=110; p2 26.2 vs 49.2 us.

- Bound: the DRAM banks. p1 runs at 510.8 GB/s, at Blackhole's nominal 512 GB/s DRAM peak, so no
  data-movement headroom is left on the DRAM-bound shapes.
- More writers per bank is slower (W=8 < 16 < 110 row-major). Placement matters most: the writer
  nearest each bank on NoC1 (`device.get_optimal_dram_bank_to_logical_worker_assignment(NOC_1)`)
  is 1.6x faster than the same 8 writers in row 0, where their NoC paths cross.
- `Z` does not move the DRAM-bound shapes between 4096 and 16384. It is fixed cost on p4 (the
  zeroing runs before the first write), so the default caps it at 4096.
- The `dram_adjacent` group is 8 single-core ranges (two columns, 4 rows each), not one rectangle
  (BB-17). The 8 ranges cost no measurable kernel time on p4.

The design's bank-affine split, bank-major range interface, block schedule and kernel are
unchanged; only the host knob defaults moved.

### Kernel check for dtypes outside Phase 0

`test_g6_zeros_debug.py::test_dirty_destination_outside_phase0` runs p4 bfloat16 and p1 bfloat8_b
through the same program past `validate()`, over a poisoned range: every element reads back +0.0.
Adding those dtypes is a SUPPORTED-only change.
