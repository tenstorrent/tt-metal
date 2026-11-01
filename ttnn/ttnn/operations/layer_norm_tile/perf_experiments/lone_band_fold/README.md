# lone_band_fold

Halve the compute of an R3 core's lone 16-row band: fold it into Wt/2 full tiles instead of computing Wt
tiles whose bottom half is stale L1. BH p100a (110 cores), base 9daa4c30ecf.

## Mechanism
- Reader / writer (residual on either RISC): source column 2j's band -> faces 0-1 of folded tile j, column 2j+1
  -> faces 2-3. Same half-tile reads/writes as the unfolded lone band; each chunk of C source columns fills C/2
  tiles. The row's CB slot stays Wt pages (producers push Wt/2 unwritten pad pages after the row, consumers pop
  them), so ring offsets, the work_a/work_b alias and every `row * Wt` index are unchanged.
- Stats: before the row collapse, `fold_row_halves` (SFPU) adds faces 0/1 and 2/3 element-wise and writes the
  sum to both halves (rounded to nearest under bf16 DEST: a plain SFPU store truncates). The unchanged
  collapse + column-0 finalize then give whole-row mean / inv_std in rows r and r+16. (v1 combined the two
  collapsed half sums instead: same speed, but 2x the mean error under bf16 DEST at |mean| = 100.)
- Shifted one-pass (balanced + DRAM residual): the folded row's shift tile copies face 0 over face 2.
- Affine: reader `scatter_folded` writes gamma/beta column 2j+1's row 0 into faces 2/3 of slot 2j (12 extra
  self-NoC reads per tensor, only on cores with a lone band). `folded_affine` (raw LLK) unpacks B with the NONE
  MOP (faces 0..3) and runs the ROW-broadcast math, so the bottom half broadcasts column 2j+1. The helpers'
  BroadcastDim::Row binds the ROW unpack MOP, which clears the B face counter per face row (re-reads faces 0/1).
- Gate (`_fold`): R3, Wt even, C even, stats window G == Wt or G/2 even. Odd Wt / odd C keep the unfolded path.

## Files
- `lbf_descriptor.py`, `kernels/` - op copies; `graduate.diff` - diff vs the op at base.
- `lbf_swap.py` - pytest plugin (`-p lbf_swap`): swaps the descriptor copy into the real op (also in
  sys.modules, so op tests that patch descriptor knobs patch the copy). `LBF_FOLD=0|1`, `LBF_KNOBS=K=v,...`.
- `test_lbf.py` + `bench.sh <tag>` - test_perf_bench rows, golden gate + PCC/rel-RMS, profiled.
- `test_lbf_domain.py` - 19 extra regimes (widths 64..1536, odd Wt, LoFi, TILE affine, bf16 DEST, |mean|=100,
  multi-block, skip bands). `test_lbf_decompose.py` - per-row mean-error decomposition.
- `parse.py`, `slow_core.py` - log / profiler readers. `logs/` - raw runs.

## Results (DEVICE KERNEL DURATION us, medians)
| row | base | fold | |
|---|---|---|---|
| focus1 inL1/resDRAM | 35.85 (4 runs) | 35.73 (3) | flat |
| focus2 allDRAM res | 48.14 (4) | 48.57 (3) | flat |
| focus3 allDRAM nores | 33.67 (4) | 33.65 (3) | flat |
| affineTILE_DRAM | 47.41 (4) | 48.40 (3) | flat (+2.1%, in band) |
| focus1 bf16 DEST | 35.47 (4) | 35.70 (3) | flat |
| allL1_res | 24.79 (3) | 22.31 (2) | -10% |
| L1_nores | 20.74 (3) | 19.07 (2) | -8% |
| 32x32 allDRAM res | 19.29 (2) | 17.13 (2) | -11% |
| 3x500 inL1/resDRAM | 19.05 (2) | 17.04 (2) | -10.5% |
| 2x512 allL1 res | 13.59 (3) | 10.94 (2) | -19.5% |
| 8x384, 64x17 | R1, no lone band | identical code path | n/a |

The DRAM-bound focus rows do not move: on the heavy cores the fold ends compute ~1 us (focus1) to ~4 us earlier,
but the critical path is the row-0 DRAM write (writer_store_block 23 -> 34 us on focus1). HALF_BAND_STREAM on
top of the fold did not cash it in (focus1 39.3 vs 38.7, focus3 36.9 vs 36.0 us, one run).

Precision: identical to the unfolded op on every row and domain case (fp32 and bf16 DEST, |mean| = 100).
