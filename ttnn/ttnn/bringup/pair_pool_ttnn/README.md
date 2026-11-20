# ttnn.bringup.pair_pool (pair_pool_ttnn)

## What it does
New op (no TTNN source; deepseek_v41_flash_d_p P.24, TINY_KERNELS). The softmax-gated pooling of row pairs of a
`[kv | gate]` tensor in one program:

    x [.., R, 2 hd] fp32 TILE  ->  out [.., R / 2, hd] fp32 TILE
    out[t] = kv[2t] w0 + kv[2t + 1] w1,   (w0, w1) = softmax(gate[2t], gate[2t + 1])   (per column)

The reference: DeepSeek-V4.1's ratio-2 compressor pooling (inference model.py: kv * softmax(score) summed over each
group of 2 tokens). Arithmetic: the op chain it replaces in the same fp32 steps with the stock ops' SFPU bodies (max,
subtract, the fp32 accurate exp, add, the fp32 SFPU divide, multiply, add), each step one fp32 rounding, results kept
in DEST between steps (no fused multiply-add): bit-identical to the chain. Work: units (output tile-row, column tile)
over the grid from common runtime args only (no per-core args); the reader reads a unit's 4 input tiles and splits
their rows by parity in L1 (word copies), the compute runs the SFPU steps, the writer writes one tile. Always
dispatched through ttnn.bringup.prepare_program / run_prepared (built once per shape, memory configs and grid; the
two addresses patched a call).

## Consider it instead of
- `ttnn.reshape` + 4 x `ttnn.slice` + `ttnn.maximum` + 2 x `ttnn.subtract` + 2 x `ttnn.exp` + `ttnn.add` +
  2 x `ttnn.divide` + 2 x `ttnn.multiply` + `ttnn.add` (a ratio-2 compressor's pair pooling; 16 launches).

## Options
- `output_tensor` (ttnn.Tensor, default None = the op allocates): the caller's buffer, exactly the output's spec
  (fp32 TILE interleaved [.., R / 2, hd]); returned as is.
- `memory_config` (default None = DRAM interleaved): the allocated output's memory config.
- `max_grid` (ttnn.CoreCoord or (x, y), default None = the whole compute grid): the cores inside (0, 0) ..
  (max_grid - 1) (a Galaxy's 110-core cap, agent rule 18).
Realistic (use by default): every option (the chain's own arithmetic).

## Measured
- 8x4 Galaxy (tests/unit/test_pair_pool.py): R 640 / 1280 / 64 / 192, hd 512 / 512 / 32 / 96: 0 ulp from the device op
  chain (5.2M values at R 640 hd 512), rel vs float64 5.3e-8. Device / host time: not measured by the agent (the
  orchestrator's A/B of deepseek_v41_flash_d_p P.24 TINY_KERNELS on / off).
- Cores: min(units, grid) = 110 a chip on the 8x4 at R 640, hd 512 (160 units), max_grid 11 x 10.

## Limits
- float32 TILE DRAM interleaved input; R % 64 == 0 (whole output tile rows), hd % 32 == 0; rank 2..4 (leading dims
  flattened with the rows: every 64 flattened rows one output tile row).
- The reader's parity split is RISC word copies (4096 words a unit): DM-bound for wide inputs; fine at the
  compressor's 640 x 1024.

## How to call
```python
pooled = ttnn.bringup.pair_pool(p, output_tensor=buf_or_None, max_grid=ttnn.CoreCoord(11, 10))
# models/demos/deepseek_v41_flash_d_p/tt/compressor.py TtCompressor._pool
```

## Used by
- deepseek_v41_flash_d_p: `tt/compressor.py` (TtCompressor._pool, setting TINY_KERNELS=on; the load-time output
  `fork.compressor.pool` from tt/host_alloc.reserve_forks), task P.24. Model cases: none yet (task O.1).

## History
CHANGELOG.md.
