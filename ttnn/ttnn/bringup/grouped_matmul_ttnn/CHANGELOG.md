# grouped_matmul_ttnn changelog

Source: none (new op), deepseek_v41_flash_d_p P.81 (OWNER 2026-11-18: attention at its roofline; the grouped wo_a).

## Changes

### Initial op (2026-11-19)
- What: `ttnn.bringup.grouped_matmul(x, w, dtype=None, compute_kernel_config=None, output_tensor=None,
  memory_config=None, max_grid=None, block_w=None, depth=2, rt=1)`: y[g] = x[g] @ w[g] for x [1, G, M, K], w
  [1, G, K, N] (TILE, DRAM interleaved), one grid row a group, GX columns (N split as evenly as possible); the row's
  column-0 core reads x's [M x block_w] K blocks and multicasts them along the row (READY / VALID semaphores, bounded
  waits); each core reads its own [block_w x PN] weight blocks; fp32 DEST per block, Float32 partials accumulated by the
  packer in L1, one copy + pack to the output dtype at the end. `supported(...)`: whether a call has a layout.
- Why: the stock 2D multicast batched matmul splits only M over the grid rows: at M = 160 it runs on 5 x 11 = 55 cores,
  compute-bound (0.175 ms for deepseek_v41_flash_d_p's wo_a vs ~0.1 ms of DRAM). Here 8 x 11 = 88 cores.
- Measured (8x4, real-time profiler, slowest chip, 20 iterations; models/demos/deepseek_v41_flash_d_p/tests/perf/
  test_attn_proj_probe.py): wo_a 8 x [160, 4096] @ [4096, 1024] bfloat8_b: stock 0.175 ms; block_w 32 0.150, 16 0.144,
  8 0.140, 4 0.133 (depth 2 and 4 alike); rt 2 (full-sync DEST) 0.135. A reader issuing depth - 1 blocks ahead under
  NoC transaction ids: 0.143-0.155 (slower, not kept). 42.6 MB moved a call: ~320 GB/s at 0.133 ms.
- Tests: tests/unit/test_grouped_matmul.py (5 cases: the 8x4's wo_a at two block widths, one group, a 4 x 3 grid with
  N not divisible by the columns, a 1 x 2 grid; vs the stock batched ttnn.matmul (bit-identical on all) and float64;
  every chip equal; a cache hit on fresh buffers into a caller's output: the same bytes; refusals). Run on the model's
  mesh through the resident runner: models/demos/deepseek_v41_flash_d_p/tests/test_grouped_matmul_wo_a.py: passed.
- Needed by: deepseek_v41_flash_d_p P.81 (ATTN_ROOFLINE)
- Files: grouped_matmul.py, __init__.py, kernels/gmm_reader.cpp, kernels/gmm_compute.cpp, kernels/gmm_writer.cpp,
  tests/unit/test_grouped_matmul.py, README.md; ../__init__.py (PYTHON_OPS), ../INDEX.md
