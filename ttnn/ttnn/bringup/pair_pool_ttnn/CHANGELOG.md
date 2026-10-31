# pair_pool_ttnn changelog

Source: none (new op), deepseek_v41_flash_d_p P.24 (TINY_KERNELS, owner 2026-10-31, opportunity #10).

## Changes

### Initial op (2026-10-31)
- What: `ttnn.bringup.pair_pool(x, output_tensor=None, memory_config=None, max_grid=None)`: [.., R, 2 hd] = [kv | gate]
  fp32 -> [.., R / 2, hd], the softmax-gated pair pool, one prepared program (common runtime args only), the op
  chain's fp32 steps with the stock SFPU bodies.
- Why: the compressor's pooling was 16 launches a layer (12 element-wise ops, 4 slices, a reshape) on 640 x 1024 fp32,
  ~2.4 ms of host for one subtract alone a chunk (opportunity report #10).
- Tests: tests/unit/test_pair_pool.py (8x4: 4 shapes 0 ulp from the device op chain, rel vs float64 <= 1e-6, negative
  controls; prepared program on fresh buffers, output buffer, determinism; refusals): 6 passed.
- Needed by: deepseek_v41_flash_d_p P.24
- Files: pair_pool.py, __init__.py, kernels/*, tests/unit/test_pair_pool.py, README.md; ../__init__.py (PYTHON_OPS)
