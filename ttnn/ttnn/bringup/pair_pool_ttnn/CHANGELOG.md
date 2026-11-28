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

### Native C++ op (2026-11-28, deepseek_v41_flash_d_p NC.5)
- What: ttnn.bringup.pair_pool is now the native C++ device op ttnn.pair_pool (ttnn/cpp/ttnn/operations/pair_pool:
  device operation, program factory, nanobind; kernels moved to device/kernels with an args header). Same kernels,
  cores, runtime args and signature; the program cache keys on shapes / memory configs / grid (no addresses), a hit
  rewrites the two addresses. The pre-port Python route kept for tests only in generic/ (with its kernels).
- Why: OWNER 2026-11-27 C++ port series (bringup/NATIVE_PORT_RULES.md): the Python prepared program's wrapper, key
  and run_prepared patching cost host time a call.
- How: tt_ops_code_gen native_migrate (separate checkout, 27 min, ported; its acceptance suite in
  tests/ttnn/unit_tests/operations/pair_pool/pre_migration, golden tests/unit/test_pair_pool.py no worse).
- Measured (8x4, tests/native/test_native_lowbit.py, models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/results/NC.5_layer.md): bit-identical to the Python route, 0 cache growth on fresh
  buffers; host a call 73.3 -> 24.1 us (1024 rows), 60.2 -> 20.0 us (640); device equal (108.3 / 76.3 us).
- Files: pair_pool.py, kernels/ -> ttnn/cpp/ttnn/operations/pair_pool, generic/ (new, test only), README.md;
  ../INDEX.md; ttnn/cpp/ttnn-nanobind/__init__.cpp, ttnn/cpp/ttnn/operations/CMakeLists.txt.
