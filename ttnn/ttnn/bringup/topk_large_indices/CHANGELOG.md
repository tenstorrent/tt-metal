# topk_large_indices (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/topk_large_indices`
- Source SHA: `df92663a019f3d26e807fc780f73a6b78a5423ea`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.*`)
- Forked for: deepseek_v41_flash_d_p DS41-151
- Used by: deepseek_v41_flash_d_p (tt/indexer.py, INDEX_TOPK_OP)

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_topk_large_indices`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### Upstream's split LLK (#62035 / #62488 / #62411), with the fork's own LLK copy
- What: the op's code as of upstream f59dc7fe77d, 82a48165b8a, 47ca7fb46b6 and 416ff2dfd8e (not yet on this branch's
  base): fused macros at K' 1024 and 2048, the row split across MATH and PACK and across core pairs, `dram_read_noc`.
  The four top-k LLK headers those commits change are copied verbatim into `device/kernels/llk/` (`topk_xl.h` from
  tt_metal/hw/inc/api/compute/experimental, `llk_math_eltwise_unary_sfpu_topk_xl.h` from the Blackhole llk_api,
  `ckernel_sfpu_topk_xl.h` and the new `ckernel_sfpu_topk_xl_split.h` from tt-llk), clang-format off, their includes
  pointed at the copies; compute.cpp includes `llk/topk_xl.h`. No option: the fork has no other user.
- Why: DeepSeek-V4.1's candidate blocks' top-k (k 2048) 182 -> 138 us (2x2) and 279 -> 166 us (8x4) on one p150b.
- Needed by: deepseek_v41_flash_d_p DS41-151
- Files: device/ (all), topk_large_indices_nanobind.cpp, device/kernels/llk/*.h, CMakeLists.txt (kernels *.h)

### K' 512 rows past one fused segment on the split segmented body
- What: compute_body_mode takes the fused segmented body from K' 512 (rows past 32 chunks fell back to Classic);
  split_whole_rows refuses long rows only at K' 1024; the accumulator CB is made for K' 512; compute.cpp's
  reduce_split_segmented_row and split_segments take K 512; the LLK copy's `_topk_large_indices_move_segment_result_`
  is templated on K (result tiles 2 * ceil(K / 1024): K 512 tiles 0, 1 -> 2, 3). Default behaviour (no option).
- Why: DeepSeek-V4.1's index top-k (k 512 over the compressed keys): 512 x 27648 596 -> 337 us, 512 x 55296
  1190 -> 675 us, 160 x 262144 (8x4 at 1M) 2256 -> 954 us on one p150b.
- Needed by: deepseek_v41_flash_d_p DS41-151
- Files: device/topk_large_indices_program_factory.cpp, device/kernels/compute.cpp,
  device/kernels/llk/ckernel_sfpu_topk_xl.h, device/kernels/llk/llk_math_eltwise_unary_sfpu_topk_xl.h

### DeepSeek-V4.1 model cases
- What: tests/cases.py, reference.py, test_topk_large_indices.py: one random-input case per captured call (2x2,
  rung last: index top-k 512 over 27648 / 55296, candidate blocks 2048 over 6912), the values the ids point at vs
  torch's top-k position by position, ids distinct and in range, sentinels, determinism. 3 / 3 pass; one id
  corrupted by hand fails them.
- Needed by: deepseek_v41_flash_d_p DS41-151
- Files: tests/cases.py, tests/reference.py, tests/test_topk_large_indices.py

### Galaxy 8x4 sizes and the stock-op comparison (no op change)
- What: tests/unit/test_stock_identity.py runs fork and stock on the same input at DeepSeek-V4.1's 8x4 per-chip calls
  (160 rows: 28160 / 56320 keys k 512, 7040 blocks k 2048, and the 1M widths 522240 / 1044480 / 130560) and the 2x2's:
  the same values position by position (31 / 31 pass), ids differ only between tied bf16 keys (README "Ties"). The
  op needed no change at 8x4 sizes. tests/unit on one Galaxy p150b: 217 / 218 pass; the one other is upstream's
  production perf band (test_upstream_split.py prefill 0.848 ms, band 0.887-0.905: faster than the band).
- Why: P.13 asked for the 8x4 sizes and a fork-vs-stock selection check.
- Needed by: deepseek_v41_flash_d_p P.13 (galaxy 8x4)
- Files: tests/unit/test_stock_identity.py, README.md
