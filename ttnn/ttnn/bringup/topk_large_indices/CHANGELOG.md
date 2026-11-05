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

### Upstream sync to source @ `107f838f379` (2026-11-05): full sync, is_allocated(), the split LLK race fix (#62675)
- What: the fork already carried #62035 / #62488 / #62435 / #62411 (`416ff2dfd8e`), which are now upstream; so the
  base is fork_op.py's mechanical copy of the source @ `416ff2dfd8e` (= the code the fork carried), upstream the same
  copy @ `107f838f379`, ours this fork. For the fork's LLK copies (`device/kernels/llk/`) the same 3-way merge of each
  upstream header @ `416ff2dfd8e` -> `107f838f379`. Everything merged clean, no conflicts. Taken:
  - `19da1202a49` #62567: `buffer() != nullptr` -> `is_allocated()` in validation (device/topk_large_indices_device_operation.cpp);
  - `19443882604` #62675 (bug fix, #62561) into `llk/ckernel_sfpu_topk_xl_split.h`: in the MATH / PACK row split, the
    stage-6 merge of chunk c - 1 stores into the survivor bank that MATH's next step (chunk c's stage-2 transposes, or
    chunk 2's copy after the paired head) works in; the merge now takes MATH's next token first (`take_next`), so a late
    MATH can no longer race it (upstream: 12 wrong K' 512 cases under TT-NOP delays; K' 1024 with longer delays);
  - `ab31699e992` #60881: comment only in `llk/topk_xl.h` (`copy_tile_init` -> `copy_init`).
  The fork's K' 512 segmented-body extension (templated `_topk_large_indices_move_segment_result_`, compute.cpp's
  segmented path) merged clean; it goes through the same `pack_slot` / `sfpu_step` paths the fix changes.
- Behaviour: a correctness fix of a timing race (results change only where MATH ran late, which gave wrong top-k ids);
  one more MATH token wait per merge, expected to cost little. To re-check: tests/unit (incl. test_upstream_split.py
  perf band) and the DeepSeek-V4.1 model cases (index top-k k 512, candidate blocks k 2048).
- Files: device/topk_large_indices_device_operation.cpp, device/kernels/llk/ckernel_sfpu_topk_xl_split.h, device/kernels/llk/topk_xl.h
