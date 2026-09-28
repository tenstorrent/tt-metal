# sdpa (fork)

- Source: `ttnn/cpp/ttnn/operations/transformer/sdpa`
- Source SHA: `99f7e834cea8bc42f0b478b9cc59a7b89f2f9c1a`
- Python: `ttnn.bringup.*` (was `ttnn.transformer.*`)
- Forked for: mimo_v2_6_d_p attention V head dim 128 (non-MLA GQA)
- Used by: none yet (mimo_v2_6_d_p will switch to it)

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_sdpa`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

The source is a subfolder of the `ttnn_op_transformer` target, so it had no CMake target of its own; fork_op.py
wrote `CMakeLists.txt` and `sources.cmake` for the fork (every host `.cpp` except the nanobind one; the public headers
`sdpa.hpp`, `sparse_sdpa.hpp`, `sparse_sdpa_msa.hpp`). The whole folder is copied, so every Python name it binds
(scaled_dot_product_attention, chunked_scaled_dot_product_attention, the joint, ring, MLA and sparse variants) is in
`ttnn.bringup.`; only the first two are carried and tested. `sdpa_config.hpp` (SDPAProgramConfig,
PagedCacheGeometryOverride) is shared with the other transformer ops and stays the source's, so the fork takes the same
`ttnn.SDPAProgramConfig`.

Build fixes after fork_op.py (no behaviour change):
- `ttnn::operations::transformer::{SDPAProgramConfig, PagedCacheGeometryOverride}` restored (the namespace rewrite had
  moved them to `ttnn::operations::bringup`); `sdpa_nanobind.cpp` gets `using` declarations for the two.
- Relative qualifications into the nested namespaces: `prim::*_IDX` -> `prim::bringup::*_IDX` (sdpa.cpp),
  `transformer::SparseKVFormat` -> `transformer::bringup::SparseKVFormat`, `operations::transformer::sdpa::` ->
  `operations::bringup::sdpa::`.
- Sub-namespaces nested twice (`ttnn::prim::bringup::detail::bringup`, `...::sdpa_cb::bringup`,
  `ttnn::transformer::bringup::sdpa::bringup`) collapsed to one `bringup`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### Non-MLA V head dim narrower than K's (GQA)
- What: `scaled_dot_product_attention` (causal, sliding window, attention sink) and
  `chunked_scaled_dot_product_attention` (paged, chunk_start_idx as an int or a tensor) take a V whose last dim is
  smaller than Q/K's; the output then has V's width. No new argument: a narrower V tensor turns it on. V must be a
  multiple of 32 and at most K's head dim (refused otherwise); the paged-cache geometry override and windowed mode
  still require V as wide as K. Host-only:
  - program factory: non-MLA `vDHt` = V's own width (was always DHt), except under a geometry override;
  - reader compile-time arg 23 (`use_mla`) is passed as `use_mla || vDHt != DHt`. In the reader it only selects V's
    addressing width (`v_tile_shape`, the paged `head_dim`); the K/V overlap it also gates needs arg 24
    (`mla_kv_overlap`), which stays false. Chosen over editing the two reader conditions so that the kernels stay the
    source's byte for byte (no kernel hash change, no recompile of any other program);
  - validation: `v_shape[3] <= DH`, `% 32 == 0` (plain and paged); windowed mode refuses a narrower V;
  - `compute_output_specs`: non-MLA output last dim = V's (unless a geometry override is active).
  The kernels already handled vDHt < DHt (the MLA path uses them).
- Default unchanged: with V as wide as K (every existing caller) vDHt == DHt, arg 23 == use_mla and the output spec
  are what they were, so the program (kernels, compile-time and runtime args, CBs) is the source's; the unit tests
  check the fork's output is bit-identical to the source op's (causal and chunked, both compute paths), and the
  source-test check reports 0 regressions (118 tests, warm kernel cache hits). The V shape is in the default
  program-cache key (tensor specs are hashed); a test runs V 192 and V 128 with the same Q/K and gets 2 programs.
- Why: MiMo-V2 attention has Q/K head dim 192 and V 128; the model pads V to 192 with zeros. With the narrow V the
  output equals the padded run's first 128 columns bit for bit, and device time at MiMo's full-attention chunk
  (Q 16x5120x192, paged cache with a 51200-token prefix, q512/k128, HiFi2, approx exp, fp32 dest off = streaming,
  1 chip) drops from 27.15 ms (V 192, source and fork alike) to 22.49 ms (V 128), -17%; chunk 0 (causal S 5120)
  from 1.505 ms to 1.256 ms, -17%.
- Tests: `tests/unit/test_sdpa_narrow_v.py` (20 cases, MiMo shapes: 16 Q heads, 1 or 2 KV heads, K 192, V 128,
  scale 192^-0.5, bf16): causal S 5120 / 2048, sliding window 128 + sink, chunked paged at prefixes 4096 and 49152,
  fp32 dest on (non-streaming) and off (streaming), each against a torch reference and bit-identical to the padded
  source op; V == K bit-identical to the source; the program cache; refusals (V 112, V 224). With arg 23 reverted to
  `use_mla` (the reader addressing V at K's width) all 11 narrow-V cases fail (PCC 0.16-0.83).
- Needed by: mimo_v2_6_d_p (attention V head dim 128; the model still calls ttnn.transformer with padded V)
- Files: `device/sdpa_program_factory.cpp`, `device/sdpa_device_operation.cpp`, `tests/unit/`
