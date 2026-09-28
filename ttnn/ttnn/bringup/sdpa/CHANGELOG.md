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
