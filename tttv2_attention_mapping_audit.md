# Attention2D Mesh/Permutation Audit

## Scope

Read-only audit of the remaining Milestone A decode mismatch. No TT hardware was
used and no shared implementation/test file was changed. The evidence below uses
the current Attention2D test/implementation and the local tt-buddy CCL guidance.

## Finding

The observed failure is **not a global user permutation or final mesh-composition
error**. The strongest remaining mapping defect is the test's intra-device core
geometry: Q/SDPA scheduling and SDPA output user shards are built from
`row_wise=False` core selections, while the production Galaxy path uses row-wise
user cores throughout decode SDPA.

The diagnostic had high per-row PCC only for local users `(0, 3, 6)` in
each column group, repeated globally as `(0,3,6)`, `(8,11,14)`, `(16,19,22)`, and
`(24,27,30)` (`tttv2_2d_modules_work_log.md:1705-1707`). A global permutation
would move whole rows consistently; this repeating within-column pattern instead
tracks local user/core indices.

The main hardware lane subsequently applied the production row-wise SDPA schedule
and output grids; both repeated Llama decode cache and output PCC checks passed
(`tttv2_2d_modules_work_log.md:1710-1714`). This independently confirms the
audit's core-layout diagnosis. The next failure is a prefill tensor-ownership
alias, not a decode mesh/user permutation.

## Mesh-order evidence

- Decode input uses `ShardTensor2dMesh(dims=(None, 3))`: users are replicated over
  mesh rows and hidden width is split over mesh columns
  (`models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py:846-855`).
- Fused QKV uses column-local offsets `[0,8,16,24]`, sharded over mesh columns by
  `dims=(None,0)`, with `slice_size=8`
  (`models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py:411-438`).
  Passing all 32 K/V cache rows, composed with `(1,0)`, independently validates
  those column slices and their global order
  (`models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py:870-875`,
  `:961-973`; work log `:1688-1694`).
- Synchronous all-gather indexes a line participant by its physical coordinate on
  `cluster_axis`; axis 1 therefore assigns indices `0..3` in column order
  (`ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:190-205`). Its multicast writer
  places each participant at `device_idx * output_chunks_per_stripe`
  (`ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_multicast_factory.cpp:291-307`,
  `:383-398`). Thus the expected gathered users are `[0..7,8..15,16..23,24..31]`,
  not ring-traversal order.
- TTNN has a directly analogous Wormhole "Before Concat Heads" all-gather case:
  dim 1, four devices, logical output `[1,32,32,128]`, height-sharded `(32,128)`
  (`tests/ttnn/unit_tests/operations/ccl/test_minimals.py:291-320`). The test maps
  the source with `PlacementShard(dim)` and validates every device against the
  original full tensor (`tests/ttnn/unit_tests/operations/ccl/test_minimals.py:195-232`,
  `:265-280`).
- Final output composition `(1,3)` concatenates the axis-0-replicated row copies
  on singleton dim 1 and the column hidden shards on dim 3, then selects the first
  row copy (`models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py:862-867`).
  It does not reorder dim-2 users. This matches the production Galaxy readback
  convention `(1,-1)` (`models/demos/llama3_70b_galaxy/tt/llama_common.py:280-285`).

## Concrete layout discrepancy

The test derives `head_cores` (32 cores) and `kv_cores` (8 cores) with
`row_wise=False`, and uses the former as the decode SDPA subgrid and the latter as
the SDPA output placement
(`models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py:627-638`,
`:692-705`, `:803-812`). SDPA itself enumerates a supplied subgrid row-wise
(`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp:174-178`,
`:294`). Supplying a column-wise-selected sparse set and then enumerating it
row-wise changes the user/core schedule.

Production Galaxy instead:

- gives fused head creation the full worker subgrid; the fused op internally
  selects Q, K, and V batch grids row-wise
  (`models/demos/llama3_70b_galaxy/tt/model_config.py:496-507`, `:2803-2813`;
  `ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/all_reduce_create_qkv_heads_device_operation.cpp:217-241`);
- selects all 32 decode SDPA workers with `row_wise=True`
  (`models/demos/llama3_70b_galaxy/tt/model_config.py:1210-1218`); and
- selects one SDPA output core per local user with `row_wise=True`
  (`models/demos/llama3_70b_galaxy/tt/model_config.py:1252-1262`).

The current test also sets decode `q_chunk_size=k_chunk_size=0`, whereas production
uses `256` for both (`test_attention_2d_wh_galaxy.py:203-218` versus
`model_config.py:1210-1218`). This is not a mesh permutation, but it should be
aligned in the same experiment because it changes the decode recipe.

## Recommended next experiment

1. Build the fused final head memory config over the full worker grid, allowing
   `all_reduce_create_qkv_heads` to derive its own row-wise Q/K/V grids.
2. Use a distinct 32-core SDPA subgrid selected from the worker grid with
   `row_wise=True`.
3. Build the eight-core SDPA output/user grid with `row_wise=True`, and use the
   same resulting grid at the gather/concat boundary.
4. Match production decode chunks (`q_chunk_size=k_chunk_size=256`).
5. Before trying any user reorder, capture one per-device post-SDPA or post-gather
   readback and compare rows locally. The expected global mapping is already fixed
   by physical column index; a software permutation would mask the core-layout
   defect rather than correct it.

tt-buddy's applicable rule is to derive collective topology/link parameters from
the sibling model and treat wrong values as deadlock risks
(`/tmp/tt-buddy-access-audit-20260819/knowledge/ccl.md:19-37`). Those parameters
are no longer the numerical differentiator here; the current Ring gather completes,
and the repeating local-user pattern points to SDPA core layout.
