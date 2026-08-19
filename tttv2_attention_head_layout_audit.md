# Milestone A Attention2D Head-Layout Audit

Date: 2026-08-19
Scope: read-only source audit; no TT hardware used; no shared implementation files edited

## Conclusion

The highest-probability remaining mismatch is the decode Q/SDPA placement boundary, not QKV weight fusion, K/V cache selection, concat's logical transform, or the WO mesh mapper.

The hardware test builds the fused-head containing grid and SDPA output grid with `row_wise=False` (`test_attention_2d_wh_galaxy.py:627-638`). Production Galaxy instead:

- gives create-heads the full worker subdevice (`model_config.py:2803-2814`);
- configures decode SDPA on 32 row-wise worker cores (`model_config.py:1210-1218`);
- emits SDPA output on one **row-wise** core per local user, shard `(32, 128)` (`model_config.py:1252-1261`);
- gathers users on axis 1 before concat (`llama_attention.py:790-825`).

The current test passes a column-wise 32-core subset as both the fused create-heads containing grid and the SDPA `sub_core_grids`, then requests SDPA output on a separately constructed column-wise eight-core grid (`test_attention_2d_wh_galaxy.py:630-638,692-705,803-812`). Passing K/V cache PCC does not validate Q's user/head placement or SDPA's output user order.

## Contract Findings

### Q head ordering

The fused weight construction is source-aligned. `_fused_qkv_weight` forms eight contiguous `[Q_row, K_row, V_row]` blocks (`test_attention_2d_wh_galaxy.py:499-506`), matching production's per-row Q/K/V chunk, transpose, and concatenate sequence (`llama_attention.py:154-192`). With `wqkv` mapped row->output and column->input, row `r` owns global Q heads `[8r, 8r+8)` and KV head `r`.

`all_reduce_create_qkv_heads` logically returns Q `[1, 8 users, 8 heads, 128]`; its output-spec code allocates Q, K, and V as consecutive row-wise batch grids selected from `final_memory_config` (`all_reduce_create_qkv_heads_device_operation.cpp:201-253`). The current `final_memory_config` uses only a column-wise-derived 32-core subset. It is large enough, but differs from the production full worker containing set and should be removed as a variable.

Recommended fix:

1. Pass `worker_cores`, not `head_cores`, as the fused op's `final_memory_config` containing grid.
2. Define the SDPA 32-core domain independently as `num_cores_to_corerangeset_in_subcoregrids(start_core, 32, worker_cores, row_wise=True)`.
3. Before changing algorithms, compose Q per device and compare each mesh row/column against `q_ref[users=col*8:(col+1)*8, heads=row*8:(row+1)*8]`. Report an 8x8 user correlation matrix per device. This is the first missing numerical boundary check.

### SDPA sharding and program config

Production non-paged Galaxy decode uses `SDPAProgramConfig((8,4), 32 row-wise subcores, q_chunk_size=256, k_chunk_size=256)` and an eight-core row-wise height-sharded output (`model_config.py:1210-1218,1252-1261`). The test uses the same grid size but `q_chunk_size=k_chunk_size=0`, a column-wise containing grid, and a column-wise output grid (`test_attention_2d_wh_galaxy.py:110-125,630-638,697-700,810-812`). Zero chunks are used by production's paged config, not its non-paged config.

Recommended first adapter patch:

```python
sdpa_compute_cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(
    ttnn.CoreCoord(1, 0), 32, worker_cores, row_wise=True
)
sdpa_output_cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(
    ttnn.CoreCoord(1, 0), 8, worker_cores, row_wise=True
)
```

Use `sdpa_compute_cores` in `SDPAProgramConfig`, use `sdpa_output_cores` for a distinct height-sharded `(32,128)` SDPA output memcfg, and keep the K/V cache-update memcfg separate. Set non-paged decode chunks to `256,256` to match production. If direct sharded output remains suspect, use the shared-Llama diagnostic path: request DRAM SDPA output, then explicitly `to_memory_config` into the row-wise eight-user layout before gathering (`models/tt_transformers/tt/attention.py:704-743`).

Add an SDPA-stage assertion before user gather. Compose local `[1,8,8,128]` and compare against the Torch attention intermediate for exactly the column's eight users and row's eight Q heads. This determines whether the mismatch is before or after SDPA without involving concat or WO.

### Concat-heads contract

`nlp_concat_heads_decode` consumes tiled BF16/FP32 height-sharded `[1, users, padded_heads, head_dim]`, with exactly one input core per user and shard `(padded_heads, head_dim)` (`nlp_concat_heads_decode_device_operation.cpp:22-69`). It emits `[1,1,max(users,32),num_heads*head_dim]` in intrinsic width-sharded L1; its `memory_config` argument is ignored without a preallocated output (`...device_operation.cpp:72-130`).

The current logical order is correct: gather local users from 8 to 32 on tensor dimension 1, then concat eight local heads (`attention_2d.py:900-918`). Production does the same (`llama_attention.py:790-825`). The gathered memcfg is also correct in shape: 32 row-wise user cores, shard `(32,128)`.

Recommended checks:

1. Derive `sub_core_grids` from `gathered_users.memory_config().shard_spec.grid` inside the adapter, as production does, rather than relying on a separately configured equal-looking grid.
2. Capture gathered users and concat output once. Verify gathered device `(row,col)` equals all 32 users for that row's eight heads, then verify concat equals `attention.transpose(1,2).reshape(1,1,32,1024)`.
3. Keep the explicit post-concat transition. Do not treat `decode_concat_memory_config=DRAM` as an op output override.

### WO input layout

The WO mapper is conceptually correct. The test maps mesh row to weight K and mesh column to weight N (`PlacementShard(0), PlacementShard(1)` on the rank-2 `[K,N]` source at `test_attention_2d_wh_galaxy.py:1015-1027`). This is equivalent to production's rank-4 mapper `dims=(2,3)` (`llama_attention.py:224-256`). Concat on row `r` supplies global hidden slice `[r*1024:(r+1)*1024]`, matching that row's WO K shard; axis-0 reduction then sums row partials.

Recommended final localization check: compare each pre-reduction WO partial on device `(row,col)` with

```python
local_attention[:, :, :, row * 1024 : (row + 1) * 1024] @ \
    wo[row * 1024 : (row + 1) * 1024, col * 2048 : (col + 1) * 2048]
```

Only change the WO mapper if this local partial fails after concat passes. The output composer `dims=(1,3)` is reasonable because row replicas are concatenated into dimension 1 and the test selects the first row (`test_attention_2d_wh_galaxy.py:862-867`).

## Recommended Execution Order

1. Add failure-only Q and SDPA intermediate checks; run one Llama decode invocation.
2. Align fused final grid, SDPA compute grid, SDPA output grid, and non-paged chunk sizes with production.
3. If SDPA passes, check gathered-users then concat numerically before WO.
4. If concat passes, check per-device WO partials before axis-0 reduction.
5. Remove temporary captures after the failing boundary is fixed; retain compact stage assertions if practical.

This order separates head selection, attention, head concatenation, and projection. It avoids using final-output PCC to infer a layout error several operations upstream.

## tt-buddy Notes

`/tmp/tt-buddy-access-audit-20260819` is at `ba9021417442d59756aa8cdf154a25648c9a0de5`. Its applicable guidance is to copy exact same-mode sibling layouts/program configs and to treat CCL topology/link values as hardware contracts (`knowledge/ccl.md`, `knowledge/matmul.md`). It has no operation-specific head-layout rule beyond the upstream source patterns above. No TT commands, tests, or resets were run for this audit.
