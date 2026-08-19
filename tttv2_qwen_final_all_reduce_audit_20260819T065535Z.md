# Qwen Final All-Reduce Field Audit

- Timestamp: 2026-08-19 06:55:35 UTC
- Scope: read-only comparison of the Qwen decode final all-reduce in
  `models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py`, its Galaxy resource factory,
  and the 6U `ff2_qwen` case in `tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py`
  through `run_all_reduce_impl` in `test_new_all_reduce.py`.
- Hardware: not used.
- Shared source files: not modified.

## Reference Derivation

The 6U `ff2_qwen` parameters are:

| Field | Known-good value |
|---|---|
| logical per-device input/output | `[1, 1, 32, 1280]` |
| mesh shape | `(8, 4)` |
| cluster axis / ring size | `0` / `8` |
| links / topology | `4` / `ttnn.Topology.Ring` (`linear=False`) |
| input cores | `24`, ordered `RING_CRS` from `PREFETCHER_NOC1_GRID` |
| input shard | `[32, 64]`; padded width `64 * 24 = 1536` |
| output cores | `10`, `NORM_CRS_QWEN = (1,0)..(2,4)` |
| output shard | `[32, 128]`; width `128 * 10 = 1280` |
| persistent intermediate logical shape | `[8, 4, 32, 10240]` |
| persistent intermediate local shard grid | the 10 output cores |
| persistent intermediate shard | `[32, 1024]` (`128 * ring_size`) |
| input / intermediate / output dtype | `bfloat16` / `bfloat16` / `bfloat16` |
| layout / buffer type | tile / L1 width-sharded |
| mesh mapper | `ShardTensor2dMesh(dims=(0,1), mesh_shape=(8,4))` |
| semaphore sets | 8 single semaphores, each allocated on `SUB_DEVICE_CRS` at value 0 |
| subdevice | one worker subdevice over `SUB_DEVICE_CRS`, id 0, stall group `[0]` |

The input ring coordinate sequence in the local `_decode_ring_config` is byte-for-byte the same
coordinate sequence as `PREFETCHER_NOC1_GRID`; both construct `RING_CRS` from a list.

## Field-by-Field Diff

| Field | Local MLP/Galaxy setup | 6U `ff2_qwen` | Assessment |
|---|---|---|---|
| logical all-reduce input | `(1,1,32,1280)` plan key; W2 logical output | `[1,1,32,1280]` | Match |
| padded input | 1536 (`padded_dim`) | 1536 | Match |
| input memory config | 24 ordered ring points, `[32,64]`, width-sharded L1, row-major | Same | Match |
| input dtype | `bfloat16` | `bfloat16` | Match |
| output logical shape | `(1,1,32,1280)` after all-reduce/reshape | `[1,1,32,1280]` | Match |
| output memory config | 10 cores `(1,0)..(2,4)`, `[32,128]`, width-sharded L1 | Same | Match in members and geometry |
| output dtype | `bfloat16` | `bfloat16` | Match |
| cluster axis | 0 | 0 | Match |
| links | 4 | 4 | Match |
| topology | Ring | Ring | Match |
| operation variant | `use_optimal_ccl_for_llama=True` | omitted/default false | **Different execution contract** |
| persistent local shape | `[1,1,32,51200]` after mesh sharding | `[1,1,32,10240]` | **5x wider locally** |
| persistent global source shape | `(8,4,32,51200)` | `(8,4,32,10240)` | **5x wider** |
| persistent core set | all 50 worker cores | 10 output cores | **Different** |
| persistent shard | `[32,1024]` | `[32,1024]` | Per-core shard matches |
| persistent dtype | `bfloat16` | `bfloat16` | Match |
| persistent mapper | shard global dims `(0,1)` over `(8,4)` | Same | Match |
| persistent buffer count | 1 | 8 | Different reuse model |
| operation semaphore slots | 2, cycled | 8, cycled with the 8 buffers | Different reuse model |
| semaphores per slot | 1 | 1 | Match |
| semaphore initial value | 0 | 0 | Match |
| semaphore core members | 50 worker cores | same 50 worker cores | Match in membership |
| semaphore `CoreRangeSet` construction | Python set of two ranges | ordered list of two ranges | **Ordering risk** |
| worker `CoreRangeSet` construction | Python set of two ranges | ordered list of two ranges | **Ordering risk** |
| subdevices | sender id 0 plus worker id 1 | worker-only id 0 | Different due to prefetch partition |
| worker core membership | `(1,0)..(3,9)` plus `(5,0)..(6,9)` | Same | Match |
| worker subdevice passed to op | id 1 | id 0 | Semantically expected difference; same cores |
| stall group | worker id 1 only | worker id 0 only | Semantically expected difference |
| local L1 size override | 0 | manager created with 0 | Match |
| fabric | `FABRIC_1D_RING` | `FABRIC_1D_RING` | Match |
| dispatch axis | column | column | Match |
| extra barrier semaphores | 2 allocated but not passed to this overload | none in reference call | Resource-only difference |

## Ranked Discrepancies

### 1. High: hybrid operation/buffer contract is not the cited known-good case

The local call enables `use_optimal_ccl_for_llama=True`, while the 6U `ff2_qwen` reference call does
not. Correspondingly, the local persistent buffer is allocated across all 50 worker cores with local
width 51200, whereas the reference buffer uses the 10 output cores with local width 10240.

The local 50-core allocation is not arbitrary: it matches the legacy optimized decode allocator in
`models/demos/llama3_70b_galaxy/tt/llama_ccl.py:get_persistent_buffers`, which uses all worker cores,
shard `[32,1024]`, and global source shape `(8,4,32,1024*50)`. Therefore this is an intentional
optimized-path contract, but it is not validated by the cited `test_all_reduce_6U_llama[ff2_qwen]`
case. A direct known-good comparison must either disable the optimal flag and use the 10-core buffer,
or cite/add an optimized Qwen case with the 50-core buffer.

### 2. High: optimized Qwen persistent dtype differs from the legacy optimized allocator

The local optimized buffer is `bfloat16`; the legacy `llama_ccl.py` optimized persistent allocator
creates this buffer as `bfloat8_b`, including for Qwen, while the generic 6U test uses `bfloat16`.
This reinforces that the current setup mixes generic-Qwen dtype policy with optimized-buffer geometry.
It may be supported, but neither compared baseline proves that exact combination.

### 3. Medium: unordered worker/semaphore `CoreRangeSet`

The reference `SUB_DEVICE_CRS` is constructed from an ordered list:
`[(1,0)..(3,9), (5,0)..(6,9)]`. The local decode worker and semaphore cores are constructed from a
Python set. Membership is identical, but iteration/hash order is not guaranteed. This codebase has
already demonstrated that list-versus-set `CoreRangeSet` construction can alter ordered shard/ring
mapping. The two-range worker set should preserve the reference list order unless TTNN explicitly
guarantees canonical ordering for subdevice and semaphore grids.

### 4. Medium: one persistent buffer is cycled against two semaphore slots

The reference pairs 8 intermediate buffers with 8 semaphore handles. The local owner cycles 2
semaphores while reusing one persistent buffer. This matches the broad legacy model pattern (one
persistent buffer, double-buffered gather semaphores), and the local harness synchronizes the worker
subdevice before readback. It is nevertheless a concurrency difference from the cited trace test and
must not be treated as equivalent under overlapped or unsynchronized invocations.

### 5. Low: additional sender subdevice and shifted worker id

The reference owns only worker subdevice id 0. The local prefetch configuration owns sender id 0 and
worker id 1. The worker core membership, stall policy, and `subdevice_id` passed to all-reduce remain
internally consistent. This is expected integration plumbing, not a shape mismatch.

## Bottom Line

The final Qwen all-reduce input, output, dtype, ring order, topology, links, output sharding, and mesh
mapping match the 6U `ff2_qwen` case. The material unresolved issue is that the local path combines
the legacy optimized 50-core buffer and optimal flag with generic-test Qwen `bfloat16` policy and a
different buffer/semaphore reuse scheme. The exact combined contract is not covered by either cited
known-good baseline. The only clear construction defect visible statically is unordered set-based
creation of the two-range worker/semaphore `CoreRangeSet`.
