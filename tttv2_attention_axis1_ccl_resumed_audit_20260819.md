# Attention2D WH Galaxy Axis-1 CCL Resumed Audit

## Goal and constraints

Audit local `tt-metal` for passing Wormhole Galaxy axis-1 CCL examples matching or nearest to per-device input shape `(1, 1, 32, 1280)`, mesh `(8, 4)`, `cluster_axis=1`, and COL dispatch. Compare them exactly with `models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py`, without running hardware or editing shared files, and recommend one concrete next experiment.

No TT hardware command or test was run. This new work log is the only file edited by this audit.

## Checkpoint 1: exact local references

Two 6U/Galaxy references match the blocked QKV geometry exactly.

### Fused QKV all-reduce and head creation

`tests/ttnn/unit_tests/operations/ccl/test_qkv_all_reduce_minimal.py:488-560` is explicitly restricted to 6U and configures:

- mesh `(8, 4)` and COL dispatch;
- `FABRIC_1D_RING` with `Topology.Ring`;
- per-device QKV shape `(1, 1, 32, 1280)` and `cluster_axis=1`;
- BF8 tiled WIDTH_SHARDED L1 input over the 24 `PREFETCHER_NOC1_GRID` cores;
- 10 output cores, local shard `(32, 128)`, BF16 output;
- three links;
- a fused `ttnn.experimental.all_reduce_create_qkv_heads` call;
- two cycling global semaphores and two persistent BF8 intermediate buffers;
- 30 traced iterations with 10 warmups.

The helper at lines 101-160 constructs global shapes `(8, 4, 32, 1280)` for input and `(8, 4, 32, 5120)` for scratch, then maps mesh dimensions `(0, 1)`. Each device therefore sees the exact blocked local QKV geometry. Lines 164-220 also fuse the reduction with decode head creation and column-specific `batch_offset=[0, 8, 16, 24]`, `slice_size=8`.

### Generic asynchronous all-reduce

`tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py:850-942` contains a 6U QKV case with the same shape, axis, mesh, COL dispatch, BF8 input, BF16 output, and 10 output cores. It uses:

- `FABRIC_1D_RING`, `Topology.Ring`, and four links;
- the 50-core Llama worker envelope `(1,0)-(3,9)` plus `(5,0)-(6,9)`;
- 24 `PREFETCHER_NOC1_GRID` input cores;
- `ttnn.experimental.all_reduce_async` via `run_all_reduce_impl(..., linear=False)`;
- persistent BF8 scratch with local shard `(32, 512)` on the 10 output cores;
- cycling semaphores/buffers and 75 traced iterations with 10 warmups.

This is the nearest reference to a non-fused reduction. The fused test is the nearest reference to the complete QKV reduction/head-creation boundary.

## Checkpoint 2: exact target differences

The current target is `models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py`.

| Property | Passing 6U references | Current Attention2D test |
| --- | --- | --- |
| Fabric | Explicit `FABRIC_1D_RING` | Boolean `True` at line 856 |
| Effective fabric | Ring with dateline deadlock avoidance | `FabricConfig(True)` resolves to `FABRIC_1D_NEIGHBOR_EXCHANGE` |
| Operation topology | Ring | Current CCL-only path uses Linear; earlier retries also requested Ring over the same neighbor-exchange fabric |
| Links | 3 fused or 4 generic async | Current standard RS/AG uses 1; earlier variants used 1 or 4 |
| QKV input | BF8 tiled WIDTH_SHARDED L1, 24 exact ring cores, shard width 64 after padding 1280 to 1536 | Current active branch passes the BF16 DRAM matmul output directly to standard RS; persistent variants perform separate placement conversions |
| Mesh distribution | Explicit global `(8,4,32,1280)` with `ShardTensor2dMesh(dims=(0,1))` | Inherited from the 2D activation/weight matmul; no standalone assertion compares its distributed config with the reference input |
| Reduction API | Proven fused QKV op or minimal-buffer `all_reduce_async` | Current active branch is standard `reduce_scatter` then `all_gather`; prior minimal RS/AG and async all-reduce retries were not reference-identical |
| Scratch | BF8 WIDTH_SHARDED L1 `(32,512)` on 10 output cores | Current RS plan owns reduced `(32,320)` output plus a DRAM input-shaped intermediate; AG owns a separate output |
| Worker subdevice | One explicit CCL worker subdevice; generic reference uses the canonical 50-core worker set | Current CCL-only plan uses one full available compute-grid subdevice |
| Output/head layout | BF16 reduced output on 10 width-sharded cores, then fused HEIGHT_SHARDED heads | Standard RS/AG restores/returns a separate reduced tensor before `nlp_create_qkv_heads_decode` |
| Batch slicing | `batch_offset=[0,8,16,24]`, `slice_size=8` in the fused reference | The common module's separate head call does not pass these arguments |
| Reuse | Many traced calls with cyclic semaphores and persistent buffers | Two eager module invocations with Galaxy resource cycling |

The shape itself is not an unsupported corner: both Llama and Qwen in the target produce the exact local width `1280`, and local 6U tests cover that geometry directly.

## Checkpoint 3: primary causal finding

The target's fabric configuration invalidates comparison with every known-passing 6U Ring reference.

- `tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp:17-25` assigns enum value `1` to `FABRIC_1D_NEIGHBOR_EXCHANGE`, value `2` to `FABRIC_1D`, and value `3` to `FABRIC_1D_RING`.
- A host-only enum check confirms `ttnn.FabricConfig(True) == ttnn.FabricConfig.FABRIC_1D_NEIGHBOR_EXCHANGE`.
- Neighbor exchange explicitly has no forwarding between non-adjacent devices. The passing 6U QKV tests explicitly request ring fabric and Ring operation topology.
- The prior Ring, Linear, standard, minimal, and async experiments all retained the target fixture's boolean `True`; they therefore did not reproduce the passing reference's fabric/topology pair.

This is a high-confidence configuration defect. It does not yet prove that correcting the fabric alone makes the full Attention2D test pass, because the current input placement, scratch geometry, subdevice envelope, and API still differ.

## Checkpoint 4: tt-buddy access

The supplied `https://github.com/tenstorrent/tt-buddy/tree/main` URL returns 404 through the web endpoint. Direct `git ls-remote` requests credentials unavailable in this environment, and an earlier local audit found no checkout, cache, Git ref, or installed skill from that repository. No `tt-buddy` content could therefore be used as evidence.

The investigation instead applied the same evidence-first workflow locally: locate a passing exact-shape test, reconstruct its entire mesh/fabric/subdevice/tensor/semaphore contract, and change one boundary at a time.

## Recommended next experiment

Run one bounded standalone A/B hardware probe under `FABRIC_1D_RING`, before rerunning the full module:

1. In a single `(8,4)` COL-dispatch process, reproduce the 6U generic async QKV reference exactly: BF8 WIDTH_SHARDED L1 `(1,1,32,1280)` on the 24 `PREFETCHER_NOC1_GRID` cores, 10-core `(32,128)` BF16 output, `(32,512)` BF8 persistent scratch, canonical 50-core worker subdevice, `cluster_axis=1`, Ring, four links, one invocation and worker-scoped synchronization.
2. In the same process after the synthetic case passes, run the Attention QKV matmul once, convert only its output to that exact reference BF8 input memory config, and invoke the same already-compiled collective contract with a fresh semaphore/scratch slot.

Interpretation is decisive:

- If the synthetic case stalls, the local runtime/hardware no longer satisfies its repository reference and the issue belongs below Attention2D.
- If synthetic passes but the matmul-derived tensor stalls, inspect distributed tensor metadata/mapping and producer-consumer queue state; do not change CCL algorithms again.
- If both pass, replace the target's boolean fabric setting and current standard RS/AG adapter with the proven Ring async contract, then proceed to the separate BF16 head-creation boundary.

Do not begin with the full fused QKV primitive. The generic async A/B probe changes fewer semantics and directly tests whether tensor lineage, rather than shape or fabric transport, distinguishes the blocked Attention output from the passing reference.

## Audit conclusion

The sole blocker is not accurately characterized as “all axis-1 APIs stall for `(1,1,32,1280)`.” Local WH Galaxy coverage proves that exact geometry in both generic async and fused QKV forms. The failed experiments were run with neighbor-exchange fabric and did not match the explicit Ring fabric/topology contract of the 6U references. The next hardware action should be the bounded reference-identical A/B probe above, with hardware reset only if that process fails to tear down cleanly.
