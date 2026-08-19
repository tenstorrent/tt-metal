# WH Galaxy Attention2D Axis-1 CCL Audit

## Scope

Read-only source audit of known WH Galaxy `(8, 4)` axis-1 collective tests closest to the Attention2D decode-local tensor `(1, 1, 32, 1280)`. No TT hardware was run and no shared production or test file was edited.

## Checkpoints

### Checkpoint 1: Candidate inventory

- Searched WH/TG/Galaxy CCL tests for `reduce_scatter`, `all_gather`, `all_reduce`, `cluster_axis`, subdevice, semaphore, worker, link, and buffer configuration.
- Identified two exact-shape WH Galaxy references:
  - generic persistent `experimental.all_reduce_async`: `tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py`
  - fused production QKV all-reduce plus head creation: `tests/ttnn/unit_tests/operations/ccl/test_qkv_all_reduce_minimal.py`
- Identified the production Galaxy implementation behind the fused path: `models/demos/llama3_70b_galaxy/tt/llama_ccl.py`.
- The broad nightly line-all-reduce tests are useful corroboration but are less exact: their closest axis-1 case uses per-chip shape `[1, 8, 32, 1280]`, not `[1, 1, 32, 1280]` (`tests/nightly/tg/ccl/test_all_reduce.py:76-98`).

### Checkpoint 2: Exact-shape generic all-reduce comparison

The generic WH Galaxy test explicitly includes `(output_shape=[1,1,32,1280], cluster_axis=1)` with both `num_links=3` and `num_links=1` (`tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py:305-318`). It runs on mesh `(8,4)`, dispatch axis `COL`, and `FABRIC_1D` (`tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py:333-350`).

Its material contract is:

- **Topology/operation:** Linear `ttnn.experimental.all_reduce_async`, not a separately submitted synchronous RS/AG pair (`test_new_all_reduce.py:84-89`, `192-203`).
- **Mesh mapping:** global source shape `[8,4,32,1280]`, mapped with `ShardTensor2dMesh(dims=(0,1), mesh_shape=(8,4))`, yielding local `[1,1,32,1280]` (`test_new_all_reduce.py:109-114`, `145-153`). Thus axis 1 reduces the four column-local partials and preserves the eight row shards.
- **Input memory:** width-sharded L1 over 24 ring/prefetch receiver cores, with shard width `round_up(ceil(1280/24),32)=64` (`test_new_all_reduce.py:109-124`).
- **Output memory:** width-sharded L1 over ten QKV cores, shard `(32,128)` (`test_new_all_reduce.py:111-133`; QKV core set at `25-42`).
- **Persistent intermediate:** width-sharded L1 over those ten output cores, shard `(32,512)`, exactly `output shard volume * axis length 4` (`test_new_all_reduce.py:114`, `134-165`).
- **Subdevice:** only worker cores `(x=1..3,y=0..9)` plus `(x=5..6,y=0..9)` are in subdevice 0; manager is loaded and that ID is the stall group (`test_new_all_reduce.py:25-32`, `91-101`).
- **Semaphores/buffers:** eight global semaphore handles and eight persistent intermediates, round-robin by iteration (`test_new_all_reduce.py:99-101`, `155-167`, `185-203`). The API does not expose `chunks_per_sync`, `num_workers_per_link`, or `num_buffers_per_channel`; it uses the operation defaults.
- **Synchronization:** every non-trace invocation synchronizes the full mesh immediately after submission (`test_new_all_reduce.py:204-205`); trace compile/capture/execute boundaries also synchronize (`test_new_all_reduce.py:218-249`).

### Checkpoint 3: Exact-shape fused QKV comparison

The fused WH Galaxy test uses exact output shape `[1,1,32,1280]`, `cluster_axis=1`, three links, BF8 input, BF16 output, 24 input cores, ten output cores, mesh `(8,4)`, dispatch `COL`, and `FABRIC_1D` (`tests/ttnn/unit_tests/operations/ccl/test_qkv_all_reduce_minimal.py:337-377`).

- **Operation/topology:** `ttnn.experimental.all_reduce_create_qkv_heads` with Linear topology (`test_qkv_all_reduce_minimal.py:56-62`, `192-220`).
- **Mapping:** same `[8,4,32,1280]` global tensor and `ShardTensor2dMesh(dims=(0,1))`, hence local `[1,1,32,1280]` (`test_qkv_all_reduce_minimal.py:98-102`, `139-149`).
- **Memory:** BF8 width-sharded L1 input over 24 ring cores; BF16 width-sharded L1 reduced output over ten cores; persistent BF8 L1 intermediate has four times the output shard width (`test_qkv_all_reduce_minimal.py:98-137`, `151-173`).
- **Subdevice:** the fused test derives the worker grid from production `CREATE_HEAD_OUTPUT_MEMCFG`, loads a one-subdevice manager, and stalls subdevice 0 (`test_qkv_all_reduce_minimal.py:67-88`).
- **Semaphores/buffers:** two global semaphores and two persistent intermediates, round-robin (`test_qkv_all_reduce_minimal.py:84-88`, `151-162`, `198-220`).
- **Synchronization:** non-trace mode synchronizes the full mesh after each call (`test_qkv_all_reduce_minimal.py:221-223`); trace boundaries synchronize explicitly (`test_qkv_all_reduce_minimal.py:234-265`).
- **Production precedent:** Galaxy `llama_ccl.py` sizes the axis-1 QKV persistent shard as `1280 / 10 * 4 = 512` over its worker core set (`models/demos/llama3_70b_galaxy/tt/llama_ccl.py:424-445`) and calls the same fused operation with the production semaphore ring and worker subdevice (`llama_ccl.py:867-905`).

### Checkpoint 4: Attention2D delta analysis

The Attention2D test uses the same physical mesh `(8,4)`, dispatch `COL`, and fabric enabled (`models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py:49-53`, `855-861`). Its input/weight mapping is coherent with an axis-1 QKV reduction: activations shard hidden dim over mesh columns (`:740-753`), while QKV weights shard K over columns and output heads over rows (`:887-894`). The local Llama QKV projection is therefore `(1,1,32,1280)` and must sum four column partials.

However, its active path differs from both passing exact-shape paths:

- `persistent_decode=False` selects the fallback branch (`test_attention_2d_wh_galaxy.py:652-662`).
- The fallback submits `ttnn.reduce_scatter(dim=3)` followed by `ttnn.all_gather(dim=3)`, each Linear and one-link (`:347-370`). Neither exact-shape reference uses this synchronous composition.
- It passes the projection tensor directly as the collective input. Decode QKV matmul output is configured as interleaved DRAM (`:693-703`), whereas both exact references require width-sharded L1 input over 24 ring cores.
- Its fallback reduce-scatter output is interleaved DRAM (`:347-356`), whereas the exact references use a width-sharded L1 output and an input-sized persistent L1 intermediate.
- A full-grid one-subdevice manager is used by `galaxy_mode_plan` (`models/common/tests/modules/_wh_galaxy_hardware.py:281-302`) through the CCL-only owner (`:72-129`, `168-185`). The generic passing test instead constrains the worker subdevice to the canonical 50-core set (`test_new_all_reduce.py:25-32`, `91-97`).
- The dormant persistent branch is also not an exact match: it uses RS/AG, one link, `chunks_per_sync=10`, two workers/link, two buffers/channel, and a DRAM intermediate (`test_attention_2d_wh_galaxy.py:263-324`, `476-521`). The exact generic all-reduce exposes none of those tuning overrides and uses its persistent intermediate in L1.
- Attention does not synchronize between projection and the fallback RS, or between RS and AG; it relies on queue ordering and later host conversion/fixture synchronization. Exact references synchronize after each non-trace all-reduce (`test_new_all_reduce.py:204-205`; `test_qkv_all_reduce_minimal.py:221-223`). Queue ordering should normally suffice, so this is a diagnostic delta rather than the leading root-cause candidate.
- Attention's `qkv_cores` are selected `row_wise=False` (`test_attention_2d_wh_galaxy.py:551-560`). The generic passing helper defines `QKV_CRS` with `row_wise=True` (`test_new_all_reduce.py:25-32`), while the fused exact-shape test also uses `row_wise=False` for its ten output cores (`test_qkv_all_reduce_minimal.py:114-128`). Therefore core ordering must follow the chosen operation; it is not independently proven wrong.

The fused operation is the closest production behavior, but `Attention2D.decode_forward` currently expects `reduce_qkv` to return a reduced QKV tensor and then separately invokes `nlp_create_qkv_heads_decode` (`models/common/modules/attention/attention_2d.py:804-827`). Substituting the fused API therefore requires a deliberate low-level contract extension; it is not a drop-in test-only change.

## Ranked Next Experiments

1. **Reproduce the exact generic passing recipe in a focused, single-call hardware test before re-entering Attention2D.** Copy the `(1,1,32,1280)`, axis-1, BF8, Linear, one-link case from `test_new_all_reduce.py`: canonical 50-core worker subdevice, 24-core width-sharded L1 input `(32,64)`, ten-core L1 output `(32,128)`, ten-core persistent L1 intermediate `(32,512)`, `experimental.all_reduce_async`, one global semaphore, and an immediate subdevice/full-mesh synchronize. This isolates whether the current machine/build still supports the repository's exact recipe. It is the highest-information experiment because shape, mesh, axis, link count, dtype, topology, dispatch, and fabric all match.
2. **Integrate that exact `experimental.all_reduce_async` recipe into the Attention test adapter.** Convert the DRAM QKV projection to the 24-core L1 input config, use the correctly sized persistent L1 buffer from the Galaxy resource plan, return BF16 in the ten-core output config, and synchronize once immediately after the call for the first diagnostic run. Do not introduce RS/AG tuning knobs. First use the canonical 50-core subdevice; only after it passes, test the full-grid CCL-only manager independently.
3. **A/B one link versus three links without changing anything else.** The generic exact-shape matrix explicitly covers both (`test_new_all_reduce.py:310`, `314`). Start with one link because it matches current resources; use three links only to distinguish a route/link allocation defect from an input/buffer contract defect.
4. **After generic all-reduce passes, remove the diagnostic synchronize and restore normal queue ordering.** This determines whether the barrier is necessary or merely masks resource lifetime/ordering. Keep tensors alive through synchronization; the passing tests do not deallocate inputs or persistent intermediates immediately after enqueue.
5. **Treat fused `all_reduce_create_qkv_heads` as the production-alignment follow-up, not the first unblocker.** It exactly matches QKV geometry and naturally emits heads, but requires extending `Attention2DLowLevelCallables` so decode can consume pre-created `(q,k,v)` without calling `nlp_create_qkv_heads_decode` again. Its test-proven three-link recipe is at `test_qkv_all_reduce_minimal.py:192-220`, `337-409`, and production usage at `llama_ccl.py:867-905`.
6. **Only if the exact generic recipe fails, reduce further to the nightly whole-grid line all-reduce.** That path uses a full-grid subdevice and explicit synchronization (`tests/nightly/tg/ccl/test_all_reduce_async.py:127-161`, `181-230`), but its nearest standard matrix has a different local batch dimension (`tests/nightly/tg/ccl/test_all_reduce.py:76-127`), so it is weaker evidence for Attention's exact geometry.

## Conclusion

The strongest source evidence does not indicate that `(1,1,32,1280)`, axis 1, Linear topology, or one link is intrinsically unsupported on WH Galaxy. The repository has an exact generic passing case for that combination. The leading mismatch is that Attention's active fallback bypasses the tested QKV CCL data contract: width-sharded L1 input, correctly sized persistent L1 intermediate, width-sharded L1 output, and a single `experimental.all_reduce_async` submission. The next hardware run should reproduce that contract exactly before varying subdevices, links, or synchronization.
