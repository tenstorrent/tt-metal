# Galaxy/Llama `nlp_concat_heads_decode` Usage Audit

## Scope

Host-only source audit for the Milestone A Attention2D decode failure at
`ttnn.experimental.nlp_concat_heads_decode` with logical input shape
`(1, 8, 8, 128)`. No hardware was used. No production or test source was
edited as part of this audit.

## API contract

The primitive expects the padded input shape to be `(1, B, Hpad, D)` where
`B <= 32`, `Hpad` is tile-height padded, and the logical head count is passed
as `num_heads`. A logical `(1, 8, 8, 128)` tensor is therefore valid only when
its padded head dimension is 32. The nanobind description states that the op
unpads the requested heads and produces a width-sharded output
([`nlp_concat_heads_decode_nanobind.cpp:18`](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/nlp_concat_heads_decode_nanobind.cpp#L18),
[`nlp_concat_heads_decode_nanobind.cpp:22`](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/nlp_concat_heads_decode_nanobind.cpp#L22)).

The device validation requires:

- device-resident BF16/FP32 tile input
  ([`nlp_concat_heads_decode_device_operation.cpp:22`](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp#L22));
- padded shape `[0] == 1`, `[1] <= 32`, and `[2] % 32 == 0`
  ([`nlp_concat_heads_decode_device_operation.cpp:38`](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp#L38));
- HEIGHT_SHARDED input with one input core per user and shard shape
  `(padded_heads, head_dim)`
  ([`nlp_concat_heads_decode_device_operation.cpp:43`](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp#L43),
  [`nlp_concat_heads_decode_device_operation.cpp:49`](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp#L49),
  [`nlp_concat_heads_decode_device_operation.cpp:60`](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp#L60));
- when the input grid is not origin-based or `sub_core_grids` is supplied, a
  compute subgrid containing at least `num_heads` cores
  ([`nlp_concat_heads_decode_device_operation.cpp:61`](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp#L61)).

The op pads output batch to 32, emits logical shape
`(1, 1, 32, num_heads * head_dim)`, and intrinsically creates an L1
WIDTH_SHARDED output with one output core per head and shard shape
`(32, head_dim)`
([`nlp_concat_heads_decode_device_operation.cpp:72`](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp#L72),
[`nlp_concat_heads_decode_device_operation.cpp:85`](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp#L85),
[`nlp_concat_heads_decode_device_operation.cpp:92`](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp#L92),
[`nlp_concat_heads_decode_device_operation.cpp:105`](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp#L105)).

Most importantly, the `memory_config` argument is ignored by the primitive
([`nlp_concat_heads_decode_device_operation.cpp:125`](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp#L125),
[`nlp_concat_heads_decode_device_operation.cpp:128`](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp#L128)). A DRAM argument cannot make concat itself write DRAM. Any requested
downstream layout must be applied after concat.

## Production Galaxy/Llama usage

### Galaxy Llama/Qwen model configuration

Both Galaxy configs reserve the same Wormhole worker subdevice:
`(1,0)-(3,9)` plus `(5,0)-(6,9)`, with start core `(1,0)`
([`model_config.py:496`](models/demos/llama3_70b_galaxy/tt/model_config.py#L496),
[`model_config.py:507`](models/demos/llama3_70b_galaxy/tt/model_config.py#L507),
[`qwen_model_config.py:48`](models/demos/llama3_70b_galaxy/tt/qwen_model_config.py#L48),
[`qwen_model_config.py:222`](models/demos/llama3_70b_galaxy/tt/qwen_model_config.py#L222)).

Their decode SDPA output config is HEIGHT_SHARDED L1 with shard shape
`(ceil(n_local_heads / 32) * 32, head_dim)` and one core per local user. The
user cores are selected from the full worker subdevice with `row_wise=True`
([`model_config.py:1252`](models/demos/llama3_70b_galaxy/tt/model_config.py#L1252),
[`model_config.py:1255`](models/demos/llama3_70b_galaxy/tt/model_config.py#L1255),
[`model_config.py:1256`](models/demos/llama3_70b_galaxy/tt/model_config.py#L1256),
[`qwen_model_config.py:1000`](models/demos/llama3_70b_galaxy/tt/qwen_model_config.py#L1000),
[`qwen_model_config.py:1003`](models/demos/llama3_70b_galaxy/tt/qwen_model_config.py#L1003)). For `(1,8,8,128)`, this means eight row-wise input cores and per-core shard
shape `(32,128)`.

The direct no-prefetch Galaxy attention path requests that SDPA layout
([`llama_attention.py:764`](models/demos/llama3_70b_galaxy/tt/llama_attention.py#L764),
[`llama_attention.py:778`](models/demos/llama3_70b_galaxy/tt/llama_attention.py#L778)). It then gathers users from 8 to 32 before concat
([`llama_attention.py:790`](models/demos/llama3_70b_galaxy/tt/llama_attention.py#L790),
[`llama_attention.py:803`](models/demos/llama3_70b_galaxy/tt/llama_attention.py#L803)), derives `sub_core_grids` from the gathered input's shard grid, and passes
that grid to concat without a memory config
([`llama_attention.py:817`](models/demos/llama3_70b_galaxy/tt/llama_attention.py#L817),
[`llama_attention.py:821`](models/demos/llama3_70b_galaxy/tt/llama_attention.py#L821)). It explicitly moves concat output to DRAM for the following matmul
([`llama_attention.py:838`](models/demos/llama3_70b_galaxy/tt/llama_attention.py#L838),
[`llama_attention.py:845`](models/demos/llama3_70b_galaxy/tt/llama_attention.py#L845)).

The Galaxy CCL fallback follows the same rule: derive the decode compute grid
from the actual sharded concat input
([`llama_ccl.py:1526`](models/demos/llama3_70b_galaxy/tt/llama_ccl.py#L1526)), invoke concat first, and when a WIDTH_SHARDED target is requested, reshard in a
second step because passing that target directly is unsafe/ineffective
([`llama_ccl.py:1533`](models/demos/llama3_70b_galaxy/tt/llama_ccl.py#L1533),
[`llama_ccl.py:1544`](models/demos/llama3_70b_galaxy/tt/llama_ccl.py#L1544),
[`llama_ccl.py:1583`](models/demos/llama3_70b_galaxy/tt/llama_ccl.py#L1583)).

### Shared production Llama

The current `tt_transformers` attention path explicitly converts SDPA output
to its decode HEIGHT_SHARDED layout before concat and passes the full
prefetcher worker subdevice as `sub_core_grids`
([`attention.py:732`](models/tt_transformers/tt/attention.py#L732),
[`attention.py:739`](models/tt_transformers/tt/attention.py#L739),
[`attention.py:742`](models/tt_transformers/tt/attention.py#L742)). Its config selects one row-wise core per user from the full worker set, with shard
shape `(padded_local_heads, head_dim)`
([`model_config.py:1790`](models/tt_transformers/tt/model_config.py#L1790),
[`model_config.py:1797`](models/tt_transformers/tt/model_config.py#L1797),
[`model_config.py:1799`](models/tt_transformers/tt/model_config.py#L1799),
[`model_config.py:1805`](models/tt_transformers/tt/model_config.py#L1805)). It applies the downstream concat-output memory config only after the op
([`attention.py:747`](models/tt_transformers/tt/attention.py#L747)); the non-prefetch target is L1 WIDTH_SHARDED
([`model_config.py:1823`](models/tt_transformers/tt/model_config.py#L1823),
[`model_config.py:1840`](models/tt_transformers/tt/model_config.py#L1840)).

The common 1D and Quasar Llama implementations likewise explicitly convert
SDPA output to their decode score memcfg before calling concat
([`attention_1d.py:816`](models/common/modules/attention/attention_1d.py#L816),
[`attention_1d.py:822`](models/common/modules/attention/attention_1d.py#L822),
[`attention_1d.py:728`](models/experimental/llama32_1b_quasar/modules/attention/attention_1d.py#L728),
[`attention_1d.py:732`](models/experimental/llama32_1b_quasar/modules/attention/attention_1d.py#L732)).

### Shape-specific coverage

The Galaxy distributed unit explicitly covers `batch=8`, `head_dim=128`,
`n_local_heads=8`, and padded heads 32, exactly matching the local
`(1,8,8,128)` logical case
([`test_multidevice_TG.py:1029`](tests/ttnn/distributed/test_multidevice_TG.py#L1029),
[`test_multidevice_TG.py:1031`](tests/ttnn/distributed/test_multidevice_TG.py#L1031)). It uses eight origin-based input cores, HEIGHT_SHARDED L1, shard
shape `(32,128)`, and no explicit subgrid
([`test_multidevice_TG.py:1043`](tests/ttnn/distributed/test_multidevice_TG.py#L1043),
[`test_multidevice_TG.py:1045`](tests/ttnn/distributed/test_multidevice_TG.py#L1045),
[`test_multidevice_TG.py:1046`](tests/ttnn/distributed/test_multidevice_TG.py#L1046),
[`test_multidevice_TG.py:1068`](tests/ttnn/distributed/test_multidevice_TG.py#L1068)). The generic op test supplies an explicit compute subgrid whenever input cores are
not sufficient/appropriate and documents the requirement that it contain at
least `n_local_heads` cores
([`test_nlp_concat_heads_decode.py:45`](tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_concat_heads_decode.py#L45),
[`test_nlp_concat_heads_decode.py:50`](tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_concat_heads_decode.py#L50),
[`test_nlp_concat_heads_decode.py:88`](tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_concat_heads_decode.py#L88)).

## Current adapter mismatch

The Attention2D hardware adapter currently creates `head_cores` with
`row_wise=False` and creates `kv_cores` with only
`_BATCH_SIZE // mesh_columns == 2` cores
([`test_attention_2d_wh_galaxy.py:596`](models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py#L596),
[`test_attention_2d_wh_galaxy.py:599`](models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py#L599)). It correctly defines the eight-core Q-head layout at
[`test_attention_2d_wh_galaxy.py:655`](models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py#L655), but currently selects the two-core KV layout for SDPA output and passes the
same two-core `kv_cores` set as concat's compute subgrid
([`test_attention_2d_wh_galaxy.py:760`](models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py#L760),
[`test_attention_2d_wh_galaxy.py:762`](models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py#L762),
[`test_attention_2d_wh_galaxy.py:764`](models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py#L764)). For input batch 8, that directly violates both hard concat constraints: the
HEIGHT_SHARDED input must use exactly eight cores, and the compute subgrid must
contain at least eight cores. This differs from production Galaxy in three
coupled ways:

1. Production assigns user shards row-wise.
2. Production keeps the eight-user Q/SDPA layout distinct from the two-core KV
   cache-update layout.
3. Production treats the full worker set as the allowed compute subdevice; the
   concat op independently chooses `num_heads` row-wise output cores from it.

The output-grid selection is explicitly row-wise even when the input grid was
not: it starts at the input grid's first core and calls
`num_cores_to_corerangeset_in_subcoregrids(..., num_heads, ..., true)`
([`nlp_concat_heads_decode_device_operation.cpp:92`](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp#L92),
[`nlp_concat_heads_decode_device_operation.cpp:98`](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp#L98)). Constraining that row-wise allocation to a column-wise eight-core input subset is
not the production contract. The current two-core SDPA/concat selections are
the direct source-level contract violation; the column-wise head grid is an
additional mismatch to correct at the same time.

## Recommended adapter patch

Patch only `models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py`:

```diff
@@ def _decode_all_reduce_config(...):
     head_cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(
-        ttnn.CoreCoord(1, 0), _BATCH_SIZE, worker_cores, row_wise=False
+        ttnn.CoreCoord(1, 0), _BATCH_SIZE, worker_cores, row_wise=True
     )

@@ return {
         "head_cores": head_cores,
         "worker_cores": worker_cores,

@@ Attention2DConfig(...):
-    decode_sdpa_output_memory_config=decode_all_reduce["kv_output_memcfg"],
+    decode_sdpa_output_memory_config=decode_all_reduce["heads_output_memcfg"],
     decode_concat_memory_config=ttnn.DRAM_MEMORY_CONFIG,
-    decode_concat_sub_core_grids=decode_all_reduce["kv_cores"],
+    decode_concat_sub_core_grids=decode_all_reduce["worker_cores"],
```

Keep `decode_concat_memory_config=ttnn.DRAM_MEMORY_CONFIG`. In Attention2D it
is correctly enforced after the primitive when the intrinsic output differs
([`attention_2d.py:897`](models/common/modules/attention/attention_2d.py#L897),
[`attention_2d.py:907`](models/common/modules/attention/attention_2d.py#L907)). It should not be interpreted as an output override for concat itself.

If the same failure remains after this adapter-only correction, the next
source-aligned diagnostic is to inspect the actual SDPA output memory config
immediately before concat. It must be HEIGHT_SHARDED on exactly eight cores
with shard shape `(32,128)`. If SDPA did not honor the requested output layout,
insert an explicit `ttnn.to_memory_config(attention,
decode_all_reduce["heads_output_memcfg"])` adapter transition before concat,
matching shared production Llama's explicit conversion at
[`attention.py:732`](models/tt_transformers/tt/attention.py#L732). That is a secondary fallback; the row-wise/full-worker correction is the minimal first patch.

## Conclusion

`(1,8,8,128)` is a supported concat input when padded to
`(1,8,32,128)`. The required input is BF16 tile, HEIGHT_SHARDED L1 across
eight row-wise user cores with shard shape `(32,128)`. On the Galaxy worker
subdevice, pass the full worker `CoreRangeSet` as `sub_core_grids`; concat will
choose eight row-wise output cores and intrinsically return L1 WIDTH_SHARDED
`(1,1,32,1024)`. Apply DRAM or another WO layout in a separate post-concat
transition.
