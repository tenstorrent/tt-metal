# Qwen3-32B WH Galaxy MLP Decode Audit

Timestamp: 2026-08-19 06:58:15 UTC

## Scope

Read-only source audit of Qwen3-32B decode MLP precision, mesh mapping, ring padding, collectives, and final output assembly. The legacy Wormhole Galaxy path under `models/demos/llama3_70b_galaxy` was compared with the common `MLP2D` implementation and its `(8, 4)` hardware case. No TT hardware was used and no shared implementation or test file was edited.

Model geometry:

- Global model width `D = 5120`.
- Global intermediate width `H = 25600`.
- Mesh shape `(8, 4)`: eight rows shard intermediate/K2, four columns shard input/output model width.
- Decode tensor shape `[1, 1, 32, D]`.
- `PAD_MLP_CORES` must be unset or `0` for Qwen3-32B. The model's logical hidden width remains `25600`; the `3840` ring width described below is storage/program padding only.

## Executive Finding

The common Qwen hardware case is not a literal legacy-precision reproduction, and one weight allocation is not a literal legacy-padding reproduction:

1. Legacy Qwen loads all three MLP weights as BF16, but runs the decode input, W1/W3 outputs, reduce-scatter outputs, SiLU/multiply, W2 output, and final all-reduce in BF8. The common case currently selects one `model_dtype = bfloat16` for weights and every activation/CCL stage.
2. Legacy forces each local W1/W3 weight allocation to physical N `3840`, matching the 24-core ring matmul program. The common helper minimally rounds logical N `3200` over 12 DRAM cores to only `3456`. Its ring program and output memory configuration still use `3840`.
3. Qwen's reduce-scatter result is logically `800` channels per mesh column, not `960`. Because logical `3200` is divisible by the 160-channel input shard width, the primitive uses `3200 / 4 = 800`; only 25 of the 30 output cores carry logical pages. The all-gather must consume logical `800` and write a logical `3200` persistent output into a ring memory configuration with physical capacity `3840`.
4. The final all-reduce must use the same ordered core sequences as the legacy/CCL tests. Multi-range `CoreRangeSet` construction from a Python `set` is not equivalent for these kernels even when the coordinate set is equal. The worker ranges, ring points, receiver points, and persistent-buffer grid must be built from ordered lists.

## Exact Mesh Mapping

At logical mesh coordinate `(r, c)`, where `0 <= r < 8` and `0 <= c < 4`:

```python
x_rc  = x[..., c * 1280 : (c + 1) * 1280]
w1_rc = w1[c * 1280 : (c + 1) * 1280, r * 3200 : (r + 1) * 3200]
w3_rc = w3[c * 1280 : (c + 1) * 1280, r * 3200 : (r + 1) * 3200]
w2_rc = w2[r * 3200 : (r + 1) * 3200, c * 1280 : (c + 1) * 1280]
```

The raw W1/W3 outputs are K-partial products of logical shape `[1, 1, 32, 3200]`. Summing the four column partials produces one row's 3200-channel gate/up block. After axis-1 reduce-scatter and all-gather, each `(r, c)` again receives that row's full logical 3200 channels for W2. W2 produces a logical 1280-channel model-width shard, and the axis-0 all-reduce sums the eight row/K2 partials. Concatenating the four column shards reconstructs global width 5120.

## End-to-End Shape Contract

| Stage | Logical per-device shape | Physical width/capacity | Required layout |
|---|---|---:|---|
| Decode input | `[1, 1, 32, 1280]` | `1536` | 24 ordered ring cores, shard `[32, 64]`; 256 right-tail storage channels |
| W1/W3 weight | `[1280, 3200]` | N `3840` | 12 DRAM cores, shard `[1280, 320]` |
| Raw W1/W3 projection | `[1, 1, 32, 3200]` | `3840` | 24 ordered receiver cores, shard `[32, 160]` |
| W1/W3 reduce-scatter | `[1, 1, 32, 800]` | output memcfg capacity `960` | ordered 30-core output grid, shard `[32, 32]`; primitive uses 25 logical cores |
| SiLU(W1) * W3 | `[1, 1, 32, 800]` | same as RS output | same RS output memory config |
| Axis-1 all-gather | `[1, 1, 32, 3200]` | `3840` | persistent output is logical 3200 in 24-core ring memcfg `[32, 160]` |
| W2 weight | `[3200, 1280]` | N `1536` | 12 DRAM cores, shard `[3200, 128]` |
| W2 output | `[1, 1, 32, 1280]` | `1536` | 24 ordered receiver cores, shard `[32, 64]` |
| Axis-0 all-reduce output | `[1, 1, 32, 1280]` | exact `1280` | 10 ordered output cores, shard `[32, 128]` |
| Composed result | `[1, 1, 32, 5120]` | exact `5120` | concatenate mesh columns; mesh rows are replicas after all-reduce |

### Why Qwen RS Is 800, Not 960

`LlamaReduceScatterDeviceOperation::compute_output_specs` uses:

```text
final_width = input_width / ring_devices
```

when `input_width % input_shard_width == 0`. For Qwen, `3200 % 160 == 0`, so `final_width = 800`. In the program factory, `ceil(3200 / 160) = 20` logical input cores, or five per ring device; source offsets are therefore `0`, `800`, `1600`, and `2400`. This differs from Llama's `3584` case, whose non-divisible width is padded to 3840 and scattered as four 960-channel slices.

## Legacy Precision Contract

| Stage | Legacy dtype/config |
|---|---|
| Materialized W1/W3/W2 weights | `ttnn.bfloat16` because `args.is_qwen` overrides the requested weight dtype |
| Decode input | `ttnn.bfloat8_b` |
| FF1/FF3 compute | HiFi2, `math_approx_mode=True`, `fp32_dest_acc_en=True`, `packer_l1_acc=True`, `dst_full_sync_en=True` |
| Raw W1/W3 output | `ttnn.bfloat8_b` |
| W1/W3 reduce-scatter and interim buffers | `ttnn.bfloat8_b` |
| SiLU/multiply | `ttnn.bfloat8_b` |
| Axis-1 all-gather persistent output | `ttnn.bfloat8_b` |
| FF2 compute/output | same HiFi2 kernel; `ttnn.bfloat8_b` output |
| Axis-0 all-reduce | no dtype override; preserves BF8 input/output |

The common default `_compute_kernel_config_hifi2_fp16()` is also not identical: it uses `math_approx_mode=False` and `fp32_dest_acc_en=False`. Literal legacy parity requires an explicit decode FF1/FF3 and FF2 kernel config rather than the common default.

## Required Common `MLP2D` Configuration

For literal legacy parity, configure:

- Logical dimensions: `dim=5120`, `hidden_dim=25600`, `max_batch_size=32`.
- Weight sources: W1/W3 `(5120, 25600)`, W2 `(25600, 5120)`, with the existing 2D mesh placements.
- Weight dtypes: BF16 for W1, W3, and W2.
- Decode activation, multiply, and CCL dtypes: BF8.
- Explicit legacy HiFi2 compute kernel for both decode matmul stages.
- W1/W3 DRAM memory config forced from ring N `3840`, producing shard `(1280, 320)` on 12 DRAM cores. Do not derive it by minimally padding logical N `3200`.
- W2 DRAM memory config from `(K=3200, padded N=1536)`, producing shard `(3200, 128)`.
- W1/W3 program: `K=1280`, padded `N=3840`, 24-core gather-in0 ring, per-core N five tiles.
- W2 program: logical `K=3200`, padded `N=1536`, per-core K four tiles and per-core N two tiles.
- Ring, receiver, worker, output, and semaphore `CoreRangeSet` values constructed from ordered lists matching legacy order.
- Ring topology and four links for all three decode collectives on WH Galaxy.
- Reduce-scatter resource key/input logical width `3200`, all-gather resource key/input logical width `800`, all-gather persistent output logical width `3200`, and final all-reduce key/input logical width `1280`.
- Final all-reduce output memcfg: ten cores with shard `[32, 128]`.
- Final all-reduce persistent interim buffer: worker subdevice's ordered 50-core grid with shard `[32, 1024]`, materialized from global shape `(8, 4, 32, 50 * 1024)` using mesh mapper `(0, 1)`.

## Common Hardware Case Deltas

The current common test has the following material differences:

1. `_weight_lazies(..., model_dtype)` gives Qwen BF16 to every weight, which matches legacy weight materialization.
2. The same `model_dtype` is then assigned to input upload, activation output, CCL resources, and multiply. This does not match legacy BF8 stage precision.
3. Generic `dram_sharded(local_k, local_n)` minimally pads Qwen W1/W3 local N `3200` to `3456`, yielding a 12-core shard width `288`; legacy forces physical N `3840`, shard width `320`.
4. W2 generic padding is correct by coincidence: local N `1280` rounds to `1536`, shard width `128`.
5. The ring and receiver point sets are now ordered lists, as required. The helper's sender and worker subdevice sets are also ordered. Remaining multi-range CCL memory-config grids must retain the same list discipline; the current all-reduce persistent-buffer helper still constructs its two worker ranges from a set.
6. The common all-gather resource contract correctly uses Qwen logical RS width `800` and a logical persistent output width `3200`.

## BF16 Qualification Evidence

The CCL 6U qualification matrix separately tests Qwen FF2/DO axis-0 all-reduce shape `[1, 1, 32, 1280]` in BF16. That is valid evidence for a qualified primitive recipe, but it is not evidence that the legacy MLP itself used BF16 throughout; the legacy MLP passes BF8 into the final all-reduce.

If literal BF8 parity still stalls after the weight geometry and core ordering are corrected, the narrow fallback is:

- keep BF16 weights;
- keep W1/W3, RS, multiply, and all-gather in BF8;
- produce/cast W2 output to BF16 and allocate only the final all-reduce resources as BF16.

Changing every Qwen stage to BF16 obscures whether the geometry/order defect is fixed and does not reproduce the legacy precision contract. The existing work log already shows that all-stage BF16 alone did not resolve the final all-reduce timeout.

## Source Evidence

- Legacy weight dtype override and 2D placements: `models/demos/llama3_70b_galaxy/tt/llama_mlp.py:55-105`.
- Legacy decode operation order and dtypes: `models/demos/llama3_70b_galaxy/tt/llama_mlp.py:119-295`.
- Legacy ring W1/W3/W2 geometry: `models/demos/llama3_70b_galaxy/tt/qwen_model_config.py:1187-1270`.
- Legacy RS output and persistent-buffer configs: `models/demos/llama3_70b_galaxy/tt/qwen_model_config.py:1505-1536` and `models/demos/llama3_70b_galaxy/tt/llama_ccl.py:350-512`.
- Legacy HiFi2 kernel: `models/demos/llama3_70b_galaxy/tt/qwen_model_config.py:362-382`.
- Qwen logical hidden width and optional model-level padding: `models/demos/llama3_70b_galaxy/tt/qwen_model_config.py:1705-1755`.
- RS logical-width branch: `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_device_operation.cpp:60-90`.
- RS logical-core and source-offset calculation: `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_program_factory.cpp:397-419,744-835`.
- Common decode pipeline: `models/common/modules/mlp/mlp_2d.py:337-470`.
- Common test geometry and dtype selection: `models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py:99-165,176-280,356-434`.
- Qualified Qwen BF16 all-reduce case: `tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py:755-785,850-870`.

## Conclusion

The exact logical contract is `3200 -> RS 800 -> AG 3200 -> W2 1280 -> AR 1280`, while ring storage remains padded to `3840` for the first four stages and `1536` for model-width input/output. The highest-priority correction is to force W1/W3 physical N to `3840` and preserve ordered CCL core ranges. Precision should then be tested with literal legacy BF16 weights plus BF8 stages; only the final W2/all-reduce should move to BF16 if the separately qualified primitive recipe is still required.
