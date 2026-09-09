# Reduce-helper migration sanity suite

**77 exact test cases: 62 Python cases and 15 C++ GTests.** They provide one primary numerical case for each of **130 of 144 kernel entries**. Shared coverage is deduplicated: one case can cover compute plus reader/writer kernels. The 14 uncovered entries are listed below.

| Kernel category | Covered | Inventory total |
| --- | ---: | ---: |
| Compute | 51 | 58 |
| Dataflow calling old helpers | 62 | 69 |
| Manual auxiliary readers | 17 | 17 |
| Total | 130 | 144 |

This is a subset of the existing [full unit suite](unit_test_suite.html), which collected 18,452 cases. The original inventory counts source kernels, including three compute/dataflow pairs embedded in Python. A kernel can run incidentally in additional cases; one primary case is assigned to it in the coverage map. This does not cover every template instantiation, layout, accumulation mode or helper call site.

```bash
# Inspect the selection without importing TTNN or opening devices.
python scripts/run_reduce_migration_sanity.py --list

# Usual single-device selections; choose the line for your architecture.
python scripts/run_reduce_migration_sanity.py --lane common --lane wormhole
python scripts/run_reduce_migration_sanity.py --lane common --lane blackhole

# Other lanes run separately on the matching hardware, for example:
python scripts/run_reduce_migration_sanity.py --lane wormhole-n300

# Inspect/collect the whole suite, or run a kernel's primary case.
python scripts/run_reduce_migration_sanity.py --dry-run
python scripts/run_reduce_migration_sanity.py --collect-only
python scripts/run_reduce_migration_sanity.py --kernel S071

# Select every case, including those needing other hardware.
python scripts/run_reduce_migration_sanity.py
```

Every invocation goes through `scripts/run_safe_pytest.sh`, with serial execution, `--run-all` and `--no-precompile` by default. The C++ adapter runs one named GTest per pytest case. Required binaries are `build/test/ttnn/unit_tests_ttnn` and, for UDM, `build/test/ttnn/unit_tests_ttnn_udm`. The build must match the checkout being tested.

Each group must collect exactly one case. During execution, a skip, xfail, missing test or any failure makes the runner return nonzero; it continues other groups and saves their results. Collection does not establish that a case will run on the current hardware. Hardware lanes are explicit selectors, not hardware discovery or a guarantee that every SKU/debug configuration is supported. Running all lanes on one machine is generally not possible.

The `{arch}` token in a few node IDs substitutes only the `silicon_arch_name` fixture from pytest's `--tt-arch`; all numerical parameters remain fixed. The plugin selects the resulting exact node and the runner checks its cardinality. For a collection targeting another build, pass e.g. `-- --tt-arch=blackhole`.

The inline examples retain their existing fixed width loops (compute fusion: 2 tile counts; reduce accumulate: 4; row reduce accumulate: 6), but the runner pins their supported environment selectors to the helper variants. DiT's correctness sweep is pinned to `cross_k_prompt_L512` and `CORR_DET_REPEATS=0`, keeping its numerical check and removing extra determinism launches. Moreh callback tests and some other tests also make several calls internally. **77 is a pytest-case count, not a device-launch count.**

Results go to a new directory under `generated/test_reports/`, with per-case logs, JUnit XML, collection metadata and `summary.json`. `--output-dir` must name a new directory, so previous results are preserved.

## Hardware lanes

| Lane | Cases | Requirements |
| --- | ---: | --- |
| `common` | 58 | One device; shared Wormhole/Blackhole test sources. Individual core-grid and debug-mode constraints still apply. |
| `wormhole` | 4 | One Wormhole device: two Moreh layernorm backward cases, the Falcon causal-mask operation case and the BGE encoder SDPA case. |
| `blackhole` | 5 | One Blackhole device: KDA, indexer and sparse attention cases. |
| `quasar` | 2 | Single logical Quasar device/emulator; requires that environment and build. |
| `fabric-1x4` | 2 | Four devices, 1x4 mesh with 2D fabric; requires unit_tests_ttnn_udm. |
| `wormhole-n300` | 1 | Exactly two Wormhole devices (N300), 2x1 mesh. |
| `blackhole-2x4` | 1 | Exactly eight Blackhole devices in a 2x4 box. |
| `galaxy` | 2 | Existing DiT fixture opens a 4x8 Galaxy mesh (32 devices), then TP1/TP2 submeshes. |
| `wormhole-t3k` | 1 | Eight Wormhole devices (2x4 T3K). |
| `blackhole-galaxy` | 1 | Blackhole Galaxy cluster; the selected experimental ring case uses a 1x4 submesh. |

Common denotes shared single-device test sources, not verification on every SKU or debug configuration. Running every lane on one machine is generally not possible.

## Coverage gaps

These are missing executable paths, not cases silently skipped by the suite.

- `S045`, `S046`, `S047`, `DF011`, `DF012`, `DF013`, `DF015`, `DF016`: No direct unit test calls the experimental Quasar generic-reduction entry points. Quasar-named ResNet sum/mean tests call the standard entry points.
- `DF014`: No host factory references this Quasar transpose reader.
- `DF017`: No direct unit test found for experimental Quasar joint SDPA.
- `S066`, `S067`: Legacy Moreh norm H/W sources have no current host factory; ord_other variants are covered separately.
- `S089`: Legacy non-metal2 RMS post-all-gather appears only in a source-composition test; that test does not execute the kernel.
- `S092`: Legacy RMS pre-all-gather 2D source has no current host reference; the active 2D factory uses layernorm_pre_allgather_2d.cpp.

The disabled general large-H softmax factory does not add a kernel gap: `SM036` explicitly runs Moreh `LARGE_H`, which uses the same compute and reader sources (`S071`, `DF029`).

## Verification

Collected 77 cases across 77 groups with 0 failed groups. No on-device test bodies were run. Host checks verified failure propagation for skips, xfails, assertions and case-count drift, plus continuation to a later passing case. 4 architecture-dependent selections also collected exactly once with --tt-arch=blackhole. Selection was checked against factory/test sources at `f808380a87b320e24457c600cc79b05d7a0b8f73`. Python/script-only additions require no C++ build.

## Exact selections and kernel map

[Collection evidence](sanity_test_collection.json). See the [HTML report](sanity_test_suite.html), [manifest](sanity_test_suite.json), and [one-row-per-kernel CSV](sanity_kernel_coverage.csv).

### SM001 — C++ reductions

Lane: `common`. Kernels: `S099`, `DF055`.

BF16 tiled [64,64], mean over H: interleaved H transpose reader and reduce.cpp.

```text
tests/ttnn/unit_tests/gtests/test_reduction.cpp::ReductionSmoke.MeanReduceH
```

### SM002 — C++ reductions

Lane: `common`. Kernels: `DF053`.

BF16 tiled [64,64], mean over W: universal reduce reader (compute already covered).

```text
tests/ttnn/unit_tests/gtests/test_reduction.cpp::ReductionSmoke.MeanReduceW
```

### SM003 — C++ reductions

Lane: `common`. Kernels: `S102`, `DF052`.

ROW_MAJOR [1,1,64,64], sum W: row-major reduce compute and reader.

```text
tests/ttnn/unit_tests/gtests/test_reduction.cpp::ReductionSmoke.RowMajorSumW
```

### SM004 — C++ reductions

Lane: `common`. Kernels: `S105`, `DF057`.

32 users, width 64, top-k=1: deterministic sampled indices. Sampling still runs its MAX/SUM softmax helpers.

```text
tests/ttnn/unit_tests/gtests/test_reduction.cpp::ReductionSmoke.SamplingGreedyTopK1
```

### SM005 — C++ normalization

Lane: `common`. Kernels: `S079`, `DF035`.

TILE [1,1,32,64], two groups, 1x1 core grid, legacy (non-Welford) groupnorm.

```text
tests/ttnn/unit_tests/gtests/test_normalization.cpp::NormalizationSmoke.GroupNormNoMcastInterleaved
```

### SM006 — GroupNorm sharded formats

Lane: `common`. Kernels: `S080`, `DF036`.

ROW_MAJOR [1,1,512,128], four HEIGHT_SHARDED cores, BF16 input/affine, BF8 mask and FP32 destination accumulation. Four tiles per group force native mean reduction and guard mask-to-reduce format reconfiguration.

```text
tests/ttnn/unit_tests/operations/fused/test_group_norm.py::test_group_norm_sharded_all_config[legacy-row_major-bf16-gb_bf16-N=1-C=128-H=1-W=512-num_groups=16-grid_y=1-grid_x=4]
```

### SM007 — C++ normalization

Lane: `common`. Kernels: `DF037`.

Tiled BF16 [1,1,32,64], no gamma/beta: ordinary interleaved layernorm reader.

```text
tests/ttnn/unit_tests/gtests/test_normalization.cpp::NormalizationSmoke.LayerNormInterleaved
```

### SM008 — C++ normalization

Lane: `common`. Kernels: `DF039`.

Tiled BF16 [1,1,32,64], ROW_MAJOR gamma/beta: rm_gb interleaved reader.

```text
tests/ttnn/unit_tests/gtests/test_normalization.cpp::NormalizationSmoke.LayerNormRowMajorGammaBeta
```

### SM009 — C++ normalization

Lane: `common`. Kernels: `S083`, `DF040`.

Tiled BF16 [1,1,32,64], WIDTH_SHARDED 1x1, use_welford defaults false, no ROW_MAJOR weights.

```text
tests/ttnn/unit_tests/gtests/test_normalization.cpp::NormalizationSmoke.LayerNormSharded1x1
```

### SM010 — C++ normalization

Lane: `common`. Kernels: `S087`, `DF045`.

One device, tiled [1,1,32,64]: layernorm pre-all-gather statistics, two tiles wide.

```text
tests/ttnn/unit_tests/gtests/test_normalization.cpp::NormalizationSmoke.DistributedLayerNormStats
```

### SM011 — C++ normalization

Lane: `common`. Kernels: `S090`, `S091`, `DF044`.

One-device pre/post RMSNorm pipeline [1,1,32,64], num_devices=1; post uses metal2 compute.

```text
tests/ttnn/unit_tests/gtests/test_normalization.cpp::NormalizationSmoke.DistributedRmsNormEndToEnd
```

### SM012 — C++ normalization

Lane: `common`. Kernels: `S088`, `DF043`.

One device, [1,1,32,64], use_2d_core_grid=true: 2D pre-all-gather worker and merge code.

```text
tests/ttnn/unit_tests/gtests/test_normalization.cpp::NormalizationSmoke.DistributedRmsNorm2DGridStats
```

### SM013 — C++ normalization

Lane: `common`. Kernels: `S093`, `DF046`.

Tiled [1,1,32,32], last-dimension stable attention softmax.

```text
tests/ttnn/unit_tests/gtests/test_normalization.cpp::NormalizationSmoke.SoftmaxInterleavedUniform
```

### SM014 — C++ normalization

Lane: `common`. Kernels: `S095`, `DF048`.

Tiled [1,1,32,32], one BLOCK_SHARDED core, no mask: ordinary sharded reader.

```text
tests/ttnn/unit_tests/gtests/test_normalization.cpp::NormalizationSmoke.SoftmaxShardedInPlace
```

### SM015 — generic reductions

Lane: `common`. Kernels: `DF054`.

BF16 [8,1,64,2048], WIDTH_SHARDED input/output: sharded input-column transpose reader.

```text
tests/ttnn/nightly/unit_tests/operations/reduction/test_reduce.py::test_sharded_reduce_h[dtype=DataType.BFLOAT16-out_sharded-in0_sharded-N=8]
```

### SM016 — generic reductions

Lane: `common`. Kernels: `DF051`.

Matched HEIGHT_SHARDED input/output on a 2x4 core grid, int32 MAX W: sharded reader with REDUCE_SCALER=1; only dataflow coverage is claimed.

```text
tests/ttnn/nightly/unit_tests/operations/reduction/test_min_max.py::test_reduce_w_height_sharded_orientation_matched[op=max-row_major]
```

### SM017 — MoE reduction

Lane: `common`. Kernels: `S104`, `DF056`.

Smallest existing MoE numerical shape: [1,1,32,64], eight experts, two selected.

```text
tests/ttnn/unit_tests/operations/reduce/test_moe.py::test_moe[N=1-C=1-H=32-W=64-k=32-E=8-e=2-BFLOAT16_B]
```

### SM018 — DeepSeek grouped gate

Lane: `common`. Kernels: `S051`, `DF020`.

Existing minimal grouped-gate numerical configuration.

```text
tests/ttnn/nightly/unit_tests/operations/reduction/test_deepseek_grouped_gate.py::test_grouped_gate[minimal_case]
```

### SM019 — SSM sum reduction

Lane: `common`. Kernels: `S052`, `DF021`.

BF16 H=32, W=1024, latent_size=32, DRAM input/output.

```text
tests/ttnn/nightly/unit_tests/operations/ssm/test_ssm_1d_sum_reduce.py::test_ssm_reduce[H=32-W=1024-latent_size=32-dtype=DataType.BFLOAT16-in_mem_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0,per_core_allocation=0,range_lockstep_allocation=0)-out_mem_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0,per_core_allocation=0,range_lockstep_allocation=0)]
```

### SM020 — Moreh clip_grad_norm

Lane: `common`. Kernels: `S056`, `DP002`.

Smallest existing clip-grad case, p=2, max_norm=2; step1 plus subsequent steps are checked. Contains two internal repetitions.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_clip_grad_norm.py::test_moreh_clip_grad_norm[num_parameters=32-norm_type=2.0-max_norm=2.0-range_of_wt=(1, 4)-range_of_ht=(1, 4)-range_of_c=(1, 4)-range_of_n=(1, 4)-range_of_padding=(0, 21, 10)-num_iters_of_each_case=2]
```

### SM021 — Moreh dot

Lane: `common`. Kernels: `S057`, `DF025`.

One aligned BF16 tile row, dot length 32.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_dot.py::test_moreh_dot[dtype=DataType.BFLOAT16-input_shape=[1, 1, 1, 32]]
```

### SM022 — Moreh group_norm

Lane: `common`. Kernels: `S059`, `DP004`.

Groupnorm small algorithm; N=2,C=4,groups=1,H=W=23, affine and statistics checked; direct case avoids the upstream helper’s unconditional skip.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_normalization_reduce_boundaries.py::test_moreh_group_norm_reduce_boundaries[fp32_dest_acc_en=False-affine=True-groups=1-small]
```

### SM023 — Moreh group_norm

Lane: `common`. Kernels: `S058`, `DP003`.

Groupnorm large algorithm; N=2,C=4,groups=1,H=W=500, affine and statistics checked; direct case avoids the upstream helper’s unconditional skip.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_normalization_reduce_boundaries.py::test_moreh_group_norm_reduce_boundaries[fp32_dest_acc_en=False-affine=True-groups=1-large]
```

### SM024 — Moreh group_norm

Lane: `common`. Kernels: `S060`, `S062`, `DP005`, `DP007`.

Small groupnorm backward, all gradients enabled: gamma/beta kernel plus small input-grad kernel.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_group_norm.py::test_moreh_group_norm_backward[beta_requires_grad=True-gamma_requires_grad=True-input_requires_grad=True-affine=True-eps=1e-05-HW=[23, 23]-C_num_groups=[4, 1]-N=2]
```

### SM025 — Moreh group_norm

Lane: `common`. Kernels: `S061`, `DP006`.

Dedicated large groupnorm backward case; input gradient only, H=W=500.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_group_norm.py::test_moreh_group_norm_backward_large_algorithm[beta_requires_grad=False-gamma_requires_grad=False-input_requires_grad=True-affine=False-eps=1e-05-HW=[500, 500]-C_num_groups=[4, 1]-N=2]
```

### SM026 — Moreh layer_norm

Lane: `common`. Kernels: `DP009`.

Layernorm small reader; [1,20], one normalized dimension.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py::test_moreh_layer_norm[input_shape_normalized_dims=([1, 20], 1)-elementwise_affine=False-bfloat16-1e-5]
```

### SM027 — Moreh layer_norm

Lane: `common`. Kernels: `DP008`.

Layernorm large reader: all four dimensions normalized, 512 inner tiles exceed resident buffers.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py::test_moreh_layer_norm_compute_kernel_options[fp32_dest_acc_en=False-input_shape_normalized_dims=([4, 8, 111, 113], 4)-elementwise_affine=False-bfloat16-0.05]
```

### SM028 — Moreh layer_norm

Lane: `wormhole`. Kernels: `DP010`, `DP012`.

Layernorm backward with width normalization and affine gradients; exercises the parameter H reduction and small input-gradient kernel.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py::test_moreh_layer_norm_backward[input_shape_normalized_dims=([2, 20, 30], 1)-elementwise_affine=True-bfloat16-1e-5]
```

### SM029 — Moreh layer_norm

Lane: `wormhole`. Kernels: `DP011`.

Dedicated layernorm large backward [1,2,500,1000]; input gradient, two normalized dimensions.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py::test_moreh_layer_norm_backward_large_algorithm[fp32_dest_acc_en=False-[1,2,500,1000]-normalized_dims=2-elementwise_affine=False-bfloat16-1e-5]
```

### SM030 — Moreh linear

Lane: `common`. Kernels: `S063`, `DF026`.

Bias gradient with width 30: reduce H.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_linear.py::test_moreh_linear_backward[BFP16-fp32_dest_acc_en=False-requires_bias_grad=True-requires_grads=(True, False)-shapes=([31, 31], [30, 31], [1, 30], [31, 30])]
```

### SM031 — Moreh linear

Lane: `common`. Kernels: `S064`, `DP013`.

Scalar bias gradient: reduce H and W.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_linear.py::test_moreh_linear_backward[BFP16-fp32_dest_acc_en=False-requires_bias_grad=True-requires_grads=(True, False)-shapes=([31, 31], [30, 31], [1, 1], [31, 30])]
```

### SM032 — Moreh mean

Lane: `common`. Kernels: `S065`, `DF027`.

BF16 [17,22], dim=0 (height), keepdim=true.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_mean.py::test_moreh_mean_ttnn_dtype[ttnn_dtype=DataType.BFLOAT16-keepdim=True-input_shape_dim=[[17, 22], [0]]]
```

### SM033 — Moreh norm

Lane: `common`. Kernels: `S068`, `DP014`.

BF16 rank-2 [32,32], p=0.0, dim=0. Upstream IDs N/C mean dim 0/1 here, therefore H/W; ord_other kernel.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_norm.py::test_moreh_norm[is_linalg_vector_norm=False-ttnn_dtype=DataType.BFLOAT16-keepdim=True-input_shape=[32, 32]-N-p=0.0]
```

### SM034 — Moreh norm

Lane: `common`. Kernels: `S069`, `DP015`.

BF16 rank-2 [32,32], p=0.0, dim=1. Upstream IDs N/C mean dim 0/1 here, therefore H/W; ord_other kernel.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_norm.py::test_moreh_norm[is_linalg_vector_norm=False-ttnn_dtype=DataType.BFLOAT16-keepdim=True-input_shape=[32, 32]-C-p=0.0]
```

### SM035 — Moreh softmax

Lane: `common`. Kernels: `S070`, `DF028`.

Explicit Moreh softmax strategy SMALL_H, BF16 [32,32]. The explicit strategy reaches large kernels without a large tensor.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py::test_softmax_callback[dtype=DataType.BFLOAT16-shape_dim_strategy=[[32, 32], 0, MorehSoftmaxOpParallelizationStrategy.SMALL_H]]
```

### SM036 — Moreh softmax

Lane: `common`. Kernels: `S071`, `DF029`.

Explicit Moreh softmax strategy LARGE_H, BF16 [32,32]. The explicit strategy reaches large kernels without a large tensor.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py::test_softmax_callback[dtype=DataType.BFLOAT16-shape_dim_strategy=[[32, 32], 0, MorehSoftmaxOpParallelizationStrategy.LARGE_H]]
```

### SM037 — Moreh softmax

Lane: `common`. Kernels: `S072`, `DF030`.

Explicit Moreh softmax strategy SMALL_W, BF16 [32,32]. The explicit strategy reaches large kernels without a large tensor.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py::test_softmax_callback[dtype=DataType.BFLOAT16-shape_dim_strategy=[[32, 32], 1, MorehSoftmaxOpParallelizationStrategy.SMALL_W]]
```

### SM038 — Moreh softmax

Lane: `common`. Kernels: `S073`, `DF031`.

Explicit Moreh softmax strategy LARGE_W, BF16 [32,32]. The explicit strategy reaches large kernels without a large tensor.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py::test_softmax_callback[dtype=DataType.BFLOAT16-shape_dim_strategy=[[32, 32], 1, MorehSoftmaxOpParallelizationStrategy.LARGE_W]]
```

### SM039 — Moreh softmax

Lane: `common`. Kernels: `S074`, `DP017`.

Explicit Moreh softmax backward strategy SMALL_H, BF16 [32,32].

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py::test_softmax_backward_callback[dtype=DataType.BFLOAT16-shape_dim_strategy=[[32, 32], 0, MorehSoftmaxBackwardOpParallelizationStrategy.SMALL_H]]
```

### SM040 — Moreh softmax

Lane: `common`. Kernels: `S075`, `DP016`.

Explicit Moreh softmax backward strategy LARGE_H, BF16 [32,32].

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py::test_softmax_backward_callback[dtype=DataType.BFLOAT16-shape_dim_strategy=[[32, 32], 0, MorehSoftmaxBackwardOpParallelizationStrategy.LARGE_H]]
```

### SM041 — Moreh softmax

Lane: `common`. Kernels: `S076`, `DF032`.

Explicit Moreh softmax backward strategy SMALL_W, BF16 [32,32].

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py::test_softmax_backward_callback[dtype=DataType.BFLOAT16-shape_dim_strategy=[[32, 32], 1, MorehSoftmaxBackwardOpParallelizationStrategy.SMALL_W]]
```

### SM042 — Moreh softmax

Lane: `common`. Kernels: `S077`, `DF033`.

Explicit Moreh softmax backward strategy LARGE_W, BF16 [32,32].

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py::test_softmax_backward_callback[dtype=DataType.BFLOAT16-shape_dim_strategy=[[32, 32], 1, MorehSoftmaxBackwardOpParallelizationStrategy.LARGE_W]]
```

### SM043 — Moreh sum

Lane: `common`. Kernels: `S078`, `DF034`.

BF16 [3,2,319,319], dim=2: Moreh height sum compute/reader.

```text
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_sum.py::test_moreh_sum[bfloat16-keepdim-true-fp32_dest_acc_en=False-2-3, 2, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1]
```

### SM044 — layer norm / RMS norm

Lane: `common`. Kernels: `DF042`.

Sharded layernorm with ROW_MAJOR gamma/beta and legacy compute selects rm_gb writer.

```text
tests/ttnn/unit_tests/operations/fused/test_layer_norm_sharded.py::test_layer_norm_sharded_with_weight_and_bias_row_major[dtype=torch.bfloat16-tensor_type=ascending_values_repeated_rows-two_stage=False-use_welford=False]
```

### SM045 — layer norm / RMS norm

Lane: `common`. Kernels: `S085`, `DF041`.

One physical device simulates four partitions; sharded layernorm PRE_ALL_GATHER selects pre-all-gather compute/writer.

```text
tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py::test_pre_allgather_layernorm[fuse_residual=False-max_atol_ex2=0.04-min_pcc_ex2=0.982-min_pcc_residual_add=0.997-min_pcc_ex=0.9997-max_atol_ex=0.01-core_grid=(8, 4)-mean=0-std=1-input_df=DataType.BFLOAT16-num_devices=4-input_width=2048-seed=0-is_rmsnorm=False]
```

### SM046 — layer norm / RMS norm

Lane: `common`. Kernels: `DF038`.

FP32 tiled input with TILE gamma/beta, W padded to 4096: >2 MiB resident buffer demand forces the large-tensor reader; legacy compute.

```text
tests/ttnn/unit_tests/operations/fused/test_layer_norm.py::test_large_layer_norm_with_weight_bias_and_residual_input[dtype=torch.float32-use_welford=False-h=19-w=4083]
```

### SM047 — softmax

Lane: `common`. Kernels: `S094`, `DF047`.

Width 637 tiles: resident stable-softmax buffers exceed 90% of L1; dedicated large-kernel numerical test.

```text
tests/ttnn/unit_tests/operations/fused/test_softmax.py::test_softmax_large_non_divisible_width[Wt=637]
```

### SM048 — softmax

Lane: `common`. Kernels: `DF050`.

Sharded softmax with ROW_MAJOR attention mask selects the row-major-mask reader.

```text
tests/ttnn/nightly/unit_tests/operations/fused/test_softmax_sharded.py::test_scale_mask_softmax_rm[bfloat8_b-in0_DRAM-no-causal-device_params={'l1_small_size': 8192}]
```

### SM049 — model operation unit tests

Lane: `wormhole`. Kernels: `DF049`.

Single-device Falcon operation test, seq=64; calls scale_causal_mask_hw_dims_softmax_in_place with tiled mask.

```text
models/demos/t3000/falcon40b/tests/unit_tests/test_falcon_softmax.py::test_FalconSoftmax_inference[{arch}-1x1_grid-0.99]
```

### SM050 — indexer score

Lane: `blackhole`. Kernels: `S039`, `DF009`.

Indexer block-pool fallback, one group and 16 blocks per unit (>8), reaches the reduce-helper branch.

```text
tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_indexer_score.py::test_indexer_score_block_pool_large_blocks_per_unit[g1-bpu16]
```

### SM051 — KDA recurrence preparation

Lane: `blackhole`. Kernels: `S040`, `DP001`.

Small KDA contract case: 2 heads, 4 chunks, K=32,V=64; output and determinism checked.

```text
tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_prepare_chunk_recurrence.py::test_prepare_chunk_recurrence_contract_accuracy_and_determinism[unit-h2-n4-k32-v64]
```

### SM052 — KDA gated RMS norm

Lane: `blackhole`. Kernels: `S041`, `DF010`.

Basic BF16 gated RMSNorm numerical case.

```text
tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_sigmoid_gated_rms_norm.py::test_sigmoid_gated_rms_norm_is_accurate_and_deterministic[bf16-output-bf16-input]
```

### SM053 — fused distributed RMS norm

Lane: `common`. Kernels: `S054`, `S055`, `DF023`, `DF024`.

Single-device simulated distributed RMSNorm, one head, sequence 128, hidden 1024, pre/post fusion.

```text
tests/ttnn/nightly/unit_tests/operations/transformers/test_distributed_fused_rmsnorm.py::test_distributed_fused_rmsnorm_sweep_shapes[num_simulated_devices8-has_rope-has_weight-num_heads1-hidden_dim1024-seqlen128-BFLOAT16_stats-BFLOAT16_in]
```

### SM054 — DiT distributed layer norm

Lane: `common`. Kernels: `DF022`.

Single-device DiT layernorm pipeline, TP=1, sequence 512, width 2048, no affine.

```text
tests/ttnn/nightly/unit_tests/operations/transformers/test_distributed_dit_layernorm.py::test_distributed_dit_layernorm_use_cases[tp1-tile_layout-no_affine-len512-dim2048-BFLOAT16_stats-BFLOAT16_in]
```

### SM055 — kernel library examples

Lane: `common`. Kernels: `S108`, `DF065`.

Inline reduce+reciprocal correctness scenario; two fixed tile counts, fused/unfused variants.

```text
tests/ttnn/unit_tests/operations/examples/test_compute_fusion.py::test_compute_fusion_correctness
```

Pinned environment: `CF_SCENARIOS=reduce_recip`.

### SM056 — kernel library examples

Lane: `common`. Kernels: `S109`, `DF066`.

Inline helper variant, row reduction, BF16 accumulation; four fixed tile counts.

```text
tests/ttnn/unit_tests/operations/examples/test_reduce_accumulate.py::test_reduce_accumulate_correctness
```

Pinned environment: `RA_VARIANTS=helper`, `RA_DIMS=row`, `RA_ACCUMS=bf16`.

### SM057 — kernel library examples

Lane: `common`. Kernels: `S111`, `DF067`.

Inline reduce_fold method, BF16 input/accumulation, positive inputs; six fixed tile counts.

```text
tests/ttnn/unit_tests/operations/examples/test_row_reduce_accumulate.py::test_row_reduce_accumulate_correctness
```

Pinned environment: `RRA_METHODS=reduce_fold`, `RRA_PRECISIONS=bf16-bf16`, `RRA_DISTS=positive`.

### SM058 — toy variance

Lane: `common`. Kernels: `S113`, `DF068`.

Interleaved toy variance, aligned width 256.

```text
tests/ttnn/unit_tests/operations/toy_variance/test_toy_variance.py::test_toy_variance[variance-W=256_aligned]
```

### SM059 — toy variance

Lane: `common`. Kernels: `S114`, `DF069`.

Sharded toy variance, two cores on one device, width 64 (one tile per core).

```text
tests/ttnn/unit_tests/operations/toy_variance/test_toy_variance.py::test_toy_variance_width_sharded[variance-P=2_W=64_one_tile_each]
```

### SM060 — SDPA prefill / chunked

Lane: `common`. Kernels: `DF063`.

Standard SDPA, small causal BF16 sequence 256, head_dim 32; numerical check includes attention sink.

```text
tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py::test_sdpa_with_attention_sink[b=1-nh=8-nkv=1-s=256-d=32-k128-q32-causal-bf16]
```

### SM061 — SDPA / MLA decode

Lane: `common`. Kernels: `DF064`.

Decode SDPA, batch 4, sequence 1024, head_dim128, 8x4 grid, sliding window128.

```text
tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py::test_sdpa_decode_sliding_window[cur_pos_tensor-b=4-nh=8-nkv=1-s=1024-d=128-grid_size=(8, 4)-sliding_window_size=128-kv_bfp8_q_bf16]
```

### SM062 — joint SDPA

Lane: `common`. Kernels: `DF059`.

Smallest joint SDPA shape: sequence15 + joint19, one head and one batch.

```text
tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_joint.py::test_joint_sdpa[d128-seq_len=15-joint_seq_len=19-nh1-b1-k128-q32-bfp8]
```

### SM063 — sparse SDPA

Lane: `blackhole`. Kernels: `DF062`.

Single-chunk sparse SDPA with all indices valid; PCC against a golden reference.

```text
tests/ttnn/unit_tests/operations/sdpa/test_sparse_sdpa.py::test_sparse_sdpa_pcc_single_chunk[-all_valid]
```

### SM064 — sparse SDPA MSA

Lane: `blackhole`. Kernels: `DF061`.

Native MSA sparse SDPA random selection, H=16,S=8,topk=16; compares device output with golden.

```text
tests/ttnn/unit_tests/operations/sdpa/test_sparse_sdpa_msa.py::test_msa_native_pcc_random_selection
```

### SM065 — Quasar model per-op tests

Lane: `quasar`. Kernels: `DF018`.

Quasar SDPA per-op numerical case, seq128, batch1.

```text
models/experimental/llama32_1b_quasar/tests/ops/test_scaled_dot_product_attention.py::test_scaled_dot_product_attention[seq128-ttnn_mesh_device0]
```

### SM066 — Quasar model per-op tests

Lane: `quasar`. Kernels: `DF019`.

Quasar decode SDPA per-op numerical case, batch1, KV capacity256.

```text
models/experimental/llama32_1b_quasar/tests/ops/test_scaled_dot_product_attention_decode.py::test_scaled_dot_product_attention_decode[batch1-ttnn_mesh_device0]
```

### SM067 — UDM interleaved reduction

Lane: `fabric-1x4`. Kernels: `DF002`.

Small interleaved UDM width reduction.

```text
tests/ttnn/unit_tests/gtests/udm/reduction/interleaved/test_udm_reduction_interleaved.cpp::MeshDevice1x4Fabric2DUDMFixture.TestWidthReductionInterleaved2D_Small
```

### SM068 — UDM sharded reduction

Lane: `fabric-1x4`. Kernels: `DF003`, `DF004`.

Small sharded UDM width reduction instantiates sender and receiver kernels.

```text
tests/ttnn/unit_tests/gtests/udm/reduction/sharded/test_udm_reduction.cpp::MeshDevice1x4Fabric2DUDMFixture.TestWidthReduction2D_Small
```

### SM069 — RMS all-gather

Lane: `wormhole-n300`. Kernels: `S035`, `DF007`.

Fused RMS all-gather on N300, two devices, BF16, no residual add; smallest existing numeric case.

```text
tests/ttnn/unit_tests/operations/ccl/test_minimals.py::test_rms_fuse_n300[silicon_arch_name={arch}-topology=Topology.Linear-device_params={'fabric_config': FabricConfig.FABRIC_1D}-output_dtype=DataType.BFLOAT16-residual_dtype=DataType.BFLOAT16-input_dtype=DataType.BFLOAT16-2x1_grid-use_noc1_only=False-fused_add=False-num_iters=5-num_links=1-fp32_dest_acc_en=False-num_devices=2-elements_per_batch=2048-input_shard_grid={[0-0 - 3-7]}-output_shard_grid=None]
```

### SM070 — attention-residual gather softmax

Lane: `blackhole-2x4`. Kernels: `S036`, `DF008`.

Plain AttnRes read, the smallest existing 2x4 fabric arm; 640 tokens per chip.

```text
tests/ttnn/unit_tests/operations/experimental/test_attn_res_gather_softmax.py::test_matches_torch[silicon_arch_name={arch}-plain-fabric2d-mesh-2x4]
```

### SM071 — DiT fused distributed RMS norm

Lane: `galaxy`. Kernels: `S034`, `DF006`.

WAN whole-row RMSNorm, TP=1, prompt length512, width5120. TP1 uses the drain-only writer; TP2 uses the all-gather worker writer.

```text
models/tt_dit/tests/unit/test_distributed_rmsnorm_fused.py::test_corr_det[{arch}-wan_tp1]
```

Pinned environment: `CORR_ONLY=cross_k_prompt_L512`, `CORR_DET_REPEATS=0`.

### SM072 — DiT fused distributed RMS norm

Lane: `galaxy`. Kernels: `DF005`.

WAN whole-row RMSNorm, TP=2, prompt length512, width5120. TP1 uses the drain-only writer; TP2 uses the all-gather worker writer.

```text
models/tt_dit/tests/unit/test_distributed_rmsnorm_fused.py::test_corr_det[{arch}-wan_tp2]
```

Pinned environment: `CORR_ONLY=cross_k_prompt_L512`, `CORR_DET_REPEATS=0`.

### SM073 — ring joint SDPA

Lane: `wormhole-t3k`. Kernels: `DF060`.

Existing numerical multi-batch ring-joint SDPA case on T3K.

```text
tests/nightly/t3000/ccl/test_ring_joint_attention.py::test_ring_joint_sdpa_multi_batch_wh_t3k[{arch}-2x4-line]
```

### SM074 — experimental ring joint SDPA

Lane: `blackhole-galaxy`. Kernels: `DF058`.

Smallest fixed experimental ring-joint SDPA case: 1x4 submesh, sequence8960,10heads; five numerical iterations.

```text
models/tt_dit/tests/unit/test_exp_ring_joint_attention.py::test_exp_ring_joint_sdpa_dit_bh_glx_custom[{arch}-1x4-ring]
```

### SM075 — BGE model-local SDPA

Lane: `wormhole`. Kernels: `DF001`.

Small standard SDPA with two query heads and 128 KV tokens.

```text
tests/ttnn/unit_tests/operations/sdpa/test_bge_encoder_sdpa_reduce_migration.py::test_bge_encoder_sdpa_reduce_auxiliary[runtime_lengths=False-streaming=False]
```

### SM076 — distributed Welford layer norm

Lane: `common`. Kernels: `DF044` (primary case SM011).

One-device Welford post-all-gather with hand-built mean/variance, BF16 input and weights, FP32 destination; exercises the shared reader without an auxiliary recipe.

```text
tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_post_allgather.py::test_layer_norm_post_all_gather_welford_with_program_cache[one_stats_pair-small-bf16]
```

### SM077 — distributed RMS norm

Lane: `common`. Kernels: `S090` (primary case SM011), `DF044` (primary case SM011).

One-device RMSNorm post-all-gather with BF16 input/weights, FP32 statistics and destination; guards the planned auxiliary buffer unpack format.

```text
tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_post_allgather.py::test_post_all_gather_mixed_stats_dtype_with_program_cache[one_stats_pair-rmsnorm-bf16_input_fp32_stats]
```
