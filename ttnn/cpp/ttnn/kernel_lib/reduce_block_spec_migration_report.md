# Local reduction block descriptor migration

Date: 2026-09-11. PR: 56063. Parent: `f4cefb9940a2ad583653e8e041a06f03ab266f72`.

## Problem and result

The reduce planner previously accepted input and output `TensorSpec` objects. It read the whole tensor's logical and padded shapes to determine compute work, but could read a per-core shard shape to size an aliased circular buffer. For a global 128×64 tensor stored as two 64×64 height shards, that could describe eight input tiles of work against a four-tile local allocation. A synthetic local TensorSpec avoided the mismatch, but its type did not distinguish local work from the global tensor.

Both planning entry points now consume `ReduceBlockSpec`: the block processed by **one reduction invocation on one core**. Factories explicitly describe local work after partitioning it. The planner has no TensorSpec overload and does not infer core assignments, memory placement, or shard orientation. Existing C++ and Python callers have been migrated together.

This is a host planning API change. It does not add runtime reduction arguments or change the device argument schema. Existing kernel loops, compile-time specialization, CB bindings, auxiliary preparation, accumulation, and Full/Tail dispatch remain in place.

## Descriptor contract

The definition is in [host/reduce_host.hpp](host/reduce_host.hpp); the implementation and validation are in [host/reduce_host.cpp](host/reduce_host.cpp).

| Field | Meaning |
| --- | --- |
| `logical_h`, `logical_w` | Valid local height and width in elements. Partial reduction-axis elements are derived from these. |
| `padded_h`, `padded_w` | Extents traversed by the local call, in elements. Tiled extents must be multiples of the input tile dimensions. |
| `batches` | Number of equal local H×W blocks traversed by this call; leading dimensions are already flattened by the factory. |
| `input_dtype`, `output_dtype` | Compute input and packed output formats, including intermediate formats in fused operations. |
| `input_layout`, `output_layout` | Tiled or dense row-major representation. Existing supported row-major staging paths are preserved. |
| `input_tile`, `output_tile` | Tile geometry used to calculate tile counts and input/output page sizes. Both default to 32×32. |
| `input_row_stride_tiles` | Physical pitch between resident input rows in tiles. Zero means the contiguous padded width. Batches are separated by `Ht * row_stride` tiles. |
| `resident_input_tiles` | Optional capacity of a caller-owned local tiled allocation. When present, the default plan uses `NoWaitNoPop`; the input must already be available to compute. |
| `resident_output_tiles` | Optional capacity of a caller-owned local tiled output allocation. Output retains the ordinary pack protocol. |

When resident capacities are absent, the planner sizes its FIFO or staging requirements under the reduction-owned L1 budget. Resident allocations are excluded from that budget. A positive `max_input_cb_bytes` caps planner-owned input storage; zero is no longer an alias request. The factory must supply the real local capacity when selecting resident storage.

`ReduceBlockSpec::tiled(h, w, input_dtype, output_dtype, batches, tile)` rounds the padded dimensions to whole tiles. It sets both input and output tile geometry to the supplied tile. The struct also supports direct field initialization. Python exposes the same fields through `ttnn.reduce_planner.ReduceBlockSpec`, with optional padded dimensions and keyword-only storage/layout details.

Reduction math, dimension, scalar, FP32 mode, hardware configuration, and CB IDs remain separate. `ReduceCallConfig` contains one `block`, replacing `input_spec` and `output_spec`. A sequence entry remains `(input_cb_id, ReduceCallConfig)`.

## Shape inference and validation

The output logical shape is derived from the local block and reduction dimension:

| Dimension | Logical output per batch | Tiled output pages per batch |
| --- | --- | --- |
| W / row reduction | H×1 | Ht |
| H / column reduction | 1×W | Wt |
| HW / scalar reduction | 1×1 | 1 |

Accumulated calls must agree on the derived output, including the unreduced logical and padded dimension, batch count, output format/layout/tile, and output allocation contract. They may have different reduction-axis lengths, such as a full block followed by a partial block.

Validation rejects zero work, padded extents smaller than logical extents, nonintegral tiled extents, overflowed tile counts, and insufficient resident input/output capacity. It also rejects a stride smaller than the local width, a stride on streamed input, and noncontiguous HW reductions. The padded W or H reduction axis must end at its last logical tile, even when the logical length is tile-aligned; spare physical columns belong in the stride, not in the padded reduction width. Existing backend restrictions on SFPU partial reductions and math combinations remain enforced.

For resident input, the allocation must contain `batches * Ht * physical_row_stride` tiles. The output allocation must contain the derived number of output tiles. These are checks against the capacity supplied by the factory; they cannot verify a deliberately incorrect capacity or partition description.

Input and output CB page sizes now use the supplied Tile's `get_tile_size(format)`. For example, BF16 16×32 input pages are 1,024 bytes. Auxiliary tiles keep their existing recipe geometry. This corrects page-size accounting; it does not establish numerical coverage for every custom tile geometry.

## Illustrative usage

A core reducing a 64×70 logical block stored in a 64×128 allocation describes the work and physical pitch separately:

```cpp
namespace rh = ttnn::kernel_lib::host;
auto block = rh::ReduceBlockSpec::tiled(
    64, 70, DataType::BFLOAT16, DataType::BFLOAT16);
block.input_row_stride_tiles = 4;
block.resident_input_tiles = 8;
block.resident_output_tiles = 2;

auto plan = rh::make_reduce_plan(
    block, ReduceOpMath::SUM, ReduceOpDim::W,
    1.0F, ReduceFp32Mode::Fast, hardware);
```

The call traverses two rows and three columns of tiles, uses six valid elements in each final width tile, and skips the fourth physical tile column. No separate `valid_elements` argument is needed. The factory still creates the CBs and binds the serialized call to its compute and dataflow kernels.

Good starting points for reading the implementation:

1. [Moreh mean H factory](../operations/moreh/moreh_mean/device/moreh_mean_h_program_factory.cpp): a single streamed H reduction with explicit local height and a two-tile input budget.
2. [Moreh reduce blocks](../operations/moreh/moreh_reduce.hpp): a sequence of reusable local blocks, a caller-owned transformed-input buffer, and the existing `WaitUpfrontNoPop` override for fused producer/consumer synchronization.
3. [Sharded LayerNorm factory](../operations/normalization/layernorm/device/layernorm_op_multi_core_sharded.cpp): full and tail widths share an allocation pitch; the planner receives capacity and stride before serialization.
4. [Sharded toy variance](../../../ttnn/operations/toy_variance/toy_variance_sharded_program_artifacts.py): global width partitioning remains in the factory; the mean reads a resident local block and the variance pass consumes a streamed transformed block. Its scalar still normalizes by the full global width.
5. [Device helper tests](../../../../tests/ttnn/unit_tests/kernel_lib/reduce/test_reduce_helpers.py), `test_reduce_local_blocks_on_multiple_cores`: three actual cores receive distinct compile-time block descriptions while sharing allocation sizes.

## Caller migration

All 43 C++ caller files and seven Python operation/example files were ported. The two generic-reduction adapters still receive the operation's TensorSpecs as factory inputs, extract dtype/tile information and their already-partitioned extents, and pass a ReduceBlockSpec to the planner. Synthetic TensorLayouts and output TensorSpecs used only for planning have been removed.

| Area | What changed |
| --- | --- |
| Moreh | Local tile blocks for mean/sum, norm, dot, clip-grad, moments, softmax and its backward paths, and bias backward now use explicit descriptors. Existing masks and fused synchronization overrides are preserved. |
| Normalization | GroupNorm, distributed LayerNorm/RMSNorm, and softmax use local blocks. Sharded LayerNorm supplies resident capacity and physical stride before planning, replacing its post-planning stride edit. |
| Generic reductions | Standard and Quasar adapters construct local descriptors for tiled work or the already-tilized row-major chunks. The existing reader partitioning and identity-padding rules are preserved. |
| Other fused operations | Sampling, MoE, grouped gate, indexer score, attention reductions, KDA, SSM, and distributed RMS operations retain their existing local extents and intermediate formats. |
| Python toy operations | Partial reduce and variance now construct blocks explicitly. Sharded variance no longer invents a one-core height-sharded TensorSpec to request resident input. |
| Python examples | Reduce, accumulate, row-accumulate, and compute-fusion examples use the new contract. `reduce_block` passes its existing row-stride option through the block descriptor; the single-call and accumulated strided paths can now be tested through this API. |

No factory's global scheduling or cross-core collective algorithm was redesigned. Global normalization factors remain the factory's responsibility. If several cores process different local shapes, factories can continue emitting distinct compile-time descriptions for those core groups. A single kernel whose work size changes between invocations still requires the existing dispatch choices or a separate future runtime-argument design.

## Tests and validation

Validation used the local Release build and Wormhole N300 hardware. The full migration matrix contains other hardware lanes; the smaller selection below covers the lanes available here.

Five new host tests cover local versus global allocation bounds, partial local blocks with physical stride and batches, custom-tile page sizes, invalid descriptor geometry/capacity, and compatible versus incompatible sequence outputs. The existing host planner tests were migrated off TensorSpec.

Four new numerical tests run SUM and MAX along W and H on three cores. Each core reduces two batches with a different local shape. The full, short, and partial blocks share a larger physical input pitch/allocation. Unvisited input tiles and partial edges contain a large sentinel; unused output pages start at a separate sentinel and must remain untouched. References are computed from only the valid local elements.

The existing device matrix was migrated to explicit resident capacities and local block dimensions. Its stale per-call threshold assertions were corrected to match the already-existing sequence-wide additive threshold and odd-call zero-pair recipe. The repeated-input fixture now sizes its auxiliary CB from the recipe instead of hard-coding one tile. The reduction algorithm itself was not changed for that correction.

| Check | Result |
| --- | --- |
| Direct `build_metal.sh` Release build with TTNN tests enabled | Passed |
| `ReduceHostPlanner.*` | 14 passed |
| Device reduce helper suite | 151 passed, including four new multicore cases |
| Existing single-call and accumulated row-stride example tests | 2 passed |
| Smaller migration sanity suite: common + Wormhole + Wormhole N300 | 63 passed; 0 failed or skipped |
| Repository pre-commit hooks on all changed files | Passed |
| `git diff --check` | Passed |

The initial checks caught a missing layout include and an accidentally omitted Python CB-ID binding during migration; both were fixed and the direct build rerun. The repeated-input fixture's stale algorithm/recipe assumptions and auxiliary allocation were also fixed, followed by a passing full helper run. The final host and helper suites passed after the last geometry-validation change.

Commands used (run from the repository root):

```bash
CMAKE_BUILD_PARALLEL_LEVEL=8 PYTHONDONTWRITEBYTECODE=1 \
    ./build_metal.sh --enable-ccache --build-ttnn-tests

PYTHONDONTWRITEBYTECODE=1 TT_METAL_HOME="$PWD" \
    build/test/ttnn/unit_tests_ttnn --gtest_filter='ReduceHostPlanner.*'

PYTHONDONTWRITEBYTECODE=1 TT_METAL_HOME="$PWD" \
    bash scripts/run_safe_pytest.sh --no-precompile \
    tests/ttnn/unit_tests/kernel_lib/reduce/test_reduce_helpers.py -q --maxfail=1

PYTHONDONTWRITEBYTECODE=1 TT_METAL_HOME="$PWD" \
    bash scripts/run_safe_pytest.sh --no-precompile \
    tests/ttnn/unit_tests/operations/examples/test_reduce_block.py::test_reduce_block_row_stride \
    tests/ttnn/unit_tests/operations/examples/test_reduce_block.py::test_reduce_block_accumulate_row_stride -q
```

The sanity run used the archived manifest byte-for-byte from the parent of the inventory-deletion commit. Its SHA256 is `6d47f61b63e4d703358e8c0355e22c6c47765c90168d0241a139c5e2ac1b7c5c`. To reproduce that selection without restoring the inventory directory:

```bash
reduce_validation_dir=$(mktemp -d)
git show 311e3c03accd5b8323222705423922c505e011c4^:ttnn/cpp/ttnn/kernel_lib/reduce_migration_inventory_2026-09-08_f808380a87b/sanity_test_suite.json \
    > "$reduce_validation_dir/sanity_test_suite.json"
PYTHONDONTWRITEBYTECODE=1 TT_METAL_HOME="$PWD" \
    python_env/bin/python scripts/run_reduce_migration_sanity.py \
    --manifest "$reduce_validation_dir/sanity_test_suite.json" \
    --lane common --lane wormhole --lane wormhole-n300 \
    --output-dir "$reduce_validation_dir/results"
```

That selection contains 50 Python cases and 13 C++ cases in 63 groups. The other 14 groups require Blackhole, Quasar, or larger multi-device topologies and were not executed here. This is a numerical sanity selection, not the full performance or model validation matrix.

Build logs, host results, helper results, both example results, per-group sanity logs/JSON/JUnit XML, and pre-commit output are retained locally under `/localdev/malimpic/reviews/pr56063-local-spec-20260911-oer_2vst/`. The relevant final logs are `build-final.log`, `host-tests-final.log`, `helper-tests-final.log`, `strided-example-tests.log`, `sanity-run.log`, `sanity-results/`, and `pre-commit-final.log`.

## Scope and remaining limits

- The planner describes a call, not a distributed tensor. Factories still own partitioning, addresses, core assignment, and inter-core communication.
- Runtime reduction dimensions were not introduced. No discarded runtime-argument commit was restored.
- Existing SFPU partial-axis and two-dimensional HW masking restrictions remain. Some fused callers intentionally describe already-masked full tiles.
- Existing caller overrides of synchronization/reconfiguration policy remain, except where resident storage now supplies the same default directly.
- Custom tile page accounting is covered by host tests; the new multicore numerical cases use ordinary 32×32 tiles.
- The inventory directory removed earlier remains deleted. The smaller suite is run with its archived manifest supplied explicitly; the runner's deleted default manifest path is a separate existing issue.
- The other review topics, including CB-ID remapping, general sequence L1 accounting, and unrelated legacy example controls, remain separate follow-ups.

## Complete changed-file inventory

Paths below are relative to the repository root. This report is also included in the commit.

| File | Change category |
| --- | --- |
| [tests/ttnn/unit_tests/gtests/test_reduction.cpp](../../../../tests/ttnn/unit_tests/gtests/test_reduction.cpp) | Host planner tests |
| [tests/ttnn/unit_tests/kernel_lib/reduce/test_reduce_helpers.py](../../../../tests/ttnn/unit_tests/kernel_lib/reduce/test_reduce_helpers.py) | Device helper and multicore tests |
| [ttnn/cpp/ttnn/kernel_lib/host/reduce_host.cpp](host/reduce_host.cpp) | Local descriptor and planning contract |
| [ttnn/cpp/ttnn/kernel_lib/host/reduce_host.hpp](host/reduce_host.hpp) | Local descriptor and planning contract |
| [ttnn/cpp/ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/device/dit_fused_distributed_rmsnorm_program_factory.cpp](../operations/experimental/ccl/dit_fused_distributed_rmsnorm/device/dit_fused_distributed_rmsnorm_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/rms_allgather_program_factory.cpp](../operations/experimental/ccl/rms_allgather/device/rms_allgather_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/attn_res_gather_softmax/device/attn_res_gather_softmax_program_factory.cpp](../operations/experimental/deepseek_prefill/attn_res_gather_softmax/device/attn_res_gather_softmax_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/experimental/indexer_score/device/indexer_score_program_factory.cpp](../operations/experimental/indexer_score/device/indexer_score_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/experimental/indexer_score/device/ring_indexer_score_dsa_program_factory.cpp](../operations/experimental/indexer_score/device/ring_indexer_score_dsa_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/prepare_chunk_recurrence_program_factory.cpp](../operations/experimental/kda/prepare_chunk_recurrence/device/prepare_chunk_recurrence_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/device/sigmoid_gated_rms_norm_program_factory.cpp](../operations/experimental/kda/sigmoid_gated_rms_norm/device/sigmoid_gated_rms_norm_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/experimental/quasar/reduction/generic/device/common.cpp](../operations/experimental/quasar/reduction/generic/device/common.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/deepseek_grouped_gate_program_factory.cpp](../operations/experimental/reduction/deepseek_grouped_gate/device/deepseek_grouped_gate_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce/device/hc_sum_reduce_program_factory.cpp](../operations/experimental/ssm/hc_sum_reduce/device/hc_sum_reduce_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/fused_rmsnorm_post_all_gather_program_factory.cpp](../operations/experimental/transformer/fused_distributed_rmsnorm/device/fused_rmsnorm_post_all_gather_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/fused_rmsnorm_pre_all_gather_program_factory.cpp](../operations/experimental/transformer/fused_distributed_rmsnorm/device/fused_rmsnorm_pre_all_gather_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/moreh_clip_grad_norm_step1_program_factory.cpp](../operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/moreh_clip_grad_norm_step1_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/moreh_dot_program_factory.cpp](../operations/moreh/moreh_dot/device/moreh_dot_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/moreh_group_norm_program_factory.cpp](../operations/moreh/moreh_group_norm/device/moreh_group_norm_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/moreh_layer_norm_program_factory.cpp](../operations/moreh/moreh_layer_norm/device/moreh_layer_norm_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/moreh_linear_backward_multi_core_program_factory.cpp](../operations/moreh/moreh_linear_backward/device/moreh_linear_backward_multi_core_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/moreh_linear_backward_single_core_program_factory.cpp](../operations/moreh/moreh_linear_backward/device/moreh_linear_backward_single_core_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/moreh_mean_h_program_factory.cpp](../operations/moreh/moreh_mean/device/moreh_mean_h_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_program_factory_h_other.cpp](../operations/moreh/moreh_norm/device/ord_other/moreh_norm_program_factory_h_other.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_program_factory_w_other.cpp](../operations/moreh/moreh_norm/device/ord_other/moreh_norm_program_factory_w_other.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_reduce.hpp](../operations/moreh/moreh_reduce.hpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/softmax_h_large/softmax_h_large.cpp](../operations/moreh/moreh_softmax/device/softmax_h_large/softmax_h_large.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/softmax_h_small/softmax_h_small.cpp](../operations/moreh/moreh_softmax/device/softmax_h_small/softmax_h_small.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/softmax_w_large/softmax_w_large.cpp](../operations/moreh/moreh_softmax/device/softmax_w_large/softmax_w_large.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/softmax_w_small/softmax_w_small.cpp](../operations/moreh/moreh_softmax/device/softmax_w_small/softmax_w_small.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/softmax_backward_h_large/softmax_backward_h_large.cpp](../operations/moreh/moreh_softmax_backward/device/softmax_backward_h_large/softmax_backward_h_large.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/softmax_backward_h_small/softmax_backward_h_small.cpp](../operations/moreh/moreh_softmax_backward/device/softmax_backward_h_small/softmax_backward_h_small.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/softmax_backward_w_large/softmax_backward_w_large.cpp](../operations/moreh/moreh_softmax_backward/device/softmax_backward_w_large/softmax_backward_w_large.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/softmax_backward_w_small/softmax_backward_w_small.cpp](../operations/moreh/moreh_softmax_backward/device/softmax_backward_w_small/softmax_backward_w_small.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_program_factory.cpp](../operations/moreh/moreh_sum/device/moreh_sum_h_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_reduce_plans.hpp](../operations/normalization/groupnorm/device/groupnorm_reduce_plans.hpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core_sharded.cpp](../operations/normalization/layernorm/device/layernorm_op_multi_core_sharded.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_post_all_gather_program_factory.cpp](../operations/normalization/layernorm_distributed/device/layernorm_post_all_gather_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp](../operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_h_large.cpp](../operations/normalization/softmax/device/softmax_program_factory_general_h_large.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_h_small.cpp](../operations/normalization/softmax/device/softmax_program_factory_general_h_small.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_large.cpp](../operations/normalization/softmax/device/softmax_program_factory_general_w_large.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_small.cpp](../operations/normalization/softmax/device/softmax_program_factory_general_w_small.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_reduce_plans.hpp](../operations/normalization/softmax/device/softmax_reduce_plans.hpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/reduction/generic/device/common.cpp](../operations/reduction/generic/device/common.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp](../operations/reduction/moe/device/moe_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/cpp/ttnn/operations/reduction/reduce_planner_nanobind.cpp](../operations/reduction/reduce_planner_nanobind.cpp) | Python planner bindings |
| [ttnn/cpp/ttnn/operations/reduction/sampling/device/sampling_program_factory.cpp](../operations/reduction/sampling/device/sampling_program_factory.cpp) | C++ local-block caller migration |
| [ttnn/ttnn/operations/examples/compute_fusion/program_descriptor_with_inline_kernels.py](../../../ttnn/operations/examples/compute_fusion/program_descriptor_with_inline_kernels.py) | Python factory / example migration |
| [ttnn/ttnn/operations/examples/reduce_accumulate/program_descriptor_with_inline_kernels.py](../../../ttnn/operations/examples/reduce_accumulate/program_descriptor_with_inline_kernels.py) | Python factory / example migration |
| [ttnn/ttnn/operations/examples/reduce_block/program_descriptor_with_inline_kernels.py](../../../ttnn/operations/examples/reduce_block/program_descriptor_with_inline_kernels.py) | Python factory / example migration |
| [ttnn/ttnn/operations/examples/row_reduce_accumulate/program_descriptor_with_inline_kernels.py](../../../ttnn/operations/examples/row_reduce_accumulate/program_descriptor_with_inline_kernels.py) | Python factory / example migration |
| [ttnn/ttnn/operations/toy_reduce_partial/toy_reduce_partial_program_descriptor.py](../../../ttnn/operations/toy_reduce_partial/toy_reduce_partial_program_descriptor.py) | Python factory / example migration |
| [ttnn/ttnn/operations/toy_variance/toy_variance_program_artifacts.py](../../../ttnn/operations/toy_variance/toy_variance_program_artifacts.py) | Python factory / example migration |
| [ttnn/ttnn/operations/toy_variance/toy_variance_sharded_program_artifacts.py](../../../ttnn/operations/toy_variance/toy_variance_sharded_program_artifacts.py) | Python factory / example migration |
