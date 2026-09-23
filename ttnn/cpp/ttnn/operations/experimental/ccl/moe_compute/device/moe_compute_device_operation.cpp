// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/moe_compute/moe_core_placement.hpp"
#include "kernels/moe_ring_common.h"
#include "moe_compute_device_operation.hpp"
#include "moe_compute_program_factory.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include "ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_align.hpp>

#include <umd/device/types/arch.hpp>

namespace ttnn::experimental::prim {
namespace detail {

constexpr auto TOKEN_SIZE = 32;  // This does not mean we only support 32 tokens, just hardcoding the shared buffer size
constexpr auto DOUBLE_BUFFER_SIZE = 2;

// LocalOutput: dm1 addresses the [k, T, H] output through a TensorAccessor built from the actual
// buffer, one page per token row (2 x H bytes) plus the column offset of its slice. A row-major
// tensor has one-row pages when it is INTERLEAVED or HEIGHT_SHARDED (tt_metal page_config.cpp,
// get_page_shape_rm), DRAM or L1, so both work with the same kernel. WIDTH_SHARDED, BLOCK_SHARDED
// and ND sharding give (1, shard width) pages: a row then spans several pages and a core's slice
// would have to be split at the shard boundaries, which this path does not implement (the fused
// combine writer addresses its output the same way and has the same limitation).
void validate_local_output_memory_config(
    const tt::tt_metal::MemoryConfig& memory_config, uint32_t num_rows, uint32_t hidden_size, const char* what) {
    using tt::tt_metal::TensorMemoryLayout;
    const auto layout = memory_config.memory_layout();
    TT_FATAL(
        layout == TensorMemoryLayout::INTERLEAVED || layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "moe_compute over a mesh axis of extent 1 writes each token row of the [k, T, H] output as one TensorAccessor "
        "page and does not split a row across column pages, so the {} must be INTERLEAVED or HEIGHT_SHARDED "
        "row-major (WIDTH_SHARDED, BLOCK_SHARDED and ND sharding need a column-page split that is not implemented on "
        "this path, the same limitation as the combine writer); got {}",
        what,
        memory_config);
    if (layout != TensorMemoryLayout::HEIGHT_SHARDED) {
        return;
    }
    const auto& shard_spec = memory_config.shard_spec();
    TT_FATAL(shard_spec.has_value(), "the {} is HEIGHT_SHARDED without a shard spec: {}", what, memory_config);
    TT_FATAL(
        shard_spec->shape[1] == hidden_size,
        "a height shard of the {} must hold whole token rows: shard width {} != hidden size {}",
        what,
        shard_spec->shape[1],
        hidden_size);
    const uint32_t shard_rows = shard_spec->shape[0];
    const uint32_t num_shard_cores = shard_spec->grid.num_cores();
    TT_FATAL(
        shard_rows > 0 && num_shard_cores * shard_rows >= num_rows,
        "the {} shard grid ({} cores x {} rows per shard) does not cover the k x T = {} token rows",
        what,
        num_shard_cores,
        shard_rows,
        num_rows);
}

}  // namespace detail
MoEComputeDeviceOperation::program_factory_t MoEComputeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MoEComputeMeshWorkloadFactory{};
}

void MoEComputeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MoEComputeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Tilize
    TT_FATAL(
        tensor_args.tilize_input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Input tensor must be in row major layout");
    TT_FATAL(
        tensor_args.tilize_input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16, "Input tensor must be bfloat16");
    TT_FATAL(
        tensor_args.tilize_expert_indices_tensor.dtype() == tt::tt_metal::DataType::UINT16,
        "Indices tensor must be uint16");

    // Input tensor rank guards. Exact ranks are enforced where every caller agrees; lenient
    // minimum ranks are used for the token/index/score tensors, which legitimately arrive as
    // rank-3 from some dispatch paths and rank-4 from others (the op only indexes [0]/[1]/[-1]).
    const auto rank_of = [](const ttnn::Tensor& t) { return t.logical_shape().rank(); };
    TT_FATAL(
        rank_of(tensor_args.tilize_input_tensor) >= 3,
        "tilize_input_tensor must be rank >= 3 ([..., tokens, hidden]); got rank {}",
        rank_of(tensor_args.tilize_input_tensor));
    TT_FATAL(
        rank_of(tensor_args.tilize_expert_indices_tensor) >= 2,
        "tilize_expert_indices_tensor must be rank >= 2 ([..., tokens, K]); got rank {}",
        rank_of(tensor_args.tilize_expert_indices_tensor));
    TT_FATAL(
        rank_of(tensor_args.tilize_expert_scores_tensor) >= 2,
        "tilize_expert_scores_tensor must be rank >= 2 ([..., tokens, K]); got rank {}",
        rank_of(tensor_args.tilize_expert_scores_tensor));
    TT_FATAL(
        rank_of(tensor_args.tilize_expert_mapping_tensor) == 2,
        "tilize_expert_mapping_tensor must be rank 2 ([num_devices, experts]); got rank {}",
        rank_of(tensor_args.tilize_expert_mapping_tensor));
    TT_FATAL(
        rank_of(tensor_args.matmul_w0_w1_tensor) == 6,
        "matmul_w0_w1_tensor must be rank 6 ([num_cores, L, E, groups_per_core, K, 4*TILE_SIZE]); got rank {}",
        rank_of(tensor_args.matmul_w0_w1_tensor));
    TT_FATAL(
        rank_of(tensor_args.matmul_w2_tensor) == 6,
        "matmul_w2_tensor must be rank 6 ([num_cores, L, E, groups_per_core, N, 4*TILE_SIZE]); got rank {}",
        rank_of(tensor_args.matmul_w2_tensor));

    // When has_bias=True, dm0 derives per-expert byte strides using ceil((K+1)/W0W1_TXN)*W0W1_TXN and
    // ceil((N+1)/W2_TXN)*W2_TXN. The physical tensors must be padded to those tile counts; if not,
    // dm0 silently reads from wrong expert boundaries after the first expert.
    if (args.has_bias) {
        constexpr uint32_t tile_h = tt::constants::TILE_HEIGHT;
        constexpr uint32_t w0w1_txn = moe_ring::W0_W1_BLOCK_TILES_H * tile_h;  // bytes per transaction row
        constexpr uint32_t w2_txn = moe_ring::W2_TILES_PER_A2A_ITER_H * tile_h;

        const auto& w0_w1_shape = tensor_args.matmul_w0_w1_tensor.tensor_spec().logical_shape();
        const uint32_t w0_w1_k = w0_w1_shape[-2];
        TT_FATAL(
            w0_w1_k % w0w1_txn == 0,
            "matmul_w0_w1_tensor K-dimension ({}) must be a multiple of {} elements ({} tiles * {} rows/tile) "
            "when has_bias=True. Use moe_compute_utils.prepare_w0_w1_tensor_with_bias() to prepare the tensor.",
            w0_w1_k,
            w0w1_txn,
            moe_ring::W0_W1_TILES_PER_TXN,
            tile_h);

        const auto& w2_shape = tensor_args.matmul_w2_tensor.tensor_spec().logical_shape();
        const uint32_t w2_n = w2_shape[-2];
        TT_FATAL(
            w2_n % w2_txn == 0,
            "matmul_w2_tensor N-dimension ({}) must be a multiple of {} elements ({} tiles * {} rows/tile) "
            "when has_bias=True. Use moe_compute_utils.prepare_w2_tensor_with_bias() to prepare the tensor.",
            w2_n,
            w2_txn,
            moe_ring::W2_TILES_PER_TXN,
            tile_h);
    }

    // validate that 32 (token dim) * output_shard_width * output_shard_height >= total tokens
    const auto& tilize_input_shape = tensor_args.tilize_input_tensor.logical_shape();
    const auto total_tokens = tilize_input_shape[0] * tilize_input_shape[1];
    const auto combine_token_parallel_cores = args.num_token_parallel_cores;
    const auto combine_data_parallel_cores = args.num_data_parallel_cores;

    // make sure the shared L1 buffer is sufficiently large enough to contain all output tokens
    const auto max_tokens = detail::TOKEN_SIZE * combine_data_parallel_cores * combine_token_parallel_cores;
    TT_FATAL(
        max_tokens >= total_tokens, "Too many tokens in input, got: {} but expected max: {}", total_tokens, max_tokens);

    // Mode-specific validation of combine_params and optional_output_tensor.
    // - ComputeOnly: no combine_params, no optional_output_tensor (5 outputs).
    // - FullLocal: combine_params must be set with local_combine=true; only on a 1x1 mesh;
    //   optional_output_tensor is allowed as the combine output sink (6 outputs, no CCL).
    // - LocalOutput: combine_params describes the final [k, T, H] output (6 outputs, no combine
    //   kernels: dm1 writes the output). The cluster_axis has extent 1; on a multi-device mesh
    //   that leaves one partial per coordinate for the caller to reduce, so the token set must be
    //   replicated (every coordinate sees the same tokens). The output is written one token row
    //   (2 x H bytes) at a time through a TensorAccessor page, so its memory config must give
    //   one-row pages: row-major INTERLEAVED or HEIGHT_SHARDED with whole rows per shard (see
    //   detail::validate_local_output_memory_config). The optional_output_tensor, when given, is
    //   the buffer dm1 addresses (create_output_tensors returns it as slot 5).
    // - FullCcl: combine_params must be set with local_combine=false (6 outputs, CCL path).
    auto* mesh_device = tensor_args.tilize_input_tensor.device();
    if (args.path == MoEComputePath::ComputeOnly) {
        TT_FATAL(!args.combine_params.has_value(), "path=ComputeOnly requires combine_params to be std::nullopt");
        TT_FATAL(
            !tensor_args.optional_output_tensor.has_value(),
            "path=ComputeOnly requires optional_output_tensor to be std::nullopt (no combine output is produced)");
    } else {
        TT_FATAL(args.combine_params.has_value(), "path=Full requires combine_params to be set");
        const auto& mesh_shape = mesh_device->shape();
        TT_FATAL(
            args.combine_params->axis < mesh_shape.dims(),
            "cluster_axis {} is out of range for a mesh with {} axes",
            args.combine_params->axis,
            mesh_shape.dims());
        if (args.path == MoEComputePath::FullLocal) {
            TT_FATAL(
                mesh_device->num_devices() == 1,
                "path=FullLocal is only supported on a 1x1 mesh, got num_devices={}",
                mesh_device->num_devices());
            TT_FATAL(
                args.combine_params->local_combine, "path=FullLocal requires combine_params->local_combine to be true");
        } else if (args.path == MoEComputePath::LocalOutput) {
            // The CCL knobs are accepted and unused on this path; num_links keeps its range check.
            TT_FATAL(args.combine_params->num_links > 0, "num_links must be greater than 0");
            TT_FATAL(
                mesh_shape[args.combine_params->axis] == 1,
                "path=LocalOutput requires cluster_axis {} to have extent 1, got {}",
                args.combine_params->axis,
                mesh_shape[args.combine_params->axis]);
            // The local output is one partial per mesh coordinate that the caller sums, so a shared
            // expert would be counted once per coordinate; dm1's expert loop also only covers the
            // routed experts' e_t pages. The public wrapper rejects this at the op boundary; this
            // repeats it for direct ttnn::prim callers.
            TT_FATAL(
                args.num_shared_experts_per_device.value_or(0) == 0,
                "moe_compute over a mesh axis of extent 1 writes a local output and does not support shared experts; "
                "got num_shared_experts_per_device={}",
                args.num_shared_experts_per_device.value_or(0));
            if (mesh_device->num_devices() > 1) {
                const auto& input_topology = tensor_args.tilize_input_tensor.tensor_topology();
                for (const auto& placement : input_topology.placements()) {
                    TT_FATAL(
                        std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Replicate>(placement),
                        "moe_compute over a mesh axis of extent 1 on a multi-device mesh requires a fully replicated "
                        "input topology; the caller reduces the per-device partials");
                }
            }
            const auto& output_memory_config = args.combine_params->output_memory_config;
            const uint32_t num_output_rows =
                args.combine_params->select_experts_k * args.combine_params->batch_size * args.combine_params->seq_size;
            const uint32_t output_hidden_size = tilize_input_shape[-1];
            detail::validate_local_output_memory_config(
                output_memory_config, num_output_rows, output_hidden_size, "output memory config");
            if (tensor_args.optional_output_tensor.has_value()) {
                const auto& out = *tensor_args.optional_output_tensor;
                const ttnn::Shape expected_shape(
                    {args.combine_params->select_experts_k,
                     args.combine_params->batch_size * args.combine_params->seq_size,
                     output_hidden_size});
                TT_FATAL(
                    out.logical_shape() == expected_shape && out.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
                        out.dtype() == tensor_args.tilize_input_tensor.dtype(),
                    "optional_output_tensor must be a {} row-major {} tensor for the local output path; got shape {} "
                    "layout {} dtype {}",
                    expected_shape,
                    tensor_args.tilize_input_tensor.dtype(),
                    out.logical_shape(),
                    out.layout(),
                    out.dtype());
                // combine_params.output_memory_config is slot 5's spec and part of the program hash;
                // the tensor is the buffer dm1 writes. ttnn::prim::moe_compute takes the tensor's
                // config when the caller leaves output_memory_config unset, so a difference here is
                // an explicit disagreement, never a silent default.
                TT_FATAL(
                    out.memory_config() == output_memory_config,
                    "optional_output_tensor memory config {} differs from output_memory_config {} on the local output "
                    "path; pass one or the other (an unset output_memory_config takes the tensor's)",
                    out.memory_config(),
                    output_memory_config);
                // The tensor's own (TensorSpec-populated) config is what dm1's accessor is built from.
                detail::validate_local_output_memory_config(
                    out.memory_config(), num_output_rows, output_hidden_size, "optional_output_tensor memory config");
            }
        } else {
            TT_FATAL(
                !args.combine_params->local_combine, "path=FullCcl requires combine_params->local_combine to be false");
            TT_FATAL(args.combine_params->num_links > 0, "num_links must be greater than 0");
            TT_FATAL(args.combine_params->axis < 2, "cluster_axis must be 0 or 1");
            TT_FATAL(
                mesh_shape[args.combine_params->axis] > 1,
                "path=FullCcl requires a cluster_axis of extent > 1; an axis of extent 1 takes the LocalOutput path");
        }
    }

    // Validate hidden_size
    const uint32_t hidden_size = tilize_input_shape[-1];
    TT_FATAL(
        hidden_size > 0 && hidden_size % 32 == 0,
        "hidden_size ({}) must be a positive multiple of 32 (TILE_SIZE)",
        hidden_size);

    // Validate intermediate_size
    const uint32_t intermediate_size = args.intermediate_size;
    TT_FATAL(
        intermediate_size > 0 && intermediate_size % 32 == 0,
        "intermediate_size ({}) must be a positive multiple of 32 (TILE_SIZE)",
        intermediate_size);

    // Validate intermediate_tiles >= matmul_num_cores (at least 1 tile per ring core).
    // Both Full and ComputeOnly paths use the same matmul ring kernels, so this applies in both modes.
    //
    // matmul_num_cores must match the actual matmul ring size produced by program_factory:
    //   - WH: ring is always 12 (no DRAM-bank harvesting).
    //   - BH: ring = live DRAM-bank count (7 or 8). args.bh_ring_size is resolved by invoke()
    //     to this value before validate runs.
    const uint32_t matmul_num_cores = args.bh_ring_size;
    const uint32_t intermediate_tiles = intermediate_size / 32;
    TT_FATAL(
        intermediate_tiles >= matmul_num_cores,
        "intermediate_size ({}) must yield at least 1 tile per ring core ({} tiles < {} cores)",
        intermediate_size,
        intermediate_tiles,
        matmul_num_cores);

    TT_FATAL(
        matmul_num_cores % combine_data_parallel_cores == 0,
        "matmul_num_cores ({}) must be divisible by num_data_parallel_cores ({}) "
        "so RING_CORES_PER_COMBINE_COL is integral",
        matmul_num_cores,
        combine_data_parallel_cores);
    const uint32_t hidden_tiles = hidden_size / 32;
    TT_FATAL(
        hidden_tiles % combine_data_parallel_cores == 0,
        "hidden_tiles ({}) must be divisible by num_data_parallel_cores ({}) "
        "so output width shards are tile-aligned",
        hidden_tiles,
        combine_data_parallel_cores);

    // dm1 auto-splits each ring A2A transfer into enough noc_async_write_one_packet calls
    // to fit within NOC_MAX_BURST_SIZE (arch-dependent). Validate tiles_per_step matches
    // the round-up formula used in MoeRingConfig::in2_tiles_per_step.
    const uint32_t tiles_per_step_raw = (intermediate_tiles + matmul_num_cores - 1) / matmul_num_cores;
    const uint32_t tiles_per_step = moe_ring::even_stride_at_least_a2a_width(tiles_per_step_raw);
    TT_FATAL(
        tiles_per_step >= moe_ring::W2_TILES_PER_A2A_ITER_W && tiles_per_step % 2 == 0,
        "tiles_per_step ({}) must be even and >= W2_TILES_PER_A2A_ITER_W ({})",
        tiles_per_step,
        moe_ring::W2_TILES_PER_A2A_ITER_W);

    const uint32_t experts_per_device = tensor_args.matmul_w0_w1_tensor.logical_shape()[2];
    TT_FATAL(
        args.num_shared_experts_per_device <= experts_per_device,
        "num_shared_experts_per_device ({}) must be <= experts_per_device ({})",
        args.num_shared_experts_per_device,
        experts_per_device);

    // Validate that dynamic core placement succeeds for this hidden size and combine grid.
    // mux_core_range_set comes from combine_params when in Full mode; ComputeOnly uses an empty set.
    const CoreRangeSet validate_mux_cores =
        args.combine_params.has_value() ? args.combine_params->mux_core_range_set : CoreRangeSet{};
    ttnn::operations::ccl::common::select_moe_compute_cores(
        mesh_device,
        combine_token_parallel_cores,
        combine_data_parallel_cores,
        hidden_size,
        validate_mux_cores,
        args.bh_ring_size);
}

MoEComputeDeviceOperation::spec_return_value_t MoEComputeDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    const ttnn::Tensor& tilize_input_tensor = tensor_args.tilize_input_tensor;
    const auto& tilize_input_shape = tilize_input_tensor.tensor_spec().logical_shape();
    auto* mesh_device = tilize_input_tensor.device();

    uint32_t experts_per_device = tensor_args.matmul_w0_w1_tensor.logical_shape()[2];
    uint32_t total_tokens =
        tilize_input_shape[0] *
        tilize_input_shape[1];  // tokens_per_device from input, total tokens across all dispatch devices

    const uint32_t hidden_size = tilize_input_shape[-1];

    const CoreCoord worker_grid_size = mesh_device->compute_with_storage_grid_size();
    const CoreRangeSet shard_cores =
        CoreRangeSet({CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1})});
    const auto num_cores = shard_cores.num_cores();

    //-------------------------------------------------------------------------
    // Tilize outputs
    //-------------------------------------------------------------------------
    // Output 0: Per expert total tokens tensor
    // This data will be replicated on all cores
    auto per_expert_total_tokens_row_bytes = tt::align(experts_per_device * sizeof(uint32_t), l1_alignment);
    auto per_expert_total_tokens_row_elements = tt::div_up(per_expert_total_tokens_row_bytes, sizeof(uint32_t));
    auto tilize_per_expert_total_tokens_shape = ttnn::Shape({num_cores, per_expert_total_tokens_row_elements});

    const ttnn::MemoryConfig tilize_per_expert_total_tokens_sharded_memory_config = ttnn::MemoryConfig{
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::ShardSpec(
            shard_cores, {1, per_expert_total_tokens_row_elements}, tt::tt_metal::ShardOrientation::ROW_MAJOR),
    };

    auto tilize_per_expert_total_tokens_spec = tt::tt_metal::TensorSpec(
        tilize_per_expert_total_tokens_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tilize_per_expert_total_tokens_sharded_memory_config));

    // Output 1: Expert activation tensor
    // Each row: [token_id, k_indices[experts_per_device], scores[experts_per_device]]
    // Row size in uint32_t elements: 2 * experts_per_device + 1
    // Total size: total_tokens * aligned_row_bytes, stored as a single DRAM page
    uint32_t activation_row_elements = (2 * experts_per_device) + 1;
    uint32_t activation_row_bytes = tt::align(activation_row_elements * sizeof(uint32_t), l1_alignment);
    uint32_t activation_total_bytes = total_tokens * activation_row_bytes;
    auto tilize_expert_activation_shape = ttnn::Shape({1, activation_total_bytes / sizeof(uint32_t)});
    auto tilize_expert_activation_spec = tt::tt_metal::TensorSpec(
        tilize_expert_activation_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1)));

    // Output 2: Token indices tensor
    // 1 page per expert per device
    // Each index is at a 16B offset due to NoC DMA restrictions
    // (tokens + 1) -> 1 extra element per page for -1 terminator
    uint32_t e_t_row_bytes = (total_tokens + 1) * tt::align(sizeof(uint32_t), l1_alignment);
    uint32_t e_t_row_elements = e_t_row_bytes / sizeof(uint32_t);
    auto tilize_e_t_shape = ttnn::Shape({experts_per_device, e_t_row_elements});
    auto tilize_e_t_spec = tt::tt_metal::TensorSpec(
        Shape(tilize_e_t_shape),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1)));

    //-------------------------------------------------------------------------
    // Shared tilize output (sharded)
    //-------------------------------------------------------------------------
    /*
     * Tilize: Used as output CB of tilize operation
     * MM: Used as input CB (where tilized chunks arrive)
     * Combine: Stores output of MM, for input to combine
     */
    ttnn::MemoryConfig output_sharded_memory_config = ttnn::MemoryConfig{
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::ShardSpec(
            shard_cores,
            {detail::DOUBLE_BUFFER_SIZE * detail::TOKEN_SIZE, hidden_size},
            tt::tt_metal::ShardOrientation::ROW_MAJOR),
    };

    auto tilize_output_shape =
        ttnn::Shape({shard_cores.num_cores(), detail::DOUBLE_BUFFER_SIZE, detail::TOKEN_SIZE, hidden_size});
    auto tilize_output_spec = tt::tt_metal::TensorSpec(
        Shape(tilize_output_shape),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            output_sharded_memory_config));

    //-------------------------------------------------------------------------
    // Shared output (sharded)
    //-------------------------------------------------------------------------
    /*
     * This will be an alias to the buffer used by Shared tilize output.
     * But re-perceived as RM. This is not strictly necessary but facilitates
     * torch interop and unit testing
     */

    const auto& tilize_output_layout = tilize_output_spec.tensor_layout();
    const tt::tt_metal::TensorLayout matmul_output_layout(
        tilize_output_layout.get_data_type(), ROW_MAJOR_LAYOUT, tilize_output_layout.get_memory_config());
    const auto matmul_output_spec = tt::tt_metal::TensorSpec(tilize_output_shape, matmul_output_layout);

    //-------------------------------------------------------------------------
    // a2a combine output
    //-------------------------------------------------------------------------
    using namespace tt::tt_metal;

    if (args.path == MoEComputePath::ComputeOnly) {
        // No combine output in ComputeOnly mode; matmul_output_spec is the final output (slot 4).
        return {
            tilize_per_expert_total_tokens_spec,
            tilize_expert_activation_spec,
            tilize_e_t_spec,
            tilize_output_spec,
            matmul_output_spec};
    }

    TT_FATAL(args.combine_params.has_value(), "combine_params required when path is not ComputeOnly");

    if (args.path == MoEComputePath::LocalOutput) {
        // Output 5: the final [k, T, H] row-major tensor that dm1 writes directly (T is the whole
        // replicated token set: the axis has extent 1). Same shape the combine returns on that axis.
        const auto& combine_params = *args.combine_params;
        const auto local_output_spec = TensorSpec(
            ttnn::Shape(
                {combine_params.select_experts_k, combine_params.batch_size * combine_params.seq_size, hidden_size}),
            TensorLayout(
                tilize_input_tensor.dtype(), PageConfig(Layout::ROW_MAJOR), combine_params.output_memory_config));
        return {
            tilize_per_expert_total_tokens_spec,
            tilize_expert_activation_spec,
            tilize_e_t_spec,
            tilize_output_spec,
            matmul_output_spec,
            local_output_spec};
    }

    ttnn::experimental::prim::SelectiveReduceCombineTensors combine_tensor_args{
        .dense_input_tensor = tilize_input_tensor,
        .dense_activations_tensor = tilize_input_tensor,
        .dense_token_maps_tensor = tilize_input_tensor,
        .dense_token_counts_tensor = tilize_input_tensor,
        .optional_output_tensor = std::nullopt,
    };
    const auto output_spec = ttnn::experimental::prim::SelectiveReduceCombineDeviceOperation::compute_output_specs(
        args.combine_params.value(), combine_tensor_args);

    return {
        tilize_per_expert_total_tokens_spec,
        tilize_expert_activation_spec,
        tilize_e_t_spec,
        tilize_output_spec,
        matmul_output_spec,
        output_spec};
}

MoEComputeDeviceOperation::topology_return_value_t MoEComputeDeviceOperation::compute_output_topologies(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto* mesh_device = tensor_args.tilize_input_tensor.device();
    const bool local_output_on_mesh = args.path == MoEComputePath::LocalOutput && mesh_device->num_devices() > 1;
    if (!local_output_on_mesh) {
        // Keep the default topology inference for ComputeOnly, FullLocal, FullCcl and a 1x1 LocalOutput.
        return {};
    }

    // An axis of extent 1 on a multi-device mesh leaves one full-width partial at every
    // coordinate. No tensor dimension is sharded, so the outputs keep the (replicated) input
    // topology instead of inheriting an expert-sharded placement from the weight inputs.
    return topology_return_value_t(6, tensor_args.tilize_input_tensor.tensor_topology());
}

MoEComputeDeviceOperation::tensor_return_value_t MoEComputeDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const std::vector<tt::tt_metal::TensorSpec>& output_specs = compute_output_specs(args, tensor_args);

    const auto tilize_output_tensor = create_device_tensor(output_specs[3], tensor_args.tilize_input_tensor.device());

    // re-percieve tilize output tensor as RM for output
    const auto matmul_output_tensor =
        ttnn::unchecked_reinterpret_layout(tilize_output_tensor, tt::tt_metal::Layout::ROW_MAJOR);
    TT_FATAL(
        matmul_output_tensor.tensor_spec() == output_specs[4],
        "Reinterpreted tensor spec does not match expected output_specs[4]");

    if (args.path == MoEComputePath::ComputeOnly) {
        // 5-tensor return: matmul_output is the final output (no combine output produced).
        return {
            create_device_tensor(output_specs[0], tensor_args.tilize_input_tensor.device()),
            create_device_tensor(output_specs[1], tensor_args.tilize_input_tensor.device()),
            create_device_tensor(output_specs[2], tensor_args.tilize_input_tensor.device()),
            tilize_output_tensor,
            matmul_output_tensor};
    }

    const auto& combine_output_tensor = tensor_args.optional_output_tensor.value_or(
        create_device_tensor(output_specs[5], tensor_args.tilize_input_tensor.device()));

    return {
        create_device_tensor(output_specs[0], tensor_args.tilize_input_tensor.device()),
        create_device_tensor(output_specs[1], tensor_args.tilize_input_tensor.device()),
        create_device_tensor(output_specs[2], tensor_args.tilize_input_tensor.device()),
        tilize_output_tensor,
        matmul_output_tensor,
        combine_output_tensor};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<ttnn::Tensor> moe_compute(
    const ttnn::Tensor& tilize_input_tensor,
    const ttnn::Tensor& tilize_expert_indices_tensor,
    const ttnn::Tensor& tilize_expert_scores_tensor,
    const ttnn::Tensor& tilize_expert_mapping_tensor,
    const ttnn::Tensor& matmul_w0_w1_tensor,
    const ttnn::Tensor& matmul_w2_tensor,
    const uint32_t layer_id,
    const uint32_t output_height_shard_dim,
    const uint32_t intermediate_size,
    const bool has_bias,
    const std::optional<uint32_t>& cluster_axis,
    const std::optional<tt::tt_fabric::Topology>& topology,
    const std::optional<uint32_t>& num_links,
    const std::optional<CoreRangeSet>& mux_core_range_set,
    const std::optional<ttnn::MemoryConfig>& output_memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const std::optional<GlobalSemaphore>& optional_cross_device_semaphore,
    const std::optional<ttnn::experimental::prim::detail::MoEActivationFunction>& activation_type,
    const bool compute_only,
    const std::optional<uint32_t>& bh_ring_size,
    const std::optional<uint32_t>& num_shared_experts_per_device) {
    using OperationType = ttnn::experimental::prim::MoEComputeDeviceOperation;

    const auto& input_shape = tilize_input_tensor.tensor_spec().logical_shape();
    const auto& indices_shape = tilize_expert_indices_tensor.tensor_spec().logical_shape();
    const uint32_t hidden_size = input_shape[-1];
    const uint32_t select_experts_k = indices_shape[-1];
    const uint32_t total_tokens = input_shape[0] * input_shape[1];

    const auto& num_token_parallel_cores = output_height_shard_dim;

    auto* mesh_device = tilize_input_tensor.device();

    // Ring size is 12 on Wormhole (no DRAM-bank harvesting). On Blackhole it is the live
    // DRAM-bank count (7 or 8). Resolved by the public API before invocation, but keep a
    // fallback to the live bank count for direct prim callers.
    const uint32_t ring_n = bh_ring_size.value_or(
        mesh_device->arch() == tt::ARCH::BLACKHOLE
            ? mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default).size()
            : 12u);
    // NOTE: the public API auto-detects the ring from the device and does not expose it as a knob.

    // Auto-compute num_data_parallel_cores: largest divisor d of hidden_tiles with d <= 4
    // AND ring_n % d == 0. dm1 maps ring cores to combine columns via
    // RING_CORES_PER_COMBINE_COL = num_cores / width_shard_dim, so both must divide evenly.
    // E.g. GPT-OSS (Ht=90) picks d=3 on WH (N=12) but falls back to d=2 on BH (N=8/7).
    const uint32_t hidden_tiles = hidden_size / 32;
    uint32_t num_data_parallel_cores = 1;
    for (uint32_t d = 4; d >= 1; --d) {
        if (hidden_tiles % d == 0 && ring_n % d == 0) {
            num_data_parallel_cores = d;
            break;
        }
    }

    // Determine the MoE compute path from compute_only and cluster_axis.
    // - ComputeOnly: compute_only=true, cluster_axis must be None, no CCL options.
    // - FullLocal: compute_only=false, cluster_axis=None, only valid on a 1x1 mesh. No CCL
    //   options; combine runs as a local reduction with no fabric.
    // - LocalOutput: compute_only=false, cluster_axis names an axis of extent 1. Nothing to
    //   combine: dm1 writes the final output, the fabric is not consulted and the CCL options
    //   are accepted and unused.
    // - FullCcl: compute_only=false, cluster_axis names an axis of extent > 1. CCL options apply.
    const uint32_t num_devices = mesh_device->num_devices();
    const auto& mesh_shape = mesh_device->shape();
    if (cluster_axis.has_value()) {
        TT_FATAL(
            *cluster_axis < mesh_shape.dims(),
            "cluster_axis {} is out of range for a mesh with {} axes",
            *cluster_axis,
            mesh_shape.dims());
    }
    const bool full_local = !compute_only && !cluster_axis.has_value();
    const bool local_output = !compute_only && cluster_axis.has_value() && mesh_shape[*cluster_axis] == 1;
    if (full_local) {
        TT_FATAL(
            num_devices == 1,
            "moe_compute(compute_only=false, cluster_axis=None) is only supported on a 1x1 mesh, "
            "got num_devices={}. Pass cluster_axis for multi-device fused compute+combine.",
            num_devices);
    }

    if (compute_only) {
        TT_FATAL(!cluster_axis.has_value(), "moe_compute(compute_only=true) requires cluster_axis to be std::nullopt");
        TT_FATAL(!topology.has_value(), "moe_compute(compute_only=true) requires topology to be std::nullopt");
        TT_FATAL(!num_links.has_value(), "moe_compute(compute_only=true) requires num_links to be std::nullopt");
        TT_FATAL(
            !mux_core_range_set.has_value(),
            "moe_compute(compute_only=true) requires mux_core_range_set to be std::nullopt");
        TT_FATAL(
            !optional_cross_device_semaphore.has_value(),
            "moe_compute(compute_only=true) requires optional_cross_device_semaphore to be std::nullopt");
        TT_FATAL(
            !optional_output_tensor.has_value(),
            "moe_compute(compute_only=true) requires optional_output_tensor to be std::nullopt");
    } else if (full_local) {
        TT_FATAL(!topology.has_value(), "moe_compute(cluster_axis=None) requires topology to be std::nullopt");
        TT_FATAL(!num_links.has_value(), "moe_compute(cluster_axis=None) requires num_links to be std::nullopt");
        TT_FATAL(
            !mux_core_range_set.has_value(),
            "moe_compute(cluster_axis=None) requires mux_core_range_set to be std::nullopt");
        TT_FATAL(
            !optional_cross_device_semaphore.has_value(),
            "moe_compute(cluster_axis=None) requires optional_cross_device_semaphore to be std::nullopt");
    } else {
        TT_FATAL(cluster_axis.has_value(), "moe_compute(compute_only=false) requires cluster_axis to be provided");
    }

    const auto& combine_cores = get_moe_combine_cores(
        mesh_device,
        num_token_parallel_cores,
        num_data_parallel_cores,
        hidden_size,
        mux_core_range_set.value_or(CoreRangeSet{}),
        ring_n);

    std::optional<ttnn::experimental::prim::SelectiveReduceCombineParams> combine_params;
    if (full_local) {
        // Local combine: no fabric, no mux, no cross-device semaphore. axis=0 names an axis of
        // extent 1 on the 1x1 mesh, which the combine program factory builds as the local
        // combine (and mesh_shape[1-axis]=1 for shared_expert_tp_factor).
        combine_params = ttnn::experimental::prim::SelectiveReduceCombineParams{
            .hidden_size = hidden_size,
            .batch_size = 1,
            .seq_size = total_tokens,
            .select_experts_k = select_experts_k,
            .num_links = 1,
            .axis = 0,
            .topology = tt::tt_fabric::Topology::Linear,
            .num_token_parallel_cores = num_token_parallel_cores,
            .num_data_parallel_cores = num_data_parallel_cores,
            .worker_cores = combine_cores,
            .mux_core_range_set = CoreRangeSet{},
            .output_memory_config = output_memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
            .optional_cross_device_semaphore = std::nullopt,
            .local_combine = true};
    } else if (!compute_only) {
        // An axis of extent 1 has nothing to combine (LocalOutput: dm1 writes the final output), so
        // skip link discovery and the fabric topology lookup, both of which need an initialized
        // fabric context that a mesh opened without a fabric config lacks. Explicit CCL options are
        // accepted and unused there; num_links is still checked > 0 below.
        const bool axis_has_neighbours = !local_output;
        // The local output leaves one partial per mesh coordinate that the caller sums, so a
        // shared expert would be counted once per coordinate.
        TT_FATAL(
            axis_has_neighbours || num_shared_experts_per_device.value_or(0) == 0,
            "moe_compute over a mesh axis of extent 1 writes a local output and does not support shared experts; "
            "run them separately and add their partial before the cross-device reduction");
        // see #27196 for potential limitations
        const uint32_t resolved_num_links = num_links.value_or(
            axis_has_neighbours ? ttnn::operations::ccl::common::get_num_links(*mesh_device, *cluster_axis) : 1);
        // Resolve `topology` via the shared CCL helper. This (a) substitutes the fabric
        // default when `topology` is nullopt, (b) maps Torus → Mesh when the tensor doesn't
        // span a wrap edge so the TT_FATAL below can reject it, and (c) downgrades Ring → Linear
        // for the trivial `mesh_shape[cluster_axis] == 2` case. Notably, it does NOT detect
        // physically-LINE meshes whose tensor still spans the full cluster axis (e.g. BH single
        // Loudbox 2x4 LINE/LINE with cluster_axis=1) — that case still resolves to Ring here.
        // BH LB callers must pass topology=Linear explicitly; the kernel-side `Topology` template
        // guard in fabric_multicast_bidirectional_atomic_inc_1d (moe_utils.hpp) then routes the
        // multicast through the line-aware code path. (Fixing get_usable_topology() to consult
        // physical mesh wrap capability is a separate follow-up that affects all CCL ops.)
        // An axis of extent 1 is a trivial topology: Linear unless the caller named one.
        const auto resolved_topology = axis_has_neighbours
                                           ? ttnn::ccl::get_usable_topology(tilize_input_tensor, topology, cluster_axis)
                                           : topology.value_or(tt::tt_fabric::Topology::Linear);
        // Mirror the kernel-side static_assert in fabric_multicast_bidirectional_atomic_inc_1d
        // (moe_utils.hpp). `get_usable_topology` can return Mesh when the fabric default is Torus
        // and the tensor doesn't span a wrap edge; the combine kernel only handles Ring/Linear and
        // would silently produce wrong wait counts → on-device hang. Reject at the host boundary
        // with a clear message instead of waiting for a JIT compile failure or a hang.
        TT_FATAL(
            resolved_topology == tt::tt_fabric::Topology::Linear || resolved_topology == tt::tt_fabric::Topology::Ring,
            "moe_compute: combine kernel only supports Topology::Linear or Topology::Ring, got {}. "
            "If the fabric default is Torus/Mesh, pass topology=ttnn.Topology.Linear or "
            "ttnn.Topology.Ring explicitly to ttnn.experimental.moe_compute.",
            resolved_topology);
        // LocalOutput: slot 5 is the caller's optional_output_tensor when one is given, so an unset
        // output_memory_config takes that tensor's config instead of a DRAM default that would
        // describe a buffer it does not match; when both are given the device op requires them to
        // agree. FullCcl keeps the DRAM default (the combine's own validation applies there).
        const ttnn::MemoryConfig resolved_output_memory_config =
            output_memory_config.has_value()                       ? *output_memory_config
            : (local_output && optional_output_tensor.has_value()) ? optional_output_tensor->memory_config()
                                                                   : ttnn::DRAM_MEMORY_CONFIG;
        combine_params = ttnn::experimental::prim::SelectiveReduceCombineParams{
            .hidden_size = hidden_size,
            .batch_size = 1,
            .seq_size = total_tokens,
            .select_experts_k = select_experts_k,
            .num_links = resolved_num_links,
            .axis = cluster_axis.value(),
            .topology = resolved_topology,
            .num_token_parallel_cores = num_token_parallel_cores,
            .num_data_parallel_cores = num_data_parallel_cores,
            .worker_cores = combine_cores,
            .mux_core_range_set = mux_core_range_set.value_or(CoreRangeSet{}),
            .output_memory_config = resolved_output_memory_config,
            .optional_cross_device_semaphore = optional_cross_device_semaphore};
    }

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .layer_id = layer_id,
            .output_height_shard_dim = output_height_shard_dim,
            .intermediate_size = intermediate_size,
            .num_shared_experts_per_device = num_shared_experts_per_device,
            .has_bias = has_bias,
            .num_token_parallel_cores = num_token_parallel_cores,
            .num_data_parallel_cores = num_data_parallel_cores,
            .path = compute_only   ? experimental::prim::MoEComputePath::ComputeOnly
                    : full_local   ? experimental::prim::MoEComputePath::FullLocal
                    : local_output ? experimental::prim::MoEComputePath::LocalOutput
                                   : experimental::prim::MoEComputePath::FullCcl,
            .bh_ring_size = ring_n,
            .combine_params = combine_params,
            .activation_type = activation_type.value_or(experimental::prim::detail::MoEActivationFunction::SILU)},
        OperationType::tensor_args_t{
            .tilize_input_tensor = tilize_input_tensor,
            .tilize_expert_indices_tensor = tilize_expert_indices_tensor,
            .tilize_expert_scores_tensor = tilize_expert_scores_tensor,
            .tilize_expert_mapping_tensor = tilize_expert_mapping_tensor,
            .matmul_w0_w1_tensor = matmul_w0_w1_tensor,
            .matmul_w2_tensor = matmul_w2_tensor,
            .optional_output_tensor = optional_output_tensor});
}

}  // namespace ttnn::prim
