// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_stats/device/attn_res_block_stats_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;

namespace {

uint32_t num_candidates(const AttnResBlockStatsInputs& tensor_args) {
    uint32_t count = 0;
    for (const auto& snapshot : tensor_args.sealed) {
        count += snapshot.logical_shape()[1];
    }
    return count;
}

// The semaphore is hashed by its cores, so a cache hit can carry one the program was not built
// against. Peers count into it at the gather cores' address; built without one of them, or in
// DRAM, nothing arrives where the core waits and the op hangs.
void validate_semaphore(const AttnResBlockStatsParams& args, const AttnResBlockStatsInputs& tensor_args) {
    TT_FATAL(
        args.cluster_axis.has_value() == args.semaphore.has_value(),
        "AttnResBlockStats gathers across an axis with the caller's semaphore: give both or neither");
    if (!args.semaphore.has_value()) {
        return;
    }
    auto* device = tensor_args.queries.device();
    TT_FATAL(args.semaphore->device() == device, "AttnResBlockStats requires the semaphore on its operands' device");
    const auto [semaphore_cores, semaphore_buffer_type] = args.semaphore->attribute_values();
    const auto grid = device->compute_with_storage_grid_size();
    for (uint32_t link = 0; link < args.num_links; ++link) {
        const CoreCoord gather_core{grid.x - 1 - link, grid.y - 1};
        TT_FATAL(
            semaphore_cores.contains(gather_core),
            "AttnResBlockStats gathers on core {} and needs the semaphore built over it, but it covers {}",
            gather_core,
            semaphore_cores);
    }
    TT_FATAL(
        semaphore_buffer_type == tt::tt_metal::BufferType::L1,
        "AttnResBlockStats waits on the semaphore in L1, got buffer type {}",
        semaphore_buffer_type);
}

}  // namespace

void AttnResBlockStatsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_semaphore(args, tensor_args);
}

void AttnResBlockStatsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& queries = tensor_args.queries;
    const tt::ARCH arch = queries.device()->arch();
    TT_FATAL(arch == tt::ARCH::BLACKHOLE, "AttnResBlockStats is only supported on Blackhole, got {}", arch);
    TT_FATAL(!tensor_args.sealed.empty(), "AttnResBlockStats needs at least one sealed snapshot tensor");
    TT_FATAL(!args.output_mem_config.is_sharded(), "AttnResBlockStats writes an interleaved output");

    operations::check_tensor(queries, "AttnResBlockStats", "queries", {DataType::BFLOAT16});
    const auto& first = tensor_args.sealed.front();
    const auto& shape = first.padded_shape();
    for (const auto& snapshot : tensor_args.sealed) {
        operations::check_tensor(snapshot, "AttnResBlockStats", "sealed", {DataType::BFLOAT16});
        TT_FATAL(snapshot.device() == queries.device(), "AttnResBlockStats operands must share a device");
        // One accessor configuration reads every snapshot, so they must agree on everything
        // but their address and their plane count.
        TT_FATAL(
            snapshot.memory_config() == first.memory_config() && !snapshot.memory_config().is_sharded(),
            "AttnResBlockStats reads interleaved snapshots with one memory config");
        const auto& snapshot_shape = snapshot.padded_shape();
        TT_FATAL(
            snapshot_shape.rank() == 4 && snapshot_shape[0] == 1 && snapshot_shape[2] == shape[2] &&
                snapshot_shape[3] == shape[3] && snapshot.logical_shape()[2] == first.logical_shape()[2],
            "AttnResBlockStats snapshots must be [1, S, N, d] with one N and d, got {} against {}",
            snapshot_shape,
            shape);
    }

    const auto& queries_shape = queries.padded_shape();
    TT_FATAL(
        queries_shape.rank() == 4 && queries_shape[0] == 1 && queries_shape[1] == 1 && queries_shape[2] == shape[3] &&
            queries_shape[3] == TILE_WIDTH,
        "AttnResBlockStats queries must be [1, 1, d, {}] for d = {}, got {}",
        TILE_WIDTH,
        shape[3],
        queries_shape);

    TT_FATAL(
        args.cluster_axis.has_value() == tensor_args.gathered.has_value(),
        "AttnResBlockStats gathers across an axis into a caller's buffer: give both or neither");
    validate_semaphore(args, tensor_args);
    if (tensor_args.gathered.has_value()) {
        const auto& gathered = *tensor_args.gathered;
        const auto view = queries.device()->get_view();
        TT_FATAL(view.is_mesh_2d() && *args.cluster_axis < 2, "AttnResBlockStats gathers across a 2D mesh axis");
        const uint32_t ring_size = *args.cluster_axis == 0 ? view.num_rows() : view.num_cols();
        TT_FATAL(ring_size > 1, "AttnResBlockStats gathers across an axis of more than one device, got {}", ring_size);
        // A gather core per link, on the grid's last row; without one nothing is sent.
        const auto grid = queries.device()->compute_with_storage_grid_size();
        TT_FATAL(
            args.num_links >= 1 && args.num_links <= grid.x,
            "AttnResBlockStats gathers over 1 to {} links, got {}",
            grid.x,
            args.num_links);
        operations::check_tensor(gathered, "AttnResBlockStats", "gathered", {DataType::FLOAT32});
        TT_FATAL(!gathered.memory_config().is_sharded(), "AttnResBlockStats gathers into an interleaved buffer");
        const auto expected = ttnn::Shape({1, num_candidates(tensor_args), shape[2], TILE_WIDTH * ring_size});
        TT_FATAL(
            gathered.padded_shape() == expected,
            "AttnResBlockStats gathers into {}, got {}",
            expected,
            gathered.padded_shape());
    }
}

tt::tt_metal::TensorSpec AttnResBlockStatsDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.gathered.has_value()) {
        return tensor_args.gathered->tensor_spec();
    }
    const uint32_t num_tokens = tensor_args.sealed.front().logical_shape()[2];
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, num_candidates(tensor_args), num_tokens, TILE_WIDTH}),
        operations::TensorLayout(DataType::FLOAT32, operations::PageConfig(Layout::TILE), args.output_mem_config));
}

Tensor AttnResBlockStatsDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.gathered.has_value()) {
        return *tensor_args.gathered;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.queries.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor attn_res_block_stats(
    const std::vector<Tensor>& sealed,
    const Tensor& queries,
    const std::optional<Tensor>& gathered,
    std::optional<uint32_t> cluster_axis,
    const std::optional<GlobalSemaphore>& semaphore,
    ttnn::ccl::Topology topology,
    uint32_t num_links,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::AttnResBlockStatsDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
        .cluster_axis = cluster_axis,
        .topology = topology,
        .num_links = num_links,
        .semaphore = semaphore};
    auto tensor_args = OperationType::tensor_args_t{.sealed = sealed, .queries = queries, .gathered = gathered};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
