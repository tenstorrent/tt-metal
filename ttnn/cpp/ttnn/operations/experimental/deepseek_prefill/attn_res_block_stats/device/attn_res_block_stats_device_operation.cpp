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

}  // namespace

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
    if (tensor_args.gathered.has_value()) {
        const auto& gathered = *tensor_args.gathered;
        const auto view = queries.device()->get_view();
        TT_FATAL(view.is_mesh_2d() && *args.cluster_axis < 2, "AttnResBlockStats gathers across a 2D mesh axis");
        const uint32_t ring_size = *args.cluster_axis == 0 ? view.num_rows() : view.num_cols();
        operations::check_tensor(gathered, "AttnResBlockStats", "gathered", {DataType::FLOAT32});
        TT_FATAL(!gathered.memory_config().is_sharded(), "AttnResBlockStats gathers into an interleaved buffer");
        TT_FATAL(
            gathered.padded_shape() == ttnn::Shape({1, num_candidates(tensor_args), shape[2], TILE_WIDTH * ring_size}),
            "AttnResBlockStats gathers into [1, {}, {}, {}], got {}",
            num_candidates(tensor_args),
            shape[2],
            TILE_WIDTH * ring_size,
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
        .num_links = num_links};
    auto tensor_args = OperationType::tensor_args_t{.sealed = sealed, .queries = queries, .gathered = gathered};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
