// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/device/attn_res_block_mix_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;

namespace {

// A partial is a whole tile per (candidate, row-tile) in L1 until it is summed.
constexpr uint32_t kMaxPartials = 8;

}  // namespace

void AttnResBlockMixDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& stats = tensor_args.stats;
    const tt::ARCH arch = stats.device()->arch();
    TT_FATAL(arch == tt::ARCH::BLACKHOLE, "AttnResBlockMix is only supported on Blackhole, got {}", arch);
    TT_FATAL(!tensor_args.sealed.empty(), "AttnResBlockMix needs at least one sealed snapshot tensor");
    TT_FATAL(!args.output_mem_config.is_sharded(), "AttnResBlockMix writes interleaved outputs");

    const auto& first = tensor_args.sealed.front();
    const auto& shape = first.padded_shape();
    uint32_t num_candidates = 0;
    for (const auto& snapshot : tensor_args.sealed) {
        operations::check_tensor(snapshot, "AttnResBlockMix", "sealed", {DataType::BFLOAT16});
        TT_FATAL(snapshot.device() == stats.device(), "AttnResBlockMix operands must share a device");
        // One accessor configuration reads every snapshot, so they must agree on everything
        // but their address and their plane count.
        TT_FATAL(
            snapshot.memory_config() == first.memory_config() && !snapshot.memory_config().is_sharded(),
            "AttnResBlockMix reads interleaved snapshots with one memory config");
        const auto& snapshot_shape = snapshot.padded_shape();
        TT_FATAL(
            snapshot_shape.rank() == 4 && snapshot_shape[0] == 1 && snapshot_shape[2] == shape[2] &&
                snapshot_shape[3] == shape[3] && snapshot.logical_shape()[2] == first.logical_shape()[2],
            "AttnResBlockMix snapshots must be [1, S, N, d] with one N and d, got {} against {}",
            snapshot_shape,
            shape);
        num_candidates += snapshot_shape[1];
    }

    // fp32, because shift and mass come out fp32 and their consumer reads them through the
    // same circular buffer as fp32 statistics.
    operations::check_tensor(stats, "AttnResBlockMix", "stats", {DataType::FLOAT32});
    const auto& stats_shape = stats.padded_shape();
    TT_FATAL(
        stats_shape.rank() == 4 && stats_shape[0] == 1 && stats_shape[1] == num_candidates &&
            stats_shape[2] == shape[2],
        "AttnResBlockMix stats must be [1, {}, {}, 32 * P] for the sealed set, got {}",
        num_candidates,
        shape[2],
        stats_shape);
    // One tile column per tensor-parallel partial, unsummed. Within a partial, a site is a
    // column, so the sites and the sum of squares share one tile column.
    TT_FATAL(
        stats_shape[-1] % TILE_WIDTH == 0 && stats_shape[-1] / TILE_WIDTH <= kMaxPartials,
        "AttnResBlockMix reads up to {} tile columns of statistics, one per partial, got a padded last dim of {}",
        kMaxPartials,
        stats_shape[-1]);
    TT_FATAL(
        args.num_sites >= 1 && 1 + args.num_sites <= TILE_WIDTH,
        "AttnResBlockMix mixes {} sites, but a statistics tile column holds the sum of squares and at most {}",
        args.num_sites,
        TILE_WIDTH - 1);
    TT_FATAL(
        args.num_closed_sites <= args.num_sites,
        "AttnResBlockMix closes {} of its {} sites",
        args.num_closed_sites,
        args.num_sites);
}

AttnResBlockMixDeviceOperation::spec_return_value_t AttnResBlockMixDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& first = tensor_args.sealed.front();
    const uint32_t num_tokens = first.logical_shape()[2];
    const uint32_t width = first.logical_shape()[3];
    const auto scalar_layout =
        operations::TensorLayout(DataType::FLOAT32, operations::PageConfig(Layout::TILE), args.output_mem_config);
    return {
        tt::tt_metal::TensorSpec(
            ttnn::Shape({args.num_sites, 1, num_tokens, width}),
            operations::TensorLayout(first.dtype(), operations::PageConfig(Layout::TILE), args.output_mem_config)),
        tt::tt_metal::TensorSpec(ttnn::Shape({args.num_sites, 1, num_tokens, 1}), scalar_layout),
        tt::tt_metal::TensorSpec(ttnn::Shape({args.num_sites, 1, num_tokens, 1}), scalar_layout)};
}

AttnResBlockMixDeviceOperation::tensor_return_value_t AttnResBlockMixDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.stats.device();
    return {
        create_device_tensor(specs[0], device),
        create_device_tensor(specs[1], device),
        create_device_tensor(specs[2], device)};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::array<Tensor, 3> attn_res_block_mix(
    const std::vector<Tensor>& sealed,
    const Tensor& stats,
    uint32_t num_sites,
    float inv_hidden_size,
    float eps,
    uint32_t num_closed_sites,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::AttnResBlockMixDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .num_sites = num_sites,
        .num_closed_sites = num_closed_sites,
        .inv_hidden_size = inv_hidden_size,
        .eps = eps,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config};
    auto tensor_args = OperationType::tensor_args_t{.sealed = sealed, .stats = stats};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
