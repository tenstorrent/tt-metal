// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_scores/device/attn_res_scores_device_operation.hpp"

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

void AttnResScoresDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& stats = tensor_args.stats;

    // HiFi4 with fp32 dest is the configuration the chain is written for, and it is
    // only correct on Blackhole.
    const tt::ARCH arch = stats.device()->arch();
    TT_FATAL(arch == tt::ARCH::BLACKHOLE, "AttnResScores is only supported on Blackhole, got {}", arch);

    // fp32 in, because every output is fp32 and the outputs' consumers read shift and
    // mass through the same circular buffer as fp32 statistics. A bf16 input would
    // have rounded the sums of squares before they got here.
    operations::check_tensor(stats, "AttnResScores", "stats", {DataType::FLOAT32});
    TT_FATAL(
        !stats.memory_config().is_sharded() && !args.output_mem_config.is_sharded(),
        "AttnResScores supports interleaved operands only");

    const auto& shape = stats.padded_shape();
    TT_FATAL(shape.rank() == 4, "AttnResScores requires a rank-4 stats tensor, got rank {}", shape.rank());
    TT_FATAL(shape[0] == 1, "AttnResScores takes one statistics plane per candidate, got dim 0 of {}", shape[0]);
    // The last dim is one tile column per tensor-parallel partial, unsummed — the op sums
    // them, which is what lets the caller cross the axis with an all-gather instead of an
    // all-reduce. Within a partial, a site is a column, so the sites and the sum of
    // squares share one tile column.
    TT_FATAL(
        shape[-1] % TILE_WIDTH == 0 && shape[-1] / TILE_WIDTH <= kMaxPartials,
        "AttnResScores reads up to {} tile columns of statistics, one per partial, got a padded last dim of {}",
        kMaxPartials,
        shape[-1]);
    TT_FATAL(
        args.num_sites >= 1 && 1 + args.num_sites <= TILE_WIDTH,
        "AttnResScores scores {} sites, but a statistics tile column holds the sum of squares and at most {}",
        args.num_sites,
        TILE_WIDTH - 1);
}

AttnResScoresDeviceOperation::spec_return_value_t AttnResScoresDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& logical = tensor_args.stats.logical_shape();
    const uint32_t num_candidates = logical[1];
    const uint32_t num_tokens = logical[2];
    const auto layout =
        operations::TensorLayout(DataType::FLOAT32, operations::PageConfig(Layout::TILE), args.output_mem_config);
    return {
        tt::tt_metal::TensorSpec(ttnn::Shape({args.num_sites, num_candidates, num_tokens, 1}), layout),
        tt::tt_metal::TensorSpec(ttnn::Shape({args.num_sites, 1, num_tokens, 1}), layout),
        tt::tt_metal::TensorSpec(ttnn::Shape({args.num_sites, 1, num_tokens, 1}), layout)};
}

AttnResScoresDeviceOperation::tensor_return_value_t AttnResScoresDeviceOperation::create_output_tensors(
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

std::array<Tensor, 3> attn_res_scores(
    const Tensor& stats,
    uint32_t num_sites,
    float inv_hidden_size,
    float eps,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::AttnResScoresDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .num_sites = num_sites,
        .inv_hidden_size = inv_hidden_size,
        .eps = eps,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config};
    auto tensor_args = OperationType::tensor_args_t{.stats = stats};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
