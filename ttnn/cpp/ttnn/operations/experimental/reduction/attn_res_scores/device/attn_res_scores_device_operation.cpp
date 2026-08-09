// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/attn_res_scores/device/attn_res_scores_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;

void AttnResScoresDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& stats = tensor_args.stats;

    operations::check_tensor(stats, "AttnResScores", "stats", {DataType::BFLOAT16, DataType::FLOAT32});
    TT_FATAL(
        args.dtype == DataType::BFLOAT16 || args.dtype == DataType::FLOAT32,
        "AttnResScores supports a BFLOAT16 or FLOAT32 output, got {}",
        args.dtype);
    TT_FATAL(
        stats.storage_type() == StorageType::DEVICE,
        "AttnResScores requires stats on device, got {}",
        stats.storage_type());
    TT_FATAL(stats.layout() == Layout::TILE, "AttnResScores requires TILE layout for stats");
    TT_FATAL(!stats.memory_config().is_sharded(), "AttnResScores supports an interleaved input only");
    TT_FATAL(!args.output_mem_config.is_sharded(), "AttnResScores supports an interleaved output only");

    const auto& shape = stats.padded_shape();
    TT_FATAL(shape.rank() == 4, "AttnResScores requires a rank-4 input, got rank {}", shape.rank());

    // The two statistics are split by page arithmetic: candidate c reads page c
    // for its sum of squares and page c + C for its dot, so dim 1 has to be the
    // stacked pair and dim 0 has to be flat.
    TT_FATAL(shape[0] == 1, "AttnResScores requires a leading dim of 1, got {}", shape[0]);
    TT_FATAL(
        shape[1] % 2 == 0 && shape[1] > 0,
        "AttnResScores requires dim 1 to hold both statistics stacked, so it must be even and non-zero, got {}",
        shape[1]);

    TT_FATAL(
        shape[-1] % TILE_WIDTH == 0 && shape[-2] % TILE_HEIGHT == 0,
        "AttnResScores requires tile-aligned inner dims, got {} x {}",
        shape[-2],
        shape[-1]);
}

tt::tt_metal::TensorSpec AttnResScoresDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // From the logical shape, not the padded one, so that tile padding stays
    // labelled as padding rather than becoming readable data.
    auto shape = tensor_args.stats.logical_shape();
    shape[1] /= 2;

    return tt::tt_metal::TensorSpec(
        shape, operations::TensorLayout(args.dtype, operations::PageConfig(Layout::TILE), args.output_mem_config));
}

Tensor AttnResScoresDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.stats.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor attn_res_scores(
    const Tensor& stats,
    float inv_hidden_size,
    float eps,
    DataType dtype,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::AttnResScoresDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .inv_hidden_size = inv_hidden_size,
        .eps = eps,
        .dtype = dtype,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config};
    auto tensor_args = OperationType::tensor_args_t{.stats = stats};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
