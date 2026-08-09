// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/attn_res_stats/device/attn_res_stats_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;

void AttnResStatsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& v = tensor_args.v;
    const auto& q = tensor_args.q;

    operations::check_tensor(v, "AttnResStats", "v", {DataType::BFLOAT16, DataType::FLOAT32});
    operations::check_tensor(q, "AttnResStats", "q", {DataType::BFLOAT16, DataType::FLOAT32});
    TT_FATAL(
        args.dtype == DataType::BFLOAT16 || args.dtype == DataType::FLOAT32,
        "AttnResStats supports a BFLOAT16 or FLOAT32 output, got {}",
        args.dtype);
    TT_FATAL(v.storage_type() == StorageType::DEVICE, "AttnResStats requires v on device, got {}", v.storage_type());
    TT_FATAL(q.storage_type() == StorageType::DEVICE, "AttnResStats requires q on device, got {}", q.storage_type());
    TT_FATAL(v.layout() == Layout::TILE && q.layout() == Layout::TILE, "AttnResStats requires TILE layout");
    TT_FATAL(
        !v.memory_config().is_sharded() && !q.memory_config().is_sharded(),
        "AttnResStats supports interleaved inputs only");
    TT_FATAL(!args.output_mem_config.is_sharded(), "AttnResStats supports an interleaved output only");

    const auto& v_shape = v.padded_shape();
    const auto& q_shape = q.padded_shape();
    TT_FATAL(v_shape.rank() == 4, "AttnResStats requires a rank-4 v, got rank {}", v_shape.rank());
    TT_FATAL(q_shape.rank() == 4, "AttnResStats requires a rank-4 q, got rank {}", q_shape.rank());

    // The candidate axis is the output's stacking axis, so dim 0 has to be flat.
    TT_FATAL(v_shape[0] == 1, "AttnResStats requires a leading dim of 1 on v, got {}", v_shape[0]);
    TT_FATAL(
        q_shape[0] == 1 && q_shape[1] == 1,
        "AttnResStats requires a single query row, got q dims [{}, {}, ...]",
        q_shape[0],
        q_shape[1]);
    TT_FATAL(
        q_shape[-2] == TILE_HEIGHT,
        "AttnResStats broadcasts q down the tokens, so it must occupy exactly one tile row, got {}",
        q_shape[-2]);
    TT_FATAL(
        v_shape[-1] == q_shape[-1],
        "AttnResStats contracts v and q over d, got {} against {}",
        v_shape[-1],
        q_shape[-1]);

    TT_FATAL(
        v_shape[-1] % TILE_WIDTH == 0 && v_shape[-2] % TILE_HEIGHT == 0,
        "AttnResStats requires tile-aligned inner dims on v, got {} x {}",
        v_shape[-2],
        v_shape[-1]);
}

tt::tt_metal::TensorSpec AttnResStatsDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // From the logical shape, not the padded one, so that tile padding stays
    // labelled as padding rather than becoming readable data. The two statistics
    // stack on the candidate axis in the order `attn_res_scores` splits them.
    const auto& v_shape = tensor_args.v.logical_shape();
    const ttnn::Shape shape({v_shape[0], v_shape[1] * 2, v_shape[2], 1});

    return tt::tt_metal::TensorSpec(
        shape, operations::TensorLayout(args.dtype, operations::PageConfig(Layout::TILE), args.output_mem_config));
}

Tensor AttnResStatsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.v.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor attn_res_stats(
    const Tensor& v,
    const Tensor& q,
    DataType dtype,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::AttnResStatsDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .dtype = dtype, .output_mem_config = output_mem_config, .compute_kernel_config = compute_kernel_config};
    auto tensor_args = OperationType::tensor_args_t{.v = v, .q = q};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
