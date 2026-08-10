// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/attn_res_accum_stats/device/attn_res_accum_stats_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;

void AttnResAccumStatsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;
    const auto& q = tensor_args.q;

    // The compute config defaults to HiFi4 with fp32 dest accumulation, which is only
    // correct on Blackhole; elsewhere the op compiles, runs, and returns silently wrong
    // values. Reject the device rather than let a caller reach that path.
    const tt::ARCH arch = tt::tt_metal::hal::get_arch();
    TT_FATAL(arch == tt::ARCH::BLACKHOLE, "AttnResAccumStats is only supported on Blackhole, got {}", arch);

    // The sum runs on the FPU, whose source registers hold about eleven mantissa bits, so
    // a Float32 addend would come back short of what an unfused add returns — that one
    // reaches the SFPU and is exact. Bfloat16 fits the registers whole. `q` is only ever
    // contracted into a reduction, which absorbs the same rounding, so it is unrestricted.
    operations::check_tensor(a, "AttnResAccumStats", "a", {DataType::BFLOAT16});
    operations::check_tensor(b, "AttnResAccumStats", "b", {DataType::BFLOAT16});
    operations::check_tensor(q, "AttnResAccumStats", "q", {DataType::BFLOAT16, DataType::FLOAT32});
    TT_FATAL(
        args.stats_dtype == DataType::BFLOAT16 || args.stats_dtype == DataType::FLOAT32,
        "AttnResAccumStats supports a BFLOAT16 or FLOAT32 statistics output, got {}",
        args.stats_dtype);

    for (const auto& [tensor, name] : {std::pair{&a, "a"}, std::pair{&b, "b"}, std::pair{&q, "q"}}) {
        TT_FATAL(
            tensor->storage_type() == StorageType::DEVICE,
            "AttnResAccumStats requires {} on device, got {}",
            name,
            tensor->storage_type());
        TT_FATAL(tensor->layout() == Layout::TILE, "AttnResAccumStats requires TILE layout on {}", name);
        TT_FATAL(!tensor->memory_config().is_sharded(), "AttnResAccumStats supports interleaved inputs only");
    }
    TT_FATAL(
        !args.total_mem_config.is_sharded() && !args.stats_mem_config.is_sharded(),
        "AttnResAccumStats supports interleaved outputs only");

    TT_FATAL(
        a.logical_shape() == b.logical_shape(),
        "AttnResAccumStats sums matching shapes, got {} against {}",
        a.logical_shape(),
        b.logical_shape());

    const auto& a_shape = a.padded_shape();
    const auto& q_shape = q.padded_shape();
    TT_FATAL(a_shape.rank() == 4, "AttnResAccumStats requires a rank-4 a, got rank {}", a_shape.rank());
    TT_FATAL(q_shape.rank() == 4, "AttnResAccumStats requires a rank-4 q, got rank {}", q_shape.rank());

    // The candidate axis is the statistics output's stacking axis, so dim 0 has to be flat.
    TT_FATAL(a_shape[0] == 1, "AttnResAccumStats requires a leading dim of 1 on a, got {}", a_shape[0]);
    TT_FATAL(
        q_shape[0] == 1 && q_shape[1] == 1,
        "AttnResAccumStats requires a single query row, got q dims [{}, {}, ...]",
        q_shape[0],
        q_shape[1]);
    // Logical rather than padded on both counts. The kernel broadcasts q's first row down
    // the tokens, and a padded height cannot tell one query row from thirty-two; a padded
    // width would likewise admit d = 63 against d = 33, whose dot contracts q's zero
    // padding against real data.
    const auto& a_logical = a.logical_shape();
    const auto& q_logical = q.logical_shape();
    TT_FATAL(
        q_logical[-2] == 1,
        "AttnResAccumStats broadcasts a single query row down the tokens, got {} rows",
        q_logical[-2]);
    TT_FATAL(
        a_logical[-1] == q_logical[-1],
        "AttnResAccumStats contracts the sum and q over d, got {} against {}",
        a_logical[-1],
        q_logical[-1]);

    TT_FATAL(
        a_shape[-1] % TILE_WIDTH == 0 && a_shape[-2] % TILE_HEIGHT == 0,
        "AttnResAccumStats requires tile-aligned inner dims on a, got {} x {}",
        a_shape[-2],
        a_shape[-1]);
}

AttnResAccumStatsDeviceOperation::spec_return_value_t AttnResAccumStatsDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // From the logical shape, not the padded one, so that tile padding stays labelled as
    // padding rather than becoming readable data. The two statistics stack on the
    // candidate axis in the order a gathering collective leaves them.
    const auto& a_shape = tensor_args.a.logical_shape();
    const ttnn::Shape stats_shape({a_shape[0], a_shape[1] * 2, a_shape[2], 1});

    return {
        tt::tt_metal::TensorSpec(
            a_shape,
            operations::TensorLayout(
                tensor_args.a.dtype(), operations::PageConfig(Layout::TILE), args.total_mem_config)),
        tt::tt_metal::TensorSpec(
            stats_shape,
            operations::TensorLayout(args.stats_dtype, operations::PageConfig(Layout::TILE), args.stats_mem_config))};
}

AttnResAccumStatsDeviceOperation::tensor_return_value_t AttnResAccumStatsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(operation_attributes, tensor_args);
    auto* device = tensor_args.a.device();
    return {create_device_tensor(specs[0], device), create_device_tensor(specs[1], device)};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::array<Tensor, 2> attn_res_accum_stats(
    const Tensor& a,
    const Tensor& b,
    const Tensor& q,
    DataType stats_dtype,
    const MemoryConfig& total_mem_config,
    const MemoryConfig& stats_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::AttnResAccumStatsDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .stats_dtype = stats_dtype,
        .total_mem_config = total_mem_config,
        .stats_mem_config = stats_mem_config,
        .compute_kernel_config = compute_kernel_config};
    auto tensor_args = OperationType::tensor_args_t{.a = a, .b = b, .q = q};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
