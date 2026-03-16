// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op_device_operation.hpp"

#include <optional>
#include <string>

#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace reduce_op_utils {

std::map<std::string, std::string> get_defines(
    tt::tt_metal::ReduceOpMath reduce_op, tt::tt_metal::ReduceOpDim reduce_dim) {
    std::map<std::string, std::string> defines;
    // TODO(AP): need a sync with Reduce::Max from HLK headers
    bool do_max = reduce_op == tt::tt_metal::ReduceOpMath::MAX;
    std::string reduce_dim_str;
    switch (reduce_dim) {
        case tt::tt_metal::ReduceOpDim::W: reduce_dim_str = "ReduceDim::REDUCE_ROW"; break;
        case tt::tt_metal::ReduceOpDim::H: reduce_dim_str = "ReduceDim::REDUCE_COL"; break;
        case tt::tt_metal::ReduceOpDim::HW: reduce_dim_str = "ReduceDim::REDUCE_SCALAR"; break;
        default: TT_THROW("Invalid reduce_op!");
    }
    defines["REDUCE_OP"] = (do_max ? "PoolType::MAX" : "PoolType::SUM");
    defines["REDUCE_DIM"] = reduce_dim_str;
    if (reduce_dim == tt::tt_metal::ReduceOpDim::W && reduce_op == tt::tt_metal::ReduceOpMath::SUM) {
        defines["REDUCE_ROW_SUM_VIA_MM"] = "1";
    }
    return defines;
}

}  // namespace reduce_op_utils
namespace ttnn::operations::reduction::generic::detail {

// reduce min
// reduce min = - reduce_max( -x )
Tensor reduce_min(
    const Tensor& input_tensor,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scaler = 1.0f,
    const tt::tt_metal::MemoryConfig& output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt) {
    Tensor input = input_tensor;
    if (input.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
        input.storage_type() == tt::tt_metal::StorageType::DEVICE) {
        // Changing layout to TILE with +inf padding
        auto pad_shape = ttnn::operations::data_movement::pad_to_tile_shape(input.padded_shape());
        input =
            ttnn::tilize_with_val_padding(input, pad_shape, std::numeric_limits<float>::infinity(), output_mem_config);
    }
    return detail::reduce(
        input,
        tt::tt_metal::ReduceOpMath::MAX,
        reduce_dim,
        scaler,
        output_mem_config,
        std::nullopt,
        compute_kernel_config,
        std::nullopt,
        true);
}

Tensor reduce(
    const Tensor& input_tensor,
    tt::tt_metal::ReduceOpMath reduce_math,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scaler,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::DataType>& output_dtype,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    bool negate) {
    if (reduce_math == tt::tt_metal::ReduceOpMath::MIN) {
        return reduce_min(input_tensor, reduce_dim, scaler, output_mem_config);
    }

    auto parallelization_strategy = ttnn::prim::get_parallelization_strategy(input_tensor, reduce_dim);
    auto is_multicore_hw = parallelization_strategy == tt::tt_metal::ReduceOpParallelizationStrategy::MULTI_CORE_HW;
    float pad_value = reduce_math == tt::tt_metal::ReduceOpMath::MAX ? -std::numeric_limits<float>::infinity() : 0;

    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Expected input tensor to be on device");
    TT_FATAL(
        input_tensor.device() != nullptr,
        "input_tensor.device() == nullptr, No device found, move input_tensor to device");

    // Due to hardware bug (#38306), HiFi4 + fp32_dest_acc_en produces incorrect results on Wormhole B0.
    // fp32_dest_acc_en defaults to True here, so always use HiFi3 as default on Wormhole B0.
    const auto arch = input_tensor.device()->arch();
    const auto is_wormhole = arch == tt::ARCH::WORMHOLE_B0;
    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(ttnn::init_device_compute_kernel_config(
        arch,
        std::nullopt,
        is_wormhole ? MathFidelity::HiFi3 : MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true));
    if (is_wormhole && compute_kernel_config.has_value() && compute_kernel_config->fp32_dest_acc_en &&
        compute_kernel_config->math_fidelity == MathFidelity::HiFi4) {
        log_warning(
            tt::LogOp,
            "HiFi4 + fp32_dest_acc_en on Wormhole B0 may produce incorrect results "
            "(hw bug #38306). Prefer HiFi3.");
    }

    // Reduce only works with tile layout, so we need to tilize the input tensor if necessary
    auto padded_shape = ttnn::operations::data_movement::pad_to_tile_shape(input_tensor.padded_shape());
    auto tilized_input =
        ttnn::tilize_with_val_padding(input_tensor, padded_shape, pad_value, input_tensor.memory_config());
    if (is_multicore_hw) {
        // Multi-core HW reduction: first reduce W, then reduce H on the result
        const Tensor output_tensor = ttnn::prim::reduce(
            tilized_input,
            reduce_math,
            tt::tt_metal::ReduceOpDim::W,
            1.0f,
            output_mem_config,
            output_dtype.value_or(input_tensor.dtype()),
            config,
            sub_core_grids,
            negate);

        return ttnn::prim::reduce(
            output_tensor,
            reduce_math,
            tt::tt_metal::ReduceOpDim::H,
            scaler,
            output_mem_config,
            output_dtype.value_or(input_tensor.dtype()),
            config,
            sub_core_grids,
            negate);
    }
    return ttnn::prim::reduce(
        tilized_input,
        reduce_math,
        reduce_dim,
        scaler,
        output_mem_config,
        output_dtype.value_or(input_tensor.dtype()),
        config,
        sub_core_grids,
        negate);
}

}  // namespace ttnn::operations::reduction::generic::detail
