// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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
    std::string reduce_dim_str;
    switch (reduce_dim) {
        case tt::tt_metal::ReduceOpDim::W: reduce_dim_str = "ckernel::ReduceDim::REDUCE_ROW"; break;
        case tt::tt_metal::ReduceOpDim::H: reduce_dim_str = "ckernel::ReduceDim::REDUCE_COL"; break;
        case tt::tt_metal::ReduceOpDim::HW: reduce_dim_str = "ckernel::ReduceDim::REDUCE_SCALAR"; break;
        default: TT_THROW("Invalid reduce_dim!");
    }
    switch (reduce_op) {
        case tt::tt_metal::ReduceOpMath::MAX: defines["REDUCE_OP"] = "ckernel::PoolType::MAX"; break;
        case tt::tt_metal::ReduceOpMath::AVG: defines["REDUCE_OP"] = "ckernel::PoolType::AVG"; break;
        default: defines["REDUCE_OP"] = "ckernel::PoolType::SUM"; break;
    }
    defines["REDUCE_DIM"] = reduce_dim_str;
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
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt) {
    Tensor input = input_tensor;
    if (input.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
        input.storage_type() == tt::tt_metal::StorageType::DEVICE) {
        // Changing layout to TILE with +inf padding
        auto pad_shape = ttnn::operations::data_movement::pad_to_tile_shape(input.padded_shape());
        input = ttnn::tilize_with_val_padding(
            input,
            pad_shape,
            std::numeric_limits<float>::infinity(),
            output_mem_config,
            std::nullopt,
            true,
            sub_core_grids);
    }
    return detail::reduce(
        input,
        tt::tt_metal::ReduceOpMath::MAX,
        reduce_dim,
        scaler,
        output_mem_config,
        std::nullopt,
        compute_kernel_config,
        sub_core_grids,
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
        return reduce_min(input_tensor, reduce_dim, scaler, output_mem_config, compute_kernel_config, sub_core_grids);
    }

    auto parallelization_strategy = ttnn::prim::get_parallelization_strategy(input_tensor, reduce_dim);
    auto is_multicore_hw = parallelization_strategy == tt::tt_metal::ReduceOpParallelizationStrategy::MULTI_CORE_HW;
    float pad_value = reduce_math == tt::tt_metal::ReduceOpMath::MAX ? -std::numeric_limits<float>::infinity() : 0;

    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Expected input tensor to be on device");
    TT_FATAL(
        input_tensor.device() != nullptr,
        "input_tensor.device() == nullptr, No device found, move input_tensor to device");

    // Due to hardware bug (#38306), HiFi4 + fp32_dest_acc_en can sometime produce incorrect results on Wormhole.
    // fp32_dest_acc_en defaults to True here, so always use HiFi3 as default on Wormhole B0.
    const auto arch = input_tensor.device()->arch();
    const auto is_wormhole = arch == tt::ARCH::WORMHOLE_B0;
    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(ttnn::init_device_compute_kernel_config(
        arch,
        std::nullopt,
        is_wormhole ? tt::tt_metal::MathFidelity::HiFi3 : tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true));
    ttnn::verify_numerical_configuration(arch, compute_kernel_config);

    // Reduce only works with tile layout, so we need to tilize the input tensor if necessary
    auto padded_shape = ttnn::operations::data_movement::pad_to_tile_shape(input_tensor.padded_shape());
    auto tilized_input = ttnn::tilize_with_val_padding(
        input_tensor, padded_shape, pad_value, input_tensor.memory_config(), std::nullopt, true, sub_core_grids);

    // GMPOOL applies exp2(floor(log2(|s|))) of the scalar (only the exponent), so for
    // MAX/MIN with non-unity scalar we instead reduce with scaler=1.0 and apply the user
    // scalar after reduction via post-multiplication. See issue #40498. The flag also
    // covers reduce_min (math_op=MAX with negate=true) since high-level dispatch lowers
    // min through reduce_min before reaching here.
    const bool use_post_mul = (reduce_math == tt::tt_metal::ReduceOpMath::MAX) && (scaler != 1.0f);
    const float reduce_scaler = use_post_mul ? 1.0f : scaler;
    const float post_mul = use_post_mul ? scaler : 1.0f;

    // The single-core HW path uses REDUCE_SCALAR mode, which applies the
    // scaler twice internally (once per dimension).  The host compensates with
    // sqrt(scaler) in ReduceSingleCoreHwProgramFactory::create.
    // However, sqrt of a negative number is NaN, so negative scalers
    // must take the two-step W-then-H path where the scaler is applied once.
    if (is_multicore_hw || (reduce_dim == tt::tt_metal::ReduceOpDim::HW && reduce_scaler < 0)) {
        // Multi-core HW reduction: first reduce W, then reduce H on the result.
        // For the Sum chain's terminal fp32->bf16 stage, keep W in fp32 so only H packs to bf16.
        const auto out_final_dtype = output_dtype.value_or(input_tensor.dtype());
        const bool keep_w_fp32 = output_dtype.has_value() && out_final_dtype == tt::tt_metal::DataType::BFLOAT16 &&
                                 tilized_input.dtype() == tt::tt_metal::DataType::FLOAT32;
        const auto out_w_dtype = keep_w_fp32 ? tt::tt_metal::DataType::FLOAT32 : out_final_dtype;

        const Tensor output_tensor = ttnn::prim::reduce(
            tilized_input,
            reduce_math,
            tt::tt_metal::ReduceOpDim::W,
            1.0f,
            output_mem_config,
            out_w_dtype,
            config,
            sub_core_grids,
            negate,
            /*post_mul_scaler=*/1.0f);

        return ttnn::prim::reduce(
            output_tensor,
            reduce_math,
            tt::tt_metal::ReduceOpDim::H,
            reduce_scaler,
            output_mem_config,
            out_final_dtype,
            config,
            sub_core_grids,
            negate,
            /*post_mul_scaler=*/post_mul);
    }
    return ttnn::prim::reduce(
        tilized_input,
        reduce_math,
        reduce_dim,
        reduce_scaler,
        output_mem_config,
        output_dtype.value_or(input_tensor.dtype()),
        config,
        sub_core_grids,
        negate,
        /*post_mul_scaler=*/post_mul);
}

}  // namespace ttnn::operations::reduction::generic::detail
