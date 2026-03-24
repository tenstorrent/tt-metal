// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/reduction/reduction_common/reduction_common.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::reduction {

// Does not support ReduceType::Prod (handled separately in prod.cpp).
template <reduction_common::ReduceType reduce_type>
Tensor reduce(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttnn::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

// input_shape has original shape while output_shape has reduction applied and last 2 dims padded.
// Need to get slice parameters based on the minimum of the two shapes.
std::tuple<ttnn::SmallVector<int>, ttnn::SmallVector<int>, ttnn::SmallVector<int>> get_slice_parameters(
    Shape input_shape, Shape output_shape) {
    ttnn::SmallVector<int> start{}, end{}, step{};
    TT_FATAL(
        input_shape.size() == output_shape.size(),
        "Input shape size {} and output shape size {} need to be equal.",
        input_shape.size(),
        output_shape.size());
    for (int i = 0; i < input_shape.size(); i++) {
        start.push_back(0);
        end.push_back(std::min(input_shape[i], output_shape[i]));
        step.push_back(1);
    }
    return {start, end, step};
}

std::pair<ttnn::SmallVector<int>, ttnn::SmallVector<int>> split_height_width_dims(
    const ttnn::SmallVector<int>& dim, const Tensor& input_tensor_arg) {
    ttnn::SmallVector<int> non_height_width_dims{}, height_width_dims{};
    const auto& input_shape = input_tensor_arg.logical_shape();
    int rank = input_shape.size();
    for (int d : dim) {
        if (d >= (rank - 2)) {
            height_width_dims.push_back(d);
        } else {
            non_height_width_dims.push_back(d);
        }
    }
    return {non_height_width_dims, height_width_dims};
}

float get_pad_value(reduction_common::ReduceType reduce_type) {
    // Prod reduction is handled separately in prod.cpp.
    TT_FATAL(reduce_type != reduction_common::ReduceType::Prod, "Prod reduction is not supported");
    return reduce_type == reduction_common::ReduceType::Max
               ? -std::numeric_limits<float>::infinity()
               : (reduce_type == reduction_common::ReduceType::Min ? std::numeric_limits<float>::infinity() : 0);
}

Tensor adjust_shape(
    const Tensor& tensor,
    const Shape& input_shape,
    bool keepdim,
    const ttnn::SmallVector<int>& height_width_dims,
    const ttnn::SmallVector<int>& non_height_width_dims) {
    ttnn::SmallVector<uint32_t> output_shape;
    for (int axis = 0; axis < input_shape.size(); axis++) {
        bool in_height_width_dims =
            std::find(height_width_dims.begin(), height_width_dims.end(), axis) != height_width_dims.end();
        bool in_non_height_width_dims =
            std::find(non_height_width_dims.begin(), non_height_width_dims.end(), axis) != non_height_width_dims.end();
        if (in_height_width_dims || in_non_height_width_dims) {
            if (keepdim) {
                output_shape.push_back(1);
            }
        } else {
            // Get the shape for the output tensor
            output_shape.push_back(input_shape[axis]);
        }
    }
    auto output_tensor = ttnn::reshape(tensor, ttnn::Shape{output_shape});
    return output_tensor;
}

template <reduction_common::ReduceType reduce_type>
static Tensor reduce_impl(
    const Tensor& input_tensor_arg,
    const ttnn::SmallVector<int>& dim,
    const bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    const ttnn::SmallVector<int>& non_height_width_dims,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    auto input_shape = input_tensor_arg.logical_shape();
    auto rank = input_shape.rank();
    auto memory_config = memory_config_arg.value_or(input_tensor_arg.memory_config());

    Tensor output_tensor;

    // If the input is a rank 0 tensor (scalar), return a copy of it
    if (rank == 0) {
        // Return a copy of the input tensor.
        return ttnn::clone(input_tensor_arg, /*dtype=*/std::nullopt, memory_config, compute_kernel_config);
    }

    // If the input is a zero volume tensor, return output with shape adjusted for keepdim
    if (input_tensor_arg.logical_volume() == 0) {
        return reduction_common::zero_volume_reduce<reduce_type>(input_tensor_arg, dim, keepdim, memory_config);
    }

    float pad_value = get_pad_value(reduce_type);
    bool single_reduce_op = (dim.empty()) || (dim.size() == 1 && (dim[0] == rank - 1 || dim[0] == rank - 2)) ||
                            (dim.size() == 2 && dim[1] == rank - 1 && dim[0] == rank - 2);
    if (!single_reduce_op) {
        auto reduce_nd_loop = [&](const bool use_reduce_type, float scalar) -> Tensor {
            Tensor output_tensor = input_tensor_arg;
            bool first = true;
            for (int i_dim = rank - 1; i_dim >= 0; i_dim--) {
                bool found = std::find(dim.begin(), dim.end(), i_dim) != dim.end();
                if (found) {
                    // Only apply the scalar once when reducing dim-by-dim,
                    // otherwise the result will be scaled multiple times.
                    float effective_scalar = first ? scalar : 1.0;
                    first = false;

                    bool transpose = i_dim < rank - 2;
                    int reduce_dim = i_dim;
                    if (transpose) {
                        output_tensor = ttnn::transpose(output_tensor, i_dim, -2, memory_config, pad_value);
                        reduce_dim = rank - 2;
                    }
                    if (use_reduce_type) {
                        output_tensor = reduce_impl<reduce_type>(
                            output_tensor,
                            {reduce_dim},
                            /*keepdim=*/true,
                            memory_config,
                            compute_kernel_config,
                            effective_scalar,
                            non_height_width_dims,
                            sub_core_grids);
                    } else {
                        output_tensor = reduce_impl<reduction_common::ReduceType::Sum>(
                            output_tensor,
                            {reduce_dim},
                            /*keepdim=*/true,
                            memory_config,
                            compute_kernel_config,
                            effective_scalar,
                            non_height_width_dims,
                            sub_core_grids);
                    }
                    if (transpose) {
                        output_tensor = ttnn::transpose(output_tensor, i_dim, -2, memory_config, pad_value);
                    }
                }
            }
            return output_tensor;
        };
        constexpr bool linear_type = reduce_type == reduction_common::ReduceType::Sum ||
                                     reduce_type == reduction_common::ReduceType::Max ||
                                     reduce_type == reduction_common::ReduceType::Min;
        if (dim.size() == 1 || linear_type) {
            output_tensor = reduce_nd_loop(/*use_reduce_type=*/true, scalar);
        } else if constexpr (reduce_type == reduction_common::ReduceType::Mean) {
            int reduced_volume = 1;
            for (int axis : dim) {
                reduced_volume *= input_shape[axis];
            }
            output_tensor = reduce_nd_loop(
                /*use_reduce_type=*/false, scalar / reduced_volume);
        } else {
            TT_THROW("Unsupported reduction operation");
        }
    } else {
        tt::tt_metal::ReduceOpDim reduce_op_dim;
        if ((dim.empty()) || (dim.size() == 1 and dim[0] == rank - 1)) {
            reduce_op_dim = tt::tt_metal::ReduceOpDim::W;
        } else if (dim.size() == 1 and dim[0] == rank - 2) {
            reduce_op_dim = tt::tt_metal::ReduceOpDim::H;
        } else if (dim.size() == 2 and dim[0] == rank - 2 and dim[1] == rank - 1) {
            reduce_op_dim = tt::tt_metal::ReduceOpDim::HW;
        } else {
            TT_THROW("Unsupported dim");
        }

        int reduced_volume = 1;
        for (int axis : dim) {
            reduced_volume *= input_shape[axis];
        }

        auto input_tensor = (rank > 4)   ? data_movement::squeeze_from_ND_to_4D(input_tensor_arg)
                            : (rank < 4) ? ttnn::unsqueeze_to_4D(input_tensor_arg)
                                         : input_tensor_arg;

        if constexpr (reduce_type == reduction_common::ReduceType::Sum) {
            output_tensor = ttnn::operations::reduction::generic::detail::reduce(
                input_tensor,
                tt::tt_metal::ReduceOpMath::SUM,
                reduce_op_dim,
                scalar,
                memory_config,
                std::nullopt,
                compute_kernel_config,
                sub_core_grids);
        } else if constexpr (reduce_type == reduction_common::ReduceType::Mean) {
            output_tensor = ttnn::operations::reduction::generic::detail::reduce(
                input_tensor,
                tt::tt_metal::ReduceOpMath::SUM,
                reduce_op_dim,
                scalar / reduced_volume,
                memory_config,
                std::nullopt,
                compute_kernel_config,
                sub_core_grids);
        } else if constexpr (reduce_type == reduction_common::ReduceType::Max) {
            output_tensor = ttnn::operations::reduction::generic::detail::reduce(
                input_tensor,
                tt::tt_metal::ReduceOpMath::MAX,
                reduce_op_dim,
                scalar,
                memory_config,
                std::nullopt,
                compute_kernel_config,
                sub_core_grids);
        } else if constexpr (reduce_type == reduction_common::ReduceType::Min) {
            output_tensor = ttnn::operations::reduction::generic::detail::reduce(
                input_tensor,
                tt::tt_metal::ReduceOpMath::MIN,
                reduce_op_dim,
                scalar,
                memory_config,
                std::nullopt,
                compute_kernel_config,
                sub_core_grids);
        } else {
            TT_THROW("Unsupported reduction operation");
        }
    }
    return adjust_shape(output_tensor, input_shape, keepdim, dim, non_height_width_dims);
}

template <reduction_common::ReduceType reduce_type>
static Tensor std_var_impl(
    const Tensor& input_tensor_arg,
    const ttnn::SmallVector<int>& dim,
    const bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    const ttnn::SmallVector<int>& non_height_width_dims,
    bool correction,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    auto input_shape = input_tensor_arg.logical_shape();
    auto rank = input_shape.size();
    auto memory_config = memory_config_arg.value_or(input_tensor_arg.memory_config());

    // If the input tensor is a rank 0 tensor, return NaN
    if (rank == 0) {
        // Create an output tensor with same shape and attributes as input tensor
        auto output_tensor = ttnn::full_like(
            input_tensor_arg,
            std::numeric_limits<float>::quiet_NaN(),
            /*dtype=*/std::nullopt,
            /*layout=*/std::nullopt,
            /*device=*/std::nullopt,
            memory_config);
        return adjust_shape(output_tensor, input_shape, keepdim, dim, non_height_width_dims);
    }

    // If the input is a zero volume tensor, return output with shape adjusted for keepdim
    if (input_tensor_arg.logical_volume() == 0) {
        return reduction_common::zero_volume_reduce<reduce_type>(input_tensor_arg, dim, keepdim, memory_config);
    }

    int reduced_volume = 1;
    for (int axis : dim) {
        reduced_volume *= input_shape[axis];
    }

    // Bessel's correction (i.e. divisor of N-1)
    if (correction) {
        reduced_volume -= 1;
    }
    TT_FATAL(reduced_volume > 0, "Reduction is performed on too few elements, yielding divisor of {}", reduced_volume);

    scalar /= reduced_volume;

    auto mean_tensor = reduce_impl<reduction_common::ReduceType::Sum>(
        input_tensor_arg,
        dim,
        keepdim,
        memory_config_arg,
        compute_kernel_config,
        scalar,
        non_height_width_dims,
        sub_core_grids);

    auto mean_square_tensor = reduce_impl<reduction_common::ReduceType::Sum>(
        ttnn::pow(input_tensor_arg, 2.0f, memory_config),
        dim,
        keepdim,
        memory_config_arg,
        compute_kernel_config,
        scalar,
        non_height_width_dims,
        sub_core_grids);
    Tensor output_tensor =
        ttnn::subtract(mean_square_tensor, ttnn::pow(mean_tensor, 2.0f, memory_config), std::nullopt, memory_config);
    if constexpr (reduce_type == reduction_common::ReduceType::Std) {
        output_tensor = ttnn::sqrt(output_tensor, false, memory_config);
    }
    return output_tensor;
}

template <reduction_common::ReduceType reduce_type>
bool call_fast_nc(DataType dtype) {
    if constexpr (reduce_type != reduction_common::ReduceType::Sum) {
        return false;
    }
    return dtype == DataType::BFLOAT16 || dtype == DataType::BFLOAT8_B;
}

template <reduction_common::ReduceType reduce_type>
constexpr bool supports_native_reduce_padding_for_type() {
    return reduce_type == reduction_common::ReduceType::Sum || reduce_type == reduction_common::ReduceType::Mean ||
           reduce_type == reduction_common::ReduceType::Max;
}

Tensor non_height_width_reduce(
    const ttnn::Tensor& input_tensor,
    ttnn::SmallVector<int> dims,
    const std::optional<MemoryConfig>& memory_config_arg,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
    const auto& input_shape = input_tensor.logical_shape();
    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(ttnn::init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        std::nullopt,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true));
    Tensor output_tensor = ttnn::experimental::reduction::fast_reduce_nc(
        input_tensor, dims, /*output=*/std::nullopt, memory_config, config);
    auto [start, end, step] = get_slice_parameters(input_shape, output_tensor.logical_shape());
    output_tensor = ttnn::slice(output_tensor, start, end, step);
    return output_tensor;
}

// Does not support ReduceType::Prod (handled separately in prod.cpp).
template <reduction_common::ReduceType reduce_type>
Tensor reduce(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttnn::SmallVector<int>>>& dim_arg,
    const bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    bool correction,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    ttnn::SmallVector<int> dim = reduction_common::generate_reduce_dim(input_tensor_arg, dim_arg);
    float pad_value = get_pad_value(reduce_type);
    bool is_tiled = input_tensor_arg.layout() == TILE_LAYOUT;
    ttnn::SmallVector<int> non_height_width_dims{}, height_width_dims{};
    if (call_fast_nc<reduce_type>(input_tensor_arg.dtype())) {
        auto dims = split_height_width_dims(dim, input_tensor_arg);
        non_height_width_dims = dims.first;
        height_width_dims = dims.second;
    }

    const bool uses_fast_nc_non_hw_dims =
        call_fast_nc<reduce_type>(input_tensor_arg.dtype()) && !non_height_width_dims.empty();
    const bool use_native_padding =
        is_tiled && supports_native_reduce_padding_for_type<reduce_type>() &&
        ttnn::operations::reduction::supports_native_reduce_padding(input_tensor_arg.dtype(), /*negate=*/false) &&
        !uses_fast_nc_non_hw_dims;
    auto input_tensor = use_native_padding ? input_tensor_arg
                                           : (is_tiled ? ttnn::fill_implicit_tile_padding(input_tensor_arg, pad_value)
                                                       : input_tensor_arg);

    // TODO: generalize to support all types, parameters, and formats. Issue #18566
    if (call_fast_nc<reduce_type>(input_tensor.dtype())) {

        if (!non_height_width_dims.empty()) {
            auto rank = input_tensor_arg.logical_shape().rank();
            auto memory_config = memory_config_arg.value_or(input_tensor_arg.memory_config());

            // If the input is a rank 0 tensor (scalar), return a copy of it
            if (rank == 0) {
                // Return a copy of the input tensor.
                return ttnn::clone(input_tensor_arg, /*dtype=*/std::nullopt, memory_config, compute_kernel_config);
            }

            // If the input is a zero volume tensor, return output with shape adjusted for keepdim
            if (input_tensor_arg.logical_volume() == 0) {
                return reduction_common::zero_volume_reduce<reduce_type>(input_tensor_arg, dim, keepdim, memory_config);
            }

            input_tensor =
                non_height_width_reduce(input_tensor, non_height_width_dims, memory_config_arg, compute_kernel_config);

            if (height_width_dims.empty()) {
                return adjust_shape(
                    input_tensor, input_tensor_arg.logical_shape(), keepdim, height_width_dims, non_height_width_dims);
            }
            dim = height_width_dims;
        }
    }
    if constexpr (
        reduce_type == reduction_common::ReduceType::Std || reduce_type == reduction_common::ReduceType::Var) {
        return std_var_impl<reduce_type>(
            input_tensor,
            dim,
            keepdim,
            memory_config_arg,
            compute_kernel_config,
            scalar,
            non_height_width_dims,
            correction,
            sub_core_grids);
    }
    return reduce_impl<reduce_type>(
        input_tensor,
        dim,
        keepdim,
        memory_config_arg,
        compute_kernel_config,
        scalar,
        non_height_width_dims,
        sub_core_grids);
}

Tensor pool_sum(
    const Tensor& input_tensor_arg,
    int dim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar) {
    return reduce_impl<reduction_common::ReduceType::Sum>(
        input_tensor_arg,
        ttnn::SmallVector<int>({dim}),
        /*keepdim=*/true,
        memory_config_arg,
        compute_kernel_config,
        scalar,
        /*non_height_width_dims=*/{},
        /*sub_core_grids=*/std::nullopt);
}

}  // namespace ttnn::operations::reduction

namespace ttnn {

Tensor sum(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, SmallVector<int>>>& dim_arg,
    bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    bool correction,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::reduction::reduce<reduction_common::ReduceType::Sum>(
        input_tensor_arg,
        dim_arg,
        keepdim,
        memory_config_arg,
        compute_kernel_config,
        scalar,
        correction,
        sub_core_grids);
}

Tensor mean(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, SmallVector<int>>>& dim_arg,
    bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    bool correction,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::reduction::reduce<reduction_common::ReduceType::Mean>(
        input_tensor_arg,
        dim_arg,
        keepdim,
        memory_config_arg,
        compute_kernel_config,
        scalar,
        correction,
        sub_core_grids);
}

Tensor max(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, SmallVector<int>>>& dim_arg,
    bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    bool correction,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::reduction::reduce<reduction_common::ReduceType::Max>(
        input_tensor_arg,
        dim_arg,
        keepdim,
        memory_config_arg,
        compute_kernel_config,
        scalar,
        correction,
        sub_core_grids);
}

Tensor min(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, SmallVector<int>>>& dim_arg,
    bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    bool correction,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::reduction::reduce<reduction_common::ReduceType::Min>(
        input_tensor_arg,
        dim_arg,
        keepdim,
        memory_config_arg,
        compute_kernel_config,
        scalar,
        correction,
        sub_core_grids);
}

Tensor std(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, SmallVector<int>>>& dim_arg,
    bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    bool correction,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::reduction::reduce<reduction_common::ReduceType::Std>(
        input_tensor_arg,
        dim_arg,
        keepdim,
        memory_config_arg,
        compute_kernel_config,
        scalar,
        correction,
        sub_core_grids);
}

Tensor var(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, SmallVector<int>>>& dim_arg,
    bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    bool correction,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::reduction::reduce<reduction_common::ReduceType::Var>(
        input_tensor_arg,
        dim_arg,
        keepdim,
        memory_config_arg,
        compute_kernel_config,
        scalar,
        correction,
        sub_core_grids);
}

}  // namespace ttnn
