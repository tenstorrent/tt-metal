// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::reduction {

ttnn::SmallVector<int> generate_reduce_dim(
    const Tensor& input_tensor_arg, const std::optional<std::variant<int, ttnn::SmallVector<int>>>& dim_arg) {
    auto input_shape = input_tensor_arg.get_logical_shape();
    auto rank = input_shape.size();
    ttnn::SmallVector<int> dim{};
    if (dim_arg.has_value()) {
        if (not std::holds_alternative<ttnn::SmallVector<int>>(dim_arg.value())) {
            auto dim_as_int = std::get<int>(dim_arg.value());
            dim = ttnn::SmallVector<int>({dim_as_int});
        } else {
            dim = std::get<ttnn::SmallVector<int>>(dim_arg.value());
        }
    }
    if (dim.empty()) {
        dim = ttnn::SmallVector<int>(rank);
        for (int i = 0; i < rank; i++) {
            dim[i] = i;
        }
    }

    for (int i = 0; i < dim.size(); i++) {
        if (dim[i] < 0) {
            dim[i] += rank;
        }
        int dim_i = dim[i];
        TT_FATAL(
            dim_i >= 0 && dim_i < rank,
            "Unsupported dim {} at index {}. After possible adjustment, needs to be at least 0 and less than rank {}",
            dim_i,
            i,
            rank);
    }

    std::sort(dim.begin(), dim.end());
    return dim;
}

float get_pad_value(ReduceType reduce_type) {
    return reduce_type == ReduceType::Max
               ? -std::numeric_limits<float>::infinity()
               : (reduce_type == ReduceType::Min ? std::numeric_limits<float>::infinity() : 0);
}

template <ReduceType reduce_type>
static Tensor reduce_impl(
    const Tensor& input_tensor_arg,
    const ttnn::SmallVector<int>& dim,
    const bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    auto input_shape = input_tensor_arg.get_logical_shape();
    auto rank = input_shape.size();
    auto memory_config = memory_config_arg.value_or(input_tensor_arg.memory_config());

    ttnn::SmallVector<uint32_t> output_shape;
    for (int axis = 0; axis < input_shape.size(); axis++) {
        if (std::find(dim.begin(), dim.end(), axis) != dim.end()) {
            if (keepdim) {
                output_shape.push_back(1);
            }
        } else {
            // Get the shape for the output tensor
            output_shape.push_back(input_shape[axis]);
        }
    }

    Tensor output_tensor;
    float pad_value = get_pad_value(reduce_type);
    bool single_reduce_op = (dim.size() == 1 && (dim[0] == rank - 1 || dim[0] == rank - 2)) ||
                            (dim.size() == 2 && dim[1] == rank - 1 && dim[0] == rank - 2);
    if (!single_reduce_op) {
        auto reduce_nd_loop = [&](const bool use_reduce_type) -> Tensor {
            Tensor output_tensor = input_tensor_arg;
            for (int i_dim = rank - 1; i_dim >= 0; i_dim--) {
                bool found = std::find(dim.begin(), dim.end(), i_dim) != dim.end();
                if (found) {
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
                            scalar);
                    } else {
                        output_tensor = reduce_impl<ReduceType::Sum>(
                            output_tensor,
                            {reduce_dim},
                            /*keepdim=*/true,
                            memory_config,
                            compute_kernel_config,
                            scalar);
                    }
                    if (transpose) {
                        output_tensor = ttnn::transpose(output_tensor, i_dim, -2, memory_config, pad_value);
                    }
                }
            }
            return output_tensor;
        };
        constexpr bool linear_type =
            reduce_type == ReduceType::Sum || reduce_type == ReduceType::Max || reduce_type == ReduceType::Min;
        if (dim.size() == 1 || linear_type) {
            output_tensor = reduce_nd_loop(/*use_reduce_type=*/true);
        } else if constexpr (reduce_type == ReduceType::Mean) {
            output_tensor = reduce_nd_loop(
                /*use_reduce_type=*/false);
            float inv_volume = 1.0f / input_tensor_arg.get_logical_volume();
            output_tensor = ttnn::mul_sfpu(inv_volume, output_tensor, memory_config);
        } else {
            TT_THROW("Unsupported reduction operation");
        }
    } else {
        tt::tt_metal::ReduceOpDim reduce_op_dim;
        if (dim.size() == 1 and dim[0] == rank - 1) {
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

        if constexpr (reduce_type == ReduceType::Sum) {
            output_tensor = tt::tt_metal::reduce(
                input_tensor,
                tt::tt_metal::ReduceOpMath::SUM,
                reduce_op_dim,
                scalar,
                memory_config,
                std::nullopt,
                compute_kernel_config);
        } else if constexpr (reduce_type == ReduceType::Mean) {
            output_tensor = tt::tt_metal::reduce(
                input_tensor,
                tt::tt_metal::ReduceOpMath::SUM,
                reduce_op_dim,
                1.0 / reduced_volume,
                memory_config,
                std::nullopt,
                compute_kernel_config);
        } else if constexpr (reduce_type == ReduceType::Max) {
            output_tensor = tt::tt_metal::reduce(
                input_tensor,
                tt::tt_metal::ReduceOpMath::MAX,
                reduce_op_dim,
                scalar,
                memory_config,
                std::nullopt,
                compute_kernel_config);
        } else if constexpr (reduce_type == ReduceType::Min) {
            output_tensor = tt::tt_metal::reduce(
                input_tensor,
                tt::tt_metal::ReduceOpMath::MIN,
                reduce_op_dim,
                scalar,
                memory_config,
                std::nullopt,
                compute_kernel_config);
        } else {
            TT_THROW("Unsupported reduction operation");
        }
    }
    output_tensor = ttnn::reshape(output_tensor, ttnn::Shape{output_shape});
    return output_tensor;
}

template <ReduceType reduce_type>
static Tensor std_var_impl(
    const Tensor& input_tensor_arg,
    const ttnn::SmallVector<int>& dim,
    const bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    auto input_shape = input_tensor_arg.get_logical_shape();
    auto rank = input_shape.size();
    auto memory_config = memory_config_arg.value_or(input_tensor_arg.memory_config());

    int reduced_volume = 1;
    for (int axis : dim) {
        reduced_volume *= input_shape[axis];
    }

    auto mean_tensor = reduce_impl<ReduceType::Sum>(
        input_tensor_arg, dim, keepdim, memory_config_arg, compute_kernel_config, 1.0 / reduced_volume);
    auto mean_square_tensor = reduce_impl<ReduceType::Sum>(
        ttnn::pow(input_tensor_arg, 2.0f, memory_config),
        dim,
        keepdim,
        memory_config_arg,
        compute_kernel_config,
        1.0 / reduced_volume);
    Tensor output_tensor =
        ttnn::subtract(mean_square_tensor, ttnn::pow(mean_tensor, 2.0f, memory_config), std::nullopt, memory_config);
    if constexpr (reduce_type == ReduceType::Std) {
        output_tensor = ttnn::sqrt(output_tensor, memory_config);
    }
    return output_tensor;
}

template <ReduceType reduce_type>
Tensor Reduce<reduce_type>::invoke(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, ttnn::SmallVector<int>>>& dim_arg,
    const bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar) {
    ttnn::SmallVector<int> dim = generate_reduce_dim(input_tensor_arg, dim_arg);
    float pad_value = get_pad_value(reduce_type);
    bool is_tiled = input_tensor_arg.get_layout() == TILE_LAYOUT;
    auto input_tensor = is_tiled ? ttnn::fill_implicit_tile_padding(input_tensor_arg, pad_value) : input_tensor_arg;
    if constexpr (reduce_type == ReduceType::Std || reduce_type == ReduceType::Var) {
        return std_var_impl<reduce_type>(input_tensor, dim, keepdim, memory_config_arg, compute_kernel_config);
    }
    return reduce_impl<reduce_type>(input_tensor, dim, keepdim, memory_config_arg, compute_kernel_config, scalar);
}

Tensor pool_sum(
    const Tensor& input_tensor_arg,
    int dim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar) {
    return reduce_impl<ReduceType::Sum>(
        input_tensor_arg,
        ttnn::SmallVector<int>({dim}),
        /*keepdim=*/true,
        memory_config_arg,
        compute_kernel_config,
        scalar);
}

template class Reduce<ReduceType::Sum>;
template class Reduce<ReduceType::Mean>;
template class Reduce<ReduceType::Max>;
template class Reduce<ReduceType::Min>;
template class Reduce<ReduceType::Std>;
template class Reduce<ReduceType::Var>;
}  // namespace operations::reduction
}  // namespace ttnn
