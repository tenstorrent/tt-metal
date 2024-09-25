// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/core/core.hpp"
namespace ttnn {
namespace operations::reduction {

template <ReduceType reduce_type>
static Tensor reduce_impl(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, std::vector<int>>>& dim_arg,
    const bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    bool reshape) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    if (not keepdim) {
        TT_THROW("keepdim=False is not supported");
    }

    auto input_shape = input_tensor_arg.get_shape();
    auto rank = input_shape.size();
    auto memory_config = memory_config_arg.value_or(input_tensor_arg.memory_config());

    std::vector<int> dim{};
    if (dim_arg.has_value()) {
        if (not std::holds_alternative<std::vector<int>>(dim_arg.value())) {
            auto dim_as_int = std::get<int>(dim_arg.value());
            dim = std::vector<int>({dim_as_int});
        } else {
            dim = std::get<std::vector<int>>(dim_arg.value());
        }
    } else {
        dim = std::vector<int>(rank);
        for (int i = 0; i < rank; i++) {
            dim[i] = i;
        }
    }

    if (dim.size() == 1) {
        if (dim[0] == rank - 3) {
            // Pad before running the op to only pay cost of formatting once
            auto input_tensor_pad_shape = AutoFormat::pad_to_tile_shape(input_tensor_arg.get_shape().with_tile_padding(), true);
            auto out_shape = input_tensor_arg.get_shape().with_tile_padding();
            out_shape[1] = 1;

            auto formatted_input_tensor = input_tensor_arg;
            float pad_value = (reduce_type == ReduceType::Max)   ? -std::numeric_limits<float>::infinity()
                              : (reduce_type == ReduceType::Min) ? std::numeric_limits<float>::infinity()
                                                                 : 0;

            if (!AutoFormat::check_input_tensor_format(input_tensor_arg, input_tensor_pad_shape)) {
                formatted_input_tensor = AutoFormat::format_input_tensor(
                    input_tensor_arg, input_tensor_arg.device(), input_tensor_pad_shape, pad_value, Layout::TILE);
            }
            Tensor output = ttnn::transpose(formatted_input_tensor, 1, -2, memory_config);
            output = reduce_impl<reduce_type>(output, 2, keepdim, memory_config, compute_kernel_config, scalar, false);
            output = ttnn::transpose(output, 1, -2, memory_config);
            return AutoFormat::format_output_tensor(output, out_shape, input_tensor_arg.device(), Layout::TILE);
        } else if (dim[0] == 0) {
            // Pad before running the op to only pay cost of formatting once
            auto input_tensor_pad_shape =
                AutoFormat::pad_to_tile_shape(input_tensor_arg.get_shape().with_tile_padding(), false, true);
            auto out_shape = input_tensor_arg.get_shape().with_tile_padding();
            out_shape[0] = 1;

            auto formatted_input_tensor = input_tensor_arg;
            if (!AutoFormat::check_input_tensor_format(input_tensor_arg, input_tensor_pad_shape)) {
                formatted_input_tensor = AutoFormat::format_input_tensor(
                    input_tensor_arg, input_tensor_arg.device(), input_tensor_pad_shape, 0.0, Layout::TILE);
            }
            Tensor output = ttnn::transpose(formatted_input_tensor, 0, -2, memory_config);
            output = reduce_impl<reduce_type>(output, 2, keepdim, memory_config, compute_kernel_config, scalar, false);
            output = ttnn::transpose(output, 0, -2, memory_config);
            return AutoFormat::format_output_tensor(output, out_shape, input_tensor_arg.device(), Layout::TILE);
        }
    }

    for (int& axis : dim) {
        if (axis < 0) {
            axis += rank;
        }
        if (axis >= rank) {
            TT_THROW("Invalid dim");
        }
    }
    std::sort(dim.begin(), dim.end());

    std::vector<uint32_t> output_shape;
    std::vector<uint32_t> padded_output_shape;
    for (int axis = 0; axis < input_shape.size(); axis++) {
        if (std::find(dim.begin(), dim.end(), axis) != dim.end()) {
            if (keepdim) {
                output_shape.push_back(1);
                padded_output_shape.push_back(axis >= rank - 2 ? ttnn::TILE_SIZE : 1);
            }
        } else {
            // Get the shape for the output tensor
            output_shape.push_back(input_shape[axis]);
            // Get the padded shape for the output tensor
            padded_output_shape.push_back(input_shape.value[axis]);
        }
    }

    auto input_tensor = ttnn::unsqueeze_to_4D(input_tensor_arg);

    Tensor output_tensor;
    if (!dim_arg.has_value()) {
        if constexpr (reduce_type == ReduceType::Sum || reduce_type == ReduceType::Max  || reduce_type == ReduceType::Min) {
            output_tensor = input_tensor;
            for (int rank = input_tensor.get_shape().with_tile_padding().rank() - 1; rank >= 0; rank--) {
                output_tensor = reduce_impl<reduce_type>(output_tensor, rank, true, memory_config, compute_kernel_config, scalar, false);
            }
        } else if constexpr (reduce_type == ReduceType::Mean) {
            output_tensor = input_tensor;
            for (int rank = input_tensor.get_shape().with_tile_padding().rank() - 1; rank >= 0; rank--) {
                output_tensor = reduce_impl<ReduceType::Sum>(output_tensor, rank, true, memory_config, compute_kernel_config, scalar, false);
            }
            float inv_volume = 1.0f/input_tensor.volume();
            return ttnn::mul_sfpu(inv_volume, output_tensor, memory_config);
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

        if constexpr (reduce_type == ReduceType::Sum) {
            output_tensor = tt::tt_metal::reduce(input_tensor, tt::tt_metal::ReduceOpMath::SUM, reduce_op_dim, scalar,
                memory_config, std::nullopt, compute_kernel_config);
        } else if constexpr (reduce_type == ReduceType::Mean) {
            output_tensor = tt::tt_metal::reduce(input_tensor, tt::tt_metal::ReduceOpMath::SUM, reduce_op_dim, 1.0 / reduced_volume,
                memory_config, std::nullopt, compute_kernel_config);
        } else if constexpr (reduce_type == ReduceType::Max) {
            output_tensor = tt::tt_metal::reduce(input_tensor, tt::tt_metal::ReduceOpMath::MAX, reduce_op_dim, scalar,
                memory_config, std::nullopt, compute_kernel_config);
        } else if constexpr (reduce_type == ReduceType::Min) {
            output_tensor = tt::tt_metal::reduce(input_tensor, tt::tt_metal::ReduceOpMath::MIN, reduce_op_dim, scalar,
                memory_config, std::nullopt, compute_kernel_config);
        } else if constexpr (reduce_type == ReduceType::Var or reduce_type == ReduceType::Std) {
            auto mean_tensor = tt::tt_metal::reduce(input_tensor, tt::tt_metal::ReduceOpMath::SUM, reduce_op_dim, 1.0 / reduced_volume,
                memory_config, std::nullopt, compute_kernel_config);
            auto mean_square_tensor = tt::tt_metal::reduce(ttnn::pow(input_tensor, 2.0f, memory_config), tt::tt_metal::ReduceOpMath::SUM, reduce_op_dim,
                1.0 / reduced_volume, memory_config, std::nullopt, compute_kernel_config);
            output_tensor = ttnn::subtract(
                mean_square_tensor, ttnn::pow(mean_tensor, 2.0f, memory_config), std::nullopt, memory_config);
            if constexpr (reduce_type == ReduceType::Std) {
                output_tensor = ttnn::sqrt(output_tensor, memory_config);
            }
        } else {
            TT_THROW("Unsupported reduction operation");
        }
    }

    if (reshape) {
        output_tensor = ttnn::reshape(output_tensor, ttnn::Shape{ttnn::Shape{output_shape, padded_output_shape}});
    }

    return output_tensor;
}

template <ReduceType reduce_type>
Tensor Reduce<reduce_type>::invoke(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, std::vector<int>>>& dim_arg,
    const bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar) {
    return reduce_impl<reduce_type>(
        input_tensor_arg, dim_arg, keepdim, memory_config_arg, compute_kernel_config, scalar, true);
}

template class Reduce<ReduceType::Sum>;
template class Reduce<ReduceType::Mean>;
template class Reduce<ReduceType::Max>;
template class Reduce<ReduceType::Min>;
template class Reduce<ReduceType::Std>;
template class Reduce<ReduceType::Var>;
}  // namespace operations::reduction
}  // namespace ttnn
