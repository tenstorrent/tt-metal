// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::reduction {

template <ReduceType reduce_type>
static Tensor reduce_impl(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, ttnn::SmallVector<int>>>& dim_arg,
    const bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    bool reshape) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    auto input_shape = input_tensor_arg.get_shape();
    auto rank = input_shape.size();
    auto memory_config = memory_config_arg.value_or(input_tensor_arg.memory_config());

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

    // bail out early if unsupported reduce op is provided
    constexpr bool reduce_op_has_linear_property = reduce_type == ReduceType::Sum || reduce_type == ReduceType::Mean ||
                                                   reduce_type == ReduceType::Max || reduce_type == ReduceType::Min;
    constexpr bool reduce_op_has_nonlinear_property = reduce_type == ReduceType::Var || reduce_type == ReduceType::Std;
    if (dim.size() == rank) {
        if constexpr (!reduce_op_has_linear_property) {
            TT_THROW("Unsupported reduction operation");
        }
    } else {
        if constexpr (!reduce_op_has_linear_property && !reduce_op_has_nonlinear_property) {
            TT_THROW("Unsupported reduction operation");
        }
    }

    std::sort(dim.begin(), dim.end());

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

    auto transpose_and_compute_reduction = [&](const Tensor& input_tensor,
                                               const int dim1,
                                               const int dim2,
                                               const std::variant<int, ttnn::SmallVector<int>>& reduce_dim,
                                               const bool need_additional_reshape) -> Tensor {
        // edge case: if both the dims are same no need of transpose
        const bool is_same_dims = (dim1 == dim2) || ((dim1 + rank) % rank == (dim2 + rank) % rank);

        ttnn::SmallVector<uint32_t> new_output_shape = {
            input_tensor.get_shape()[0],
            input_tensor.get_shape()[1],
            input_tensor.get_shape()[2],
            input_tensor.get_shape()[3]};
        new_output_shape[dim1] = 1;

        Tensor output = input_tensor;
        if (!is_same_dims) {
            output = ttnn::transpose(input_tensor, dim1, dim2, memory_config);
        }
        if constexpr (reduce_type == ReduceType::Mean) {
            output = reduce_impl<ReduceType::Sum>(
                output,
                reduce_dim,
                /*keepdim=*/true,
                memory_config,
                compute_kernel_config,
                scalar,
                need_additional_reshape);
        } else {
            output = reduce_impl<reduce_type>(
                output,
                reduce_dim,
                /*keepdim=*/true,
                memory_config,
                compute_kernel_config,
                scalar,
                need_additional_reshape);
        }
        if (!is_same_dims) {
            output = ttnn::transpose(output, dim1, dim2, memory_config);
            if (reshape) {
                output = ttnn::reshape(output, ttnn::Shape{new_output_shape});
            }
        }
        return output;
    };

    auto is_w_or_h_or_wh_case = [&]() -> bool {
        return (dim.size() == 1 and dim[0] == rank - 1) || (dim.size() == 1 and dim[0] == rank - 2) ||
               (dim.size() == 2 and dim[0] == rank - 2 and dim[1] == rank - 1);
    };

    auto input_tensor = ttnn::unsqueeze_to_4D(input_tensor_arg);
    Tensor output_tensor;

    // if we get dims as vector, we recursively call this function until we reach the case where we deal with W or H or
    // WH dimensions
    if (!is_w_or_h_or_wh_case()) {
        if constexpr (!reduce_op_has_linear_property) {
            TT_THROW("Unsupported reduction operation");
        }
        output_tensor = input_tensor;
        for (int rank = input_tensor.get_legacy_shape().rank() - 1; rank >= 0; rank--) {
            if (std::find(dim.begin(), dim.end(), rank) == dim.end()) {
                continue;
            }
            output_tensor = transpose_and_compute_reduction(output_tensor, rank, -1, -1, dim.size() != rank);
        }
        if constexpr (reduce_type == ReduceType::Mean) {
            float inv_volume = 1.0f / input_tensor.get_logical_volume();
            output_tensor = ttnn::mul_sfpu(inv_volume, output_tensor, memory_config);
        }
    } else {
        // Only deal with dimension : W, H, WH
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
        } else if constexpr (reduce_type == ReduceType::Var or reduce_type == ReduceType::Std) {
            auto mean_tensor = tt::tt_metal::reduce(
                input_tensor,
                tt::tt_metal::ReduceOpMath::SUM,
                reduce_op_dim,
                1.0 / reduced_volume,
                memory_config,
                std::nullopt,
                compute_kernel_config);
            auto mean_square_tensor = tt::tt_metal::reduce(
                ttnn::pow(input_tensor, 2.0f, memory_config),
                tt::tt_metal::ReduceOpMath::SUM,
                reduce_op_dim,
                1.0 / reduced_volume,
                memory_config,
                std::nullopt,
                compute_kernel_config);
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
        output_tensor = ttnn::reshape(output_tensor, ttnn::Shape{output_shape});
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
