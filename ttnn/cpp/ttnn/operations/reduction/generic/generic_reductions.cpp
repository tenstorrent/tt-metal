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

ttnn::SmallVector<int> generate_reduce_dim(
    const Tensor& input_tensor_arg, const std::optional<std::variant<int, ttnn::SmallVector<int>>>& dim_arg) {
    auto input_shape = input_tensor_arg.get_shape();
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

Tensor reshape_nd_to_4d_for_reduction(const Tensor& input_tensor_arg, const bool transpose, int& dim1, int dim2 = -1) {
    auto input_shape = input_tensor_arg.get_shape();
    auto rank = input_shape.size();
    auto input_tensor = input_tensor_arg;
    if (dim1 < 0) {
        dim1 += rank;
    }
    if (dim2 < 0) {
        dim2 += rank;
    }
    if (rank <= 4) {
        return input_tensor;
    }

    auto new_shape = ttnn::SmallVector<uint32_t>(4, 1);

    // if we need transpose, we need to reshape to 4D based on specific dim1 and dim2 values
    if (transpose) {
        int insert_idx = 0;
        for (int i = 0; i < rank; i++) {
            if (i == dim1 || i == dim2) {
                insert_idx++;
                new_shape[insert_idx] = input_shape[i];
                if (i == dim1) {
                    dim1 = insert_idx;
                } else {
                    dim2 = insert_idx;
                }
                insert_idx++;
                continue;
            }
            new_shape[insert_idx] *= input_shape[i];
        }
    } else {
        // transpose is not needed but we still need to reshape to 4D for reduction operation
        // So we squash all front dims into one dim until we reach 4-D shape
        int insert_idx = 3;
        for (int i = rank - 1; i >= 0; i--) {
            if (insert_idx == 0) {
                new_shape[insert_idx] *= input_shape[i];
                continue;
            }
            new_shape[insert_idx] = input_shape[i];
            insert_idx--;
        }

        int offset = rank - 4;
        dim1 -= offset;
    }
    return ttnn::reshape(input_tensor, ttnn::SimpleShape{new_shape});
    ;
}

Tensor reshape_4d_to_nd_after_reduction(
    const Tensor& input_tensor_arg, const ttnn::Shape& pre_reduction_shape, int dim) {
    ttnn::SmallVector<uint32_t> post_reduction_shape(pre_reduction_shape.size());
    for (int i = 0; i < pre_reduction_shape.size(); i++) {
        if (i == dim) {
            post_reduction_shape[dim] = 1;
        } else {
            post_reduction_shape[i] = pre_reduction_shape[i];
        }
    }
    return ttnn::reshape(input_tensor_arg, ttnn::SimpleShape{post_reduction_shape});
}

template <ReduceType reduce_type>
static Tensor reduce_impl(
    const Tensor& input_tensor_arg,
    const ttnn::SmallVector<int>& dim,
    const bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    bool reshape) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    auto input_shape = input_tensor_arg.get_shape();
    auto rank = input_shape.size();
    auto memory_config = memory_config_arg.value_or(input_tensor_arg.memory_config());
    const bool is_rank_le_4d = rank <= 4;

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

    auto input_tensor = is_rank_le_4d ? ttnn::unsqueeze_to_4D(input_tensor_arg) : input_tensor_arg;

    Tensor output_tensor;
    bool single_reduce_op = (dim.size() == 1 && (dim[0] == rank - 1 || dim[0] == rank - 2)) ||
                            (dim.size() == 2 && dim[0] == rank - 1 && dim[0] == rank - 2);
    if (!single_reduce_op) {
        auto reduce_4d_loop = [&](const bool use_reduce_type) -> Tensor {
            Tensor output_tensor = input_tensor;
            int offset = is_rank_le_4d ? 4 - rank : 0;
            for (int i_dim = rank - 1; i_dim >= 0; i_dim--) {
                if (std::find(dim.begin(), dim.end(), i_dim) == dim.end()) {
                    continue;
                }
                bool transpose = i_dim < rank - 2;
                int adjusted_dim = offset + i_dim;

                auto pre_reduction_shape = output_tensor.get_shape();
                if (!is_rank_le_4d) {
                    output_tensor = reshape_nd_to_4d_for_reduction(output_tensor, transpose, adjusted_dim);
                }

                int reduce_dim = adjusted_dim;

                if (transpose) {
                    output_tensor = ttnn::transpose(output_tensor, adjusted_dim, -1, memory_config);
                    reduce_dim = output_tensor.get_shape().size() - 1;
                }
                if (use_reduce_type) {
                    output_tensor = reduce_impl<reduce_type>(
                        output_tensor,
                        {reduce_dim},
                        /*keepdim=*/true,
                        memory_config,
                        compute_kernel_config,
                        scalar,
                        /*reshape=*/false);
                } else {
                    output_tensor = reduce_impl<ReduceType::Sum>(
                        output_tensor,
                        {reduce_dim},
                        /*keepdim=*/true,
                        memory_config,
                        compute_kernel_config,
                        scalar,
                        /*reshape=*/false);
                }
                if (transpose) {
                    output_tensor = ttnn::transpose(output_tensor, adjusted_dim, -1, memory_config);
                }
                if (!is_rank_le_4d) {
                    output_tensor = reshape_4d_to_nd_after_reduction(output_tensor, pre_reduction_shape, i_dim);
                }
            }
            return output_tensor;
        };
        constexpr bool linear_type =
            reduce_type == ReduceType::Sum || reduce_type == ReduceType::Max || reduce_type == ReduceType::Min;
        if (dim.size() == 1 || linear_type) {
            output_tensor = reduce_4d_loop(/*use_reduce_type=*/true);
        } else if constexpr (reduce_type == ReduceType::Mean) {
            output_tensor = reduce_4d_loop(
                /*use_reduce_type=*/false);
            float inv_volume = 1.0f / input_tensor.get_logical_volume();
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
        output_tensor = ttnn::reshape(output_tensor, ttnn::SimpleShape{output_shape});
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
    return reduce_impl<reduce_type>(
        input_tensor_arg, dim, keepdim, memory_config_arg, compute_kernel_config, scalar, true);
}

template class Reduce<ReduceType::Sum>;
template class Reduce<ReduceType::Mean>;
template class Reduce<ReduceType::Max>;
template class Reduce<ReduceType::Min>;
template class Reduce<ReduceType::Std>;
template class Reduce<ReduceType::Var>;
}  // namespace operations::reduction
}  // namespace ttnn
