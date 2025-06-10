// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::reduction {

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
    auto input_shape = input_tensor_arg.logical_shape();
    int rank = input_shape.size();
    for (int i = 0; i < dim.size(); i++) {
        if (dim[i] >= (rank - 2)) {
            height_width_dims.push_back(dim[i]);
        } else {
            non_height_width_dims.push_back(dim[i]);
        }
    }
    return {non_height_width_dims, height_width_dims};
}

ttnn::SmallVector<int> generate_reduce_dim(
    const Tensor& input_tensor_arg, const std::optional<std::variant<int, ttnn::SmallVector<int>>>& dim_arg) {
    auto input_shape = input_tensor_arg.logical_shape();
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

/* Creates appropriate output tensor for a given zero volume input tensor.
   The output tensor has the same shape as the input tensor, except that the dimensions
   specified in dim are reduced to 1.
   The output tensor is filled with NaN/0/inf based on the reduce_type.
*/
template <ReduceType reduce_type>
static Tensor zero_volume_reduce(
    const Tensor& input_tensor,
    const ttnn::SmallVector<int>& dim,
    const bool keepdim,
    const MemoryConfig& memory_config) {
    auto input_shape = input_tensor.logical_shape();

    // min/max is unsupported when reduction dim is zero
    if constexpr (reduce_type == ReduceType::Max || reduce_type == ReduceType::Min) {
        // Check the shape of the reduction dims
        for (auto red_dim : dim) {
            if (input_shape[red_dim] == 0) {
                TT_THROW("Expected reduction dim {} to have non-zero size", red_dim);
            }
        }
    }

    ttnn::SmallVector<uint32_t> output_shape;

    // Iterate over the input shape and adjust the output shape for keepdim
    for (int i = 0; i < input_shape.size(); i++) {
        // If this is in the reduction dims, keep it only if keepdim is true
        bool is_reduction_dim = std::find(dim.begin(), dim.end(), i) != dim.end();

        if (is_reduction_dim && keepdim) {
            output_shape.push_back(1);
        } else if (!is_reduction_dim) {
            output_shape.push_back(input_shape[i]);
        }
    }

    constexpr float fill_value = (reduce_type == ReduceType::Sum) ? 0 : NAN;

    return ttnn::full(
        ttnn::Shape(output_shape),
        fill_value,
        input_tensor.dtype(),
        input_tensor.layout(),
        *input_tensor.mesh_device(),
        memory_config);
}

template <ReduceType reduce_type>
static Tensor reduce_impl(
    const Tensor& input_tensor_arg,
    const ttnn::SmallVector<int>& dim,
    const bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    const ttnn::SmallVector<int>& non_height_width_dims) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    auto input_shape = input_tensor_arg.logical_shape();
    auto rank = input_shape.size();
    auto memory_config = memory_config_arg.value_or(input_tensor_arg.memory_config());

    Tensor output_tensor;

    // If the input is a rank 0 tensor, return a copy of it, adjusted for keepdim
    if (rank == 0) {
        // Create an output tensor with same shape and attributes as input tensor
        output_tensor = ttnn::clone(input_tensor_arg, /*dtype=*/std::nullopt, memory_config, compute_kernel_config);
        return adjust_shape(output_tensor, input_shape, keepdim, dim, non_height_width_dims);
    }

    // If the input is a zero volume tensor, return output with shape adjusted for keepdim
    if (input_tensor_arg.logical_volume() == 0) {
        return zero_volume_reduce<reduce_type>(input_tensor_arg, dim, keepdim, memory_config);
    }

    float pad_value = get_pad_value(reduce_type);
    bool single_reduce_op = (dim.size() == 0) || (dim.size() == 1 && (dim[0] == rank - 1 || dim[0] == rank - 2)) ||
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
                            scalar,
                            non_height_width_dims);
                    } else {
                        output_tensor = reduce_impl<ReduceType::Sum>(
                            output_tensor,
                            {reduce_dim},
                            /*keepdim=*/true,
                            memory_config,
                            compute_kernel_config,
                            scalar,
                            non_height_width_dims);
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
            float inv_volume = 1.0f / input_tensor_arg.logical_volume();
            output_tensor = ttnn::mul_sfpu(inv_volume, output_tensor, memory_config);
        } else {
            TT_THROW("Unsupported reduction operation");
        }
    } else {
        tt::tt_metal::ReduceOpDim reduce_op_dim;
        if ((dim.size() == 0) || (dim.size() == 1 and dim[0] == rank - 1)) {
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
                scalar / reduced_volume,
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
    return adjust_shape(output_tensor, input_shape, keepdim, dim, non_height_width_dims);
}

template <ReduceType reduce_type>
static Tensor std_var_impl(
    const Tensor& input_tensor_arg,
    const ttnn::SmallVector<int>& dim,
    const bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    const ttnn::SmallVector<int>& non_height_width_dims,
    bool correction) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    auto input_shape = input_tensor_arg.logical_shape();
    auto rank = input_shape.size();
    auto memory_config = memory_config_arg.value_or(input_tensor_arg.memory_config());

    // If the input tensor is a rank 0 tensor, return NaN
    if (rank == 0) {
        // Create an output tensor with same shape and attributes as input tensor
        auto output_tensor =
            ttnn::clone(input_tensor_arg, /*dtype=*/std::nullopt, memory_config, compute_kernel_config);
        output_tensor = ttnn::mul_sfpu(NAN, output_tensor, memory_config);
        return adjust_shape(output_tensor, input_shape, keepdim, dim, non_height_width_dims);
    }

    // If the input is a zero volume tensor, return output with shape adjusted for keepdim
    if (input_tensor_arg.logical_volume() == 0) {
        return zero_volume_reduce<reduce_type>(input_tensor_arg, dim, keepdim, memory_config);
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

    auto mean_tensor = reduce_impl<ReduceType::Sum>(
        input_tensor_arg, dim, keepdim, memory_config_arg, compute_kernel_config, scalar, non_height_width_dims);

    auto mean_square_tensor = reduce_impl<ReduceType::Sum>(
        ttnn::pow(input_tensor_arg, 2.0f, memory_config),
        dim,
        keepdim,
        memory_config_arg,
        compute_kernel_config,
        scalar,
        non_height_width_dims);
    Tensor output_tensor =
        ttnn::subtract(mean_square_tensor, ttnn::pow(mean_tensor, 2.0f, memory_config), std::nullopt, memory_config);
    if constexpr (reduce_type == ReduceType::Std) {
        output_tensor = ttnn::sqrt(output_tensor, memory_config);
    }
    return output_tensor;
}

template <ReduceType reduce_type>
bool call_fast_nc(DataType dtype) {
    if constexpr (reduce_type != ReduceType::Sum) {
        return false;
    }
    return dtype == DataType::BFLOAT16 || dtype == DataType::BFLOAT8_B;
}

Tensor non_height_width_reduce(
    const ttnn::Tensor& input_tensor,
    ttnn::SmallVector<int> dims,
    const std::optional<MemoryConfig>& memory_config_arg,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
    auto input_shape = input_tensor.logical_shape();
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

template <ReduceType reduce_type>
Tensor Reduce<reduce_type>::invoke(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, ttnn::SmallVector<int>>>& dim_arg,
    const bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    bool correction) {
    ttnn::SmallVector<int> dim = generate_reduce_dim(input_tensor_arg, dim_arg);
    float pad_value = get_pad_value(reduce_type);
    bool is_tiled = input_tensor_arg.layout() == TILE_LAYOUT;
    auto input_tensor = is_tiled ? ttnn::fill_implicit_tile_padding(input_tensor_arg, pad_value) : input_tensor_arg;
    // TODO: generalize to support all types, parameters, and formats. Issue #18566
    ttnn::SmallVector<int> non_height_width_dims{}, height_width_dims{};
    if (call_fast_nc<reduce_type>(input_tensor.dtype())) {
        auto dims = split_height_width_dims(dim, input_tensor);
        non_height_width_dims = dims.first;
        height_width_dims = dims.second;

        if (non_height_width_dims.size() > 0) {
            input_tensor =
                non_height_width_reduce(input_tensor, non_height_width_dims, memory_config_arg, compute_kernel_config);

            if (height_width_dims.size() == 0) {
                return adjust_shape(
                    input_tensor, input_tensor_arg.logical_shape(), keepdim, height_width_dims, non_height_width_dims);
            }
            dim = height_width_dims;
        }
    }
    if constexpr (reduce_type == ReduceType::Std || reduce_type == ReduceType::Var) {
        return std_var_impl<reduce_type>(
            input_tensor,
            dim,
            keepdim,
            memory_config_arg,
            compute_kernel_config,
            scalar,
            non_height_width_dims,
            correction);
    }
    return reduce_impl<reduce_type>(
        input_tensor, dim, keepdim, memory_config_arg, compute_kernel_config, scalar, non_height_width_dims);
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
        scalar,
        /*non_height_width_dims=*/{});
}

template class Reduce<ReduceType::Sum>;
template class Reduce<ReduceType::Mean>;
template class Reduce<ReduceType::Max>;
template class Reduce<ReduceType::Min>;
template class Reduce<ReduceType::Std>;
template class Reduce<ReduceType::Var>;
}  // namespace operations::reduction
}  // namespace ttnn
