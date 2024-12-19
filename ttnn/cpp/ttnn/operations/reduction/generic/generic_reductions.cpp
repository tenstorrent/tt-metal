// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/core/core.hpp"

// Some tensors are pre-padded with 0s. E.g. Those generated via from_torch.
// Therefore need to always pad tensors again. To do that, convert to row major,
// pad, and then convert back to tile layout.
// Limitations of pad require transpose, un-transpose, and then slicing to isolate values of interest.
// End result will be padded, and after reduce is done, will need to be sliced back.
ttnn::Tensor pad_tensor_with_value(const ttnn::Tensor& input_tensor, float pad_value) {
    ttnn::Shape with_padding = input_tensor.get_shape().with_tile_padding();
    ttnn::Tensor intermediate_tensor =
        ttnn::to_layout(input_tensor, Layout::ROW_MAJOR, std::nullopt, std::nullopt, input_tensor.device());
    tt::tt_metal::Array4D padded_shape = {with_padding[0], with_padding[1], with_padding[2], with_padding[3]};
    ttnn::Tensor padded_tensor =
        ttnn::pad(intermediate_tensor, padded_shape, tt::tt_metal::Array4D({0, 0, 0, 0}), pad_value);
    padded_tensor = ttnn::to_layout(padded_tensor, Layout::TILE, std::nullopt, std::nullopt, padded_tensor.device());
    tt::log_debug(tt::LogOp, "max {} {} {}", padded_shape, pad_value, padded_tensor);
    return padded_tensor;
}

// Pad tensor with values, reduce, and then slice back to un-padded size.
ttnn::Tensor reduce_with_padding(
    ttnn::Tensor& input_tensor,
    float pad_value,
    tt::tt_metal::ReduceOpMath op,
    const tt::tt_metal::ReduceOpDim reduce_op_dim,
    float scalar,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    ttnn::Tensor padded_tensor = pad_tensor_with_value(input_tensor, pad_value);
    ttnn::Tensor output_tensor = tt::tt_metal::reduce(
        padded_tensor, op, reduce_op_dim, scalar, memory_config, std::nullopt, compute_kernel_config);
    ttnn::Shape shape = input_tensor.get_shape();
    std::array<uint32_t, 4> begins = {0, 0, 0, 0};
    std::array<uint32_t, 4> ends = {shape[0], shape[1], shape[2], shape[3]};
    std::array<uint32_t, 4> step = {1, 1, 1, 1};
    if (reduce_op_dim == tt::tt_metal::ReduceOpDim::W) {
        ends[3] = 1;
    } else if (reduce_op_dim == tt::tt_metal::ReduceOpDim::H) {
        ends[2] = 1;
    } else if (reduce_op_dim == tt::tt_metal::ReduceOpDim::HW) {
        ends[2] = 1;
        ends[3] = 1;
    } else {
        TT_THROW("Unsupported reduce op dim {}", reduce_op_dim);
    }

    output_tensor = ttnn::slice(output_tensor, begins, ends, step);
    return output_tensor;
}

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
    } else {
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

    if (dim.size() == 1 && (rank == 3 || rank == 4)) {
        if (dim[0] == 1 && rank == 4) {
            Tensor output = ttnn::transpose(input_tensor_arg, 1, -2, memory_config);
            output = reduce_impl<reduce_type>(
                output, 2, /*keepdim=*/true, memory_config, compute_kernel_config, scalar, /*reshape=*/true);
            output = ttnn::transpose(output, 1, -2, memory_config);
            if (reshape) {
                output = ttnn::reshape(output, ttnn::Shape{output_shape});
            }
            return output;
        } else if (dim[0] == 0) {
            Tensor output = ttnn::transpose(input_tensor_arg, 0, -2, memory_config);
            output = reduce_impl<reduce_type>(
                output, -2, /*keepdim=*/true, memory_config, compute_kernel_config, scalar, /*reshape=*/true);
            output = ttnn::transpose(output, 0, -2, memory_config);
            if (reshape) {
                output = ttnn::reshape(output, ttnn::Shape{output_shape});
            }
            return output;
        }
    }

    auto input_tensor = ttnn::unsqueeze_to_4D(input_tensor_arg);

    Tensor output_tensor;
    if (!dim_arg.has_value()) {
        if constexpr (
            reduce_type == ReduceType::Sum || reduce_type == ReduceType::Max || reduce_type == ReduceType::Min) {
            output_tensor = input_tensor;
            for (int rank = input_tensor.get_legacy_shape().rank() - 1; rank >= 0; rank--) {
                output_tensor = reduce_impl<reduce_type>(
                    output_tensor, rank, true, memory_config, compute_kernel_config, scalar, false);
            }
        } else if constexpr (reduce_type == ReduceType::Mean) {
            output_tensor = input_tensor;
            for (int rank = input_tensor.get_legacy_shape().rank() - 1; rank >= 0; rank--) {
                output_tensor = reduce_impl<ReduceType::Sum>(
                    output_tensor, rank, true, memory_config, compute_kernel_config, scalar, false);
            }
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
            output_tensor = reduce_with_padding(
                input_tensor,
                -std::numeric_limits<float>::infinity(),
                tt::tt_metal::ReduceOpMath::MAX,
                reduce_op_dim,
                scalar,
                memory_config,
                compute_kernel_config);
        } else if constexpr (reduce_type == ReduceType::Min) {
            output_tensor = reduce_with_padding(
                input_tensor,
                std::numeric_limits<float>::infinity(),
                tt::tt_metal::ReduceOpMath::MIN,
                reduce_op_dim,
                scalar,
                memory_config,
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
