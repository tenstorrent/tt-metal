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
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/reduction/reduction_common/reduction_common.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/reduction/generic/device/welford_reduce_device_operation.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"

#include <numeric>

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

    int reduced_volume = 1;
    if (rank != 0) {
        for (int axis : dim) {
            reduced_volume *= input_shape[axis];
        }
    }

    if (rank == 0 || reduced_volume == 1) {
        // If the input tensor is a rank 0 tensor (i.e. scalar), or reduction would produce a scalar,
        // return NaN or 0.0, depending on correction. This matches PyTorch behavior.
        float fill_value = correction ? std::numeric_limits<float>::quiet_NaN() : 0.0f;
        // Create an output tensor with same shape and attributes as input tensor
        // Cannot use ttnn::full_like because it will not return a NaN tensor. Issue #40503
        auto output_tensor = operations::creation::full_impl(
            input_tensor_arg.logical_shape(),
            fill_value,
            input_tensor_arg.dtype(),
            input_tensor_arg.layout(),
            input_tensor_arg.device(),
            memory_config);

        return adjust_shape(output_tensor, input_shape, keepdim, dim, non_height_width_dims);
    }

    // If the input is a zero volume tensor, return output with shape adjusted for keepdim
    if (input_tensor_arg.logical_volume() == 0) {
        return reduction_common::zero_volume_reduce<reduce_type>(input_tensor_arg, dim, keepdim, memory_config);
    }

    // For now support only interleaved tensors.
    TT_FATAL(!input_tensor_arg.is_sharded(), "Welford variance does not yet support sharded inputs");

    // Validate that the divisor is positive (Bessel's correction subtracts 1).
    // This could fail if e.g. there is only one element across the reduction dimensions and correction is true.
    int divisor = correction ? (reduced_volume - 1) : reduced_volume;
    TT_FATAL(divisor > 0, "Reduction is performed on too few elements, yielding divisor of {}", divisor);

    bool single_h = (dim.size() == 1 && dim[0] == rank - 2);
    bool single_w = (dim.size() == 1 && dim[0] == rank - 1);

    // Determine the reduce dimension and prepare the input tensor.
    //   single H or W:      direct H-reduce or W-reduce kernel
    //   single non-H/W dim: permute to H position, H-reduce, inverse permute
    //   2+ dims:            unified HW path with reduce_batch_size
    tt::tt_metal::ReduceOpDim reduce_dim;
    ttnn::Tensor input_tensor = input_tensor_arg;
    uint32_t reduce_batch_size = 1;
    bool needs_inverse_permute = false;
    ttnn::SmallVector<int64_t> inverse_perm;

    if (single_h || single_w) {
        reduce_dim = single_w ? tt::tt_metal::ReduceOpDim::W : tt::tt_metal::ReduceOpDim::H;
        // 1D tensors need reshaping to 2D because the kernel requires at least 2 dimensions.
        if (rank == 1) {
            input_tensor = ttnn::reshape(input_tensor, ttnn::Shape{1, input_shape[0]});
        }
    } else if (dim.size() == 1) {
        // Single non-H/W dim: permute to H position, H-reduce, inverse permute.
        reduce_dim = tt::tt_metal::ReduceOpDim::H;
        int target_dim = dim[0];
        ttnn::SmallVector<int64_t> perm(rank);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[target_dim], perm[rank - 2]);
        input_tensor = ttnn::permute(input_tensor, perm, memory_config);
        needs_inverse_permute = true;
        inverse_perm = perm;  // swap is its own inverse
    } else {
        // 2+ dims: unified HW path.  Permute all reduction dims to the end,
        // last two become H and W.  Extra reduction dims (if any) fold into
        // the NC batch dimension; reduce_batch_size tells the writer kernel
        // how many consecutive NC slices to group per output element.
        reduce_dim = tt::tt_metal::ReduceOpDim::HW;

        // Build permutation: kept dims first (in original order), then all
        // reduction dims.  dim is already sorted ascending by generate_reduce_dim.
        ttnn::SmallVector<int64_t> perm;
        perm.reserve(rank);
        for (uint32_t i = 0; i < rank; ++i) {
            if (std::find(dim.begin(), dim.end(), static_cast<int>(i)) == dim.end()) {
                perm.push_back(static_cast<int64_t>(i));
            }
        }
        for (int d : dim) {
            perm.push_back(static_cast<int64_t>(d));
        }

        // ttnn::permute checks for identity internally and skips data movement if not needed.
        input_tensor = ttnn::permute(input_tensor, perm, memory_config);

        // Extra reduction dims beyond the last two contribute to reduce_batch_size.
        for (size_t i = 0; i < dim.size() - 2; ++i) {
            reduce_batch_size *= input_shape[dim[i]];
        }
    }

    if (input_tensor.layout() != Layout::TILE) {
        ttnn::Shape padded_shape = data_movement::pad_to_tile_shape(input_tensor.padded_shape());
        input_tensor = ttnn::tilize_with_val_padding(
            input_tensor, padded_shape, 0.0f, memory_config, std::nullopt, /*use_multicore=*/true, sub_core_grids);
    }

    auto reduce_math = (reduce_type == reduction_common::ReduceType::Std) ? tt::tt_metal::ReduceOpMath::STD
                                                                          : tt::tt_metal::ReduceOpMath::VAR;
    ttnn::Tensor output_tensor = ttnn::prim::welford_reduce(
        input_tensor,
        reduce_math,
        reduce_dim,
        scalar,
        memory_config,
        std::nullopt,
        compute_kernel_config,
        correction,
        sub_core_grids,
        reduce_batch_size);

    if (needs_inverse_permute) {
        output_tensor = ttnn::permute(output_tensor, inverse_perm, memory_config);
    }

    // Compensate for any shape adjustments applied to the input tensor.
    return adjust_shape(output_tensor, input_shape, keepdim, dim, non_height_width_dims);
}

template <reduction_common::ReduceType reduce_type>
bool call_fast_nc(DataType dtype) {
    if constexpr (reduce_type != reduction_common::ReduceType::Sum) {
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
    // TODO: generalize to support all types, parameters, and formats. Issue #18566
    ttnn::SmallVector<int> non_height_width_dims{}, height_width_dims{};

    if constexpr (
        reduce_type == reduction_common::ReduceType::Std || reduce_type == reduction_common::ReduceType::Var) {
        return std_var_impl<reduce_type>(
            input_tensor_arg,
            dim,
            keepdim,
            memory_config_arg,
            compute_kernel_config,
            scalar,
            non_height_width_dims,
            correction,
            sub_core_grids);
    }

    bool is_tiled = input_tensor_arg.layout() == TILE_LAYOUT;
    auto input_tensor = is_tiled ? ttnn::fill_implicit_tile_padding(input_tensor_arg, pad_value) : input_tensor_arg;

    if (call_fast_nc<reduce_type>(input_tensor.dtype())) {
        auto dims = split_height_width_dims(dim, input_tensor);
        non_height_width_dims = dims.first;
        height_width_dims = dims.second;

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
