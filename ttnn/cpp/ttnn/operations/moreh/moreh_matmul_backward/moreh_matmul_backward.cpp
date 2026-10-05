// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_matmul_backward.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/moreh/moreh_dot_backward/moreh_dot_backward.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::moreh::moreh_matmul_backward {

// Batch dims as-is, last two dims in tiles, innermost first.
void get_tensor_dim(ttsl::SmallVector<uint32_t>& dim, const ttnn::Shape& shape) {
    const auto rank = shape.rank();
    for (auto i = 0; i < rank; ++i) {
        auto idx = rank - 1 - i;
        if (idx == rank - 1 || idx == rank - 2) {
            dim[i] = shape[idx] / tt::constants::TILE_HEIGHT;
        } else {
            dim[i] = shape[idx];
        }
    }
}

ttsl::SmallVector<int64_t> find_reduce_dim(const ttnn::Shape& a_shape, const ttnn::Shape& b_shape) {
    ttsl::SmallVector<uint32_t> a_dim(ttnn::MAX_NUM_DIMENSIONS, 1);
    ttsl::SmallVector<uint32_t> b_dim(ttnn::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(a_dim, a_shape);
    get_tensor_dim(b_dim, b_shape);
    int32_t rank = std::max(a_shape.rank(), b_shape.rank());
    ttsl::SmallVector<int64_t> dims;
    for (int i = 0; i < rank - 2; ++i) {
        int idx = rank - 1 - i;
        TT_FATAL(idx >= 0, "idx < 0");
        if (a_dim[idx] != b_dim[idx]) {
            dims.push_back(i);
        }
    }
    return dims;
}

bool is_same_batch_dim(const Tensor& tensor_a, const Tensor& tensor_b) {
    ttsl::SmallVector<uint32_t> a_dim(ttnn::MAX_NUM_DIMENSIONS, 1);
    ttsl::SmallVector<uint32_t> b_dim(ttnn::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(a_dim, tensor_a.padded_shape());
    get_tensor_dim(b_dim, tensor_b.padded_shape());
    for (auto i = 2; i < ttnn::MAX_NUM_DIMENSIONS; ++i) {
        if (a_dim[i] != b_dim[i]) {
            return false;
        }
    }
    return true;
}

inline bool is_dot_backward(const Tensor& output_grad, const Tensor& input, const Tensor& other) {
    // TODO: non-4d support for dot backward.
    if (output_grad.padded_shape().rank() != 4 || input.padded_shape().rank() != 4 ||
        other.padded_shape().rank() != 4) {
        return false;
    }
    return is_scalar(output_grad) && is_1d_tensor(input) && is_1d_tensor(other) && is_same_shape(input, other);
}

}  // namespace ttnn::operations::moreh::moreh_matmul_backward

namespace ttnn {

std::vector<std::optional<Tensor>> moreh_matmul_backward(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs,
    const std::optional<const Tensor>& input_grad,
    const std::optional<const Tensor>& other_grad,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    if (operations::moreh::moreh_matmul_backward::is_dot_backward(output_grad, input, other)) {
        return ttnn::moreh_dot_backward(output_grad, input, other, input_grad, other_grad, memory_config);
    }

    std::vector<std::optional<Tensor>> outputs(2);

    const bool input_requires_grad = are_required_outputs.at(0);
    const bool other_requires_grad = are_required_outputs.at(1);

    if (input_requires_grad) {
        TT_FATAL(input_grad.has_value(), "Input gradient is marked required but not provided.");
        const auto& input_grad_tensor = input_grad.value();
        if (operations::moreh::moreh_matmul_backward::is_same_batch_dim(output_grad, input_grad_tensor)) {
            ttnn::matmul(
                output_grad,
                other,
                /*transpose_a=*/false,
                /*transpose_b=*/true,
                memory_config,
                /*dtype=*/std::nullopt,
                /*program_config=*/std::nullopt,
                /*activation=*/std::nullopt,
                compute_kernel_config,
                /*core_grid=*/std::nullopt,
                /*output_tile=*/std::nullopt,
                input_grad_tensor);
        } else {
            const auto& temp_input_grad = ttnn::matmul(
                output_grad,
                other,
                /*transpose_a=*/false,
                /*transpose_b=*/true,
                memory_config,
                /*dtype=*/std::nullopt,
                /*program_config=*/std::nullopt,
                /*activation=*/std::nullopt,
                compute_kernel_config,
                /*core_grid=*/std::nullopt,
                /*output_tile=*/std::nullopt,
                std::nullopt);
            auto reduce_dims = operations::moreh::moreh_matmul_backward::find_reduce_dim(
                temp_input_grad.padded_shape(), input_grad_tensor.padded_shape());
            ttnn::moreh_sum(
                temp_input_grad, reduce_dims, true, input_grad_tensor, memory_config, compute_kernel_config);
        }
        outputs[0] = input_grad_tensor;
    }

    if (other_requires_grad) {
        TT_FATAL(other_grad.has_value(), "Other gradient is marked required but not provided.");
        const auto& other_grad_tensor = other_grad.value();
        if (operations::moreh::moreh_matmul_backward::is_same_batch_dim(output_grad, other_grad_tensor)) {
            ttnn::matmul(
                input,
                output_grad,
                /*transpose_a=*/true,
                /*transpose_b=*/false,
                memory_config,
                /*dtype=*/std::nullopt,
                /*program_config=*/std::nullopt,
                /*activation=*/std::nullopt,
                compute_kernel_config,
                /*core_grid=*/std::nullopt,
                /*output_tile=*/std::nullopt,
                other_grad_tensor);
        } else {
            const auto& temp_other_grad = ttnn::matmul(
                input,
                output_grad,
                /*transpose_a=*/true,
                /*transpose_b=*/false,
                memory_config,
                /*dtype=*/std::nullopt,
                /*program_config=*/std::nullopt,
                /*activation=*/std::nullopt,
                compute_kernel_config,
                /*core_grid=*/std::nullopt,
                /*output_tile=*/std::nullopt,
                std::nullopt);
            auto reduce_dims = operations::moreh::moreh_matmul_backward::find_reduce_dim(
                temp_other_grad.padded_shape(), other_grad_tensor.padded_shape());
            ttnn::moreh_sum(
                temp_other_grad, reduce_dims, true, other_grad_tensor, memory_config, compute_kernel_config);
        }
        outputs[1] = other_grad_tensor;
    }

    return outputs;
}

}  // namespace ttnn
