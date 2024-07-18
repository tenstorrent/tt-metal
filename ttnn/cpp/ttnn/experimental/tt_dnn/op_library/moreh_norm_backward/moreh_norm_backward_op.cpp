// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/experimental/tt_dnn/op_library/moreh_norm_backward/moreh_norm_backward_op.hpp"

#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/experimental/tt_dnn/op_library/run_operation.hpp"
#include "ttnn/experimental/tensor/tensor.hpp"
#include "ttnn/experimental/tensor/tensor_impl.hpp"
#include "ttnn/experimental/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/experimental/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {


inline
std::tuple<uint32_t, uint32_t, uint32_t> extract_spatial_dims(const Shape& shape) {
    const auto rank = shape.rank();

    TT_FATAL(rank >= 2, "Shape must have at least two dims.");
    uint32_t W = shape[-1];
    uint32_t H = shape[-2];

    uint32_t other_dims_product = 1;
    for (auto i = 0; i < rank - 2; ++i) {
        other_dims_product *= shape[i];
    }

    return { W, H, other_dims_product};
}

inline void initialize_dims_with_range(std::vector<int64_t>& dims, uint32_t input_rank) {
    dims.resize(input_rank);
    std::iota(dims.begin(), dims.end(), 0);
}

inline std::vector<int64_t> get_dim(
    const std::optional<std::variant<int64_t, std::vector<int64_t>>>& dim,
    uint32_t input_rank
) {
    std::vector<int64_t> dims;
    if (!dim.has_value()) {
        initialize_dims_with_range(dims, input_rank);
    }
    else if (std::holds_alternative<int64_t>(dim.value())) {
        auto d = std::get<int64_t>(dim.value());
        dims.push_back(d);
    }
    else {
        dims = std::get<std::vector<int64_t>>(dim.value());
        if (dims.empty()) {
            initialize_dims_with_range(dims, input_rank);
        }
    }
    return dims;
}

inline
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_and_scale_spatial_dims(const Shape& shape, uint32_t dim) {
    const auto rank = shape.rank();

    TT_FATAL(rank >= 2, "Shape must have at least two dims.");
    uint32_t Wt = shape[-1] / TILE_WIDTH;
    uint32_t Ht = shape[-2] / TILE_HEIGHT;

    uint32_t reduce_dim = shape[dim];
    uint32_t inner_dims_product = 1;
    for (auto i = dim + 1; i < rank - 2; ++i) {
        inner_dims_product *= shape[i];
    }

    uint32_t inner_tile_size = inner_dims_product * Ht * Wt;
    uint32_t reduce_tile_size = reduce_dim * inner_tile_size;

    return { Wt, Ht, inner_tile_size, reduce_tile_size};
}

void MorehNormBackward::validate_with_output_tensors(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<Tensor>> &output_tensors) const {
    const auto &input = input_tensors.at(0);
    const auto &output = input_tensors.at(1);
    const auto &output_grad = input_tensors.at(2);

    const auto &input_grad = output_tensors.at(0);

    check_tensor(input, "moreh_norm_backward", "input");
    check_tensor(output, "moreh_norm_backward", "output");
    check_tensor(output_grad, "moreh_norm_backward", "output_grad");

    check_tensor(input_grad, "moreh_norm_backward", "input_grad");
}

std::vector<Shape> MorehNormBackward::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    return {input_tensors.at(0).get_legacy_shape()};
}

std::vector<Tensor> MorehNormBackward::create_output_tensors(const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    auto input = input_tensors.at(0);
    return operation::generic_create_output_tensors(
        *this, input_tensors, input.get_dtype(), input.get_layout(), this->memory_config);
}

operation::ProgramWithCallbacks MorehNormBackward::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input = input_tensors.at(0);
    const auto &output = input_tensors.at(1);
    const auto &output_grad = input_tensors.at(2);
    const auto &input_grad = output_tensors.at(0);

    return moreh_norm_backward_(input, output, output_grad, this->p, this->dims, this->keepdim, input_grad, this->compute_kernel_config);
}

Tensor moreh_norm_backward(
    const Tensor &input,
    const Tensor &output,
    const Tensor &output_grad,
    float p,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const bool keepdim,
    const std::optional<const Tensor> input_grad,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config
    ) {
    return moreh_norm_backward_impl(input, output, output_grad, p, dim, keepdim, input_grad, memory_config, compute_kernel_config);
}

Tensor moreh_norm_backward_impl(
    const Tensor &input,
    const Tensor &output,
    const Tensor &output_grad,
    float p,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const bool keepdim,
    const std::optional<const Tensor> input_grad,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config
    ) {

    uint32_t rank = input.get_legacy_shape().rank();
    std::vector<int64_t> dims = get_dim(dim, rank);
    std::sort(dims.begin(), dims.end());

    auto device = input.device();

    auto kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input, output, output_grad}))};
    operation::launch_op(
        [p, dims, keepdim, memory_config, kernel_config_val](
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            const std::vector<std::optional<Tensor>> &optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehNormBackward{.p = p,
                .dims=dims,
                .keepdim=keepdim,
                .memory_config=memory_config.value_or(input_tensors.at(0).memory_config()),
                .compute_kernel_config = kernel_config_val,
                }, input_tensors, optional_input_tensors, optional_output_tensors);
        },
        {input, output, output_grad},
        output_tensors,
        {},
        {input_grad});

    return output_tensors.at(0);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
