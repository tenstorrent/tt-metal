// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn {

namespace operations {

namespace data_movement {

// TODO: Accept parallelization

struct Downsample {
    std::array<uint32_t, 5> downsample_params;
    DataType output_dtype;
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("downsample_params", "output_dtype");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->downsample_params), std::cref(this->output_dtype));
    }
    static ttnn::Tensor execute_on_worker_thread(
        const Tensor& input_tensor_a, std::array<uint32_t, 5> downsample_params, std::optional<DataType> output_dtype) {
        std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor_a}))};
        operation::launch_op(
            [downsample_params, output_dtype](
                const std::vector<Tensor>& input_tensors,
                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                return operation::run_without_autoformat(
                    Downsample{downsample_params, output_dtype.value_or(input_tensors.at(0).get_dtype())}, input_tensors);
            },
            {input_tensor_a},
            output_tensors);
        return output_tensors.at(0);
    }
};

//operation::ProgramWithCallbacks downsample_multi_core(const Tensor &a, Tensor& output);
operation::ProgramWithCallbacks downsample_single_core(const Tensor &a, std::array<uint32_t, 5> downsample_params, Tensor& output);

Tensor downsample (const Tensor &a, std::array<uint32_t, 5> downsample_params, std::optional<DataType> output_dtype=std::nullopt);

// namespace downsample_helpers {
// uint32_t get_num_cores(CoreCoord grid_size, uint32_t nblocks);
// }

} // namespace data_movement

}  // namespace operations

constexpr auto downsample = ttnn::register_operation<ttnn::operations::data_movement::Downsample>("ttnn::downsample");

}  // namespace ttnn
