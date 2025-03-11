// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement {

struct ConvKnitDeviceOperation {
    const int kernel_height;
    const int num_output_channels;
    const int input_width;
    const int num_input_channels;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;

    static constexpr auto attribute_names =
        std::forward_as_tuple("kernel_height", "num_output_channels", "input_width", "num_input_channels");
    const auto attribute_values() const {
        return std::make_tuple(
            this->kernel_height, this->num_output_channels, this->input_width, this->num_input_channels);
    }
};
}  // namespace ttnn::operations::data_movement
