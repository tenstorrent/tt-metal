// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement {

struct ConvCropDeviceOperation {
    const tt::tt_metal::MemoryConfig output_mem_config;
    const int crop_height;
    const int crop_width;
    const int pre_crop_height;
    const int pre_crop_width;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;

    static constexpr auto attribute_names =
        std::forward_as_tuple("output_mem_config", "crop_height", "crop_width", "pre_crop_height", "pre_crop_width");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->output_mem_config),
            this->crop_height,
            this->crop_width,
            this->pre_crop_height,
            this->pre_crop_width);
    }
};

}  // namespace ttnn::operations::data_movement
