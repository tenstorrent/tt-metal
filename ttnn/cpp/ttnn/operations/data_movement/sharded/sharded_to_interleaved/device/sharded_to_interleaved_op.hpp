// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement {

struct ShardedToInterleavedDeviceOperation {
    const tt::tt_metal::MemoryConfig output_mem_config;
    const tt::tt_metal::DataType output_dtype;
    const bool is_l1_aligned = false;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("output_mem_config", "output_dtype");
    auto attribute_values() const {
        return std::make_tuple(std::cref(this->output_mem_config), std::cref(this->output_dtype));
    }
};

}  // namespace ttnn::operations::data_movement
