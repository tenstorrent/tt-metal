// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement {

struct ShardedToInterleavedPartialDeviceOperation {
    const uint32_t num_slices;
    const uint32_t slice_index;
    const MemoryConfig output_mem_config;
    const DataType output_dtype;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor> &input_tensors,
                                                   std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names =
        std::make_tuple("num_slices", "slice_index", "output_mem_config", "output_dtype");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->num_slices),
                               std::cref(this->slice_index),
                               std::cref(this->output_mem_config),
                               std::cref(this->output_dtype));
    }
};

}  // namespace ttnn::operations::data_movement
