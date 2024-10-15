// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement {

struct InterleavedToShardedDeviceOperation {
    const MemoryConfig output_mem_config;
    const DataType output_dtype;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names =
        std::make_tuple("output_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->output_mem_config)
            );
    }
};
}
