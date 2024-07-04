// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace ttnn::operations::data_movement {


struct Slice {
    const tt::tt_metal::Shape output_tensor_start;
    const tt::tt_metal::Shape output_tensor_end;
    const MemoryConfig output_mem_config;
    

    void validate_with_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    static constexpr auto attribute_names = std::forward_as_tuple("output_tensor_start", "output_tensor_end", "output_mem_config");
    const auto attribute_values() const {
        return std::forward_as_tuple(this->output_tensor_start, this->output_tensor_end, this->output_mem_config);
    }
};


}  // namespace ttnn::operations::data_movement
