// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement {

uint32_t get_rm_start_offset(const Tensor& tensor, const ttnn::Shape& slice_start);
uint32_t get_tiled_start_offset(const Tensor& input_tensor, const ttnn::Shape& slice_start, bool round_up = false);
uint32_t get_tiled_start_offset(const ttnn::Shape& input_shape, const ttnn::Shape& slice_start, bool round_up = false);
struct SliceDeviceOperation {
    const ttnn::Shape slice_start;
    const ttnn::Shape slice_end;
    const ttnn::Shape step;
    const tt::tt_metal::MemoryConfig output_mem_config;

    void validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::data_movement
