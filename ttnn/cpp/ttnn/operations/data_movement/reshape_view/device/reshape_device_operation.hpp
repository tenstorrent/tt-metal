// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"
namespace ttnn {

struct ReshapeDeviceOperation {
    const ttnn::Shape logical_output_shape;
    const ttnn::Shape padded_output_shape;
    tt::tt_metal::MemoryConfig output_mem_config;

    // Required functions to all tensor op functions
    void update_structure(const Tensor& input_tensor);
    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn
