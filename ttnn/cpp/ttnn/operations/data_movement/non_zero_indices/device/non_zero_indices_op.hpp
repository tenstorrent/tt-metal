// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>

#include "ttnn/run_operation.hpp"

namespace ttnn {

namespace operations::data_movement {

struct NonZeroIndices {
    const tt::tt_metal::MemoryConfig output_mem_config;
    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

tt::tt_metal::operation::ProgramWithCallbacks non_zero_indices_single_core(
    const Tensor& input, const Tensor& out_num_indices, const Tensor& out_indices);

}  // namespace operations::data_movement

}  // namespace ttnn
