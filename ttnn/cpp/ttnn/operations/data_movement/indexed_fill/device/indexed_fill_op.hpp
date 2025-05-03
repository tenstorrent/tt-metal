// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement {

struct IndexedFill {
    const tt::tt_metal::MemoryConfig output_mem_config;
    const int64_t dim;
    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

tt::tt_metal::operation::ProgramWithCallbacks indexed_fill_multi_core(
    const Tensor& batch_ids, const Tensor& input_a, const Tensor& input_b, const Tensor& output);

}  // namespace ttnn::operations::data_movement
