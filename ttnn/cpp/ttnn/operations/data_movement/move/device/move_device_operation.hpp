// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement {

enum class MoveOpParallelizationStrategy { MULTI_CORE, MULTI_CORE_OVERLAP, MULTI_CORE_SHARDED };

struct MoveDeviceOperation {
    const tt::tt_metal::MemoryConfig output_mem_config;
    const MoveOpParallelizationStrategy move_op_parallelization_strategy;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    MoveOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const;
};

tt::tt_metal::operation::ProgramWithCallbacks move_multi_core(const Tensor& input, Tensor& output);
tt::tt_metal::operation::ProgramWithCallbacks move_multi_core_with_overlap(const Tensor& input, Tensor& output);
tt::tt_metal::operation::ProgramWithCallbacks move_multi_core_sharded(const Tensor& input, Tensor& output);

}  // namespace ttnn::operations::data_movement
