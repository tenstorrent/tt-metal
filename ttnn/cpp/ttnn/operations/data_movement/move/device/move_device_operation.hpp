// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

namespace ttnn::operations::data_movement {

enum class MoveOpParallelizationStrategy {
    MULTI_CORE, MULTI_CORE_OVERLAP, MULTI_CORE_SHARDED
};

struct MoveDeviceOperation {
    const MemoryConfig output_mem_config;
    const MoveOpParallelizationStrategy move_op_parallelization_strategy;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<ttnn::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    MoveOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
};

operation::ProgramWithCallbacks move_multi_core(const Tensor &input, Tensor &output);
operation::ProgramWithCallbacks move_multi_core_with_overlap(const Tensor &input, Tensor &output);
operation::ProgramWithCallbacks move_multi_core_sharded(const Tensor &input, Tensor &output);

}  // namespace tt_metal
