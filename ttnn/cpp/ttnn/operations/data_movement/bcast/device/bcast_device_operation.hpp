// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"

namespace ttnn::operations::data_movement {



enum class BcastOpParallelizationStrategy {
    MULTI_CORE_H_SHARDED,
    MULTI_CORE_H_SHARDED_OPTIMISED,
    MULTI_CORE_H,
    MULTI_CORE_W,
    MULTI_CORE_HW,
    SINGLE_CORE
};


struct EltwiseBinaryBroadcast {
    const ttnn::BcastOpMath math_op;
    const ttnn::BcastOpDim dim;
    const MemoryConfig output_mem_config;
    const bool in_place;

    void validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<ttnn::SimpleShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    BcastOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
};

} // ttnn::operations::data_movement
