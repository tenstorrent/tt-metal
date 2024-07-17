// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt::tt_metal {

enum class ScanOpParallelizationStrategy { SHARDED_MULTI_CORE };

enum class ScanOpDirection { ROWS, COLS, ROWS_REVERSED, COLS_REVERSED };

struct Scan {
    ScanOpDirection direction = ScanOpDirection::COLS_REVERSED;
    uint32_t n_tile_columns;

    void validate(const std::vector<Tensor> &input_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
        return {};  // In-place
    }

    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const {
        return {};  // In-place
    }

    ScanOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
        return ScanOpParallelizationStrategy::SHARDED_MULTI_CORE;
    }
};

Tensor scan(Tensor &a);

}  // namespace tt::tt_metal
