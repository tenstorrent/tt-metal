// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement {

enum class CopyOpParallelizationStrategy { MULTI_CORE };

struct CopyDeviceOperation {
    const MemoryConfig output_mem_config;
    const DataType output_dtype;

    void validate_with_output_tensors(const std::vector<Tensor> &input_tensors,
                                      const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<ttnn::SimpleShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;

    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors,
                                              const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor> &input_tensors,
                                                   std::vector<Tensor> &output_tensors) const;
    CopyOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
};

operation::ProgramWithCallbacks copy_multi_core(const Tensor &input, const Tensor &output, bool backwards = false);

}  // namespace ttnn::operations::data_movement
