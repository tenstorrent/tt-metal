/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace tt {

namespace operations {

namespace primary {

/*
 * prod product
 */

struct Prod_op {
    const tt::tt_metal::MemoryConfig output_mem_config;
    const tt::tt_metal::DataType output_dtype;  // TODO: Uplift output_dtype as an option for general dot/bmm
    void validate(const std::vector<tt::tt_metal::Tensor>& input_tensors) const;
    std::vector<tt::tt_metal::TensorSpec> compute_output_specs(
        const std::vector<tt::tt_metal::Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<tt::tt_metal::Tensor>& input_tensors,
        std::vector<tt::tt_metal::Tensor>& output_tensors) const;
};

tt::tt_metal::operation::ProgramWithCallbacks prod_single_core(
    const tt::tt_metal::Tensor& input_tensor_a, const tt::tt_metal::Tensor& output_tensor);

tt::tt_metal::Tensor prod_all(
    const tt::tt_metal::Tensor& input,
    const tt::tt_metal::MemoryConfig& mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
}  // namespace primary

}  // namespace operations
}  // namespace tt
