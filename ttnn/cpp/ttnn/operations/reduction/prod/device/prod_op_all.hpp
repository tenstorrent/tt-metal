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
    const MemoryConfig output_mem_config;
    const DataType output_dtype;  // TODO: Uplift output_dtype as an option for general dot/bmm
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor> &input_tensors,
                                                   std::vector<Tensor> &output_tensors) const;
};

operation::ProgramWithCallbacks prod_single_core(const Tensor &input_tensor_a, const Tensor &output_tensor);

Tensor prod_all(const Tensor &input, const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
}  // namespace primary

}  // namespace operations
}  // namespace tt
