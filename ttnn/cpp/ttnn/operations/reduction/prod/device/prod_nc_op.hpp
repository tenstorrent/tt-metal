// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

struct Prod {
    int64_t dim;
    void validate(const std::vector<Tensor>& inputs) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& inputs) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& inputs) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const;
};

tt::tt_metal::operation::ProgramWithCallbacks prod_nc_format(const Tensor& input, const Tensor& output, int64_t dim);

Tensor prod_(
    const Tensor& input,
    std::optional<std::reference_wrapper<const Tensor>> output,
    const int64_t& dim,
    const MemoryConfig& mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor prod_nc(
    const Tensor& input,
    const Tensor& output,
    ttsl::SmallVector<int64_t>& dims,
    const MemoryConfig& output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
