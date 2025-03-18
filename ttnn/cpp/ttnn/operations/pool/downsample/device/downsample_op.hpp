// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn {

namespace operations {

namespace downsample {

// TODO: Accept parallelization

struct Downsample {
    std::array<uint32_t, 5> downsample_params;
    tt::tt_metal::DataType dtype;
    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

Tensor downsample(
    const Tensor& a,
    std::array<uint32_t, 5> downsample_params,
    std::optional<tt::tt_metal::DataType> dtype = std::nullopt);
// tt::tt_metal::operation::ProgramWithCallbacks downsample_multi_core(const Tensor &a, Tensor& output);
tt::tt_metal::operation::ProgramWithCallbacks downsample_single_core(
    const Tensor& a, std::array<uint32_t, 5> downsample_params, Tensor& output);

// namespace downsample_helpers {
// uint32_t get_num_cores(CoreCoord grid_size, uint32_t nblocks);
// }

}  // namespace downsample

}  // namespace operations

}  // namespace ttnn
