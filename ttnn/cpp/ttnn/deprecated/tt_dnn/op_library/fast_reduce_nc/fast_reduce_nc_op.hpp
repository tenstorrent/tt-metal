// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <utility>
#include <vector>
#include <tuple>

#include "ttnn/run_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {

namespace tt_metal {


inline
std::tuple<uint32_t, uint32_t, uint32_t> extract_spatial_dims(const Shape& shape) {
    const auto rank = shape.rank();

    TT_FATAL(rank >= 2, "Shape must have at least two dims.");
    uint32_t W = shape[-1];
    uint32_t H = shape[-2];

    uint32_t other_dims_product = 1;
    for (auto i = 0; i < rank - 2; ++i) {
        other_dims_product *= shape[i];
    }

    return { W, H, other_dims_product};
}

struct FastReduceNC {
    int64_t dim;
    MemoryConfig output_mem_config;
    const DeviceComputeKernelConfig compute_kernel_config;
    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) const;
};

operation::ProgramWithCallbacks reduce_nc_impl(const Tensor &input, const Tensor &output, int64_t dim, const DeviceComputeKernelConfig &compute_kernel_config);

Tensor fast_reduce_nc(
    const Tensor &input,
    std::vector<int64_t> &dims,
    const std::optional<const Tensor> output = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
}  // namespace tt-metal

}  // namespace tt
