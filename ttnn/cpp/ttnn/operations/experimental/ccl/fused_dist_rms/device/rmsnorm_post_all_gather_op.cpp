// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_post_all_gather_op.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/constants.hpp>

#include <optional>

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace ttnn::operations::experimental::ccl {

void FusedRMSNormPostAllGather::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 2, "Must have 2 input tensors");
    auto& a = input_tensors.at(0);
    auto& stats = input_tensors.at(1);

    for (const auto& tensor : input_tensors) {
        TT_FATAL(tensor.layout() == Layout::TILE, "Input tensor must have TILE layout, got: {}", tensor.layout());
        TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "Input tensor must be BFLOAT16, got: {}", tensor.dtype());
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Operands to rmsnorm need to be on device!");
        TT_FATAL(tensor.buffer() != nullptr, "Operands to rmsnorm need to be allocated in buffers on device!");
    }

    // stats has 1 tile columns per device
    TT_FATAL(
        stats.padded_shape()[-1] % TILE_WIDTH == 0,
        "Stats inner dimension must be divisible by TILE_WIDTH (32), got: {}",
        stats.padded_shape()[-1]);
    // All other dims must match
    TT_FATAL(
        stats.padded_shape().size() == a.padded_shape().size(), "Stats and input must have same number of dimensions");
    for (int i = 0; i < a.padded_shape().size() - 1; i++) {
        TT_FATAL(
            stats.padded_shape()[i] == a.padded_shape()[i],
            "Stats and input dim{} must match, got stats: {} vs input: {}",
            i,
            stats.padded_shape()[i],
            a.padded_shape()[i]);
    }
}

std::vector<TensorSpec> FusedRMSNormPostAllGather::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    auto& input_tensor = input_tensors.at(0);
    return {TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            this->dtype.value_or(input_tensor.dtype()),
            tt::tt_metal::PageConfig(Layout::TILE),
            input_tensor.memory_config()))};
}

tt::tt_metal::operation::ProgramWithCallbacks FusedRMSNormPostAllGather::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& a = input_tensors.at(0);
    const auto& stats = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    return fused_rmsnorm_post_allgather_multi_core(a, stats, output_tensor, this->eps, this->compute_kernel_config);
}
}  // namespace ttnn::operations::experimental::ccl
