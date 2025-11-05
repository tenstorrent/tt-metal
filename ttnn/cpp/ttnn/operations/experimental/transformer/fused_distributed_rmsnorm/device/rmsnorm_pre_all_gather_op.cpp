// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_pre_all_gather_op.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/constants.hpp>

#include <optional>

namespace ttnn::operations::experimental::transformer {

void FusedRMSNormPreAllGather::validate(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;

    TT_FATAL(input_tensors.size() == 1, "Must have 1 input tensor");
    auto& tensor = input_tensors.at(0);

    TT_FATAL(tensor.layout() == Layout::TILE, "Input tensor must have TILE layout, got: {}", tensor.layout());
    TT_FATAL(
        tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input tensor must use INTERLEAVED memory layout, got: {}",
        tensor.memory_config().memory_layout());
    TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "Input tensor must be BFLOAT16, got: {}", tensor.dtype());
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device, got: {}", tensor.storage_type());
    TT_FATAL(tensor.buffer() != nullptr, "Input tensor must be allocated in device buffers (buffer is null)");
}

std::vector<TensorSpec> FusedRMSNormPreAllGather::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::tt_metal;
    using namespace tt::constants;

    const auto& input_tensor = input_tensors.at(0);

    auto output_shape = input_tensors.at(0).logical_shape();
    uint32_t num_tiles_w = 1;  // RMSNorm only
    output_shape[-1] = num_tiles_w * TILE_WIDTH;

    return {TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), input_tensor.memory_config()))};
}

tt::tt_metal::operation::ProgramWithCallbacks FusedRMSNormPreAllGather::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    return fused_rmsnorm_pre_allgather_multi_core(a, output_tensor, this->compute_kernel_config);
}

}  // namespace ttnn::operations::experimental::transformer
