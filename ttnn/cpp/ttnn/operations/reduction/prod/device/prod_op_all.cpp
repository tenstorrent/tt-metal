// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <optional>
#include "prod_op_all.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include <tt-metalium/constants.hpp>
#include <ttnn/operations/functions.hpp>
#include "tools/profiler/op_profiler.hpp"
#include <ttnn/operations/core/core.hpp>
#include <umd/device/tt_cluster_descriptor.h>  // tt_ClusterDescriptor

namespace tt {
using namespace constants;
namespace operations {
namespace primary {

void Prod_op::validate(const std::vector<tt::tt_metal::Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operands need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL((input_tensor_a.get_layout() == tt::tt_metal::Layout::TILE), "Input Layout must be tilized");
    TT_FATAL(input_tensor_a.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED, "Error");
    const auto dtype = input_tensor_a.get_dtype();
    TT_FATAL(
        dtype == tt::tt_metal::DataType::BFLOAT16 || dtype == tt::tt_metal::DataType::BFLOAT8_B,
        "Error - unsupported data type for prod, expected BFLOAT16 or BFLOAT8 but got {}.",
        input_tensor_a.get_dtype());
}

std::vector<tt::tt_metal::TensorSpec> Prod_op::compute_output_specs(
    const std::vector<tt::tt_metal::Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, 1, TILE_HW}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            output_mem_config))};
}

tt::tt_metal::operation::ProgramWithCallbacks Prod_op::create_program(
    const std::vector<tt::tt_metal::Tensor>& input_tensors, std::vector<tt::tt_metal::Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return prod_single_core(input_tensor_a, output_tensor);
}

tt::tt_metal::Tensor prod_all(const tt::tt_metal::Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config) {
    tt::tt_metal::Tensor result =
        tt::tt_metal::operation::run(Prod_op{.output_mem_config = output_mem_config}, {input}).at(0);
    return ttnn::prod_result_computation_WH_B0<bfloat16>(
        result, result.get_dtype(), result.get_layout(), result.device(), output_mem_config);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
