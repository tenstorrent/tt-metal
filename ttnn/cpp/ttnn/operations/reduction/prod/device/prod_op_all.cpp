// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <algorithm>
#include <optional>

#include "prod_op_all.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "tt_metal/common/constants.hpp"
#include <ttnn/operations/functions.hpp>
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

namespace tt {
using namespace constants;
namespace operations {
namespace primary {

void Prod_op::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands need to be allocated in buffers on device!");
    TT_FATAL((input_tensor_a.get_layout() == Layout::TILE), "Input Layout must be tilized");
    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
    TT_FATAL(input_tensor_a.get_dtype() == DataType::BFLOAT16, "Error");
}

std::vector<tt::tt_metal::LegacyShape> Prod_op::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}

std::vector<Tensor> Prod_op::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks Prod_op::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return prod_single_core(input_tensor_a, output_tensor);
}

Tensor prod_all(const Tensor& input, const MemoryConfig& output_mem_config ) {
    Tensor result = ttnn::tiled_prod( operation::run(Prod_op{.output_mem_config = output_mem_config}, {input}).at(0), output_mem_config);
    auto arch_env = detect_arch();
    if(arch_env == tt::ARCH::WORMHOLE_B0){
        return tt::numpy::prod_result_computation_WH_B0<bfloat16>(result, result.get_dtype(), result.get_layout(), result.device(), output_mem_config);
    }
    //else --> GS Arch
    return tt::numpy::prod_result_computation_GS<bfloat16>(result, result.get_dtype(), result.get_layout(), result.device(), output_mem_config);
}

}
}
}
