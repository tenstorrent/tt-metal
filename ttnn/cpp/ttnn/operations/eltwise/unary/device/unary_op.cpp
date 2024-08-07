// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_op.hpp"
#include "unary_program_factory_multicore.hpp"
#include "unary_program_factory_sharded.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::operations::unary {

inline void validate_supported_arch_dtype(tt::ARCH arch, DataType input_datatype, DataType output_datatype, UnaryOpType op_type) {
    switch (op_type) {
        case UnaryOpType::REMAINDER:
        case UnaryOpType::FLOOR:
        case UnaryOpType::CEIL:
        case UnaryOpType::LEFT_SHIFT:
        case UnaryOpType::RIGHT_SHIFT:
            TT_FATAL(arch != tt::ARCH::GRAYSKULL, "Operation is not supported on Grayskull");
            break;
        case UnaryOpType::BITWISE_XOR:
        case UnaryOpType::BITWISE_NOT:
        case UnaryOpType::BITWISE_AND:
        case UnaryOpType::BITWISE_OR:
            TT_FATAL(arch != tt::ARCH::GRAYSKULL, "BITWISE operation is not supported on Grayskull");
            TT_FATAL(input_datatype == DataType::INT32, "Data type is not supported for Bitwise operations");
            TT_FATAL(output_datatype == DataType::INT32, "Data type is not supported for Bitwise operations");
            break;
        case UnaryOpType::FMOD:
            TT_FATAL(arch != tt::ARCH::GRAYSKULL, "FMOD operation is not supported on Grayskull");
            TT_FATAL(input_datatype == DataType::BFLOAT16, "Data type is not supported for Fmod operations");
            TT_FATAL(output_datatype == DataType::BFLOAT16, "Data type is not supported for Fmod operations");
            break;
        default:
            return;
    }
}

void Unary::validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto out_mem_config = (!output_tensors.empty() && output_tensors.at(0).has_value()) ? output_tensors.at(0).value().memory_config() : this->output_mem_config;
    auto output_datatype = output_dtype;
    if(!output_tensors.empty() && output_tensors.at(0).has_value()){
        const auto& out = output_tensors.at(0);
        output_datatype = out->get_dtype();
    }
    auto arch = input_tensor_a.device()->arch();
    auto input_datatype = input_tensor_a.get_dtype();
    for (const auto& unary_op : this->op_chain) {
        validate_supported_arch_dtype(arch, input_datatype, output_datatype, unary_op.op_type);
    }
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to eltwise unary need to be on device!");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr, "Operands to eltwise unary need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout == out_mem_config.memory_layout,
        "Input and output memory layout must match");
    if (!input_tensor_a.is_sharded()) {
        TT_FATAL((input_tensor_a.get_layout() == Layout::TILE), "Inputs to eltwise unary must be tilized");
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
            "Interleaved memory layout supported");
    }
    if(!output_tensors.empty() && output_tensors.at(0).has_value()){
        const auto output_shape_required = this->compute_output_shapes(input_tensors);
        const auto& out_tensor = output_tensors.at(0).value();
        TT_FATAL(out_tensor.get_legacy_shape() == output_shape_required.at(0), fmt::format("The input tensors need a shape of {}, however the output tensor is only {}", output_shape_required,  out_tensor.get_legacy_shape()));
    }
    if (!output_tensors.empty()) {
        TT_FATAL(output_tensors.size() == 1, "Must have 1 output tensors");
    }
}

std::vector<tt::tt_metal::Shape> Unary::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}

std::vector<Tensor> Unary::create_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if(!output_tensors.empty() && output_tensors.at(0).has_value()){
        return {output_tensors.at(0).value()};
    }

    const auto& input_tensor = input_tensors.at(0);
    if (this->output_mem_config.is_sharded()) {
        return {create_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            this->output_dtype,
            input_tensor.get_layout(),
            input_tensor.device(),
            this->output_mem_config)};
    }
    return tt::tt_metal::operation::generic_create_output_tensors(
        *this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks Unary::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);
    switch (parallelization_strategy) {
        case UnaryOpParallelizationStrategy::SHARDED_MULTI_CORE:
            return detail::unary_sharded(input_tensor, output_tensor, this->op_chain, this->fp32_dest_acc_en, this->preserve_fp32_precision);
        case UnaryOpParallelizationStrategy::MULTI_CORE:
        default: return detail::unary_multi_core(input_tensor, output_tensor, this->op_chain, this->fp32_dest_acc_en, this->preserve_fp32_precision);
    }
}

UnaryOpParallelizationStrategy Unary::get_parallelization_strategy(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (input_tensor.is_sharded())
        return UnaryOpParallelizationStrategy::SHARDED_MULTI_CORE;
    else {
        return UnaryOpParallelizationStrategy::MULTI_CORE;
    }
}

const tt::tt_metal::operation::Hash Unary::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& input_shape = input_tensor.get_legacy_shape();

    tt::tt_metal::operation::Hash hash = operation::hash_operation<Unary>(
        compute_volume(input_shape),
        input_tensor.dtype,
        std::get<DeviceStorage>(input_tensor.storage).memory_config(),
        this->output_mem_config);

    for (const auto& unary_with_param_op : this->op_chain) {
        hash = tt::stl::hash::hash_objects(hash, unary_with_param_op.op_type);
        if (unary_with_param_op.has_parameter()) {
            hash = tt::stl::hash::hash_objects(hash, unary_with_param_op.params);
        }
    }
    return hash;
}


}  // namespace ttnn::operations::unary
