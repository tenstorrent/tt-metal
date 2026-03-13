// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "dit_minimal_rm_binary.hpp"
#include "device/dit_minimal_rm_binary_device_operation.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor DitMinimalRmBinaryOperation::invoke(
    const Tensor& input_a,
    const Tensor& input_b,
    const std::string& op,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    using BinaryOpType = ttnn::experimental::prim::BinaryOpType;

    TT_FATAL(op == "add" || op == "mul", "dit_minimal_rm_binary: unsupported op '{}'; choose 'add' or 'mul'.", op);

    const BinaryOpType op_type = (op == "add") ? BinaryOpType::ADD : BinaryOpType::MUL;
    const auto output_memory_config = memory_config.value_or(input_a.memory_config());

    return ttnn::prim::dit_minimal_rm_binary(input_a, input_b, op_type, input_a.dtype(), output_memory_config);
}

}  // namespace ttnn::operations::experimental
