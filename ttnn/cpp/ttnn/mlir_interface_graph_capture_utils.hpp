#pragma once

#include "third_party/json/json.hpp"
#include "ttnn/operations/common/l1_interface_common.hpp"
#include "ttnn/operations/matmul/device/matmul_types.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::mlir_interface::graph_capture {
bool is_graph_capture_mode_enabled();

nlohmann::json get_unary_op_trace(const L1InterfaceOperandParams& input, const L1InterfaceOperandParams& output);

nlohmann::json get_unary_op_trace(
    const ttnn::types::Shape& shape_a,
    tt::tt_metal::DataType data_type_a,
    tt::tt_metal::Layout layout_a,
    const tt::tt_metal::MemoryConfig& memory_config_a,
    const tt::tt_metal::MemoryConfig& memory_config_o);

nlohmann::json get_binary_op_trace(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output);

nlohmann::json get_binary_op_trace(
    const ttnn::types::Shape& shape_a,
    tt::tt_metal::DataType data_type_a,
    tt::tt_metal::Layout layout_a,
    const tt::tt_metal::MemoryConfig& memory_config_a,
    const ttnn::types::Shape& shape_b,
    tt::tt_metal::DataType data_type_b,
    tt::tt_metal::Layout layout_b,
    const tt::tt_metal::MemoryConfig& memory_config_b,
    tt::tt_metal::DataType data_type_o,
    const tt::tt_metal::MemoryConfig& memory_config_o);

nlohmann::json get_softmax_op_trace(
    const L1InterfaceOperandParams& input, const int dim_arg, const L1InterfaceOperandParams& output);

nlohmann::json get_softmax_op_trace(
    const ttnn::types::Shape& shape_a,
    tt::tt_metal::DataType data_type_a,
    tt::tt_metal::Layout layout_a,
    const tt::tt_metal::MemoryConfig& memory_config_a,
    const int dim_arg,
    const tt::tt_metal::MemoryConfig& memory_config_o);

nlohmann::json get_matmul_op_trace(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output,
    const ttnn::operations::matmul::MatmulProgramConfig& program_config);

nlohmann::json get_matmul_op_trace(
    const ttnn::types::Shape& shape_a,
    tt::tt_metal::DataType data_type_a,
    tt::tt_metal::Layout layout_a,
    const tt::tt_metal::MemoryConfig& memory_config_a,
    const ttnn::types::Shape& shape_b,
    tt::tt_metal::DataType data_type_b,
    tt::tt_metal::Layout layout_b,
    const tt::tt_metal::MemoryConfig& memory_config_b,
    tt::tt_metal::DataType data_type_o,
    const tt::tt_metal::MemoryConfig& memory_config_o,
    const ttnn::operations::matmul::MatmulProgramConfig& program_config);
}  // namespace ttnn::mlir_interface::graph_capture
