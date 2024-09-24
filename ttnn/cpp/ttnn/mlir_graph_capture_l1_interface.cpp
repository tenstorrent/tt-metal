#include "mlir_graph_capture_l1_interface.hpp"

#include <cstdint>
#include <tuple>

#include "third_party/json/json.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/mlir_interface_graph_capture_utils.hpp"
#include "ttnn/operations/common/l1_interface_common.hpp"
#include "ttnn/operations/eltwise/binary/binary_l1_interface.hpp"
#include "ttnn/operations/eltwise/unary/unary_l1_interface.hpp"
#include "ttnn/operations/matmul/matmul_l1_interface.hpp"
#include "ttnn/operations/normalization/softmax/softmax_l1_interface.hpp"

namespace ttnn::mlir_interface::graph_capture {

static std::vector<std::tuple<uint32_t, uint32_t>> get_cb_allocations_from_trace(const nlohmann::json& json_trace) {
    auto graph_circular_buffer_allocations = graph::extract_circular_buffer_allocations_per_core(json_trace);

    std::vector<std::tuple<uint32_t, uint32_t>> cbs_per_core;
    for (auto cb_allocation : graph_circular_buffer_allocations) {
        cbs_per_core.emplace_back(std::make_tuple(cb_allocation, (uint32_t)64));
    }

    return cbs_per_core;
}

static std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_allocations_from_trace(const nlohmann::json& json_trace) {
    auto graph_tensor_allocations = graph::extract_l1_buffer_allocations(json_trace);

    std::vector<std::tuple<uint32_t, uint32_t>> tensors_per_core;
    for (auto cb_allocation : graph_tensor_allocations) {
        tensors_per_core.emplace_back(std::make_tuple(cb_allocation, (uint32_t)64));
    }

    return tensors_per_core;
}

GraphCaptureUnaryOpL1Usage::GraphCaptureUnaryOpL1Usage(
    const L1InterfaceOperandParams& input, const L1InterfaceOperandParams& output) :
    UnaryOpL1Usage(input, output) {
    m_json_trace = get_unary_op_trace(input, output);
}

std::vector<std::tuple<uint32_t, uint32_t>> GraphCaptureUnaryOpL1Usage::get_circular_buffer_l1_allocations_per_core()
    const {
    return get_cb_allocations_from_trace(m_json_trace);
}

std::vector<std::tuple<uint32_t, uint32_t>> GraphCaptureUnaryOpL1Usage::get_tensor_l1_allocations_per_core() const {
    return get_tensor_allocations_from_trace(m_json_trace);
}

GraphCaptureEltwiseOpL1Usage::GraphCaptureEltwiseOpL1Usage(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output) :
    EltwiseOpL1Usage(input_a, input_b, output) {
    m_json_trace = get_binary_op_trace(input_a, input_b, output);
}

std::vector<std::tuple<uint32_t, uint32_t>> GraphCaptureEltwiseOpL1Usage::get_circular_buffer_l1_allocations_per_core()
    const {
    return get_cb_allocations_from_trace(m_json_trace);
}

std::vector<std::tuple<uint32_t, uint32_t>> GraphCaptureEltwiseOpL1Usage::get_tensor_l1_allocations_per_core() const {
    return get_tensor_allocations_from_trace(m_json_trace);
}

GraphCaptureSoftmaxOpL1Usage::GraphCaptureSoftmaxOpL1Usage(
    const L1InterfaceOperandParams& input_a, const int dim_arg, const std::optional<L1InterfaceOperandParams>& output) :
    SoftmaxOpL1Usage(input_a, dim_arg, output) {
    m_json_trace = get_softmax_op_trace(input, dim_arg, this->output);
}

std::vector<std::tuple<uint32_t, uint32_t>> GraphCaptureSoftmaxOpL1Usage::get_circular_buffer_l1_allocations_per_core()
    const {
    return get_cb_allocations_from_trace(m_json_trace);
}

std::vector<std::tuple<uint32_t, uint32_t>> GraphCaptureSoftmaxOpL1Usage::get_tensor_l1_allocations_per_core() const {
    return get_tensor_allocations_from_trace(m_json_trace);
}

GraphCaptureMatmulOpL1Usage::GraphCaptureMatmulOpL1Usage(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output,
    const ttnn::operations::matmul::MatmulProgramConfig& program_config) :
    MatmulOpL1Usage(input_a, input_b, output), m_program_config(program_config) {
    m_json_trace = get_matmul_op_trace(input_a, input_b, output, m_program_config);
}

std::vector<std::tuple<uint32_t, uint32_t>> GraphCaptureMatmulOpL1Usage::get_circular_buffer_l1_allocations_per_core()
    const {
    return get_cb_allocations_from_trace(m_json_trace);
}

std::vector<std::tuple<uint32_t, uint32_t>> GraphCaptureMatmulOpL1Usage::get_tensor_l1_allocations_per_core() const {
    return get_tensor_allocations_from_trace(m_json_trace);
}
}  // namespace ttnn::mlir_interface::graph_capture
