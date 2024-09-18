#include "mlir_graph_capture_l1_interface.hpp"

#include <cstdint>
#include <tuple>

#include "host_api.hpp"
#include "impl/device/device.hpp"
#include "third_party/json/json.hpp"
#include "ttnn/graph/graph_operation_queries.hpp"
#include "ttnn/operations/common/l1_interface_common.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_l1_interface.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_l1_interface.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/matmul/matmul_l1_interface.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/normalization/softmax/softmax_l1_interface.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::mlir_interface::graph_capture {

class ScopedDeviceContext {
   public:
    ScopedDeviceContext() : m_device(tt::tt_metal::CreateDevice(0)) {}
    ~ScopedDeviceContext() { tt::tt_metal::CloseDevice(m_device); }

    tt::tt_metal::Device& get_device() { return *m_device; }

   private:
    tt::tt_metal::Device* m_device;
};

static ttnn::Tensor create_tensor(tt::tt_metal::Device& device, const L1InterfaceOperandParams& params) {
    return ttnn::zeros(
        std::get<ttnn::types::Shape>(params),
        std::get<tt::tt_metal::DataType>(params),
        std::get<tt::tt_metal::Layout>(params),
        device,
        std::get<tt::tt_metal::MemoryConfig>(params));
}

static nlohmann::json get_unary_op_trace(
    const L1InterfaceOperandParams& input, const L1InterfaceOperandParams& output) {
    ScopedDeviceContext ctx;

    auto input_tensor = create_tensor(ctx.get_device(), input);

    auto call = [&] {
        const auto output_tensor = ttnn::relu(input_tensor, std::get<tt::tt_metal::MemoryConfig>(output));
        return output_tensor;
    };

    return graph::query_trace(call);
}

static nlohmann::json get_binary_op_trace(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output) {
    ScopedDeviceContext ctx;

    auto input_tensor_a = create_tensor(ctx.get_device(), input_a);
    auto input_tensor_b = create_tensor(ctx.get_device(), input_b);

    auto call = [&] {
        const auto output_tensor = ttnn::add(
            input_tensor_a,
            input_tensor_b,
            std::get<tt::tt_metal::DataType>(output),
            std::get<tt::tt_metal::MemoryConfig>(output));
        return output_tensor;
    };

    return graph::query_trace(call);
}

static nlohmann::json get_softmax_op_trace(
    const L1InterfaceOperandParams& input, const int dim_arg, const L1InterfaceOperandParams& output) {
    ScopedDeviceContext ctx;

    auto input_tensor = create_tensor(ctx.get_device(), input);

    auto call = [&] {
        const auto output_tensor = ttnn::softmax(input_tensor, dim_arg, std::get<tt::tt_metal::MemoryConfig>(output));
        return output_tensor;
    };

    return graph::query_trace(call);
}

static nlohmann::json get_matmul_op_trace(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output,
    const ttnn::operations::matmul::MatmulProgramConfig& program_config) {
    ScopedDeviceContext ctx;

    auto input_tensor_a = create_tensor(ctx.get_device(), input_a);
    auto input_tensor_b = create_tensor(ctx.get_device(), input_b);

    auto call = [&] {
        const auto output_tensor = ttnn::matmul(
            input_tensor_a,
            input_tensor_b,
            false /* transpose_a */,
            false /* transpose_b */,
            std::get<tt::tt_metal::MemoryConfig>(output),
            std::get<tt::tt_metal::DataType>(output),
            program_config);
        return output_tensor;
    };

    return graph::query_trace(call);
}

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
