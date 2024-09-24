#include "mlir_interface_graph_capture_utils.hpp"

#include "host_api.hpp"
#include "ttnn/graph/graph_operation_queries.hpp"
#include "ttnn/operations/common/l1_interface_common.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::mlir_interface::graph_capture {
bool is_graph_capture_mode_enabled() {
    static const bool graph_capture_enabled = tt::parse_env("TTNN_MLIR_INTERFACE_USE_GRAPH_CAPTURE", false);

    return graph_capture_enabled;
}

class ScopedDeviceContext {
   public:
    ScopedDeviceContext() : m_device(tt::tt_metal::CreateDevice(0)) {}
    ~ScopedDeviceContext() { tt::tt_metal::CloseDevice(m_device); }

    tt::tt_metal::Device& get_device() { return *m_device; }

   private:
    tt::tt_metal::Device* m_device;
};

static ttnn::Tensor create_tensor(
    tt::tt_metal::Device& device,
    const ttnn::types::Shape& shape,
    tt::tt_metal::DataType data_type,
    tt::tt_metal::Layout layout,
    const tt::tt_metal::MemoryConfig& memory_config) {
    return ttnn::zeros(shape, data_type, layout, device, memory_config);
}

static ttnn::Tensor create_tensor(tt::tt_metal::Device& device, const L1InterfaceOperandParams& params) {
    return create_tensor(
        device,
        std::get<ttnn::types::Shape>(params),
        std::get<tt::tt_metal::DataType>(params),
        std::get<tt::tt_metal::Layout>(params),
        std::get<tt::tt_metal::MemoryConfig>(params));
}

nlohmann::json get_unary_op_trace(const L1InterfaceOperandParams& input, const L1InterfaceOperandParams& output) {
    return get_unary_op_trace(
        std::get<ttnn::types::Shape>(input),
        std::get<tt::tt_metal::DataType>(input),
        std::get<tt::tt_metal::Layout>(input),
        std::get<tt::tt_metal::MemoryConfig>(input),
        std::get<tt::tt_metal::MemoryConfig>(output));
}

nlohmann::json get_unary_op_trace(
    const ttnn::types::Shape& shape_a,
    tt::tt_metal::DataType data_type_a,
    tt::tt_metal::Layout layout_a,
    const tt::tt_metal::MemoryConfig& memory_config_a,
    const tt::tt_metal::MemoryConfig& memory_config_o) {
    ScopedDeviceContext ctx;

    auto input_tensor = create_tensor(ctx.get_device(), shape_a, data_type_a, layout_a, memory_config_a);

    auto call = [&] {
        // TODO: Replace relu with specific unary operation
        const auto output_tensor = ttnn::relu(input_tensor, memory_config_o);
        return output_tensor;
    };

    return graph::query_trace(call);
}

nlohmann::json get_binary_op_trace(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output) {
    return get_binary_op_trace(
        std::get<ttnn::types::Shape>(input_a),
        std::get<tt::tt_metal::DataType>(input_a),
        std::get<tt::tt_metal::Layout>(input_a),
        std::get<tt::tt_metal::MemoryConfig>(input_a),
        std::get<ttnn::types::Shape>(input_b),
        std::get<tt::tt_metal::DataType>(input_b),
        std::get<tt::tt_metal::Layout>(input_b),
        std::get<tt::tt_metal::MemoryConfig>(input_b),
        std::get<tt::tt_metal::DataType>(output),
        std::get<tt::tt_metal::MemoryConfig>(output));
}

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
    const tt::tt_metal::MemoryConfig& memory_config_o) {
    ScopedDeviceContext ctx;

    auto input_tensor_a = create_tensor(ctx.get_device(), shape_a, data_type_a, layout_a, memory_config_a);
    auto input_tensor_b = create_tensor(ctx.get_device(), shape_b, data_type_b, layout_b, memory_config_b);

    auto call = [&] {
        // TODO: Replace add with specific binary operation
        const auto output_tensor = ttnn::add(input_tensor_a, input_tensor_b, data_type_o, memory_config_o);
        return output_tensor;
    };

    return graph::query_trace(call);
}

nlohmann::json get_softmax_op_trace(
    const L1InterfaceOperandParams& input, const int dim_arg, const L1InterfaceOperandParams& output) {
    return get_softmax_op_trace(
        std::get<ttnn::types::Shape>(input),
        std::get<tt::tt_metal::DataType>(input),
        std::get<tt::tt_metal::Layout>(input),
        std::get<tt::tt_metal::MemoryConfig>(input),
        dim_arg,
        std::get<tt::tt_metal::MemoryConfig>(output));
}

nlohmann::json get_softmax_op_trace(
    const ttnn::types::Shape& shape_a,
    tt::tt_metal::DataType data_type_a,
    tt::tt_metal::Layout layout_a,
    const tt::tt_metal::MemoryConfig& memory_config_a,
    const int dim_arg,
    const tt::tt_metal::MemoryConfig& memory_config_o) {
    ScopedDeviceContext ctx;

    auto input_tensor = create_tensor(ctx.get_device(), shape_a, data_type_a, layout_a, memory_config_a);

    auto call = [&] {
        const auto output_tensor = ttnn::softmax(input_tensor, dim_arg, memory_config_o);
        return output_tensor;
    };

    return graph::query_trace(call);
}

nlohmann::json get_matmul_op_trace(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output,
    const ttnn::operations::matmul::MatmulProgramConfig& program_config) {
    return get_matmul_op_trace(
        std::get<ttnn::types::Shape>(input_a),
        std::get<tt::tt_metal::DataType>(input_a),
        std::get<tt::tt_metal::Layout>(input_a),
        std::get<tt::tt_metal::MemoryConfig>(input_a),
        std::get<ttnn::types::Shape>(input_b),
        std::get<tt::tt_metal::DataType>(input_b),
        std::get<tt::tt_metal::Layout>(input_b),
        std::get<tt::tt_metal::MemoryConfig>(input_b),
        std::get<tt::tt_metal::DataType>(output),
        std::get<tt::tt_metal::MemoryConfig>(output),
        program_config);
}

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
    const ttnn::operations::matmul::MatmulProgramConfig& program_config) {
    ScopedDeviceContext ctx;

    auto input_tensor_a = create_tensor(ctx.get_device(), shape_a, data_type_a, layout_a, memory_config_a);
    auto input_tensor_b = create_tensor(ctx.get_device(), shape_b, data_type_b, layout_b, memory_config_b);

    auto call = [&] {
        const auto output_tensor = ttnn::matmul(
            input_tensor_a,
            input_tensor_b,
            false /* transpose_a */,
            false /* transpose_b */,
            memory_config_o,
            data_type_o,
            program_config);
        return output_tensor;
    };

    return graph::query_trace(call);
}
};  // namespace ttnn::mlir_interface::graph_capture
