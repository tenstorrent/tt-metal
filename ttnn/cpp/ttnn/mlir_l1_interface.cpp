#include "mlir_l1_interface.hpp"

#include <memory>

#include "common/env_lib.hpp"
#include "ttnn/mlir_graph_capture_l1_interface.hpp"
#include "ttnn/operations/eltwise/binary/binary_l1_interface.hpp"
#include "ttnn/operations/eltwise/unary/unary_l1_interface.hpp"
#include "ttnn/operations/matmul/matmul_l1_interface.hpp"
#include "ttnn/operations/normalization/softmax/softmax_l1_interface.hpp"

namespace ttnn::mlir_interface {

bool is_graph_capture_mode_enabled() {
    static const bool graph_capture_enabled = tt::parse_env("TTNN_MLIR_INTERFACE_USE_GRAPH_CAPTURE", false);

    return graph_capture_enabled;
}

std::unique_ptr<OpL1UsageFactory> OpL1UsageAbstractFactory::Make() {
    if (is_graph_capture_mode_enabled()) {
        return std::make_unique<GraphCaptureOpL1UsageFactory>();
    } else {
        return std::make_unique<AnalyticalOpL1UsageFactory>();
    }
}

// ========= Analytical solution ==========
std::unique_ptr<UnaryOpL1Usage> AnalyticalOpL1UsageFactory::get_unary_op_l1_usage(
    const L1InterfaceOperandParams& input, const std::optional<L1InterfaceOperandParams>& output) const {
    return UnaryOpL1UsageFactory::Make(input, output);
}

std::unique_ptr<EltwiseOpL1Usage> AnalyticalOpL1UsageFactory::get_eltwise_op_l1_usage(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output) const {
    return EltwiseOpL1UsageFactory::Make(input_a, input_b, output);
}

std::unique_ptr<SoftmaxOpL1Usage> AnalyticalOpL1UsageFactory::get_softmax_op_l1_usage(
    const L1InterfaceOperandParams& input,
    const int dim_arg,
    const std::optional<L1InterfaceOperandParams>& output) const {
    return SoftmaxOpL1UsageFactory::Make(input, dim_arg, output);
}

std::unique_ptr<MatmulOpL1Usage> AnalyticalOpL1UsageFactory::get_matmul_op_l1_usage(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output,
    const ttnn::operations::matmul::MatmulProgramConfig& program_config) const {
    return MatmulOpL1UsageFactory::Make(input_a, input_b, output, program_config);
}

// ========= Graph capture solution ==========
std::unique_ptr<UnaryOpL1Usage> GraphCaptureOpL1UsageFactory::get_unary_op_l1_usage(
    const L1InterfaceOperandParams& input, const std::optional<L1InterfaceOperandParams>& output) const {
    return std::make_unique<graph_capture::GraphCaptureUnaryOpL1Usage>(input, output.value_or(input));
}

std::unique_ptr<EltwiseOpL1Usage> GraphCaptureOpL1UsageFactory::get_eltwise_op_l1_usage(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output) const {
    return std::make_unique<graph_capture::GraphCaptureEltwiseOpL1Usage>(input_a, input_b, output);
}

std::unique_ptr<SoftmaxOpL1Usage> GraphCaptureOpL1UsageFactory::get_softmax_op_l1_usage(
    const L1InterfaceOperandParams& input,
    const int dim_arg,
    const std::optional<L1InterfaceOperandParams>& output) const {
    return std::make_unique<graph_capture::GraphCaptureSoftmaxOpL1Usage>(input, dim_arg, output);
}

std::unique_ptr<MatmulOpL1Usage> GraphCaptureOpL1UsageFactory::get_matmul_op_l1_usage(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output,
    const ttnn::operations::matmul::MatmulProgramConfig& program_config) const {
    return std::make_unique<graph_capture::GraphCaptureMatmulOpL1Usage>(input_a, input_b, output, program_config);
}
};  // namespace ttnn::mlir_interface
