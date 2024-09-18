#pragma once

#include "third_party/json/json.hpp"
#include "ttnn/operations/common/l1_interface_common.hpp"
#include "ttnn/operations/eltwise/binary/binary_l1_interface.hpp"
#include "ttnn/operations/eltwise/unary/unary_l1_interface.hpp"
#include "ttnn/operations/matmul/device/matmul_types.hpp"
#include "ttnn/operations/matmul/matmul_l1_interface.hpp"
#include "ttnn/operations/normalization/softmax/softmax_l1_interface.hpp"

namespace ttnn::mlir_interface::graph_capture {
class GraphCaptureUnaryOpL1Usage : public UnaryOpL1Usage {
   public:
    GraphCaptureUnaryOpL1Usage(const L1InterfaceOperandParams& input, const L1InterfaceOperandParams& output);
    ~GraphCaptureUnaryOpL1Usage() = default;

    std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const override;
    std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const override;

   private:
    nlohmann::json m_json_trace;
};

class GraphCaptureEltwiseOpL1Usage : public EltwiseOpL1Usage {
   public:
    GraphCaptureEltwiseOpL1Usage(
        const L1InterfaceOperandParams& input_a,
        const L1InterfaceOperandParams& input_b,
        const L1InterfaceOperandParams& output);
    ~GraphCaptureEltwiseOpL1Usage() = default;

    std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const override;
    std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const override;

   private:
    nlohmann::json m_json_trace;
};

class GraphCaptureSoftmaxOpL1Usage : public SoftmaxOpL1Usage {
   public:
    GraphCaptureSoftmaxOpL1Usage(
        const L1InterfaceOperandParams& input_a,
        const int dim_arg,
        const std::optional<L1InterfaceOperandParams>& output);
    ~GraphCaptureSoftmaxOpL1Usage() = default;

    std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const override;
    std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const override;

   private:
    nlohmann::json m_json_trace;
};

class GraphCaptureMatmulOpL1Usage : public MatmulOpL1Usage {
   public:
    GraphCaptureMatmulOpL1Usage(
        const L1InterfaceOperandParams& input_a,
        const L1InterfaceOperandParams& input_b,
        const L1InterfaceOperandParams& output,
        const ttnn::operations::matmul::MatmulProgramConfig& program_config);
    ~GraphCaptureMatmulOpL1Usage() = default;

    std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const override;
    std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const override;

   private:
    ttnn::operations::matmul::MatmulProgramConfig m_program_config;
    nlohmann::json m_json_trace;
};
};  // namespace ttnn::mlir_interface::graph_capture
