#pragma once

#include <memory>
#include <optional>

#include "ttnn/operations/common/l1_interface_common.hpp"
#include "ttnn/operations/matmul/device/matmul_types.hpp"

class UnaryOpL1Usage;
class EltwiseOpL1Usage;
class SoftmaxOpL1Usage;
class MatmulOpL1Usage;

namespace ttnn::mlir_interface {

bool is_graph_capture_mode_enabled();

class OpL1UsageFactory {
   public:
    virtual ~OpL1UsageFactory() = default;

    virtual std::unique_ptr<UnaryOpL1Usage> get_unary_op_l1_usage(
        const L1InterfaceOperandParams& input,
        const std::optional<L1InterfaceOperandParams>& output = std::nullopt) const = 0;

    virtual std::unique_ptr<EltwiseOpL1Usage> get_eltwise_op_l1_usage(
        const L1InterfaceOperandParams& input_a,
        const L1InterfaceOperandParams& input_b,
        const L1InterfaceOperandParams& output) const = 0;

    virtual std::unique_ptr<SoftmaxOpL1Usage> get_softmax_op_l1_usage(
        const L1InterfaceOperandParams& input,
        const int dim_arg,
        const std::optional<L1InterfaceOperandParams>& output = std::nullopt) const = 0;

    virtual std::unique_ptr<MatmulOpL1Usage> get_matmul_op_l1_usage(
        const L1InterfaceOperandParams& input_a,
        const L1InterfaceOperandParams& input_b,
        const L1InterfaceOperandParams& output,
        const ttnn::operations::matmul::MatmulProgramConfig& program_config) const = 0;
};

class AnalyticalOpL1UsageFactory : public OpL1UsageFactory {
   public:
    ~AnalyticalOpL1UsageFactory() = default;

    std::unique_ptr<UnaryOpL1Usage> get_unary_op_l1_usage(
        const L1InterfaceOperandParams& input,
        const std::optional<L1InterfaceOperandParams>& output = std::nullopt) const override;

    std::unique_ptr<EltwiseOpL1Usage> get_eltwise_op_l1_usage(
        const L1InterfaceOperandParams& input_a,
        const L1InterfaceOperandParams& input_b,
        const L1InterfaceOperandParams& output) const override;

    std::unique_ptr<SoftmaxOpL1Usage> get_softmax_op_l1_usage(
        const L1InterfaceOperandParams& input,
        const int dim_arg,
        const std::optional<L1InterfaceOperandParams>& output = std::nullopt) const override;

    std::unique_ptr<MatmulOpL1Usage> get_matmul_op_l1_usage(
        const L1InterfaceOperandParams& input_a,
        const L1InterfaceOperandParams& input_b,
        const L1InterfaceOperandParams& output,
        const ttnn::operations::matmul::MatmulProgramConfig& program_config) const override;
};

class GraphCaptureOpL1UsageFactory : public OpL1UsageFactory {
   public:
    ~GraphCaptureOpL1UsageFactory() = default;

    std::unique_ptr<UnaryOpL1Usage> get_unary_op_l1_usage(
        const L1InterfaceOperandParams& input,
        const std::optional<L1InterfaceOperandParams>& output = std::nullopt) const override;

    std::unique_ptr<EltwiseOpL1Usage> get_eltwise_op_l1_usage(
        const L1InterfaceOperandParams& input_a,
        const L1InterfaceOperandParams& input_b,
        const L1InterfaceOperandParams& output) const override;

    std::unique_ptr<SoftmaxOpL1Usage> get_softmax_op_l1_usage(
        const L1InterfaceOperandParams& input,
        const int dim_arg,
        const std::optional<L1InterfaceOperandParams>& output = std::nullopt) const override;

    std::unique_ptr<MatmulOpL1Usage> get_matmul_op_l1_usage(
        const L1InterfaceOperandParams& input_a,
        const L1InterfaceOperandParams& input_b,
        const L1InterfaceOperandParams& output,
        const ttnn::operations::matmul::MatmulProgramConfig& program_config) const override;
};

class OpL1UsageAbstractFactory {
   public:
    OpL1UsageAbstractFactory() = delete;
    static std::unique_ptr<OpL1UsageFactory> Make();
};
};  // namespace ttnn::mlir_interface
