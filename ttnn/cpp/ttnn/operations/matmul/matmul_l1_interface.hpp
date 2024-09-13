#pragma once

#include "ttnn/cpp/ttnn/operations/common/l1_interface_common.hpp"
#include "ttnn/cpp/ttnn/operations/matmul/device/matmul_types.hpp"

class MatmulOPL1Usage {
   public:
    MatmulOPL1Usage(
        const L1InterfaceOperandParams& input_a,
        const L1InterfaceOperandParams& input_b,
        const L1InterfaceOperandParams& output);
    virtual ~MatmulOPL1Usage() = default;

    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const = 0;
    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const = 0;

   protected:
    L1InterfaceOperandParams input_a;
    L1InterfaceOperandParams input_b;
    L1InterfaceOperandParams output;
};

class MatmulMultiCoreReuseMultiCastOpL1Usage : public MatmulOPL1Usage {
   public:
    MatmulMultiCoreReuseMultiCastOpL1Usage(
        const L1InterfaceOperandParams& input_a,
        const L1InterfaceOperandParams& input_b,
        const L1InterfaceOperandParams& output,
        const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig& program_config);
    virtual ~MatmulMultiCoreReuseMultiCastOpL1Usage() = default;

    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const override;
    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const override;

   protected:
    ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig program_config;
};

class MatmulMultiCoreReuseMultiCast1DOpL1Usage : public MatmulOPL1Usage {
   public:
    MatmulMultiCoreReuseMultiCast1DOpL1Usage(
        const L1InterfaceOperandParams& input_a,
        const L1InterfaceOperandParams& input_b,
        const L1InterfaceOperandParams& output,
        const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config);
    virtual ~MatmulMultiCoreReuseMultiCast1DOpL1Usage() = default;

    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const override;
    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const override;

   protected:
    std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core_mcast_in1(
        const uint32_t B, const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t num_blocks) const;
    std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core_mcast_in0(
        const uint32_t B, const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t num_blocks) const;

    ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig program_config;
};

class MatmulOpL1UsageFactory {
   public:
    MatmulOpL1UsageFactory() = delete;
    static std::unique_ptr<MatmulOPL1Usage> Make(
        const L1InterfaceOperandParams& input_a,
        const L1InterfaceOperandParams& input_b,
        const L1InterfaceOperandParams& output,
        const ttnn::operations::matmul::MatmulProgramConfig& program_config);
};
