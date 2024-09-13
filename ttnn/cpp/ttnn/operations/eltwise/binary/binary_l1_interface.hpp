#pragma once

#include "ttnn/cpp/ttnn/operations/common/l1_interface_common.hpp"

class EltwiseOpL1Usage {
   public:
    EltwiseOpL1Usage(
        const L1InterfaceOperandParams& input_a, const L1InterfaceOperandParams& input_b, const L1InterfaceOperandParams& output);
    virtual ~EltwiseOpL1Usage() = default;

    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const = 0;
    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const = 0;

   protected:
    std::optional<L1InterfaceOperandParams> calculate_repeat_buffer_impl(
        const L1InterfaceOperandParams& input_a, const L1InterfaceOperandParams& input_b);

    std::optional<ShardSpec> get_op_shard_spec() const;

    L1InterfaceOperandParams input_a;
    L1InterfaceOperandParams input_b;
    L1InterfaceOperandParams output;
    std::optional<L1InterfaceOperandParams> repeat;
};

class ElementWiseMultiCoreOpL1Usage : public EltwiseOpL1Usage {
   public:
    ElementWiseMultiCoreOpL1Usage(
        const L1InterfaceOperandParams& input_a, const L1InterfaceOperandParams& input_b, const L1InterfaceOperandParams& output);
    virtual ~ElementWiseMultiCoreOpL1Usage() = default;

    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const override;
    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const override;
};

class BroadcastWidthMultiCoreOpL1Usage : public EltwiseOpL1Usage {
   public:
    BroadcastWidthMultiCoreOpL1Usage(
        const L1InterfaceOperandParams& input_a, const L1InterfaceOperandParams& input_b, const L1InterfaceOperandParams& output);
    virtual ~BroadcastWidthMultiCoreOpL1Usage() = default;

    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const override;
    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const override;
};

class EltwiseOpL1UsageFactory {
   public:
    EltwiseOpL1UsageFactory() = delete;
    static std::unique_ptr<EltwiseOpL1Usage> Make(
        const L1InterfaceOperandParams& input_a, const L1InterfaceOperandParams& input_b, const L1InterfaceOperandParams& output);
};
