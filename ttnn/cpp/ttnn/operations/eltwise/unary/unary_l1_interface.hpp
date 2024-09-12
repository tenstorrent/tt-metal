#pragma once

#include "ttnn/cpp/ttnn/operations/common/l1_interface_common.hpp"

class UnaryOpL1Usage {
   public:
    UnaryOpL1Usage(const L1InterfaceOperandParams& input, const L1InterfaceOperandParams& output);
    virtual ~UnaryOpL1Usage() = default;

    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const = 0;
    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const = 0;

   protected:
    L1InterfaceOperandParams input;
    L1InterfaceOperandParams output;
};

class InterleavedUnaryOpL1Usage : public UnaryOpL1Usage {
   public:
    InterleavedUnaryOpL1Usage(const L1InterfaceOperandParams& input, const std::optional<L1InterfaceOperandParams>& output);
    virtual ~InterleavedUnaryOpL1Usage() = default;

    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const override;
    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const override;
};

class ShardedUnaryOpL1Usage : public UnaryOpL1Usage {
   public:
    ShardedUnaryOpL1Usage(const L1InterfaceOperandParams& input, const std::optional<L1InterfaceOperandParams>& output);
    virtual ~ShardedUnaryOpL1Usage() = default;

    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const override;
    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const override;
};

class UnaryOpL1UsageFactory {
   public:
    UnaryOpL1UsageFactory() = delete;
    static std::unique_ptr<UnaryOpL1Usage> Make(
        const L1InterfaceOperandParams& input, const std::optional<L1InterfaceOperandParams>& output = std::nullopt);
};
