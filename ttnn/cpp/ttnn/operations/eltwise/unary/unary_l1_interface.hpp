#pragma once

#include "ttnn/cpp/ttnn/operations/eltwise/common/eltwise_l1_interface_common.hpp"

class UnaryOpL1Usage {
   public:
    UnaryOpL1Usage(const EltwiseOpParams& input, const EltwiseOpParams& output);
    virtual ~UnaryOpL1Usage() = default;

    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const = 0;
    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const = 0;

   protected:
    EltwiseOpParams input;
    EltwiseOpParams output;
};

class InterleavedUnaryOpL1Usage : public UnaryOpL1Usage {
   public:
    InterleavedUnaryOpL1Usage(const EltwiseOpParams& input, const std::optional<EltwiseOpParams>& output);
    virtual ~InterleavedUnaryOpL1Usage() = default;

    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const override;
    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const override;
};

class ShardedUnaryOpL1Usage : public UnaryOpL1Usage {
   public:
    ShardedUnaryOpL1Usage(const EltwiseOpParams& input, const std::optional<EltwiseOpParams>& output);
    virtual ~ShardedUnaryOpL1Usage() = default;

    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const override;
    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const override;
};

class UnaryOpL1UsageFactory {
   public:
    UnaryOpL1UsageFactory() = delete;
    static std::unique_ptr<UnaryOpL1Usage> Make(
        const EltwiseOpParams& input, const std::optional<EltwiseOpParams>& output = std::nullopt);
};
