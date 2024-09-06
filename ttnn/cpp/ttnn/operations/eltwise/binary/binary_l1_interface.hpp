#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>
#include "impl/buffers/buffer.hpp"
#include "ttnn/tensor/types.hpp"

// forward declarations
namespace tt {
namespace tt_metal {
enum class DataType;
enum class Layout;
struct MemoryConfig;
struct ShardSpec;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn {
namespace types {
struct Shape;
}
}  // namespace ttnn

using EltwiseOpParams =
    std::tuple<ttnn::types::Shape, tt::tt_metal::DataType, tt::tt_metal::Layout, tt::tt_metal::MemoryConfig>;

// these should go to a common tensor_l1_interface header or something
uint32_t calculate_circular_buffer_l1_allocation_size_per_core(
    const ttnn::types::Shape& shape,
    const tt::tt_metal::DataType data_type,
    const tt::tt_metal::Layout& layout,
    const tt::tt_metal::MemoryConfig& memory_config,
    const uint32_t max_block_size);

inline uint32_t calculate_circular_buffer_l1_allocation_size_per_core(EltwiseOpParams input, uint32_t max_block_size) {
    return calculate_circular_buffer_l1_allocation_size_per_core(
        std::get<ttnn::types::Shape>(input),
        std::get<tt::tt_metal::DataType>(input),
        std::get<tt::tt_metal::Layout>(input),
        std::get<tt::tt_metal::MemoryConfig>(input),
        max_block_size);
}

uint32_t calculate_tensor_l1_allocation_size_per_core(
    const ttnn::types::Shape& shape,
    const tt::tt_metal::DataType data_type,
    const tt::tt_metal::Layout& layout,
    const tt::tt_metal::MemoryConfig& memory_config);

inline uint32_t calculate_tensor_l1_allocation_size_per_core(EltwiseOpParams input) {
    return calculate_tensor_l1_allocation_size_per_core(
        std::get<ttnn::types::Shape>(input),
        std::get<tt::tt_metal::DataType>(input),
        std::get<tt::tt_metal::Layout>(input),
        std::get<tt::tt_metal::MemoryConfig>(input));
}

uint32_t get_num_of_cores(const std::optional<tt::tt_metal::ShardSpec>& shard_spec);

uint32_t get_num_pages(const tt::tt_metal::ShardSpec& shard_spec);

uint32_t calculate_repeat_circular_buffer_size(tt::tt_metal::DataType data_type);

uint32_t calculate_max_block_size(const std::optional<tt::tt_metal::ShardSpec>& shard_spec);

class EltwiseOpL1Usage {
   public:
    EltwiseOpL1Usage(const EltwiseOpParams& input_a, const EltwiseOpParams& input_b, const EltwiseOpParams& output);
    virtual ~EltwiseOpL1Usage() = default;

    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const = 0;
    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const = 0;

   protected:
    std::optional<EltwiseOpParams> calculate_repeat_buffer_impl(
        const EltwiseOpParams& input_a, const EltwiseOpParams& input_b);

    std::optional<ShardSpec> get_op_shard_spec() const;

    EltwiseOpParams input_a;
    EltwiseOpParams input_b;
    EltwiseOpParams output;
    std::optional<EltwiseOpParams> repeat;
};

class ElementWiseMultiCoreOpL1Usage : public EltwiseOpL1Usage {
   public:
    ElementWiseMultiCoreOpL1Usage(
        const EltwiseOpParams& input_a, const EltwiseOpParams& input_b, const EltwiseOpParams& output);
    virtual ~ElementWiseMultiCoreOpL1Usage() = default;

    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const override;
    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const override;
};

class BroadcastWidthMultiCoreOpL1Usage : public EltwiseOpL1Usage {
   public:
    BroadcastWidthMultiCoreOpL1Usage(
        const EltwiseOpParams& input_a, const EltwiseOpParams& input_b, const EltwiseOpParams& output);
    virtual ~BroadcastWidthMultiCoreOpL1Usage() = default;

    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const override;
    virtual std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const override;
};

class EltwiseOpL1UsageFactory {
   public:
    EltwiseOpL1UsageFactory() = delete;
    static std::unique_ptr<EltwiseOpL1Usage> Make(
        const EltwiseOpParams& input_a, const EltwiseOpParams& input_b, const EltwiseOpParams& output);
};
