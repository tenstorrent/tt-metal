#pragma once

#include <cstdint>
#include <iostream>
#include <optional>
#include <tuple>
#include <vector>

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

using L1InterfaceOpParams =
    std::tuple<ttnn::types::Shape, tt::tt_metal::DataType, tt::tt_metal::Layout, tt::tt_metal::MemoryConfig>;

uint32_t calculate_circular_buffer_l1_allocation_size_per_core(
    const ttnn::types::Shape& shape,
    const tt::tt_metal::DataType data_type,
    const tt::tt_metal::Layout& layout,
    const tt::tt_metal::MemoryConfig& memory_config,
    const uint32_t max_block_size);

inline uint32_t calculate_circular_buffer_l1_allocation_size_per_core(
    L1InterfaceOpParams input, uint32_t max_block_size) {
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

inline uint32_t calculate_tensor_l1_allocation_size_per_core(L1InterfaceOpParams input) {
    return calculate_tensor_l1_allocation_size_per_core(
        std::get<ttnn::types::Shape>(input),
        std::get<tt::tt_metal::DataType>(input),
        std::get<tt::tt_metal::Layout>(input),
        std::get<tt::tt_metal::MemoryConfig>(input));
}

uint32_t get_num_of_cores(const std::optional<tt::tt_metal::ShardSpec>& shard_spec = std::nullopt);

uint32_t get_num_pages(const tt::tt_metal::ShardSpec& shard_spec);

uint32_t calculate_repeat_circular_buffer_size(tt::tt_metal::DataType data_type);

uint32_t calculate_max_block_size(const std::optional<tt::tt_metal::ShardSpec>& shard_spec);
