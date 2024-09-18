#pragma once

#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

#include "ttnn/tensor/types.hpp"

// forward declarations
namespace tt {
namespace tt_metal {
enum class DataType;
enum class Layout;
enum class TensorMemoryLayout;
struct MemoryConfig;
struct ShardSpec;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn {
namespace types {
struct Shape;
}
}  // namespace ttnn

constexpr static uint32_t c_cb_shares_space_with_sharded_operand = (uint32_t)0;

using L1InterfaceOperandParams =
    std::tuple<ttnn::types::Shape, tt::tt_metal::DataType, tt::tt_metal::Layout, tt::tt_metal::MemoryConfig>;

uint32_t calculate_circular_buffer_l1_allocation_size_per_core(
    const ttnn::types::Shape& shape,
    const tt::tt_metal::DataType data_type,
    const tt::tt_metal::Layout& layout,
    const tt::tt_metal::MemoryConfig& memory_config,
    const uint32_t max_block_size);

inline uint32_t calculate_circular_buffer_l1_allocation_size_per_core(
    L1InterfaceOperandParams operand, uint32_t max_block_size) {
    return calculate_circular_buffer_l1_allocation_size_per_core(
        std::get<ttnn::types::Shape>(operand),
        std::get<tt::tt_metal::DataType>(operand),
        std::get<tt::tt_metal::Layout>(operand),
        std::get<tt::tt_metal::MemoryConfig>(operand),
        max_block_size);
}

uint32_t calculate_tensor_l1_allocation_size_per_core(
    const ttnn::types::Shape& shape,
    const tt::tt_metal::DataType data_type,
    const tt::tt_metal::Layout& layout,
    const tt::tt_metal::MemoryConfig& memory_config);

inline uint32_t calculate_tensor_l1_allocation_size_per_core(const L1InterfaceOperandParams& operand) {
    return calculate_tensor_l1_allocation_size_per_core(
        std::get<ttnn::types::Shape>(operand),
        std::get<tt::tt_metal::DataType>(operand),
        std::get<tt::tt_metal::Layout>(operand),
        std::get<tt::tt_metal::MemoryConfig>(operand));
}

uint32_t get_num_of_cores(const std::optional<tt::tt_metal::ShardSpec>& shard_spec = std::nullopt);

uint32_t get_num_pages(const tt::tt_metal::ShardSpec& shard_spec);

uint32_t calculate_repeat_circular_buffer_size(tt::tt_metal::DataType data_type);

uint32_t calculate_max_block_size(const std::optional<tt::tt_metal::ShardSpec>& shard_spec);

bool is_sharded(const L1InterfaceOperandParams& operand);

uint32_t get_tile_size(const L1InterfaceOperandParams& operand);

tt::tt_metal::Shape get_legacy_shape(const L1InterfaceOperandParams& operand);

bool has_layout(const L1InterfaceOperandParams& operand, TensorMemoryLayout layout);

std::optional<tt::tt_metal::ShardSpec> get_shard_spec(const L1InterfaceOperandParams& operand);
