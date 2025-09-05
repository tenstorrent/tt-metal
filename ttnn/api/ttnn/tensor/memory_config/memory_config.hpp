// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/reflection.hpp>
#include <tt_stl/span.hpp>

#include "ttnn/tensor/types.hpp"

namespace tt {

namespace tt_metal {

class MemoryConfig final {
public:
    MemoryConfig() = default;  // Interleaved DRAM
    explicit MemoryConfig(
        TensorMemoryLayout memory_layout,
        BufferType buffer_type = BufferType::DRAM,
        std::optional<ShardSpec> shard_spec = std::nullopt);
    explicit MemoryConfig(BufferType buffer_type, std::optional<NdShardSpec> nd_shard_spec = std::nullopt);
    MemoryConfig(const MemoryConfig& other) = default;
    MemoryConfig& operator=(const MemoryConfig& other) = default;
    MemoryConfig(MemoryConfig&& other) noexcept = default;
    MemoryConfig& operator=(MemoryConfig&& other) noexcept = default;

    TensorMemoryLayout memory_layout() const { return memory_layout_; }
    BufferType buffer_type() const { return buffer_type_; }
    const std::optional<ShardSpec>& shard_spec() const { return shard_spec_; }
    const std::optional<NdShardSpec>& nd_shard_spec() const { return nd_shard_spec_; }
    bool created_with_nd_shard_spec() const { return created_with_nd_shard_spec_; }

    MemoryConfig with_shard_spec(std::optional<ShardSpec> shard_spec) const {
        return MemoryConfig(memory_layout_, buffer_type_, std::move(shard_spec));
    }

    bool is_sharded() const;
    bool is_l1() const;
    bool is_dram() const;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "memory_layout", "buffer_type", "shard_spec", "nd_shard_spec", "created_with_nd_shard_spec");
    auto attribute_values() const {
        return std::forward_as_tuple(
            memory_layout_, buffer_type_, shard_spec_, nd_shard_spec_, created_with_nd_shard_spec_);
    }

    static MemoryConfig create_with_prepopulated_shard_specs(
        TensorMemoryLayout memory_layout,
        BufferType buffer_type,
        std::optional<ShardSpec> shard_spec,
        std::optional<NdShardSpec> nd_shard_spec,
        bool created_with_nd_shard_spec);

    friend std::ostream& operator<<(std::ostream& os, const MemoryConfig& config);

private:
    MemoryConfig(
        TensorMemoryLayout memory_layout,
        BufferType buffer_type,
        std::optional<ShardSpec> shard_spec,
        std::optional<NdShardSpec> nd_shard_spec,
        bool created_with_nd_shard_spec);

    TensorMemoryLayout memory_layout_ = TensorMemoryLayout::INTERLEAVED;  // Interleave the data across multiple banks
    BufferType buffer_type_ = BufferType::DRAM;                           // Can be either DRAM or L1
    std::optional<ShardSpec> shard_spec_ = std::nullopt;
    std::optional<NdShardSpec> nd_shard_spec_ = std::nullopt;
    bool created_with_nd_shard_spec_ = false;
};

std::ostream& operator<<(std::ostream& os, const MemoryConfig& config);

bool operator==(const MemoryConfig& config_a, const MemoryConfig& config_b);
bool operator!=(const MemoryConfig& config_a, const MemoryConfig& config_b);

}  // namespace tt_metal
}  // namespace tt

template <>
struct ttsl::json::to_json_t<tt::tt_metal::MemoryConfig> {
    nlohmann::json operator()(const tt::tt_metal::MemoryConfig& config) const;
};

template <>
struct ttsl::json::from_json_t<tt::tt_metal::MemoryConfig> {
    tt::tt_metal::MemoryConfig operator()(const nlohmann::json& json_object) const;
};
