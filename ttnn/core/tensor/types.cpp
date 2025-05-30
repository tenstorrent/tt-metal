// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_impl.hpp"

namespace tt::tt_metal {

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::DataType& data_type) {
    switch (data_type) {
        case DataType::BFLOAT16: return os << "DataType::BFLOAT16";
        case DataType::FLOAT32: return os << "DataType::FLOAT32";
        case DataType::UINT32: return os << "DataType::UINT32";
        case DataType::BFLOAT8_B: return os << "DataType::BFLOAT8_B";
        case DataType::BFLOAT4_B: return os << "DataType::BFLOAT4_B";
        case DataType::UINT8: return os << "DataType::UINT8";
        case DataType::UINT16: return os << "DataType::UINT16";
        case DataType::INT32: return os << "DataType::INT32";
        case DataType::INVALID:
        default: return os << "Invalid";
    }
}

bool is_floating_point(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16:
        case DataType::FLOAT32:
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: return true;
        default: return false;
    }
}

bool is_block_float(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: return true;
        default: return false;
    }
}

tt::DataFormat datatype_to_dataformat_converter(tt::tt_metal::DataType datatype) {
    switch (datatype) {
        case tt::tt_metal::DataType::BFLOAT16: return tt::DataFormat::Float16_b;
        case tt::tt_metal::DataType::BFLOAT8_B: return tt::DataFormat::Bfp8_b;
        case tt::tt_metal::DataType::BFLOAT4_B: return tt::DataFormat::Bfp4_b;
        case tt::tt_metal::DataType::FLOAT32: return tt::DataFormat::Float32;
        case tt::tt_metal::DataType::INT32: return tt::DataFormat::Int32;
        case tt::tt_metal::DataType::UINT32: return tt::DataFormat::UInt32;
        case tt::tt_metal::DataType::UINT16: return tt::DataFormat::UInt16;
        case tt::tt_metal::DataType::UINT8: return tt::DataFormat::UInt8;
        default: TT_ASSERT(false, "Unsupported DataType"); return tt::DataFormat::Float16_b;
    }
}

MemoryConfig::MemoryConfig(
    TensorMemoryLayout memory_layout, BufferType buffer_type, std::optional<ShardSpec> shard_spec) :
    memory_layout_(memory_layout), buffer_type_(buffer_type), shard_spec_(std::move(shard_spec)) {}

MemoryConfig::MemoryConfig(BufferType buffer_type, NdShardSpec nd_shard_spec) :
    memory_layout_(TensorMemoryLayout::BLOCK_SHARDED),
    buffer_type_(buffer_type),
    nd_shard_spec_(std::move(nd_shard_spec)),
    created_with_nd_shard_spec_(true) {}

MemoryConfig::MemoryConfig(
    TensorMemoryLayout memory_layout,
    BufferType buffer_type,
    std::optional<ShardSpec> shard_spec,
    std::optional<NdShardSpec> nd_shard_spec,
    bool created_with_nd_shard_spec) :
    memory_layout_(memory_layout),
    buffer_type_(buffer_type),
    shard_spec_(std::move(shard_spec)),
    nd_shard_spec_(std::move(nd_shard_spec)),
    created_with_nd_shard_spec_(created_with_nd_shard_spec) {}

MemoryConfig MemoryConfig::create_with_prepopulated_shard_specs(
    TensorMemoryLayout memory_layout,
    BufferType buffer_type,
    std::optional<ShardSpec> shard_spec,
    std::optional<NdShardSpec> nd_shard_spec,
    bool created_with_nd_shard_spec) {
    return MemoryConfig(
        memory_layout, buffer_type, std::move(shard_spec), std::move(nd_shard_spec), created_with_nd_shard_spec);
}

bool MemoryConfig::is_sharded() const {
    switch (this->memory_layout_) {
        case TensorMemoryLayout::HEIGHT_SHARDED:
        case TensorMemoryLayout::WIDTH_SHARDED:
        case TensorMemoryLayout::BLOCK_SHARDED: return true;
        default: return false;
    }
}

bool MemoryConfig::is_l1() const { return buffer_type_ == BufferType::L1 or buffer_type_ == BufferType::L1_SMALL; }

bool MemoryConfig::is_dram() const { return buffer_type_ == BufferType::DRAM; }

bool operator==(const MemoryConfig& config_a, const MemoryConfig& config_b) {
    return config_a.buffer_type() == config_b.buffer_type() && config_a.memory_layout() == config_b.memory_layout() &&
           config_a.shard_spec() == config_b.shard_spec();
}

bool operator!=(const MemoryConfig& config_a, const MemoryConfig& config_b) { return not(config_a == config_b); }

std::ostream& operator<<(std::ostream& os, const MemoryConfig& config) {
    tt::stl::reflection::operator<<(os, config);
    return os;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Layout& layout) {
    switch (layout) {
        case Layout::ROW_MAJOR: return os << "Layout::ROW_MAJOR";
        case Layout::TILE: return os << "Layout::TILE";
        case Layout::INVALID: return os << "Layout::INVALID";
        default: return os << "Layout::UNKNOWN";
    }
}

}  // namespace tt::tt_metal

nlohmann::json tt::stl::json::to_json_t<tt::tt_metal::MemoryConfig>::operator()(
    const tt::tt_metal::MemoryConfig& config) const {
    nlohmann::json json_object;
    json_object["memory_layout"] = config.memory_layout();
    json_object["buffer_type"] = config.buffer_type();
    json_object["created_with_nd_shard_spec"] = config.created_with_nd_shard_spec();
    if (config.created_with_nd_shard_spec()) {
        if (config.nd_shard_spec().has_value()) {
            json_object["nd_shard_spec"] = tt::stl::json::to_json(config.nd_shard_spec().value());
        }
    } else {
        if (config.shard_spec().has_value()) {
            json_object["shard_spec"] = tt::stl::json::to_json(config.shard_spec().value());
        }
    }
    return json_object;
}

tt::tt_metal::MemoryConfig tt::stl::json::from_json_t<tt::tt_metal::MemoryConfig>::operator()(
    const nlohmann::json& json_object) const {
    auto memory_layout = json_object["memory_layout"].get<tt::tt_metal::TensorMemoryLayout>();
    auto buffer_type = json_object["buffer_type"].get<tt::tt_metal::BufferType>();
    auto created_with_nd_shard_spec = json_object["created_with_nd_shard_spec"].get<bool>();
    if (created_with_nd_shard_spec) {
        auto nd_shard_spec = tt::stl::json::from_json<tt::tt_metal::NdShardSpec>(json_object["nd_shard_spec"]);
        return tt::tt_metal::MemoryConfig(buffer_type, std::move(nd_shard_spec));
    } else {
        std::optional<tt::tt_metal::ShardSpec> shard_spec;
        if (json_object.contains("shard_spec")) {
            shard_spec = tt::stl::json::from_json<tt::tt_metal::ShardSpec>(json_object["shard_spec"]);
        }
        return tt::tt_metal::MemoryConfig(memory_layout, buffer_type, std::move(shard_spec));
    }
}
