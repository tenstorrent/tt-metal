// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/tensor/types.hpp"

namespace tt {

namespace tt_metal {

static DistributedTensorConfig create_shard_distributed_tensor_config(const std::unordered_map<std::string, std::string>& metadata) {
    return ShardTensor(std::stoi(metadata.at("shard_dim")));
}
static DistributedTensorConfig create_shard_2d_distributed_tensor_config(const std::unordered_map<std::string, std::string>& metadata) {
    return ShardTensor2D(ShardMesh(std::stoi(metadata.at("mesh_shape_y")), std::stoi(metadata.at("mesh_shape_x"))));
}
static DistributedTensorConfig create_replicate_distributed_tensor_config(const std::unordered_map<std::string, std::string>& metadata) {
    if (auto it = metadata.find("replication_factor"); it != metadata.end()) {
        return ReplicateTensor(std::stoi(it->second));
    }
    TT_THROW("Unsupported Replication strategy:");
}

DistributedTensorConfig get_distributed_tensor_config(const std::unordered_map<std::string, std::string>& metadata) {
    if (auto it = metadata.find("strategy"); it != metadata.end()) {
        const std::string& strategy = it->second;
        if (strategy == "shard") {
            return create_shard_distributed_tensor_config(metadata);
        } else if (strategy == "shard_2d") {
            return create_shard_2d_distributed_tensor_config(metadata);
        } else if (strategy == "replicate") {
            return create_replicate_distributed_tensor_config(metadata);
        }
    }
    TT_THROW("Unsupported DistributedTensorConfig strategy:");
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

Padding::Padding(const std::size_t rank) : rank_{rank}, pad_dimensions_{}, pad_value_{} {}

Padding::Padding(const std::initializer_list<PadDimension> pad_dimensions, PadValue pad_value) :
    rank_(pad_dimensions.size()), pad_dimensions_{}, pad_value_(pad_value) {
    std::copy(std::begin(pad_dimensions), std::end(pad_dimensions), std::begin(this->pad_dimensions_));
}

Padding::Padding(const std::vector<PadDimension>& pad_dimensions, PadValue pad_value) :
    rank_(pad_dimensions.size()), pad_dimensions_{}, pad_value_(pad_value) {
    std::copy(std::begin(pad_dimensions), std::end(pad_dimensions), std::begin(this->pad_dimensions_));
}

const uint32_t Padding::get_normalized_index(std::int64_t index) const {
    std::int64_t rank = static_cast<std::int64_t>(this->rank_);
    std::uint64_t normalized_index = index >= 0 ? index : rank + index;
    TT_ASSERT(
        normalized_index >= 0 and normalized_index < rank,
        fmt::format(
            "Index is out of bounds for the rank, should be between 0 and {} however is {}",
            rank - 1,
            normalized_index));
    return normalized_index;
}

Padding::PadDimension& Padding::operator[](const std::int64_t index) {
    auto normalized_index = this->get_normalized_index(index);
    return this->pad_dimensions_[normalized_index];
}

const Padding::PadDimension Padding::operator[](const std::int64_t index) const {
    auto normalized_index = this->get_normalized_index(index);
    return this->pad_dimensions_[normalized_index];
}

Padding::PadValue Padding::pad_value() const { return this->pad_value_; }

bool operator==(const Padding& padding_a, const Padding& padding_b) {
    if (padding_a.rank_ != padding_b.rank_) {
        return false;
    }
    for (auto index = 0; index < padding_a.rank_; index++) {
        if (padding_a[index].front != padding_b[index].front) {
            return false;
        }

        if (padding_a[index].back != padding_b[index].back) {
            return false;
        }
    }
    return padding_a.pad_value_ == padding_b.pad_value_;
}

bool operator!=(const Padding& padding_a, const Padding& padding_b) { return not(padding_a == padding_b); }

Shape::Shape(const std::initializer_list<uint32_t> dimensions) :
    rank_(dimensions.size()), dimensions_{}, padding_(dimensions.size()) {
    std::copy(std::begin(dimensions), std::end(dimensions), std::begin(this->dimensions_));
    validate();
}
Shape::Shape(const std::vector<uint32_t>& dimensions) :
    rank_(dimensions.size()), dimensions_{}, padding_(dimensions.size()) {
    std::copy(std::begin(dimensions), std::end(dimensions), std::begin(this->dimensions_));
    validate();
}

Shape::Shape(const std::initializer_list<uint32_t> dimensions, const Padding& padding) :
    rank_(dimensions.size()), dimensions_{}, padding_(padding) {
    TT_ASSERT(this->padding_.rank_ == this->rank_);
    std::copy(std::begin(dimensions), std::end(dimensions), std::begin(this->dimensions_));
    validate();
}
Shape::Shape(const std::vector<uint32_t>& dimensions, const Padding& padding) :
    rank_(dimensions.size()), dimensions_{}, padding_(padding) {
    TT_ASSERT(this->padding_.rank_ == this->rank_);
    std::copy(std::begin(dimensions), std::end(dimensions), std::begin(this->dimensions_));
    validate();
}

Shape::Shape(const Shape& other, const Padding& padding) :
    dimensions_(other.dimensions_), rank_(other.rank_), padding_(padding) {
    TT_ASSERT(this->padding_.rank_ == this->rank_);
}

void Shape::validate() const {
    TT_FATAL(this->rank_ > 0, "Shape rank == 0 is not supported, {}", *this);
    for (auto index = 0; index < this->rank_; index++) {
        TT_FATAL(
            this->dimensions_[index] >= 0,
            fmt::format("Shape can't have negative dimensions, however dimension {} is {}, {}", index, this->dimensions_[index], *this));
    }
}

std::size_t Shape::rank() const { return this->rank_; }
std::size_t Shape::size() const { return this->rank_; }

uint32_t& Shape::operator[](const std::int64_t index) {
    auto normalized_index = this->get_normalized_index(index);
    return this->dimensions_[normalized_index];
}
const uint32_t Shape::operator[](const std::int64_t index) const {
    auto normalized_index = this->get_normalized_index(index);
    return this->dimensions_[normalized_index];
}

const uint32_t* Shape::begin() const { return this->dimensions_.data(); }
const uint32_t* Shape::end() const { return this->dimensions_.data() + this->rank_; }

const Padding& Shape::padding() const {
    return this->padding_;
}

const Shape Shape::without_padding() const {
    auto padding = this->padding_;
    std::vector<std::uint32_t> shape_without_padding;
    for (auto index = 0; index < this->rank(); index++) {
        const auto dimension = this->operator[](index);
        auto&& [front_pad, back_pad] = padding.pad_dimensions_[index];
        const auto new_dimension = dimension - (front_pad + back_pad);
        shape_without_padding.push_back(new_dimension);
    }
    return Shape(shape_without_padding);
}

const uint32_t Shape::get_normalized_index(std::int64_t index) const {
    std::int64_t rank = static_cast<std::int64_t>(this->rank_);
    std::uint64_t normalized_index = index >= 0 ? index : rank + index;
    TT_ASSERT(
        normalized_index >= 0 and normalized_index < rank,
        fmt::format(
            "Index is out of bounds for the rank, should be between 0 and {} however is {}",
            rank - 1,
            normalized_index));
    return normalized_index;
}

bool operator==(const ReplicateTensor& a, const ReplicateTensor& b) {
    return a.replication_factor == b.replication_factor; // All instances are considered equal because there are no data members.
}
bool operator==(const AllGatherTensor&, const AllGatherTensor&) {
    return true; // All instances are considered equal because there are no data members.
}
bool operator==(const ShardTensor& lhs, const ShardTensor& rhs) {
    return lhs.shard_dimension == rhs.shard_dimension; // Equal if they have the same shard_dimension.
}
bool operator==(const ShardTensor2D& lhs, const ShardTensor2D& rhs) {
    return lhs.shard_mesh == rhs.shard_mesh; // Equal if they have the same shard_mesh.
}

bool operator==(const Shape& shape_a, const Shape& shape_b) {
    if (shape_a.rank() != shape_b.rank()) {
        return false;
    }
    for (auto index = 0; index < shape_a.rank(); index++) {
        if (shape_a[index] != shape_b[index]) {
            return false;
        }
    }
    // TODO:(arakhmati): should the padding be ignored?
    return true;  // Ignore the padding when comparing shapes
}

bool operator!=(const Shape& shape_a, const Shape& shape_b) { return not(shape_a == shape_b); }

bool MemoryConfig::is_sharded() const {
    switch (this->memory_layout) {
        case TensorMemoryLayout::HEIGHT_SHARDED:
        case TensorMemoryLayout::WIDTH_SHARDED:
        case TensorMemoryLayout::BLOCK_SHARDED: return true;
        default: return false;
    }
}

bool MemoryConfig::is_l1() const { return buffer_type == BufferType::L1 or buffer_type == BufferType::L1_SMALL; }

bool MemoryConfig::is_dram() const { return buffer_type == BufferType::DRAM; }

bool operator==(const MemoryConfig& config_a, const MemoryConfig& config_b) {
    return config_a.buffer_type == config_b.buffer_type && config_a.memory_layout == config_b.memory_layout && config_a.shard_spec == config_b.shard_spec;
}

bool operator!=(const MemoryConfig& config_a, const MemoryConfig& config_b) { return not(config_a == config_b); }

void dump_memory_config(std::ostream& output_stream, const MemoryConfig& memory_config) {
    output_stream.write(reinterpret_cast<const char*>(&VERSION_ID), sizeof(std::uint8_t));
    output_stream.write(reinterpret_cast<const char*>(&memory_config.memory_layout), sizeof(TensorMemoryLayout));
    output_stream.write(reinterpret_cast<const char*>(&memory_config.buffer_type), sizeof(BufferType));

    bool has_shard_spec = memory_config.shard_spec.has_value();
    output_stream.write(reinterpret_cast<const char*>(&has_shard_spec), sizeof(bool));
    if (has_shard_spec) {
        const auto& shard_spec = memory_config.shard_spec.value();
        const auto& core_ranges = shard_spec.grid.ranges();
        std::size_t num_core_ranges = core_ranges.size();
        output_stream.write(reinterpret_cast<const char*>(&num_core_ranges), sizeof(std::size_t));
        for (const auto& core_range : core_ranges) {
            output_stream.write(reinterpret_cast<const char*>(&core_range), sizeof(CoreRange));
        }
        output_stream.write(reinterpret_cast<const char*>(&shard_spec.shape), sizeof(std::array<uint32_t, 2>));
        output_stream.write(reinterpret_cast<const char*>(&shard_spec.orientation), sizeof(ShardOrientation));
        output_stream.write(reinterpret_cast<const char*>(&shard_spec.halo), sizeof(bool));
    }
}

void dump_memory_config(const std::string& file_name, const MemoryConfig& memory_config) {
    std::ofstream output_stream(file_name, std::ios::out | std::ios::binary);
    if (not output_stream) {
        throw std::runtime_error(fmt::format("Cannot open \"{}\"", file_name));
    }
    dump_memory_config(output_stream, memory_config);
}

MemoryConfig load_memory_config(std::ifstream& input_stream) {
    std::uint8_t version_id;
    TensorMemoryLayout memory_layout;
    BufferType buffer_type;
    bool has_shard_spec;
    input_stream.read(reinterpret_cast<char*>(&version_id), sizeof(std::uint8_t));
    if (version_id != VERSION_ID) {
        throw std::runtime_error(fmt::format("Unsupported version_id: {}", version_id));
    }
    input_stream.read(reinterpret_cast<char*>(&memory_layout), sizeof(TensorMemoryLayout));
    input_stream.read(reinterpret_cast<char*>(&buffer_type), sizeof(BufferType));
    input_stream.read(reinterpret_cast<char*>(&has_shard_spec), sizeof(bool));

    std::optional<ShardSpec> shard_spec = std::nullopt;
    if (has_shard_spec) {
        std::size_t num_core_ranges;
        std::set<CoreRange> core_ranges;
        std::array<uint32_t, 2> shape;
        ShardOrientation orientation;
        bool halo;

        input_stream.read(reinterpret_cast<char*>(&num_core_ranges), sizeof(std::size_t));
        for (auto index = 0; index < num_core_ranges; index++) {
            CoreRange core_range{{}, {}};
            input_stream.read(reinterpret_cast<char*>(&core_range), sizeof(CoreRange));
            core_ranges.insert(core_range);
        }
        input_stream.read(reinterpret_cast<char*>(&shape), sizeof(std::array<uint32_t, 2>));
        input_stream.read(reinterpret_cast<char*>(&orientation), sizeof(ShardOrientation));
        input_stream.read(reinterpret_cast<char*>(&halo), sizeof(bool));
        shard_spec = {CoreRangeSet{core_ranges}, shape, orientation, halo};
    }
    return MemoryConfig{memory_layout, buffer_type, shard_spec};
}

MemoryConfig load_memory_config(const std::string& file_name) {
    std::ifstream input_stream(file_name, std::ios::in | std::ios::binary);
    if (not input_stream) {
        throw std::runtime_error(fmt::format("Cannot open \"{}\"", file_name));
    }
    return load_memory_config(input_stream);
}

}  // namespace tt_metal

}  // namespace tt
