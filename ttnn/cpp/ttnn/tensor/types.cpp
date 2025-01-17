// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_impl.hpp"

namespace ttnn::types {

const Shape Shape::to_rank(size_t new_rank) const {
    auto padded_shape = value;
    auto shape = value.without_padding();

    SmallVector<uint32_t> new_shape(new_rank, 1);
    SmallVector<uint32_t> new_padded_shape(new_rank, 1);

    int cur_idx = static_cast<int>(rank()) - 1;
    int new_idx = static_cast<int>(new_rank) - 1;
    for (; cur_idx >= 0 && new_idx >= 0; cur_idx--, new_idx--) {
        new_shape[new_idx] = shape[cur_idx];
        new_padded_shape[new_idx] = padded_shape[cur_idx];
    }
    for (; cur_idx >= 0; cur_idx--) {
        TT_FATAL(shape[cur_idx] == 1, "Can't convert shape rank");
        TT_FATAL(padded_shape[cur_idx] == 1, "Can't convert shape rank");
    }

    return Shape(std::move(new_shape), std::move(new_padded_shape));
}

}  // namespace ttnn::types

namespace tt::tt_metal {

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

Padding::Padding(const std::size_t rank) : rank_{rank}, pad_dimensions_{}, pad_value_{} {}

Padding::Padding(const std::initializer_list<PadDimension> pad_dimensions, PadValue pad_value) :
    rank_(pad_dimensions.size()), pad_dimensions_{}, pad_value_(pad_value) {
    std::copy(std::begin(pad_dimensions), std::end(pad_dimensions), std::begin(this->pad_dimensions_));
}

Padding::Padding(tt::stl::Span<const PadDimension> pad_dimensions, PadValue pad_value) :
    rank_(pad_dimensions.size()), pad_dimensions_{}, pad_value_(pad_value) {
    std::copy(std::begin(pad_dimensions), std::end(pad_dimensions), std::begin(this->pad_dimensions_));
}

const uint32_t Padding::get_normalized_index(std::int64_t index) const {
    std::int64_t rank = static_cast<std::int64_t>(this->rank_);
    std::uint64_t normalized_index = index >= 0 ? index : rank + index;
    TT_FATAL(
        normalized_index >= 0 and normalized_index < rank,
        "Index is out of bounds for the rank, should be between 0 and {} however is {}",
        rank - 1,
        normalized_index);
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

LegacyShape::LegacyShape(const std::initializer_list<uint32_t> dimensions) :
    rank_(dimensions.size()), dimensions_{}, padding_(dimensions.size()) {
    std::copy(std::begin(dimensions), std::end(dimensions), std::begin(this->dimensions_));
}
LegacyShape::LegacyShape(tt::stl::Span<const uint32_t> dimensions) :
    rank_(dimensions.size()), dimensions_{}, padding_(dimensions.size()) {
    std::copy(std::begin(dimensions), std::end(dimensions), std::begin(this->dimensions_));
}

LegacyShape::LegacyShape(const std::initializer_list<uint32_t> dimensions, const Padding& padding) :
    rank_(dimensions.size()), dimensions_{}, padding_(padding) {
    TT_ASSERT(this->padding_.rank_ == this->rank_);
    std::copy(std::begin(dimensions), std::end(dimensions), std::begin(this->dimensions_));
}
LegacyShape::LegacyShape(tt::stl::Span<const uint32_t> dimensions, const Padding& padding) :
    rank_(dimensions.size()), dimensions_{}, padding_(padding) {
    TT_ASSERT(this->padding_.rank_ == this->rank_);
    std::copy(std::begin(dimensions), std::end(dimensions), std::begin(this->dimensions_));
}

LegacyShape::LegacyShape(const LegacyShape& other, const Padding& padding) :
    dimensions_(other.dimensions_), rank_(other.rank_), padding_(padding) {
    TT_ASSERT(this->padding_.rank_ == this->rank_);
}

std::size_t LegacyShape::rank() const { return this->rank_; }
std::size_t LegacyShape::size() const { return this->rank_; }

uint32_t& LegacyShape::operator[](const std::int64_t index) {
    auto normalized_index = this->get_normalized_index(index);
    return this->dimensions_[normalized_index];
}
const uint32_t LegacyShape::operator[](const std::int64_t index) const {
    auto normalized_index = this->get_normalized_index(index);
    return this->dimensions_[normalized_index];
}

const uint32_t* LegacyShape::begin() const { return this->dimensions_.data(); }
const uint32_t* LegacyShape::end() const { return this->dimensions_.data() + this->rank_; }

const Padding& LegacyShape::padding() const { return this->padding_; }

const LegacyShape LegacyShape::without_padding() const {
    auto padding = this->padding_;
    ttnn::SmallVector<uint32_t> shape_without_padding;
    for (auto index = 0; index < this->rank(); index++) {
        const auto dimension = this->operator[](index);
        auto&& [front_pad, back_pad] = padding.pad_dimensions_[index];
        const auto new_dimension = dimension - (front_pad + back_pad);
        shape_without_padding.push_back(new_dimension);
    }
    return LegacyShape(shape_without_padding);
}

ttnn::SimpleShape LegacyShape::logical_shape() const {
    const LegacyShape logical = without_padding();

    ttnn::SmallVector<uint32_t> values(rank());
    for (size_t i = 0; i < values.size(); i++) {
        values[i] = logical[i];
    }
    return ttnn::SimpleShape(std::move(values));
}

ttnn::SimpleShape LegacyShape::padded_shape() const {
    ttnn::SmallVector<uint32_t> values(rank());
    for (size_t i = 0; i < values.size(); i++) {
        values[i] = (*this)[i];
    }
    return ttnn::SimpleShape(std::move(values));
}

const uint32_t LegacyShape::get_normalized_index(std::int64_t index) const {
    std::int64_t rank = static_cast<std::int64_t>(this->rank_);
    std::uint64_t normalized_index = index >= 0 ? index : rank + index;
    TT_FATAL(
        normalized_index >= 0 and normalized_index < rank,
        "Index is out of bounds for the rank, should be between 0 and {} however is {}",
        rank - 1,
        normalized_index);
    return normalized_index;
}

Array4D LegacyShape::to_array_4D() const {
    TT_FATAL(rank() == 4, "to_array_4D is only valid for 4D shapes! Called for {}.", *this);
    Array4D ret_array;
    for (int i = 0; i < rank(); i++) {
        ret_array[i] = this->operator[](i);
    }
    return ret_array;
}

bool operator==(const tt::tt_metal::LegacyShape& shape_a, const tt::tt_metal::LegacyShape& shape_b) {
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

bool operator!=(const tt::tt_metal::LegacyShape& shape_a, const tt::tt_metal::LegacyShape& shape_b) {
    return not(shape_a == shape_b);
}

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
    return config_a.buffer_type == config_b.buffer_type && config_a.memory_layout == config_b.memory_layout &&
           config_a.shard_spec == config_b.shard_spec;
}

bool operator!=(const MemoryConfig& config_a, const MemoryConfig& config_b) { return not(config_a == config_b); }

std::ostream& operator<<(std::ostream& os, const MemoryConfig& config) {
    tt::stl::reflection::operator<<(os, config);
    return os;
}

}  // namespace tt::tt_metal

namespace ttnn::types {

uint32_t Shape::operator[](std::int64_t index) const {
    const auto dimension = value[index];
    auto [front_pad, back_pad] = value.padding()[index];
    return dimension - (front_pad + back_pad);
}

}  // namespace ttnn::types
