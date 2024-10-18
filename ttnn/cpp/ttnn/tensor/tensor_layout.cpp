// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_layout.hpp"
#include <variant>
#include <vector>
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "types.hpp"

namespace tt::tt_metal {

namespace {
size_t round_up(size_t value, size_t multiple) {
    TT_FATAL(multiple != 0, "round_up: multiple should not be 0");

    // can be faster if multiple is power of 2
    // return (value + multiple - 1) & ~(multiple - 1);
    return ((value + multiple - 1) / multiple) * multiple;
};

Alignment legacyPaddedShapeToAlignment(const ttnn::SimpleShape& legacyPaddedShape) {
    const auto rank = legacyPaddedShape.rank();
    std::vector<uint32_t> values(rank);

    if(rank >= 1) {
        values[rank - 1] = legacyPaddedShape[rank - 1];
    }
    if(rank >= 2) {
        values[rank - 2] = legacyPaddedShape[rank - 2];
    }
    for (int i = rank - 3; i >= 0; i--) {
        values[i] = legacyPaddedShape[i] * values[i + 1];
    }

    Alignment result(values);
    return result;
}

// Euclidean algorithm for greatest common divisor
size_t gcd(size_t a, size_t b) {
    while (b != 0) {
        a %= b;
        std::swap(a, b);
    }
    return a;
}

// Least common multiple
size_t lcm(size_t a, size_t b) {
    return a * b / gcd(a, b);
}
}

namespace utils {
size_t element_size_bytes(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16: return sizeof(bfloat16);
        case DataType::FLOAT32: return sizeof(float);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::UINT32: return sizeof(uint32_t);
        case DataType::UINT16: return sizeof(uint16_t);
        case DataType::UINT8: return sizeof(uint8_t);
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B:
            TT_THROW("element_size_bytes() should not be used for BFLOAT8_B and BFLOAT4_B types becaues of how they are packed");

        default:
            TT_THROW("Unsupported data type!");
    }
}
}

Size::Size(size_t height, size_t width) : mHeight(height), mWidth(width) {}
Size::Size(const std::pair<size_t, size_t>& size) : mHeight(size.first), mWidth(size.second) {}
Size::Size(const std::array<size_t, 2>& size) : mHeight(size[0]), mWidth(size[1]) {}


Size Size::operator/(const Size& rhs) const {
    return Size(mHeight / rhs.mHeight, mWidth / rhs.mWidth);
}

Size Size::operator*(size_t scalar) const {
    return Size(mHeight * scalar, mWidth * scalar);
}

Size Size::operator%(const Size& rhs) const {
    return Size(mHeight % rhs.mHeight,  mWidth % rhs.mWidth);
}

Size::operator std::pair<size_t, size_t>() const {
    return {mHeight, mWidth};
}

Size::operator std::array<size_t, 2>() const {
    return {mHeight, mWidth};
}

Size::operator std::array<uint32_t, 2>() const {
    return {static_cast<uint32_t>(mHeight), static_cast<uint32_t>(mWidth)};
}

size_t Size::height() const { return mHeight; }
size_t Size::width() const { return mWidth; }

bool Size::operator==(const Size& rhs) const {
    return mHeight == rhs.mHeight && mWidth == rhs.mWidth;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Size& size)
{
    os << "(" << size.height() << ", " << size.width() << ")";
    return os;
}

std::ostream &operator<<(std::ostream &os, const tt::tt_metal::Alignment &value) {
    os << "Alignment([";
    for (size_t i = 0; i < value.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << value[i];
    }
    os << "])";
    return os;
}

TilePageConfig::TilePageConfig(const Tile& tile)
 : tile(tile) {
}

Alignment RowMajorPageConfig::createDefaultAlignment(DataType dataType) const {
{
    TT_FATAL(dataType != DataType::BFLOAT4_B && dataType != DataType::BFLOAT8_B, "BFLOAT4_B and BFLOAT8_B data types are not supported for ROW_MAJOR layout");
    return Alignment({sizeof(uint32_t) / utils::element_size_bytes(dataType)});}
}

Alignment TilePageConfig::createDefaultAlignment(DataType dataType) const {
    return Alignment({tile.get_height(), tile.get_width()});
}

void RowMajorPageConfig::validateAlignment(const Alignment& alignment, DataType dataType) const {
    TT_FATAL(alignment.size() > 0, "Alignment should have at least 1 dimension for Row Major layout");
    uint32_t widthAlignment = alignment[-1];
    uint32_t element_size = utils::element_size_bytes(dataType);
    uint32_t page_alignment = sizeof(uint32_t) / element_size;
    TT_FATAL((widthAlignment % page_alignment) == 0,
        "Wrong custom Tensor Layout alignment {}. For Row Major layout with element size {}bytes the innermost dimension must align to {}. This is because Buffer data is packed as uint32_t (4 bytes).",
        alignment, element_size, page_alignment);
}

void TilePageConfig::validateAlignment(const Alignment& alignment, DataType dataType) const {
    TT_FATAL(alignment.size() >= 2, "Alignment should have at least 2 dimensions for Tile layout");
    const auto widthAlignment = alignment[-1];
    TT_FATAL(widthAlignment % tile.get_width() == 0,
        "Wrong custom Tensor Layout alignment {}. For Tile layout innermost dimension should be multiple of tile width {}.", alignment, tile.get_width());
    auto heightAlignment = alignment[-2];
    TT_FATAL((heightAlignment % tile.get_height()) == 0,
        "Wrong custom Tensor Layout alignment {}. For Tile layout second innermost dimension should be multiple of tile height {}.", alignment, tile.get_height());
}

Size RowMajorPageConfig::get_page_shape(const Size& physical_size, const MemoryConfig& memoryConfig) const {
    if (memoryConfig.shard_spec.has_value()) {
        const auto& shard_spec = memoryConfig.shard_spec.value();
        const auto& shard_shape = shard_spec.shape;
        return Size(1, shard_shape[1]);
    }
    return Size(1, physical_size.width());
}

Size TilePageConfig::get_page_shape(const Size& physical_size, const MemoryConfig& memoryConfig) const {
    return Size(tile.get_height(), tile.get_width());
}

size_t RowMajorPageConfig::get_page_size_bytes(const Size& page_shape, DataType dataType) const {
    const auto size = page_shape.height() * page_shape.width() * utils::element_size_bytes(dataType);
    return size;
}

size_t TilePageConfig::get_page_size_bytes(const Size& page_shape, DataType dataType) const {
    const auto tiles_count = page_shape.height() / tile.get_height() * page_shape.width() / tile.get_width();
    const auto size = tiles_count * tile.get_tile_size(datatype_to_dataformat_converter(dataType));
    return size;
}

PageConfig::PageConfig(const Config& config)
    : mConfig(config) {
}

PageConfig::PageConfig(Layout layout)
    : PageConfig(layout, std::nullopt) {
}

PageConfig::PageConfig(Layout layout, const std::optional<Tile>& tile) {
    if(layout == Layout::ROW_MAJOR) {
        mConfig = RowMajorPageConfig();
    }
    else {
        mConfig =  TilePageConfig(tile.value_or(Tile()));
    }
}

Alignment PageConfig::createDefaultAlignment(DataType dataType) const {
    return std::visit([&](const auto& config) constexpr { return config.createDefaultAlignment(dataType); }, mConfig);
}

void PageConfig::validateAlignment(const Alignment& alignment, DataType dataType) const {
    std::visit([&](const auto& config) constexpr { config.validateAlignment(alignment, dataType); }, mConfig);
}

Size PageConfig::get_page_shape(const Size& physical_size, const MemoryConfig& memoryConfig) const {
    return std::visit([&](const auto& config) constexpr { return config.get_page_shape(physical_size, memoryConfig); }, mConfig);
}

size_t PageConfig::get_page_size_bytes(const Size& page_shape, DataType dataType) const {
    return std::visit([&](const auto& config) constexpr { return config.get_page_size_bytes(page_shape, dataType); }, mConfig);
}

bool PageConfig::isRowMajor() const {
    return std::holds_alternative<RowMajorPageConfig>(mConfig);
}

std::optional<Tile> PageConfig::get_tile() const
{
    if(std::holds_alternative<TilePageConfig>(mConfig)) {
        return std::get<TilePageConfig>(mConfig).tile;
    }

    return std::nullopt;
}


TensorLayout::TensorLayout(DataType dataType, const PageConfig& pageConfig, const MemoryConfig& memoryConfig, const Alignment& alignment)
    : mDataType(dataType),
      mPageConfig(pageConfig),
      mMemoryConfig(memoryConfig),
      mAlignment(alignment) {

    initializeAlignment();
    validateAlignment();
}

// Private constructor to create TensorLayout from LegacyPaddedShape
TensorLayout::TensorLayout(DataType dataType, const PageConfig& pageConfig, const MemoryConfig& memoryConfig, const ttnn::SimpleShape& legacyPaddedShape)
    : TensorLayout(dataType, pageConfig, memoryConfig, legacyPaddedShapeToAlignment(legacyPaddedShape)) {
}

TensorLayout TensorLayout::fromLegacyPaddedShape(DataType dataType, const PageConfig& pageConfig, const MemoryConfig& memoryConfig, const ttnn::SimpleShape& legacyPaddedShape) {
    return TensorLayout(dataType, pageConfig, memoryConfig, legacyPaddedShape);
}

void TensorLayout::initializeAlignment() {
    if(mAlignment.size() != 0)
        return;

    mAlignment = mPageConfig.createDefaultAlignment(mDataType);
}

void TensorLayout::validateAlignment() const
{
    return mPageConfig.validateAlignment(mAlignment, mDataType);
}

std::optional<ShardSpecBuffer> TensorLayout::get_shard_spec_buffer(const ttnn::SimpleShape& shape) const {
    if (!mMemoryConfig.is_sharded())
        return std::nullopt;

    TT_FATAL(mMemoryConfig.shard_spec.has_value(), "MemoryConfig should have Shard Spec specified for sharded memory layout");

    auto& shard_spec = mMemoryConfig.shard_spec.value();
    Size physical_shape = get_physical_shape(shape);
    Size page_shape = get_page_shape(physical_shape);
    Size tensor2d_size = physical_shape / page_shape; // looks like shard grid
    ShardSpecBuffer shard_spec_buffer(shard_spec, std::array<uint32_t, 2>(page_shape), std::array<uint32_t, 2>(tensor2d_size));

    return shard_spec_buffer;
}

size_t TensorLayout::get_packed_buffer_size_bytes(const ttnn::SimpleShape& shape) const {
    const Size physical_size = get_physical_shape(shape);
    const Size page_shape = get_page_shape(physical_size);
    const Size size_modulo = physical_size % page_shape;
    TT_FATAL(size_modulo.height() == 0 && size_modulo.width() == 0, "Physical size {} should be multiple of page size {}", physical_size, page_shape);

    const size_t physical_area = physical_size.height() * physical_size.width();
    const size_t page_area = page_shape.height() * page_shape.width();

    size_t page_count = physical_area / page_area;
    size_t page_size_bytes = get_page_size_bytes(page_shape);

    return page_count * page_size_bytes;
}

size_t TensorLayout::get_page_size_bytes(const ttnn::SimpleShape& shape) const {
    auto physical_size = get_physical_shape(shape);
    auto page_shape = get_page_shape(physical_size);
    return get_page_size_bytes(page_shape);
}

size_t TensorLayout::get_page_size_bytes(const Size& page_size) const {
    return mPageConfig.get_page_size_bytes(page_size, mDataType);
}

Size TensorLayout::get_physical_shape(const ttnn::SimpleShape& shape) const {
    const int rank = static_cast<int>(shape.rank());
    const int alignmentRank = static_cast<int>(mAlignment.size());
    const int maxRank = std::max(rank, alignmentRank);
    size_t width = 1;
    size_t height = 1;

    // Iterate dims in reverse order and ensure alignment
    // Even tensor of rank 0 or 1 must be aligned (to Tile / Page / Shard)
    for (int i = -1; i >= -maxRank; --i) {
        auto& dim = i == -1 ? width : height;
        if(i >= -rank) {
            dim *= shape[i];
        }

        // Align the current dimension if alignment is available
        if (i >= -alignmentRank) {
            dim = round_up(dim, mAlignment[i]);
        }
    }

    Size size{height, width};

    return size;
}

Size TensorLayout::get_page_shape(const Size& physical_size) const {
    if(mMemoryConfig.memory_layout == TensorMemoryLayout::SINGLE_BANK) {
        return physical_size;
    }

    return mPageConfig.get_page_shape(physical_size, mMemoryConfig);
}

Strides TensorLayout::get_strides(const ttnn::SimpleShape& shape) const {
    const int rank = static_cast<int>(shape.rank());
    const int alignmentRank = static_cast<int>(mAlignment.size());

    Strides strides(rank, 1);
    for (int i = rank - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];

        const int alignment_index = i - (rank - alignmentRank) + 1;
        if(alignment_index >= 0) {
            strides[i] = round_up(strides[i], mAlignment[alignment_index]);
        }
    }

    return strides;
}

ttnn::SimpleShape TensorLayout::get_padded_shape(const ttnn::SimpleShape& shape) const
{
    std::vector<uint32_t> padded_shape(shape.rank());
    int rankIdx = static_cast<int>(shape.rank()) - 1;
    int alignmentIdx = static_cast<int>(mAlignment.size()) - 1;
    size_t accum_alignment = 1;

    for (;rankIdx >= 0 && alignmentIdx >= 0; rankIdx--, alignmentIdx--) {
        // The last 2 dimensions of a shape are special
        if (rankIdx >= static_cast<int>(shape.rank()) - 2) {
            padded_shape[rankIdx] = round_up(shape[rankIdx], mAlignment[alignmentIdx]);
        } else {
            if (accum_alignment % mAlignment[alignmentIdx] == 0) {
                // Alignment for this dimension is redundant, ignoring
                padded_shape[rankIdx] = shape[rankIdx];
            } else if (mAlignment[alignmentIdx] % accum_alignment == 0) {
                padded_shape[rankIdx] = round_up(shape[rankIdx], mAlignment[alignmentIdx] / accum_alignment);
            } else {
                TT_THROW("Padded shape can't be deducted from TensorLayout parameters {} and Shape {}", mAlignment, shape);
            }
        }

        // Alignment doesn't accumulate on the last dimension of a shape
        if (rankIdx != static_cast<int>(shape.rank()) - 1) {
            accum_alignment *= padded_shape[rankIdx];
        }
    }
    for(; rankIdx >= 0; rankIdx--) {
        padded_shape[rankIdx] = shape[rankIdx];
    }
    return ttnn::SimpleShape(std::move(padded_shape));
}

} // namespace tt::tt_metal
