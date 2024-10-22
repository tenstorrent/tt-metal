// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
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


TensorLayout::TensorLayout(DataType dataType, const PageConfig& pageConfig, const MemoryConfig& memoryConfig, const Alignment& alignment)
    : mDataType(dataType),
      mPageConfig(pageConfig),
      mMemoryConfig(memoryConfig),
      mAlignment(alignment) {

    initialize_alignment();
    validate_alignment();
}

TensorLayout TensorLayout::fromLegacyPaddedShape(DataType dataType, const PageConfig& pageConfig, const MemoryConfig& memoryConfig, const ttnn::SimpleShape& legacyPaddedShape) {
    return TensorLayout(dataType, pageConfig, memoryConfig, legacyPaddedShapeToAlignment(legacyPaddedShape));
}

void TensorLayout::initialize_alignment() {
    if(mAlignment.size() != 0)
        return;

    mAlignment = mPageConfig.create_default_alignment(mDataType);
}

void TensorLayout::validate_alignment() const
{
    return mPageConfig.validate_alignment(mAlignment, mDataType);
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
