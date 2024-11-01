// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_layout.hpp"

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

size_t round_up(size_t value, size_t multiple) {
    TT_FATAL(multiple != 0, "round_up: multiple must not be 0");

    // can be faster if multiple is power of 2
    // return (value + multiple - 1) & ~(multiple - 1);
    return ((value + multiple - 1) / multiple) * multiple;
};

Alignment legacyShapeToAlignment(const ttnn::Shape& shape) {
    auto legacy_padded_shape = shape.padded_shape();
    if (shape.logical_shape() == legacy_padded_shape) {
        return Alignment{};
    }

    const auto rank = legacy_padded_shape.rank();
    ttnn::SmallVector<uint32_t> values(rank);

    if(rank >= 1) {
        values[rank - 1] = legacy_padded_shape[rank - 1];
    }
    if(rank >= 2) {
        values[rank - 2] = legacy_padded_shape[rank - 2];
    }
    for (int i = rank - 3; i >= 0; i--) {
        values[i] = legacy_padded_shape[i] * values[i + 1];
    }

    Alignment result(std::move(values));
    return result;
}

} // namespace CMAKE_UNIQUE_NAMESPACE
}

TensorLayout::TensorLayout(DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config)
    : TensorLayout(dtype, page_config, memory_config, {}) {
}

// Private
TensorLayout::TensorLayout(DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const Alignment& alignment)
    : dtype_(dtype),
      page_config_(page_config),
      memory_config_(memory_config),
      alignment_(alignment) {

    initialize_alignment();
    validate_alignment();
}

TensorLayout TensorLayout::fromLegacyPaddedShape(DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const ttnn::Shape& legacy_shape) {
    return TensorLayout(dtype, page_config, memory_config, CMAKE_UNIQUE_NAMESPACE::legacyShapeToAlignment(legacy_shape));
}

void TensorLayout::initialize_alignment() {
    if(!alignment_.empty()) {
        return;
    }

    alignment_ = page_config_.create_default_alignment(dtype_, memory_config_);
}

void TensorLayout::validate_alignment() const
{
    return page_config_.validate_alignment(alignment_, dtype_, memory_config_);
}

std::optional<ShardSpecBuffer> TensorLayout::compute_shard_spec_buffer(const ttnn::SimpleShape& shape) const {
    if (!memory_config_.is_sharded()) {
        return std::nullopt;
    }

    TT_FATAL(memory_config_.shard_spec.has_value(), "MemoryConfig must have Shard Spec specified for sharded memory layout");

    auto& shard_spec = memory_config_.shard_spec.value();
    const Size physical_size = compute_physical_shape(shape);
    const Size page_shape = compute_page_shape(physical_size);

    TT_FATAL(physical_size.width() % page_shape.width() == 0, "Physical width {} must be multiple of page width {}", physical_size.width(), page_shape.width());
    TT_FATAL(physical_size.height() % page_shape.height() == 0, "Physical height {} must be multiple of page height {}", physical_size.height(), page_shape.height());
    const auto width_in_pages = physical_size.width() / page_shape.width();
    const auto height_in_pages = physical_size.height() / page_shape.height();
    const std::array<uint32_t, 2> tensor2d_shape {height_in_pages, width_in_pages};

    ShardSpecBuffer shard_spec_buffer(shard_spec, std::array<uint32_t, 2>(page_shape), tensor2d_shape);
    return shard_spec_buffer;
}

size_t TensorLayout::compute_packed_buffer_size_bytes(const ttnn::SimpleShape& shape) const {
    const Size physical_size = compute_physical_shape(shape);
    const Size page_shape = compute_page_shape(physical_size);
    const auto width_remainder = physical_size.width() % page_shape.width();
    const auto height_remainder = physical_size.height() % page_shape.height();
    TT_FATAL(width_remainder == 0 && height_remainder == 0, "Physical size {} must be multiple of page size {}", physical_size, page_shape);

    const size_t physical_area = physical_size.height() * physical_size.width();
    const size_t page_area = page_shape.height() * page_shape.width();

    const size_t page_count = physical_area / page_area;
    const size_t page_size_bytes = compute_page_size_bytes(page_shape);

    return page_count * page_size_bytes;
}

size_t TensorLayout::compute_page_size_bytes(const ttnn::SimpleShape& shape) const {
    const auto physical_size = compute_physical_shape(shape);
    const auto page_shape = compute_page_shape(physical_size);
    return compute_page_size_bytes(page_shape);
}

size_t TensorLayout::compute_page_size_bytes(const Size& page_size) const {
    return page_config_.get_page_size_bytes(page_size, dtype_);
}

Size TensorLayout::compute_physical_shape(const ttnn::SimpleShape& shape) const {
    const int rank = static_cast<int>(shape.rank());
    const int alignment_rank = static_cast<int>(alignment_.size());
    const int max_rank = std::max(rank, alignment_rank);
    size_t width = 1;
    size_t height = 1;

    // Iterate dims in reverse order and ensure alignment
    // Even tensor of rank 0 or 1 must be aligned (to Tile / Page / Shard)
    for (int i = -1; i >= -max_rank; --i) {
        auto& dim = i == -1 ? width : height;
        if(i >= -rank) {
            dim *= shape[i];
        }

        // Align the current dimension if alignment is available
        if (i >= -alignment_rank) {
            dim = CMAKE_UNIQUE_NAMESPACE::round_up(dim, alignment_[i]);
        }
    }

    Size size{height, width};
    return size;
}

Size TensorLayout::compute_page_shape(const Size& physical_size) const {
    return page_config_.get_page_shape(physical_size, dtype_, memory_config_);
}

Strides TensorLayout::compute_strides(const ttnn::SimpleShape& shape) const {
    const int rank = static_cast<int>(shape.rank());
    const int alignment_rank = static_cast<int>(alignment_.size());

    Strides strides(rank, 1);
    for (int i = rank - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];

        const int alignment_index = i - (rank - alignment_rank) + 1;
        if(alignment_index >= 0) {
            strides[i] = CMAKE_UNIQUE_NAMESPACE::round_up(strides[i], alignment_[alignment_index]);
        }
    }

    return strides;
}

ttnn::SimpleShape TensorLayout::compute_padded_shape(const ttnn::SimpleShape& shape) const
{
    ttnn::SmallVector<uint32_t> padded_shape(shape.rank());
    int rank_index = static_cast<int>(shape.rank()) - 1;
    int alignment_index = static_cast<int>(alignment_.size()) - 1;
    size_t accum_alignment = 1;

    for (;rank_index >= 0 && alignment_index >= 0; rank_index--, alignment_index--) {
        // The last 2 dimensions of a shape are special
        if (rank_index >= static_cast<int>(shape.rank()) - 2) {
            padded_shape[rank_index] = CMAKE_UNIQUE_NAMESPACE::round_up(shape[rank_index], alignment_[alignment_index]);
        } else {
            if (accum_alignment % alignment_[alignment_index] == 0) {
                // Alignment for this dimension is redundant, ignoring
                padded_shape[rank_index] = shape[rank_index];
            } else if (alignment_[alignment_index] % accum_alignment == 0) {
                padded_shape[rank_index] = CMAKE_UNIQUE_NAMESPACE::round_up(shape[rank_index], alignment_[alignment_index] / accum_alignment);
            } else {
                TT_THROW("Padded shape can't be deducted from TensorLayout parameters {} and Shape {}", alignment_, shape);
            }
        }

        // Alignment doesn't accumulate on the last dimension of a shape
        if (rank_index != static_cast<int>(shape.rank()) - 1) {
            accum_alignment *= padded_shape[rank_index];
        }
    }
    for(; rank_index >= 0; rank_index--) {
        padded_shape[rank_index] = shape[rank_index];
    }
    return ttnn::SimpleShape(std::move(padded_shape));
}

} // namespace tt::tt_metal
