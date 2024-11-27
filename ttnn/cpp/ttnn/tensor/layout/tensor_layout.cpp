// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_layout.hpp"

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

size_t round_up(size_t value, size_t multiple) {
    if (multiple == 0) {
        return value;
    }

    return ((value + multiple - 1) / multiple) * multiple;
};

Alignment legacyShapeToAlignment(
    const ttnn::Shape& shape, const PageConfig& page_config, const MemoryConfig& memory_config) {
    const auto& logical_shape = shape.logical_shape();
    const auto& legacy_padded_shape = shape.padded_shape();
    if (logical_shape == legacy_padded_shape) {
        return Alignment{};
    }

    const auto rank = legacy_padded_shape.rank();
    bool alignment_can_be_2D = true;
    for (int i = rank - 3; i >= 0; i--) {
        alignment_can_be_2D &= logical_shape[i] == legacy_padded_shape[i];
    }

    // SHARDED
    if (memory_config.shard_spec.has_value()) {
        TT_FATAL(
            alignment_can_be_2D,
            "Tensor with shape {} cannot be sharded because alignment will have rank greater than 2!",
            shape);
        if (page_config.get_layout() == Layout::ROW_MAJOR) {
            const auto& shard_spec = memory_config.shard_spec.value();
            if (shard_spec.physical_shard_shape.has_value()) {
                return Alignment{shard_spec.physical_shard_shape.value()[1]};
            }
            return Alignment{shard_spec.shape[1]};
        }
        return Alignment{};
    }

    // INTERLEAVED with only height/width padding
    if (alignment_can_be_2D) {
        ttnn::SmallVector<uint32_t> values(std::min((int)rank, 2));
        const auto alignment_size = values.size();
        if (alignment_size >= 1) {
            values[alignment_size - 1] = legacy_padded_shape[-1];
        }
        if (alignment_size == 2) {
            values[alignment_size - 2] = legacy_padded_shape[-2];
        }
        Alignment result(std::move(values));
        return result;
    }

    // INTERLEAVED with (deprecated) non-height/width padding
    // NOTE: Rank > 2 is guaranteed in this case
    ttnn::SmallVector<uint32_t> values(rank);
    values[rank - 1] = legacy_padded_shape[-1];
    values[rank - 2] = legacy_padded_shape[-2];

    for (int i = rank - 3; i >= 0; i--) {
        values[i] = legacy_padded_shape[i] * values[i + 1];
    }

    for (auto& value : values) {
        if (value == 0) {
            value = 1;
        }
    }

    Alignment result(std::move(values));
    return result;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TensorLayout::TensorLayout(DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config) :
    TensorLayout(dtype, page_config, memory_config, {}) {}

// Private
TensorLayout::TensorLayout(
    DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const Alignment& alignment) :
    dtype_(dtype), page_config_(page_config), memory_config_(memory_config), alignment_(alignment) {
    initialize_alignment();
    validate_alignment();
}

TensorLayout TensorLayout::fromLegacyPaddedShape(
    DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const ttnn::Shape& legacy_shape) {
    return TensorLayout(
        dtype,
        page_config,
        memory_config,
        CMAKE_UNIQUE_NAMESPACE::legacyShapeToAlignment(legacy_shape, page_config, memory_config));
}

void TensorLayout::initialize_alignment() {
    if (!alignment_.empty()) {
        return;
    }

    alignment_ = page_config_.create_default_alignment(dtype_, memory_config_);
}

void TensorLayout::validate_alignment() const {
    TT_FATAL(
        alignment_.size() <= 2 || !memory_config_.is_sharded(),
        "Tensor must be interleaved if alignment has rank greater than 2!");
    return page_config_.validate_alignment(alignment_, dtype_, memory_config_);
}

std::optional<ShardSpecBuffer> TensorLayout::compute_shard_spec_buffer(const ttnn::SimpleShape& shape) const {
    if (!memory_config_.is_sharded()) {
        return std::nullopt;
    }

    TT_FATAL(
        memory_config_.shard_spec.has_value(), "MemoryConfig must have Shard Spec specified for sharded memory layout");

    const Size physical_size = compute_physical_shape(shape);
    const Size page_shape = compute_page_shape(physical_size);

    TT_FATAL(
        physical_size.width() % page_shape.width() == 0,
        "Physical width {} must be multiple of page width {}",
        physical_size.width(),
        page_shape.width());
    TT_FATAL(
        physical_size.height() % page_shape.height() == 0,
        "Physical height {} must be multiple of page height {}",
        physical_size.height(),
        page_shape.height());
    const auto width_in_pages = physical_size.width() / page_shape.width();
    const auto height_in_pages = physical_size.height() / page_shape.height();
    const std::array<uint32_t, 2> tensor2d_shape{height_in_pages, width_in_pages};

    auto shard_spec = memory_config_.shard_spec.value();

    switch (shard_spec.mode) {
        case ShardMode::PHYSICAL: break;
        case ShardMode::LOGICAL: {
            const auto& physical_shard_shape = compute_physical_shard_shape(shard_spec.shape);
            shard_spec.shape = physical_shard_shape;
            break;
        }
        default: TT_THROW("Unsupported shard mode {} in compute_shard_spec_buffer!", shard_spec.mode);
    }

    ShardSpecBuffer shard_spec_buffer(shard_spec, std::array<uint32_t, 2>(page_shape), tensor2d_shape);
    return shard_spec_buffer;
}

size_t TensorLayout::compute_packed_buffer_size_bytes(const ttnn::SimpleShape& shape) const {
    const Size physical_size = compute_physical_shape(shape);
    const Size page_shape = compute_page_shape(physical_size);
    const auto width_remainder = physical_size.width() % page_shape.width();
    const auto height_remainder = physical_size.height() % page_shape.height();
    TT_FATAL(
        width_remainder == 0 && height_remainder == 0,
        "Physical size {} must be multiple of page size {}",
        physical_size,
        page_shape);

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

Size TensorLayout::compute_physical_shard_shape(const Size& logical_shard_shape) const {
    TT_FATAL(
        memory_config_.shard_spec.has_value() and memory_config_.shard_spec.value().mode == ShardMode::LOGICAL,
        "TensorLayout must be logically sharded for compute_physical_shard_shape!");
    const auto& shard_spec = memory_config_.shard_spec.value();
    // TODO: If physical_shard_shape is provided, alignment_ == physical_shard_shape is guaranteed (should we store
    // physical_shard_shape instead?)
    if (shard_spec.physical_shard_shape.has_value()) {
        const auto& physical_shard_shape = shard_spec.physical_shard_shape.value();
        TT_FATAL(
            physical_shard_shape[0] == alignment_[-2] and physical_shard_shape[1] == alignment_[-1],
            "Alignment {} must be same as physical shard shape {} provided in shard spec!",
            alignment_,
            physical_shard_shape);
        return physical_shard_shape;
    }

    // TODO: Alignment is guaranteed to be rank 2 or less if tensor is sharded (remove validate?)
    const int alignment_rank = static_cast<int>(alignment_.size());
    TT_FATAL(alignment_rank <= 2, "Alignment {} must be rank 2 or less to compute physical shard shape", alignment_);
    auto physical_shard_height = CMAKE_UNIQUE_NAMESPACE::round_up(logical_shard_shape.height(), alignment_[-2]);
    auto physical_shard_width = CMAKE_UNIQUE_NAMESPACE::round_up(logical_shard_shape.width(), alignment_[-1]);
    return Size{physical_shard_height, physical_shard_width};
}

Size TensorLayout::compute_physical_shape(const ttnn::SimpleShape& shape) const {
    const int rank = static_cast<int>(shape.rank());
    const int alignment_rank = static_cast<int>(alignment_.size());

    size_t width = 1;
    size_t height = 1;

    // LOGICAL SHARDING
    if (memory_config_.shard_spec.has_value() and memory_config_.shard_spec.value().mode == ShardMode::LOGICAL) {
        // Iterate dims in reverse order
        for (int i = -1; i >= -rank; --i) {
            auto& dim = i == -1 ? width : height;
            dim *= shape[i];
        }

        const auto& logical_shard_shape = Size(memory_config_.shard_spec.value().shape);
        const auto& physical_shard_shape = compute_physical_shard_shape(logical_shard_shape);

        auto get_physical_size =
            [](auto original_size, auto logical_shard_size, auto physical_shard_size, auto alignment) -> uint32_t {
            if (logical_shard_size == 0) {
                return 0;
            }
            // If we always pad to full shards, then return:
            // auto num_shards = tt::div_up(original_size, logical_shard_size);
            // return (uint32_t) physical_shard_size * num_shards;

            // If we pad all shards except last shard up to physical size and last one only up to nearest alignment,
            // then return this: NOTE: This matches existing physical sharding where physical host data can be sharded
            // with partial shards
            auto num_full_shards = original_size / logical_shard_size;
            auto last_physical_shard_size =
                CMAKE_UNIQUE_NAMESPACE::round_up(original_size % logical_shard_size, alignment);
            return (physical_shard_size * num_full_shards + last_physical_shard_size);
        };

        auto physical_height =
            get_physical_size(height, logical_shard_shape.height(), physical_shard_shape.height(), alignment_[-2]);
        auto physical_width =
            get_physical_size(width, logical_shard_shape.width(), physical_shard_shape.width(), alignment_[-1]);

        Size size{physical_height, physical_width};
        return size;
    }

    // INTERLEAVED or deprecated PHYSICAL SHARDING
    const int max_rank = std::max(rank, alignment_rank);

    // Iterate dims in reverse order and ensure alignment
    // Even tensor of rank 0 or 1 must be aligned (to Tile / Page / Shard)
    for (int i = -1; i >= -max_rank; --i) {
        auto& dim = i == -1 ? width : height;
        if (i >= -rank) {
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
    std::optional<Size> physical_shard_shape = std::nullopt;
    if (memory_config_.shard_spec.has_value()) {
        const auto& shard_spec = memory_config_.shard_spec.value();
        switch (shard_spec.mode) {
            case ShardMode::PHYSICAL: physical_shard_shape = shard_spec.shape; break;
            case ShardMode::LOGICAL: physical_shard_shape = compute_physical_shard_shape(shard_spec.shape); break;
            default: TT_THROW("Unsupported shard mode {} in compute_shard_spec_buffer!", shard_spec.mode);
        }
    }

    return page_config_.get_page_shape(physical_size, dtype_, memory_config_, physical_shard_shape);
}

Strides TensorLayout::compute_strides(const ttnn::SimpleShape& shape) const {
    const int rank = static_cast<int>(shape.rank());
    const int alignment_rank = static_cast<int>(alignment_.size());

    Strides strides(rank, 1);
    for (int i = rank - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];

        const int alignment_index = i - (rank - alignment_rank) + 1;
        if (alignment_index >= 0) {
            strides[i] = CMAKE_UNIQUE_NAMESPACE::round_up(strides[i], alignment_[alignment_index]);
        }
    }

    return strides;
}

ttnn::SimpleShape TensorLayout::compute_padded_shape(const ttnn::SimpleShape& shape) const {
    ttnn::SmallVector<uint32_t> padded_shape(shape.rank());
    int rank_index = static_cast<int>(shape.rank()) - 1;
    int alignment_index = static_cast<int>(alignment_.size()) - 1;
    size_t accum_alignment = 1;

    for (; rank_index >= 0 && alignment_index >= 0; rank_index--, alignment_index--) {
        // The last 2 dimensions of a shape are special
        if (rank_index >= static_cast<int>(shape.rank()) - 2) {
            padded_shape[rank_index] = CMAKE_UNIQUE_NAMESPACE::round_up(shape[rank_index], alignment_[alignment_index]);
        } else {
            if (accum_alignment % alignment_[alignment_index] == 0) {
                // Alignment for this dimension is redundant, ignoring
                padded_shape[rank_index] = shape[rank_index];
            } else if (alignment_[alignment_index] % accum_alignment == 0) {
                padded_shape[rank_index] =
                    CMAKE_UNIQUE_NAMESPACE::round_up(shape[rank_index], alignment_[alignment_index] / accum_alignment);
            } else {
                TT_THROW(
                    "Padded shape can't be deducted from TensorLayout parameters {} and Shape {}", alignment_, shape);
            }
        }

        // Alignment doesn't accumulate on the last dimension of a shape
        if (rank_index != static_cast<int>(shape.rank()) - 1) {
            accum_alignment *= padded_shape[rank_index];
        }
    }
    for (; rank_index >= 0; rank_index--) {
        padded_shape[rank_index] = shape[rank_index];
    }
    return ttnn::SimpleShape(std::move(padded_shape));
}

}  // namespace tt::tt_metal
