// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/layout/tensor_layout.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/math.hpp>

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
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& legacy_padded_shape,
    const PageConfig& page_config,
    const MemoryConfig& memory_config) {
    if (logical_shape == legacy_padded_shape) {
        return Alignment{};
    }

    const int padded_rank = legacy_padded_shape.rank();
    bool alignment_can_be_2D = true;
    for (int i = -3; i >= -padded_rank; i--) {
        alignment_can_be_2D &= logical_shape[i] == legacy_padded_shape[i];
    }

    // 2D SHARDED
    if (memory_config.shard_spec().has_value()) {
        TT_FATAL(
            alignment_can_be_2D,
            "Tensor with shape {} ({}) cannot be sharded because alignment will have rank greater than 2!",
            logical_shape,
            legacy_padded_shape);
        if (page_config.get_layout() == Layout::ROW_MAJOR) {
            const auto& shard_spec = memory_config.shard_spec().value();
            if (shard_spec.physical_shard_shape.has_value()) {
                return Alignment{shard_spec.physical_shard_shape.value()[1]};
            }
            return Alignment{shard_spec.shape[1]};
        }
        return Alignment{};
    }

    // INTERLEAVED with only height/width padding
    if (alignment_can_be_2D) {
        ttnn::SmallVector<uint32_t> values(std::min((int)padded_rank, 2));
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
    ttnn::SmallVector<uint32_t> values(padded_rank);
    values[padded_rank - 1] = legacy_padded_shape[-1];
    values[padded_rank - 2] = legacy_padded_shape[-2];

    for (int i = padded_rank - 3; i >= 0; i--) {
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

void validate_alignment(const TensorLayout& tensor_layout) {
    const auto& alignment = tensor_layout.get_alignment();
    const auto& memory_config = tensor_layout.get_memory_config();
    TT_FATAL(
        alignment.size() <= 2 || !memory_config.shard_spec().has_value(),
        "Tensor must be interleaved if alignment has rank greater than 2!");

    const auto& page_config = tensor_layout.get_page_config();
    const auto& dtype = tensor_layout.get_data_type();
    return page_config.validate_alignment(alignment, dtype, memory_config);
}

void validate_shard_spec(const TensorLayout& tensor_layout) {
    const auto& memory_config = tensor_layout.get_memory_config();
    const auto& layout = tensor_layout.get_layout();
    if (memory_config.is_sharded() && layout == Layout::TILE) {
        const auto& tile_shape = tensor_layout.get_tile().get_tile_shape();
        if (memory_config.shard_spec().has_value()) {
            const auto& physical_shard_shape = tensor_layout.get_physical_shard_shape();
            TT_FATAL(
                (physical_shard_shape.height() % tile_shape[0] == 0 &&
                 physical_shard_shape.width() % tile_shape[1] == 0),
                "Physical shard shape {} must be tile {} sized!",
                physical_shard_shape,
                tile_shape);
        } else {
            const auto& shard_shape = memory_config.nd_shard_spec().value().shard_shape;
            TT_FATAL(
                (shard_shape[-2] % tile_shape[0] == 0 && shard_shape[-1] % tile_shape[1] == 0),
                "Physical shard shape {} must be tile {} sized!",
                shard_shape,
                tile_shape);
        }
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TensorLayout::TensorLayout(
    DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const Alignment& alignment) :
    dtype_(dtype), page_config_(page_config), memory_config_(memory_config), alignment_(alignment) {
    initialize_alignment();
    CMAKE_UNIQUE_NAMESPACE::validate_alignment(*this);
    CMAKE_UNIQUE_NAMESPACE::validate_shard_spec(*this);
}

TensorLayout TensorLayout::fromPaddedShape(
    DataType dtype,
    const PageConfig& page_config,
    const MemoryConfig& memory_config,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape) {
    return TensorLayout(
        dtype,
        page_config,
        memory_config,
        CMAKE_UNIQUE_NAMESPACE::legacyShapeToAlignment(logical_shape, padded_shape, page_config, memory_config));
}

TensorLayout TensorLayout::restore_from_serialized(
    DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const Alignment& alignment) {
    return TensorLayout(dtype, page_config, memory_config, alignment);
}

void TensorLayout::initialize_alignment() {
    auto default_alignment = page_config_.create_default_alignment(dtype_, memory_config_);
    if (alignment_.empty()) {
        alignment_ = default_alignment;
        return;
    }

    ttnn::SmallVector<uint32_t> result(std::max(alignment_.size(), default_alignment.size()), 1);
    for (size_t i = 0; i < alignment_.size(); i++) {
        result[i + result.size() - alignment_.size()] = alignment_[i];
    }
    for (size_t i = 0; i < default_alignment.size(); i++) {
        size_t result_idx = i + result.size() - default_alignment.size();
        result[result_idx] = CMAKE_UNIQUE_NAMESPACE::round_up(result[result_idx], default_alignment[i]);
    }
    alignment_ = Alignment(std::move(result));
}

BufferShardingArgs TensorLayout::compute_buffer_sharding_args(const ttnn::Shape& shape) const {
    if (!memory_config_.is_sharded()) {
        return {};
    }

    TT_FATAL(
        memory_config_.shard_spec().has_value() || memory_config_.nd_shard_spec().has_value(),
        "MemoryConfig must have Shard Spec specified for sharded memory layout");

    const Shape2D physical_size = compute_physical_shape(shape);
    const Shape2D page_shape = compute_page_shape(physical_size);

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

    std::optional<ShardSpecBuffer> shard_spec_buffer;
    std::optional<BufferDistributionSpec> distribution_spec;

    if (auto shard_spec = memory_config_.shard_spec()) {
        const auto width_in_pages = physical_size.width() / page_shape.width();
        const auto height_in_pages = physical_size.height() / page_shape.height();
        const std::array<uint32_t, 2> tensor2d_shape_in_pages{height_in_pages, width_in_pages};

        switch (shard_spec->mode) {
            case ShardMode::PHYSICAL: break;
            case ShardMode::LOGICAL: {
                const auto& physical_shard_shape = get_physical_shard_shape();
                shard_spec->shape =
                    std::array<uint32_t, 2>{physical_shard_shape.height(), physical_shard_shape.width()};
                break;
            }
            default: TT_THROW("Unsupported shard mode {} in compute_distribution_spec!", shard_spec->mode);
        }

        shard_spec_buffer = ShardSpecBuffer(*shard_spec, std::array<uint32_t, 2>(page_shape), tensor2d_shape_in_pages);
    }

    if (const auto& nd_shard_spec = memory_config_.nd_shard_spec()) {
        auto padded_shape = compute_padded_shape(shape);
        distribution_spec = BufferDistributionSpec::from_shard_spec(
            padded_shape,
            nd_shard_spec->shard_shape,
            page_shape,
            nd_shard_spec->grid,
            nd_shard_spec->orientation,
            nd_shard_spec->shard_distribution_strategy);
    }
    return BufferShardingArgs(
        std::move(distribution_spec), std::move(shard_spec_buffer), memory_config_.memory_layout());
}

size_t TensorLayout::compute_packed_buffer_size_bytes(const ttnn::Shape& shape) const {
    const Shape2D physical_size = compute_physical_shape(shape);
    const Shape2D page_shape = compute_page_shape(physical_size);
    const auto width_remainder = physical_size.width() % page_shape.width();
    const auto height_remainder = physical_size.height() % page_shape.height();
    TT_FATAL(
        (width_remainder == 0 && height_remainder == 0) || ((physical_size.width() * physical_size.height()) == 0),
        "Physical size {} must be multiple of page size {}",
        physical_size,
        page_shape);

    const size_t physical_area = physical_size.height() * physical_size.width();
    const size_t page_area = page_shape.height() * page_shape.width();

    const size_t page_count = physical_area / page_area;
    const size_t page_size_bytes = compute_page_size_bytes(page_shape);

    return page_count * page_size_bytes;
}

size_t TensorLayout::compute_page_size_bytes(const ttnn::Shape& shape) const {
    const auto physical_size = compute_physical_shape(shape);
    const auto page_shape = compute_page_shape(physical_size);
    return compute_page_size_bytes(page_shape);
}

size_t TensorLayout::compute_page_size_bytes(const Shape2D& page_size) const {
    return page_config_.get_page_size_bytes(page_size, dtype_);
}

size_t TensorLayout::compute_consumed_memory_bytes_per_bank(
    const ttnn::Shape& shape, size_t page_alignment, size_t num_banks) const {
    const Shape2D physical_shape = compute_physical_shape(shape);
    const Shape2D page_shape = compute_page_shape(physical_shape);

    size_t num_pages_per_bank = 0;
    if (!memory_config_.is_sharded()) {
        const size_t num_pages =
            physical_shape.height() * physical_shape.width() / page_shape.height() / page_shape.width();
        num_pages_per_bank = div_up(num_pages, num_banks);
    } else if (const auto& shard_spec = memory_config_.shard_spec()) {
        Shape2D shard_shape = Shape2D(shard_spec->shape);
        if (shard_spec->physical_shard_shape.has_value()) {
            shard_shape = shard_spec->physical_shard_shape.value();
        }
        num_pages_per_bank =
            div_up(shard_shape.height(), page_shape.height()) * div_up(shard_shape.width(), page_shape.width());
    } else {
        auto sharding_args = compute_buffer_sharding_args(shape);
        const auto& dist_spec = sharding_args.buffer_distribution_spec().value();
        num_pages_per_bank = dist_spec.max_num_dev_pages_per_core();
    }

    const size_t aligned_page_size = round_up(compute_page_size_bytes(page_shape), page_alignment);
    return num_pages_per_bank * aligned_page_size;
}

size_t TensorLayout::compute_consumed_memory_bytes_per_bank(const ttnn::Shape& shape, const IDevice& device) const {
    const size_t page_alignment = device.allocator()->get_alignment(memory_config_.buffer_type());
    size_t num_banks = 0;
    if (memory_config_.is_l1()) {
        num_banks = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y;
    } else {
        num_banks = device.num_dram_channels();
    }
    return compute_consumed_memory_bytes_per_bank(shape, page_alignment, num_banks);
}

Shape2D TensorLayout::get_logical_shard_shape() const {
    TT_FATAL(
        memory_config_.shard_spec().has_value(),
        "Shard spec must have value for TensorLayout::get_logical_shard_shape!");

    // In physical mode, shape in shard spec is logical shard shape if no padding
    // Otherwise, not possible to infer logical shard shape in general
    return Shape2D(memory_config_.shard_spec().value().shape);
}

Shape2D TensorLayout::get_physical_shard_shape() const {
    TT_FATAL(
        memory_config_.shard_spec().has_value(),
        "Shard spec must have value for TensorLayout::get_physical_shard_shape!");
    const auto& shard_spec = memory_config_.shard_spec().value();

    auto compute_physical_shard_shape_for_logical_mode = [&]() -> Shape2D {
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

        const auto& logical_shard_shape = Shape2D(shard_spec.shape);
        // TODO: Alignment is guaranteed to be rank 2 or less if tensor is sharded (remove validate?)
        const int alignment_rank = static_cast<int>(alignment_.size());
        TT_FATAL(
            alignment_rank <= 2, "Alignment {} must be rank 2 or less to compute physical shard shape", alignment_);
        auto physical_shard_height = CMAKE_UNIQUE_NAMESPACE::round_up(logical_shard_shape.height(), alignment_[-2]);
        auto physical_shard_width = CMAKE_UNIQUE_NAMESPACE::round_up(logical_shard_shape.width(), alignment_[-1]);
        return Shape2D{physical_shard_height, physical_shard_width};
    };

    switch (shard_spec.mode) {
        case ShardMode::PHYSICAL: return shard_spec.shape; break;
        case ShardMode::LOGICAL: return compute_physical_shard_shape_for_logical_mode(); break;
        default: TT_THROW("Unsupported shard mode {} in get_physical_shard_shape!", shard_spec.mode);
    }
}

Shape2D TensorLayout::compute_logical_2d_shape(const ttnn::Shape& shape) const {
    if (shape.rank() < 2) {
        return Shape2D{1, shape[-1]};
    }
    size_t width = shape[-1];
    size_t height = shape[-2];
    for (int i = -3; i >= -shape.rank(); --i) {
        height *= shape[i];
    }
    return Shape2D{height, width};
}

Shape2D TensorLayout::compute_physical_shape(const ttnn::Shape& shape) const {
    const int rank = static_cast<int>(shape.rank());
    const int alignment_rank = static_cast<int>(alignment_.size());

    size_t width = 1;
    size_t height = 1;

    // LOGICAL SHARDING
    if (memory_config_.shard_spec().has_value() and memory_config_.shard_spec().value().mode == ShardMode::LOGICAL) {
        // Iterate dims in reverse order
        for (int i = -1; i >= -rank; --i) {
            auto& dim = i == -1 ? width : height;
            dim *= shape[i];
        }

        const auto& logical_shard_shape = get_logical_shard_shape();
        const auto& physical_shard_shape = get_physical_shard_shape();

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

        Shape2D size{physical_height, physical_width};
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

    Shape2D size{height, width};
    return size;
}

Shape2D TensorLayout::compute_page_shape(const Shape2D& physical_size) const {
    std::optional<Shape2D> physical_shard_shape = std::nullopt;
    if (memory_config_.shard_spec().has_value()) {
        physical_shard_shape = get_physical_shard_shape();
    }

    return page_config_.get_page_shape(physical_size, dtype_, memory_config_, physical_shard_shape);
}

Strides TensorLayout::compute_strides(const ttnn::Shape& logical_shape) const {
    const int rank = static_cast<int>(logical_shape.rank());
    const int alignment_rank = static_cast<int>(alignment_.size());
    Strides strides(rank, 1);
    for (int i = rank - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * logical_shape[i + 1];
        const int alignment_index = i - (rank - alignment_rank) + 1;
        if (alignment_index >= 0) {
            strides[i] = CMAKE_UNIQUE_NAMESPACE::round_up(strides[i], alignment_[alignment_index]);
        }
    }
    return strides;
}

ttnn::Shape TensorLayout::compute_padded_shape(const ttnn::Shape& shape) const {
    ttnn::SmallVector<uint32_t> padded_shape(std::max(shape.rank(), alignment_.size()));
    int rank_index = static_cast<int>(shape.rank()) - 1;
    int alignment_index = static_cast<int>(alignment_.size()) - 1;
    int padded_shape_index = static_cast<int>(padded_shape.size() - 1);
    size_t accum_alignment = 1;

    for (; alignment_index >= 0; rank_index--, alignment_index--, padded_shape_index--) {
        uint32_t shape_value = rank_index >= 0 ? shape[rank_index] : 1;
        uint32_t alignment_value = alignment_[alignment_index];
        uint32_t& padded_shape_value = padded_shape[padded_shape_index];
        // The last 2 dimensions of a shape are special
        if (rank_index >= static_cast<int>(shape.rank()) - 2) {
            padded_shape_value = CMAKE_UNIQUE_NAMESPACE::round_up(shape_value, alignment_value);
        } else {
            if (accum_alignment % alignment_value == 0) {
                // Alignment for this dimension is redundant, ignoring
                padded_shape_value = shape_value;
            } else if (alignment_value % accum_alignment == 0) {
                padded_shape_value = CMAKE_UNIQUE_NAMESPACE::round_up(shape_value, alignment_value / accum_alignment);
            } else {
                TT_THROW(
                    "Padded shape can't be deducted from TensorLayout parameters {} and Shape {}", alignment_, shape);
            }
        }

        // Alignment doesn't accumulate on the last dimension of a shape
        if (rank_index != static_cast<int>(shape.rank()) - 1) {
            accum_alignment *= padded_shape_value;
        }
    }
    for (; rank_index >= 0; rank_index--, padded_shape_index--) {
        padded_shape[padded_shape_index] = shape[rank_index];
    }
    return ttnn::Shape(std::move(padded_shape));
}

}  // namespace tt::tt_metal
