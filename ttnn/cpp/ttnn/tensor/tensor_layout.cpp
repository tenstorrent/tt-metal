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

Alignment legacyPaddedShapeToAlignment(const ttnn::SimpleShape& legacy_padded_shape) {
    const auto rank = legacy_padded_shape.rank();
    std::vector<uint32_t> values(rank);

    if(rank >= 1) {
        values[rank - 1] = legacy_padded_shape[rank - 1];
    }
    if(rank >= 2) {
        values[rank - 2] = legacy_padded_shape[rank - 2];
    }
    for (int i = rank - 3; i >= 0; i--) {
        values[i] = legacy_padded_shape[i] * values[i + 1];
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


TensorLayout::TensorLayout(DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const Alignment& alignment)
    : m_dtype(dtype),
      m_page_config(page_config),
      m_memory_config(memory_config),
      m_alignment(alignment) {

    initialize_alignment();
    validate_alignment();
}

TensorLayout TensorLayout::fromLegacyPaddedShape(DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const ttnn::SimpleShape& legacy_padded_shape) {
    return TensorLayout(dtype, page_config, memory_config, legacyPaddedShapeToAlignment(legacy_padded_shape));
}

void TensorLayout::initialize_alignment() {
    if(m_alignment.size() != 0)
        return;

    m_alignment = m_page_config.create_default_alignment(m_dtype);
}

void TensorLayout::validate_alignment() const
{
    return m_page_config.validate_alignment(m_alignment, m_dtype);
}

std::optional<ShardSpecBuffer> TensorLayout::get_shard_spec_buffer(const ttnn::SimpleShape& shape) const {
    if (!m_memory_config.is_sharded())
        return std::nullopt;

    TT_FATAL(m_memory_config.shard_spec.has_value(), "MemoryConfig should have Shard Spec specified for sharded memory layout");

    auto& shard_spec = m_memory_config.shard_spec.value();
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
    return m_page_config.get_page_size_bytes(page_size, m_dtype);
}

Size TensorLayout::get_physical_shape(const ttnn::SimpleShape& shape) const {
    const int rank = static_cast<int>(shape.rank());
    const int alignment_rank = static_cast<int>(m_alignment.size());
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
            dim = round_up(dim, m_alignment[i]);
        }
    }

    Size size{height, width};

    return size;
}

Size TensorLayout::get_page_shape(const Size& physical_size) const {
    if(m_memory_config.memory_layout == TensorMemoryLayout::SINGLE_BANK) {
        return physical_size;
    }

    return m_page_config.get_page_shape(physical_size, m_memory_config);
}

Strides TensorLayout::get_strides(const ttnn::SimpleShape& shape) const {
    const int rank = static_cast<int>(shape.rank());
    const int alignment_rank = static_cast<int>(m_alignment.size());

    Strides strides(rank, 1);
    for (int i = rank - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];

        const int alignment_index = i - (rank - alignment_rank) + 1;
        if(alignment_index >= 0) {
            strides[i] = round_up(strides[i], m_alignment[alignment_index]);
        }
    }

    return strides;
}

ttnn::SimpleShape TensorLayout::get_padded_shape(const ttnn::SimpleShape& shape) const
{
    std::vector<uint32_t> padded_shape(shape.rank());
    int rank_index = static_cast<int>(shape.rank()) - 1;
    int alignment_index = static_cast<int>(m_alignment.size()) - 1;
    size_t accum_alignment = 1;

    for (;rank_index >= 0 && alignment_index >= 0; rank_index--, alignment_index--) {
        // The last 2 dimensions of a shape are special
        if (rank_index >= static_cast<int>(shape.rank()) - 2) {
            padded_shape[rank_index] = round_up(shape[rank_index], m_alignment[alignment_index]);
        } else {
            if (accum_alignment % m_alignment[alignment_index] == 0) {
                // Alignment for this dimension is redundant, ignoring
                padded_shape[rank_index] = shape[rank_index];
            } else if (m_alignment[alignment_index] % accum_alignment == 0) {
                padded_shape[rank_index] = round_up(shape[rank_index], m_alignment[alignment_index] / accum_alignment);
            } else {
                TT_THROW("Padded shape can't be deducted from TensorLayout parameters {} and Shape {}", m_alignment, shape);
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
