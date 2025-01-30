// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>
#include <algorithm>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/device_impl.hpp>
#include <tt-metalium/reflection.hpp>
#include <tt-metalium/span.hpp>
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/tensor/host_buffer/types.hpp"
#include "cpp/ttnn/tensor/enum_types.hpp"

#include "ttnn/tensor/shape/shape.hpp"

namespace tt {

namespace tt_metal {

static constexpr std::uint8_t VERSION_ID = 4;

enum class DataType {
    BFLOAT16 = 0,
    FLOAT32 = 1,
    UINT32 = 2,
    BFLOAT8_B = 3,
    BFLOAT4_B = 4,
    UINT8 = 5,
    UINT16 = 6,
    INT32 = 7,
    INVALID = 8,
};

template <typename T>
consteval inline DataType convert_to_data_type() {
    if constexpr (std::is_same_v<T, uint8_t>) {
        return DataType::UINT8;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return DataType::UINT16;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DataType::INT32;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return DataType::UINT32;
    } else if constexpr (std::is_same_v<T, float>) {
        return DataType::FLOAT32;
    } else if constexpr (std::is_same_v<T, ::bfloat16>) {
        return DataType::BFLOAT16;
    } else {
        static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported DataType!");
    }
}

bool is_floating_point(DataType dtype);

bool is_block_float(DataType dtype);

enum class StorageType {
    OWNED,
    DEVICE,
    BORROWED,           // for storing torch/numpy/etc tensors
    MULTI_DEVICE,       // on-device storage for multi-device context
    MULTI_DEVICE_HOST,  // host storage for multi-device context
};

tt::DataFormat datatype_to_dataformat_converter(DataType datatype);

static constexpr std::size_t MAX_NUM_DIMENSIONS = 8;

struct Padding {
    enum class PadValue { Any, Zero, Infinity, NegativeInfinity };

    struct PadDimension {
        std::size_t front;
        std::size_t back;
    };

    std::size_t rank_;
    std::array<PadDimension, MAX_NUM_DIMENSIONS> pad_dimensions_;
    PadValue pad_value_;

    Padding(const Padding &) = default;
    Padding &operator=(const Padding &) = default;
    Padding(Padding &&) = default;
    Padding &operator=(Padding &&) = default;
    ~Padding() = default;

    Padding(const std::size_t rank);
    Padding(const std::initializer_list<PadDimension> pad_dimensions, PadValue pad_value);
    Padding(tt::stl::Span<const PadDimension> pad_dimensions, PadValue pad_value);

    template <std::size_t Rank>
    Padding(const std::array<std::array<uint32_t, 2>, Rank> pad_dimensions, PadValue pad_value) :
        rank_(pad_dimensions.size()), pad_dimensions_{}, pad_value_(pad_value) {
        for (auto index = 0; index < Rank; index++) {
            this->pad_dimensions_[index] = {.front = pad_dimensions[index][0], .back = pad_dimensions[index][1]};
        }
    }

    const uint32_t get_normalized_index(std::int64_t index) const;

    PadDimension &operator[](const std::int64_t index);
    const PadDimension operator[](const std::int64_t index) const;

    PadValue pad_value() const;

    size_t rank() const { return rank_; }

    static constexpr auto attribute_names = std::forward_as_tuple("rank", "pad_dimensions", "pad_value");
    const auto attribute_values() const {
        return std::forward_as_tuple(this->rank_, this->pad_dimensions_, this->pad_value_);
    }
    friend std::ostream &operator<<(std::ostream &os, const Padding &padding);
};

inline std::ostream &operator<<(std::ostream &os, const Padding &padding) {
    os << "Padding(";
    os << "rank: " << padding.rank_;
    os << ", pad_dimensions: [";
    for (std::size_t i = 0; i < padding.rank_; ++i) {
        os << "{front: " << padding.pad_dimensions_[i].front << ", back: " << padding.pad_dimensions_[i].back << "}";
        if (i < padding.rank_ - 1)
            os << ", ";
    }
    os << "]";
    os << ", pad_value: ";
    switch (padding.pad_value_) {
        case Padding::PadValue::Any: os << "Any"; break;
        case Padding::PadValue::Zero: os << "Zero"; break;
        case Padding::PadValue::Infinity: os << "Infinity"; break;
        case Padding::PadValue::NegativeInfinity: os << "NegativeInfinity"; break;
        default: os << "Unknown";
    }
    os << ")";
    return os;
}

bool operator==(const Padding &, const Padding &);
bool operator!=(const Padding &, const Padding &);
typedef std::array<uint32_t, 1> Array1D;
typedef std::array<uint32_t, 2> Array2D;
typedef std::array<uint32_t, 3> Array3D;
typedef std::array<uint32_t, 4> Array4D;
typedef std::array<uint32_t, 5> Array5D;
typedef std::array<uint32_t, 6> Array6D;
typedef std::array<uint32_t, 7> Array7D;
typedef std::array<uint32_t, 8> Array8D;

class LegacyShape {
    std::size_t rank_;
    std::array<uint32_t, MAX_NUM_DIMENSIONS> dimensions_;
    Padding padding_;

   public:
    LegacyShape(const LegacyShape &) = default;
    LegacyShape &operator=(const LegacyShape &) = default;
    LegacyShape(LegacyShape &&) = default;
    LegacyShape &operator=(LegacyShape &&) = default;
    ~LegacyShape() = default;

    LegacyShape(const std::initializer_list<uint32_t>);
    LegacyShape(tt::stl::Span<const uint32_t>);
    LegacyShape(const ttnn::SmallVector<uint32_t>& vec) : LegacyShape(tt::stl::Span(vec)) {};
    LegacyShape(const std::initializer_list<uint32_t>, const Padding &);
    LegacyShape(tt::stl::Span<const uint32_t>, const Padding &);
    LegacyShape(const ttnn::SmallVector<uint32_t>& vec, const Padding &padding) : LegacyShape(tt::stl::Span(vec), padding) {};

    explicit LegacyShape(const LegacyShape &, const Padding &);

    template <std::size_t Rank>
    LegacyShape(const std::array<uint32_t, Rank> &shape) : rank_(Rank), dimensions_{}, padding_{Rank} {
        for (auto index = 0; index < Rank; index++) {
            this->dimensions_[index] = shape[index];
        }
    }

    LegacyShape(const Array4D &shape) : rank_(4), dimensions_{}, padding_{4} {
        for (auto index = 0; index < 4; index++) {
            this->dimensions_[index] = shape[index];
        }
    }

    template <std::size_t Rank>
    explicit LegacyShape(const std::array<uint32_t, Rank> &shape, const std::array<uint32_t, Rank> &shape_with_tile_padding) :
        rank_(Rank), dimensions_{}, padding_{Rank} {
        for (auto index = 0; index < Rank; index++) {
            auto padded_dimension = shape_with_tile_padding[index];
            this->dimensions_[index] = padded_dimension;
            this->padding_[index] = {.front = 0, .back = padded_dimension - shape[index]};
        }
    }
    explicit LegacyShape(tt::stl::Span<const uint32_t> shape, tt::stl::Span<const uint32_t> shape_with_tile_padding) :
    rank_(shape_with_tile_padding.size()), dimensions_{}, padding_{shape_with_tile_padding.size()} {
        for (int index = 0; index < shape_with_tile_padding.size(); index++) {
            int shape_index = index + static_cast<int>(shape.size()) - static_cast<int>(shape_with_tile_padding.size());
            int dimension = shape_index >= 0 ? shape[shape_index] : 1;
            int padded_dimension = shape_with_tile_padding[index];
            this->dimensions_[index] = padded_dimension;
            this->padding_[index] = {.front = 0, .back = static_cast<size_t>(padded_dimension - dimension)};
        }
    }
    explicit LegacyShape(const ttnn::SmallVector<uint32_t>& shape, const ttnn::SmallVector<uint32_t>& shape_with_tile_padding)
        : LegacyShape(tt::stl::Span<const uint32_t>(shape), tt::stl::Span<const uint32_t>(shape_with_tile_padding)) {}
    explicit LegacyShape(const std::initializer_list<uint32_t> shape, const std::initializer_list<uint32_t> shape_with_tile_padding)
        : LegacyShape(ttnn::SmallVector<uint32_t>(shape), ttnn::SmallVector<uint32_t>(shape_with_tile_padding)) {}

    std::size_t rank() const;
    std::size_t size() const;

    uint32_t &operator[](const std::int64_t index);
    const uint32_t operator[](const std::int64_t index) const;

    const uint32_t *begin() const;
    const uint32_t *end() const;

    const Padding &padding() const;
    const LegacyShape without_padding() const;

    ttnn::SimpleShape logical_shape() const;
    ttnn::SimpleShape padded_shape() const;

    const uint32_t get_normalized_index(std::int64_t index) const;

    static constexpr auto attribute_names = std::forward_as_tuple("rank", "dimensions", "padding");
    const auto attribute_values() const {
        return std::forward_as_tuple(this->rank_, this->dimensions_, this->padding_);
    }
    friend std::ostream &operator<<(std::ostream &os, const LegacyShape &shape);

    Array4D to_array_4D() const;
};

inline std::ostream &operator<<(std::ostream &os, const tt::tt_metal::LegacyShape &shape) {
    const auto shape_without_padding = shape.without_padding();
    const auto &padding = shape.padding();
    os << "Shape([";
    for (auto i = 0; i < shape_without_padding.rank(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << shape_without_padding[i];
        if (padding[i].back > 0) {
            os << "[" << shape[i] << "]";
        }
    }
    os << "])";
    return os;
}

bool operator==(const tt::tt_metal::LegacyShape &, const tt::tt_metal::LegacyShape &);
bool operator!=(const tt::tt_metal::LegacyShape &, const tt::tt_metal::LegacyShape &);

struct MemoryConfig {
    TensorMemoryLayout memory_layout = TensorMemoryLayout::INTERLEAVED;  // Interleave the data across multiple banks
    BufferType buffer_type = BufferType::DRAM;                           // Can be either DRAM or L1
    std::optional<ShardSpec> shard_spec = std::nullopt;
    bool is_sharded() const;
    bool is_l1() const;
    bool is_dram() const;
};

std::ostream& operator<<(std::ostream& os, const MemoryConfig& config);

bool operator==(const MemoryConfig &config_a, const MemoryConfig &config_b);
bool operator!=(const MemoryConfig &config_a, const MemoryConfig &config_b);

} // namespace tt_metal
} // namespace tt


namespace ttnn {
namespace types {

namespace detail {
template <std::size_t Rank>
static tt::tt_metal::LegacyShape compute_ttl_shape(
    const std::array<uint32_t, Rank> &shape, const std::array<std::array<uint32_t, 2>, Rank> &padding) {
    auto ttl_shape = std::array<uint32_t, Rank>{};
    for (auto index = 0; index < Rank; index++) {
        ttl_shape[index] = shape[index] + padding[index][0] + padding[index][1];
    }
    return tt::tt_metal::LegacyShape{tt::stl::Span(ttl_shape), tt::tt_metal::Padding{padding, tt::tt_metal::Padding::PadValue::Any}};
}

}  // namespace detail


struct Shape {
    // ttnn::Shape is a wrapper around tt::tt_metal::LegacyShape
    // It is used to flip the default value of operator[] to return the shape without padding
    tt::tt_metal::LegacyShape value;

    Shape(const std::initializer_list<uint32_t> dimensions) : value{dimensions} {}

    Shape(const tt::tt_metal::LegacyShape &shape) : value{shape} {}

    template <std::size_t Rank>
    Shape(const std::array<uint32_t, Rank> &shape) : value{shape} {}

    template <std::size_t Rank>
    explicit Shape(const std::array<uint32_t, Rank> &shape, const std::array<uint32_t, Rank> &shape_with_tile_padding) :
        value{tt::tt_metal::LegacyShape{shape, shape_with_tile_padding}} {}

    template <std::size_t Rank>
    explicit Shape(
        const std::array<uint32_t, Rank> &shape, const std::array<std::array<uint32_t, 2>, Rank> &tile_padding) :
        value{detail::compute_ttl_shape(shape, tile_padding)} {}

    Shape(tt::stl::Span<const uint32_t> shape) : value{tt::tt_metal::LegacyShape{shape}} {}

    Shape(const SmallVector<uint32_t>& shape) : value{tt::tt_metal::LegacyShape{shape}} {}

    explicit Shape(tt::stl::Span<const uint32_t> shape, tt::stl::Span<const uint32_t> shape_with_tile_padding) :
        value{tt::tt_metal::LegacyShape{shape, shape_with_tile_padding}} {}

    explicit Shape(const std::initializer_list<uint32_t> shape, const std::initializer_list<uint32_t> shape_with_tile_padding) :
        value{tt::tt_metal::LegacyShape{shape, shape_with_tile_padding}} {}

    explicit Shape(tt::stl::Span<const uint32_t> shape, const tt::tt_metal::Padding &padding) :
        value{tt::tt_metal::LegacyShape{shape, padding}} {}

    explicit Shape(const Shape &shape, const tt::tt_metal::Padding &padding) :
        value{tt::tt_metal::LegacyShape{shape.value, padding}} {}

    Shape(const SimpleShape& shape): value{shape.view()} {}

    const auto rank() const { return this->value.rank(); }

    const size_t size() const { return this->rank(); }

    // Returns the padded shape, padding information is stripped
    [[deprecated("Replaced by padded_shape()")]]
    const tt::tt_metal::Padding &padding() const { return this->value.padding(); }

    const uint32_t get_normalized_index(std::int64_t index) const { return this->value.get_normalized_index(index); }

    Shape with_tile_padding() const {
        return Shape{tt::tt_metal::LegacyShape{this->value, tt::tt_metal::Padding{this->value.rank()}}};
    }

    SimpleShape padded_shape() const {
        SmallVector<uint32_t> values(rank());
        for (size_t i = 0; i < values.size(); i++) {
            values[i] = this->value[i]; // value stored LegacyShape, its operator[] returns padded value
        }
        return SimpleShape(std::move(values));
    }

    // Returns the shape without padding, padding information is stripped
    SimpleShape logical_shape() const {
        SmallVector<uint32_t> values(this->rank());
        for (size_t i = 0; i < values.size(); i++) {
            values[i] = this->operator[](i); // operator[] returns the shape without padding
        }
        return SimpleShape(std::move(values));
    }

    bool has_tile_padding() const {
        auto rank = this->rank();
        for (auto index = 0; index < rank; index++) {
            if (this->has_tile_padding(index)) {
                return true;
            }
        }
        return false;
    }

    bool has_tile_padding(int dim) const {
        return this->value.padding()[dim].front > 0 or this->value.padding()[dim].back > 0;
    }

    bool operator==(const Shape &other) const {
        const auto &shape_a = this->value;
        const auto &shape_b = other.value;
        // tt::tt_metal::LegacyShape comparison doesn't take padding into account
        return (shape_a == shape_b and shape_a.without_padding() == shape_b.without_padding());
    }

    template <std::size_t Rank>
    bool operator==(const std::array<std::uint32_t, Rank> &other) const {
        return Shape{this->value.without_padding()} == Shape{other};
    }

    bool operator!=(const Shape &other) const { return not(*this == other); }

    // Returns value without padding
    uint32_t operator[](std::int64_t index) const;

    const Shape to_rank(size_t new_rank) const;

    static constexpr auto attribute_names = std::forward_as_tuple("value");
    const auto attribute_values() const { return std::forward_as_tuple(this->value); }
};

static std::ostream &operator<<(std::ostream &os, const Shape &shape) {
    const auto shape_with_tile_padding = shape.with_tile_padding();
    const auto &padding = shape.value.padding();
    os << "ttnn.Shape([";
    for (auto i = 0; i < shape.rank(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << shape[i];
        if (padding[i].back > 0) {
            os << "[" << shape_with_tile_padding[i] << "]";
        }
    }
    os << "])";
    return os;
}

}  // namespace types

using types::Shape;

}  // namespace ttnn
