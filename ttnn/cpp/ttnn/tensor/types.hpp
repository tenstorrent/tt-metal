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

#include "common/bfloat16.hpp"
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/tt_stl/concepts.hpp"
#include "tt_metal/tt_stl/reflection.hpp"
#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/cpp/ttnn/tensor/enum_types.hpp"

namespace ttnn {
    /**
SimpleShape is a temporary measure aimed at making a clear distinction between Shape/LegacyShape when padding information is stripped.
Context:
    Both Shape and LegacyShape can carry padding information.
    And we use Shape or LegacyShape to carry full shape info or separately logical or physical shape.
    Absence of distinction between full shape and logical shape leads to confusion and makes further refactoring harder.
Plan:
    We want to replace `Shape Shape::with_tile_padding() const` with `SimpleShape padded_shape() const`
    We will clearly see where full shape is used vs logical or physical shape is used.
    Need to split .hpp and .cpp
**/
class SimpleShape {
public:
    explicit SimpleShape(const std::vector<uint32_t>& shape) : value(shape) {}
    explicit SimpleShape(std::vector<uint32_t>&& shape) : value(std::move(shape)) {}
    explicit SimpleShape(std::initializer_list<uint32_t> ilist) : value(ilist) {}
    template<std::size_t N>
    explicit SimpleShape(const std::array<uint32_t, N>& arr) : value(arr.begin(), arr.end()) {}

    template<std::size_t N>
    bool operator==(const std::array<uint32_t, N> &other) const {
        bool sameSize = value.size() == N;
        return sameSize && std::equal(value.begin(), value.end(), other.begin());
    }

    bool operator==(const SimpleShape &other) const;
    bool operator==(const std::vector<uint32_t> &other) const;

    uint32_t operator[](int32_t index) const;
    uint32_t &operator[](int32_t index);

    size_t rank() const { return this->value.size(); }
    uint64_t volume() const;

    auto cbegin() const { return this->value.cbegin(); }
    auto cend() const { return this->value.cend(); }

    const std::vector<uint32_t>& as_vector() const { return this->value; }

    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("value");
    auto attribute_values() const { return std::forward_as_tuple(this->value); }

    friend std::ostream &operator<<(std::ostream &os, const SimpleShape &shape);

private:
    std::vector<uint32_t> value;
};

inline std::ostream &operator<<(std::ostream &os, const ttnn::SimpleShape &shape) {
    os << "SimpleShape([";
    for (size_t i = 0; i < shape.rank(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << shape[i];
    }
    os << "])";
    return os;
}

} // namespace ttnn

namespace tt {

namespace tt_metal {

static constexpr std::uint8_t VERSION_ID = 3;

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

inline bool is_floating_point(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16:
        case DataType::FLOAT32:
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: return true;
        default: return false;
    }
}

enum class StorageType {
    OWNED,
    DEVICE,
    BORROWED,           // for storing torch/numpy/etc tensors
    MULTI_DEVICE,       // on-device storage for multi-device context
    MULTI_DEVICE_HOST,  // host storage for multi-device context
};

struct AllGatherTensor {};
bool operator==(const AllGatherTensor &, const AllGatherTensor &);
struct ReplicateTensor {
    int replication_factor = 1;
    ReplicateTensor() = default;
    ReplicateTensor(int replication_factor) : replication_factor(replication_factor) {}
};
bool operator==(const ReplicateTensor &, const ReplicateTensor &);
struct ShardTensor {
    int shard_dimension;
    ShardTensor(int shard_dimension) : shard_dimension(shard_dimension) {}
};
bool operator==(const ShardTensor &lhs, const ShardTensor &rhs);

using ShardMesh = std::pair<std::uint16_t, std::uint16_t>;  // (y,x)
struct ShardTensor2D {
    ShardMesh shard_mesh;  // logic 2D grid that defines the mapping of shards to devices
    ShardTensor2D(ShardMesh mesh) : shard_mesh(std::move(mesh)) {}
};
bool operator==(const ShardTensor2D &lhs, const ShardTensor2D &rhs);

// DistributedTensorConfig is a variant of different ways in which a tensor can be distributed across devices.
using DistributedTensorConfig = std::variant<ReplicateTensor, ShardTensor, ShardTensor2D, AllGatherTensor>;
DistributedTensorConfig get_distributed_tensor_config(const std::unordered_map<std::string, std::string> &metadata);

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
    Padding(const std::vector<PadDimension> &pad_dimensions, PadValue pad_value);

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
    LegacyShape(const std::vector<uint32_t> &);
    LegacyShape(const std::initializer_list<uint32_t>, const Padding &);
    LegacyShape(const std::vector<uint32_t> &, const Padding &);

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
    explicit LegacyShape(const std::vector<uint32_t> &shape, const std::vector<uint32_t> &shape_with_tile_padding) :
        rank_(shape.size()), dimensions_{}, padding_{shape.size()} {
        TT_ASSERT(
            shape.size() == shape_with_tile_padding.size(),
            "Shape and shape_with_tile_padding must have the same size");
        for (auto index = 0; index < shape.size(); index++) {
            auto padded_dimension = shape_with_tile_padding[index];
            this->dimensions_[index] = padded_dimension;
            this->padding_[index] = {.front = 0, .back = padded_dimension - shape[index]};
        }
    }

    std::size_t rank() const;
    std::size_t size() const;

    uint32_t &operator[](const std::int64_t index);
    const uint32_t operator[](const std::int64_t index) const;

    const uint32_t *begin() const;
    const uint32_t *end() const;

    const Padding &padding() const;
    const LegacyShape without_padding() const;

    ttnn::SimpleShape logical_shape() const;

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

bool operator==(const MemoryConfig &config_a, const MemoryConfig &config_b);
bool operator!=(const MemoryConfig &config_a, const MemoryConfig &config_b);

void dump_memory_config(std::ostream &output_stream, const MemoryConfig &memory_config);
void dump_memory_config(const std::string &file_name, const MemoryConfig &memory_config);

MemoryConfig load_memory_config(std::ifstream &input_stream);
MemoryConfig load_memory_config(const std::string &file_name);

using OwnedBuffer = std::variant<
    owned_buffer::Buffer<uint8_t>,
    owned_buffer::Buffer<uint16_t>,
    owned_buffer::Buffer<int32_t>,
    owned_buffer::Buffer<uint32_t>,
    owned_buffer::Buffer<float>,
    owned_buffer::Buffer<bfloat16>>;

// HostDataType supports all types included in OwnedBuffer as well as void*
static_assert(
    std::variant_size_v<OwnedBuffer> + 1 == std::variant_size_v<tt::tt_metal::HostDataType>,
    "The data types supported in OwnedBuffer must match those in HostDataType.");
struct OwnedStorage {
    OwnedBuffer buffer;
    OwnedStorage() = default;
    OwnedStorage(OwnedBuffer buffer_) : buffer(std::move(buffer_)) {}

    static constexpr auto attribute_names = std::forward_as_tuple();
    const auto attribute_values() const { return std::forward_as_tuple(); }

    inline void insert_buffer(const OwnedBuffer &buffer_) { this->buffer = buffer_; }

    inline OwnedBuffer get_buffer() const { return this->buffer; }

    inline bool is_allocated() const {
        return std::visit([](auto &&buffer) -> bool { return buffer.is_allocated(); }, buffer);
    }
};

using DeviceBuffer = std::shared_ptr<Buffer>;
struct DeviceStorage {
    DeviceBuffer buffer;
    DeviceStorage() = default;
    DeviceStorage(DeviceBuffer buffer_) : buffer(std::move(buffer_)) {}

    const MemoryConfig memory_config() const {
        if (this->buffer.get() == nullptr) {
            TT_THROW("MemoryConfig can only be obtained if the buffer is not null");
        }

        std::optional<ShardSpec> shard_spec = std::nullopt;
        if (is_sharded(this->buffer->buffer_layout())) {
            shard_spec = this->buffer->shard_spec().tensor_shard_spec;
        }
        return MemoryConfig{
            .memory_layout = this->buffer->buffer_layout(),
            .buffer_type = this->buffer->buffer_type(),
            .shard_spec = shard_spec};
    }

    inline void insert_buffer(DeviceBuffer buffer_) { this->buffer = buffer_; }

    inline DeviceBuffer get_buffer() const { return this->buffer; }
    static constexpr auto attribute_names = std::forward_as_tuple("memory_config");
    const auto attribute_values() const { return std::make_tuple(this->memory_config()); }

    inline bool is_allocated() const { return buffer && buffer->size() > 0; }
};

using BorrowedBuffer = std::variant<
    borrowed_buffer::Buffer<uint8_t>,
    borrowed_buffer::Buffer<uint16_t>,
    borrowed_buffer::Buffer<int32_t>,
    borrowed_buffer::Buffer<uint32_t>,
    borrowed_buffer::Buffer<float>,
    borrowed_buffer::Buffer<bfloat16>>;
struct BorrowedStorage {
    BorrowedBuffer buffer;

    BorrowedStorage() = default;
    std::function<void()> on_creation_callback = [] {};
    std::function<void()> on_destruction_callback = [] {};

    explicit BorrowedStorage(
        const BorrowedBuffer &buffer,
        const std::function<void()> &on_creation_callback,
        const std::function<void()> &on_destruction_callback) :
        buffer(buffer), on_creation_callback(on_creation_callback), on_destruction_callback(on_destruction_callback) {
        this->on_creation_callback();
    }

    BorrowedStorage(const BorrowedStorage &other) :
        buffer(other.buffer),
        on_creation_callback(other.on_creation_callback),
        on_destruction_callback(other.on_destruction_callback) {
        this->on_creation_callback();
    }

    BorrowedStorage operator=(const BorrowedStorage &other) {
        this->buffer = other.buffer;
        this->on_creation_callback = other.on_creation_callback;
        this->on_destruction_callback = other.on_destruction_callback;
        this->on_creation_callback();
        return *this;
    }

    BorrowedStorage(BorrowedStorage &&other) :
        buffer(other.buffer),
        on_creation_callback(other.on_creation_callback),
        on_destruction_callback(other.on_destruction_callback) {
        other.on_creation_callback = [] {};
        other.on_destruction_callback = [] {};
    }

    BorrowedStorage operator=(BorrowedStorage &&other) {
        this->buffer = other.buffer;
        this->on_creation_callback = other.on_creation_callback;
        this->on_destruction_callback = other.on_destruction_callback;
        other.on_creation_callback = [] {};
        other.on_destruction_callback = [] {};
        return *this;
    }

    ~BorrowedStorage() { this->on_destruction_callback(); }

    static constexpr auto attribute_names = std::forward_as_tuple();
    const auto attribute_values() const { return std::forward_as_tuple(); }

    inline bool is_allocated() const { return true; }
};

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
    return tt::tt_metal::LegacyShape{ttl_shape, tt::tt_metal::Padding{padding, tt::tt_metal::Padding::PadValue::Any}};
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

    Shape(const std::vector<uint32_t> &shape) : value{tt::tt_metal::LegacyShape{shape}} {}

    explicit Shape(const std::vector<uint32_t> &shape, const std::vector<uint32_t> &shape_with_tile_padding) :
        value{tt::tt_metal::LegacyShape{shape, shape_with_tile_padding}} {}

    explicit Shape(const std::vector<uint32_t> &shape, const Padding &padding) :
        value{tt::tt_metal::LegacyShape{shape, padding}} {}

    explicit Shape(const Shape &shape, const Padding &padding) :
        value{tt::tt_metal::LegacyShape{shape.value, padding}} {}

    Shape(const SimpleShape& shape): value{shape.as_vector()} {}

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
        std::vector<uint32_t> values(rank());
        for (size_t i = 0; i < values.size(); i++) {
            values[i] = this->value[i]; // value stored LegacyShape, its operator[] returns padded value
        }
        return SimpleShape(std::move(values));
    }

    // Returns the shape without padding, padding information is stripped
    SimpleShape logical_shape() const {
        std::vector<uint32_t> values(this->rank());
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

SimpleShape get_physical_shape(const SimpleShape& logical_shape, DataType data_type, Layout layout, const std::optional<Tile>& tile = std::nullopt);

}  // namespace ttnn

namespace tt {
namespace tt_metal {

struct MultiDeviceHostStorage {
    DistributedTensorConfig strategy;
    std::vector<OwnedBuffer> buffers;
    std::vector<ttnn::Shape> shapes;
    mutable std::mutex mtx;

    friend void swap(MultiDeviceHostStorage &first, MultiDeviceHostStorage &second) {
        std::scoped_lock lock(first.mtx, second.mtx);
        // enable ADL (not necessary, but good practice)
        using std::swap;

        swap(first.strategy, second.strategy);
        swap(first.buffers, second.buffers);
        swap(first.shapes, second.shapes);
    }

    MultiDeviceHostStorage() = default;
    MultiDeviceHostStorage(
        DistributedTensorConfig strategy_, std::vector<OwnedBuffer> buffers_, std::vector<ttnn::Shape> shapes_) :
        strategy(strategy_), buffers(buffers_), shapes(shapes_) {}
    MultiDeviceHostStorage(MultiDeviceHostStorage &&other) { swap(*this, other); }
    // unfotunately we need to have this code written manually.
    MultiDeviceHostStorage(const MultiDeviceHostStorage &other) {
        std::scoped_lock lock(other.mtx);
        strategy = other.strategy;
        buffers = other.buffers;
        shapes = other.shapes;
    }

    MultiDeviceHostStorage &operator=(const MultiDeviceHostStorage &other) {
        MultiDeviceHostStorage temp(other);
        swap(*this, temp);
        return *this;
    }

    MultiDeviceHostStorage &operator=(MultiDeviceHostStorage &&other) {
        swap(*this, other);
        return *this;
    }

    bool operator==(const MultiDeviceHostStorage &other) {
        return this->strategy == other.strategy and this->buffers == other.buffers and this->shapes == other.shapes;
    }

    static constexpr auto attribute_names = std::forward_as_tuple();
    const auto attribute_values() const { return std::forward_as_tuple(); }

    // Helper Functions - Getters and setters to get/modify storage attributes. These are needed to
    // preinitialize empty tensor handles and use/populate them in the worker threads.
    void insert_buffer_and_shape_for_device(int buffer_index, const OwnedBuffer &buffer, const ttnn::Shape shape) {
        std::lock_guard<std::mutex> lock(mtx);
        buffers[buffer_index] = buffer;
        shapes[buffer_index] = shape;
    }

    OwnedBuffer get_buffer(int buffer_index) const {
        std::lock_guard<std::mutex> lock(mtx);
        TT_ASSERT(buffer_index < buffers.size(), "Buffer not found for buffer_index {}",buffer_index);
        return buffers[buffer_index];
    }

    OwnedBuffer &get_buffer(int buffer_index) {
        std::lock_guard<std::mutex> lock(mtx);
        TT_ASSERT(buffer_index < buffers.size(), "Buffer not found for buffer_index {}", buffer_index);
        return buffers[buffer_index];
    }

    ttnn::Shape get_tensor_shape(int shape_index) const {
        std::lock_guard<std::mutex> lock(mtx);
        TT_ASSERT(shape_index < shapes.size(), "Buffer not found for device {}", shape_index);
        return shapes[shape_index];
    }

    uint32_t num_buffers() const {
        std::lock_guard<std::mutex> lock(mtx);
        return buffers.size();
    }

    inline bool is_allocated() const {
        // not sure what is better mutex for each buffer 10 times or one here.
        // I think this one is better.
        std::lock_guard<std::mutex> lock(mtx);

        return std::all_of(buffers.begin(), buffers.end(), [](auto &&buffer) {
            return std::visit([](auto &&buffer) -> bool { return buffer.is_allocated(); }, buffer);
        });
    }
};

struct MultiDeviceStorage {
    DistributedTensorConfig strategy;
    std::vector<int> ordered_device_ids;
    std::unordered_map<int, DeviceBuffer> buffers;
    std::unordered_map<int, ttnn::Shape> shapes;
    mutable std::mutex buffer_mtx;
    mutable std::mutex shape_mtx;
    MultiDeviceStorage() = default;

    friend void swap(MultiDeviceStorage &first, MultiDeviceStorage &second) {
        std::scoped_lock lock(first.buffer_mtx, first.shape_mtx, second.buffer_mtx, second.shape_mtx);

        swap(first.strategy, second.strategy);
        swap(first.ordered_device_ids, second.ordered_device_ids);
        swap(first.buffers, second.buffers);
        swap(first.shapes, second.shapes);
    }

    MultiDeviceStorage(
        DistributedTensorConfig strategy_,
        std::vector<int> ordered_device_ids_,
        std::unordered_map<int, DeviceBuffer> buffers_,
        std::unordered_map<int, ttnn::Shape> shapes_) :
        strategy(std::move(strategy_)),
        ordered_device_ids(std::move(ordered_device_ids_)),
        buffers(std::move(buffers_)),
        shapes(std::move(shapes_)) {}

    MultiDeviceStorage(MultiDeviceStorage &&other) { swap(*this, other); }

    MultiDeviceStorage(const MultiDeviceStorage &other) {
        std::scoped_lock lock(other.buffer_mtx, other.shape_mtx);
        ordered_device_ids = other.ordered_device_ids;
        strategy = other.strategy;
        buffers = other.buffers;
        shapes = other.shapes;
    }

    MultiDeviceStorage &operator=(const MultiDeviceStorage &other) {
        MultiDeviceStorage tmp(other);
        swap(*this, tmp);
        return *this;
    }

    MultiDeviceStorage &operator=(MultiDeviceStorage &&other) {
        swap(*this, other);
        return *this;
    }

    bool operator==(const MultiDeviceStorage &other) {
        return this->ordered_device_ids == other.ordered_device_ids and this->strategy == other.strategy and
               this->buffers == other.buffers and this->shapes == other.shapes;
    }

    inline const MemoryConfig memory_config() const {
        std::lock_guard<std::mutex> lock(buffer_mtx);
        TT_FATAL(!this->ordered_device_ids.empty(), "No device ids in list. Please ensure fields are initialized properly.");
        auto first_device_id = this->ordered_device_ids[0];
        if (this->buffers.at(first_device_id).get() == nullptr) {
            TT_THROW("MemoryConfig can only be obtained if the buffer is not null");
        }
        std::optional<ShardSpec> shard_spec = std::nullopt;
        if (is_sharded(this->buffers.at(first_device_id)->buffer_layout())) {
            shard_spec = this->buffers.at(first_device_id)->shard_spec().tensor_shard_spec;
        }
        return MemoryConfig{
            .memory_layout = this->buffers.at(first_device_id)->buffer_layout(),
            .buffer_type = this->buffers.at(first_device_id)->buffer_type(),
            .shard_spec = shard_spec};
    }

    static constexpr auto attribute_names = std::forward_as_tuple();
    const auto attribute_values() const { return std::forward_as_tuple(); }

    // Helper Functions - Getters and setters to get/modify storage attributes. These are needed to
    // preinitialize empty tensor handles and use/populate them in the worker threads.

    inline void insert_buffer_and_shape_for_device(Device *device, const DeviceBuffer buffer, const ttnn::Shape shape) {
        std::scoped_lock lock(buffer_mtx, shape_mtx);
        TT_ASSERT(
            device == buffer->device(),
            "Mismatch between device derived from buffer and device derived from MultiDeviceStorage.");
        buffers.insert({device->id(), buffer});
        shapes.insert({device->id(), shape});
    }

    inline DeviceBuffer get_buffer_for_device(Device *device) const {
        std::lock_guard<std::mutex> lock(buffer_mtx);
        TT_ASSERT(
            buffers.find(device->id()) != buffers.end(), "Buffer not found for device {}",device->id());
        TT_ASSERT(
            buffers.at(device->id())->device() == device,
            "Mismatch between device derived from buffer and device derived from MultiDeviceStorage.");
        return buffers.at(device->id());
    }

    inline DeviceBuffer &get_buffer_for_device(Device *device) {
        std::lock_guard<std::mutex> lock(buffer_mtx);
        TT_ASSERT(
            buffers.find(device->id()) != buffers.end(), "Buffer not found for device {}", device->id());
        TT_ASSERT(
            buffers.at(device->id())->device() == device,
            "Mismatch between device derived from buffer and device derived from MultiDeviceStorage.");
        return buffers.at(device->id());
    }

    inline DeviceBuffer get_buffer_for_device_id(uint32_t device_id) const {
        std::lock_guard<std::mutex> lock(buffer_mtx);
        return buffers.at(device_id);
    }

    inline ttnn::Shape get_tensor_shape_for_device(Device *device) const {
        std::lock_guard<std::mutex> lock(shape_mtx);
        TT_ASSERT(
            shapes.find(device->id()) != shapes.end(), "Shape not found for device {}", device->id());
        return shapes.at(device->id());
    }

    inline uint32_t num_buffers() const {
        std::lock_guard<std::mutex> lock(buffer_mtx);
        return buffers.size();
    }

    inline bool has_buffer_for_device(Device *device) const {
        std::lock_guard<std::mutex> lock(buffer_mtx);
        return buffers.find(device->id()) != buffers.end();
    }

    inline bool has_buffer_for_device_id(uint32_t device_id) const {
        std::lock_guard<std::mutex> lock(buffer_mtx);
        return buffers.find(device_id) != buffers.end();
    }

    inline bool is_allocated() const {
        std::lock_guard<std::mutex> lock(buffer_mtx);

        return std::all_of(
            ordered_device_ids.begin(), ordered_device_ids.end(), [&buffers = this->buffers](auto &&device_id) {
                const auto &buffer = buffers.at(device_id);
                return buffer && buffer->size() > 0;
            });
    }
};

using Storage = std::variant<OwnedStorage, DeviceStorage, BorrowedStorage, MultiDeviceHostStorage, MultiDeviceStorage>;

template <typename T>
constexpr void raise_unsupported_storage() {
    static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported Storage");
}

}  // namespace tt_metal

}  // namespace tt
