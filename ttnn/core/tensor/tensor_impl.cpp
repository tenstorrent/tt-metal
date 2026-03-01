// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include <fmt/format.h>
#include <optional>

#include <sys/mman.h>
#include <unistd.h>

#include "tensor/tensor_ops.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"

#include "tt-metalium/shape.hpp"
#include "tt-metalium/math.hpp"
#include "tt-metalium/distributed_host_buffer.hpp"
#include "tt-metalium/host_buffer.hpp"
#include "tt-metalium/memory_pin.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_coord.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_command_queue.hpp"
#include "tt-metalium/experimental/tensor/tensor_apis.hpp"
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>

#include <tt_stl/overloaded.hpp>
#include <tt_stl/span.hpp>
#include <tt_stl/assert.hpp>

#include <tracy/Tracy.hpp>

using namespace tt::tt_metal;

namespace tt::tt_metal::tensor_impl {

PrintOptions TTNN_PRINT_OPTIONS;

std::ostream& operator<<(std::ostream& os, const DataType& dtype) {
    switch (dtype) {
        case DataType::BFLOAT8_B: os << "bfloat8_b"; break;
        case DataType::BFLOAT4_B: os << "bfloat4_b"; break;
        case DataType::BFLOAT16: os << "bfloat16"; break;
        case DataType::FLOAT32: os << "float32"; break;
        case DataType::UINT8: os << "uint8"; break;
        case DataType::UINT16: os << "uint16"; break;
        case DataType::UINT32: os << "uint32"; break;
        case DataType::INT32: os << "int32"; break;
        default: throw std::invalid_argument("Unknown data type");
    }
    return os;
}

// allocate_device_buffer has been moved to tt-metalium/impl/tensor/tensor_apis.cpp

// pad_bfloat8_b, unpad_bfloat8_b, pad_bfloat4_b, unpad_bfloat4_b have been moved to
// tt-metalium/impl/tensor/tensor_apis.cpp

// ======================================================================================
//                                      .to_string()
// ======================================================================================

namespace detail {

struct DimensionShortener {
    size_t size{};
    std::optional<std::size_t> max;

    bool print_parenthesis_and_advance_index_if_reached_half_of_max_and_check_if_loop_is_done(
        std::ostream& ss, std::size_t& index, const std::string& before, const std::string& after) const {
        if (this->max.has_value() and this->size > this->max.value() and index == this->max.value() / 2) {
            ss << before << "...," << after;
            index = this->size - (this->max.value() / 2);
        }
        return index < this->size;
    }
};

inline DimensionShortener get_dimension_shortener(std::size_t size) {
    switch (TTNN_PRINT_OPTIONS.profile) {
        case TensorPrintProfile::Empty: return DimensionShortener{size, 0};
        case TensorPrintProfile::Short: return DimensionShortener{size, 4};
        case TensorPrintProfile::Full: return DimensionShortener{size, std::nullopt};
        default: TT_THROW("Unrecognized TTNN_TENSOR_PRINT_PROFILE {}", TTNN_PRINT_OPTIONS.profile);
    }
}

inline void print_trailing_comma(std::ostream& ss, std::size_t index, std::size_t size, const std::string& after) {
    if (index < size - 1) {
        ss << "," << after;
    }
}

template <typename T>
inline void print_datum(std::ostream& ss, T datum, bool use_scientific = false) {
    if (std::is_integral_v<T>) {
        ss << std::setw(5) << datum;
    } else {
        int precision = TTNN_PRINT_OPTIONS.precision;
        if (use_scientific) {
            // Note: scientific required fixed width + 4 (e+/-AB, e.g. 1.23456e+08)
            ss << std::scientific << std::setw(precision + 7) << std::setprecision(precision) << datum;
        } else {
            ss << std::fixed << std::setw(precision + 3) << std::setprecision(precision) << datum;
        }
    }
}

template <>
inline void print_datum(std::ostream& ss, bfloat16 datum, bool use_scientific) {
    print_datum(ss, static_cast<float>(datum), use_scientific);
}

template <>
inline void print_datum(std::ostream& ss, uint8_t datum, bool use_scientific) {
    print_datum<uint32_t>(ss, datum, use_scientific);
}

// Helper function to determine if scientific notation should be used
template <typename T>
bool should_use_scientific_notation(tt::stl::Span<const T> buffer) {
    if (TTNN_PRINT_OPTIONS.sci_mode == SciMode::Enable) {
        return true;
    }
    if (TTNN_PRINT_OPTIONS.sci_mode == SciMode::Disable) {
        return false;
    }

    // SciMode::Default - auto-detect based on data range
    if constexpr (std::is_integral_v<T>) {
        return false;  // Never use scientific notation for integers
    } else {
        double nonzero_finite_min = std::numeric_limits<double>::max();
        double nonzero_finite_max = std::numeric_limits<double>::lowest();
        bool found_nonzero_finite = false;

        for (const auto& value : buffer) {
            double val = static_cast<double>(value);
            if (std::isfinite(val) && val != 0.0) {
                double abs_val = std::abs(val);
                nonzero_finite_min = std::min(nonzero_finite_min, abs_val);
                nonzero_finite_max = std::max(nonzero_finite_max, abs_val);
                found_nonzero_finite = true;
            }
        }

        if (!found_nonzero_finite) {
            return false;  // No nonzero finite values, don't use scientific notation
        }

        return (nonzero_finite_max / nonzero_finite_min > 1000.0) || (nonzero_finite_max > 1.0e8) ||
               (nonzero_finite_min < 1.0e-4);
    }
}

constexpr int constexpr_strlen(const char* str) { return *str ? 1 + constexpr_strlen(str + 1) : 0; }

constexpr auto TENSOR_TYPE_STRING = "ttnn.Tensor";
constexpr auto TENSOR_TYPE_STRING_PLUS_OPEN_PARENTHESIS_LENGTH = constexpr_strlen(TENSOR_TYPE_STRING) + 1;

template <typename T>
void to_string_row_major(
    std::stringstream& ss,
    tt::stl::Span<const T> buffer,
    const tt::tt_metal::Shape& shape,
    const tt::tt_metal::Strides& strides,
    std::size_t outer_index,
    const std::size_t buffer_offset,
    int64_t rank,
    int64_t dim,
    bool use_scientific) {
    auto stride = dim < strides.size() ? strides[dim] : 0;

    std::string spaces = std::string(TENSOR_TYPE_STRING_PLUS_OPEN_PARENTHESIS_LENGTH + dim, ' ');
    std::string before;
    std::string after;
    if (rank == 1) {
        before = " ";
        after = " ";
    } else if (rank == 2) {
        before = spaces + " ";
        after = "\n";
    } else {
        before = spaces + " ";
        after = "\n\n";
    }

    if (dim > 0 and outer_index > 0) {
        ss << spaces;
    }
    if (rank != 0) {
        ss << "[";
    }
    auto dimension_shortener = get_dimension_shortener(rank != 0 ? shape[-rank] : 1);
    for (std::size_t index = 0;
         dimension_shortener.print_parenthesis_and_advance_index_if_reached_half_of_max_and_check_if_loop_is_done(
             ss, index, before, after);
         index++) {
        std::string after_comma;
        if (rank == 1) {
            after_comma = " ";
        } else if (rank == 2) {
            after_comma = "\n";
        } else {
            after_comma = after;
        }

        if (rank > 1) {
            to_string_row_major(
                ss, buffer, shape, strides, index, buffer_offset + (index * stride), rank - 1, dim + 1, use_scientific);
        } else {
            print_datum(ss, buffer[buffer_offset + index], use_scientific);
        }
        print_trailing_comma(ss, index, rank != 0 ? shape[-rank] : 1, after_comma);
    }
    if (rank != 0) {
        ss << "]";
    }
}

template <typename T>
void to_string(
    std::stringstream& ss,
    tt::stl::Span<const T> buffer,
    const tt::tt_metal::Shape& shape,
    const tt::tt_metal::Strides& strides,
    DataType dtype,
    Layout layout) {
    ss << TENSOR_TYPE_STRING << "(";

    if (TTNN_PRINT_OPTIONS.profile == TensorPrintProfile::Empty) {
        ss << "...";
    } else {
        bool use_scientific = should_use_scientific_notation<T>(buffer);
        to_string_row_major<T>(ss, buffer, shape, strides, 0, 0, shape.rank(), 0, use_scientific);
    }
    ss << ", shape=" << fmt::format("{}", shape) << ", dtype=" << fmt::format("{}", dtype)
       << ", layout=" << fmt::format("{}", layout) << ")";
}

}  // namespace detail

template <typename T>
std::string to_string_impl(const Tensor& tensor) {
    const auto& shape = tensor.logical_shape();

    if (!tensor.is_allocated()) {
        return fmt::format(
            "{}(<buffer is not allocated>, shape={}, dtype={}, layout={})",
            detail::TENSOR_TYPE_STRING,
            shape,
            tensor.dtype(),
            tensor.layout());
    }

    auto get_row_major_tensor = [&](const Tensor& tensor) -> Tensor {
        if (tensor.layout() == Layout::ROW_MAJOR) {
            return tensor;
        }
        if (tensor.dtype() == DataType::BFLOAT8_B || tensor.dtype() == DataType::BFLOAT4_B) {
            Tensor float_tensor = tt::tt_metal::to_dtype(tensor, DataType::FLOAT32);
            return Tensor(tt::tt_metal::to_layout(float_tensor.host_tensor(), Layout::ROW_MAJOR));
        }
        return Tensor(tt::tt_metal::to_layout(tensor.host_tensor(), Layout::ROW_MAJOR));
    };

    auto get_host_buffers = [&](const HostStorage& storage) {
        std::vector<HostBuffer> buffers;
        storage.buffer().apply([&](const HostBuffer& shard) { buffers.push_back(shard); });
        return buffers;
    };

    if (is_cpu_tensor(tensor)) {
        const Tensor row_major_tensor = get_row_major_tensor(tensor);
        const auto strides = row_major_tensor.tensor_spec().compute_strides();
        const std::vector<HostBuffer> buffers = get_host_buffers(row_major_tensor.host_storage());
        std::stringstream ss;
        for (size_t i = 0; i < buffers.size(); i++) {
            detail::to_string(ss, buffers[i].view_as<T>(), shape, strides, tensor.dtype(), tensor.layout());
            if (i + 1 != buffers.size()) {
                ss << std::endl;
            }
        }
        return ss.str();
    }

    const auto& storage = tensor.device_storage();
    auto cpu_tensor = tensor.cpu();
    if (storage.mesh_buffer == nullptr) {
        // Use owned buffer path above.
        return to_string_impl<T>(cpu_tensor);
    }

    auto* mesh_device = storage.mesh_buffer->device();
    // TODO: Uncomment after the distributed tensors migration to tt-metal is complete.
    // if (mesh_device->num_devices() == 1) {
    //     return to_string<T>(ttnn::distributed::get_device_tensors(cpu_tensor).at(0));
    // }

    const Tensor row_major_tensor = get_row_major_tensor(cpu_tensor);
    const auto strides = row_major_tensor.tensor_spec().compute_strides();
    const auto& coords = storage.coords;
    auto coords_it = coords.begin();
    const std::vector<HostBuffer> buffers = get_host_buffers(row_major_tensor.host_storage());
    std::stringstream ss;
    for (size_t i = 0; i < buffers.size(); i++) {
        const distributed::MeshCoordinate coord = *coords_it++;
        if (mesh_device->is_local(coord)) {
            ss << "device_id: " << mesh_device->get_device(coord)->id() << ", " << coord << std::endl;
            detail::to_string(ss, buffers[i].view_as<T>(), shape, strides, tensor.dtype(), tensor.layout());
        }
        if (i + 1 != buffers.size()) {
            ss << std::endl;
        }
    }
    return ss.str();
}

template <>
std::string to_string_impl<bfloat8_b>(const Tensor& tensor) {
    return to_string_impl<float>(tensor);
}

template <>
std::string to_string_impl<bfloat4_b>(const Tensor& tensor) {
    return to_string_impl<float>(tensor);
}

std::string to_string(const Tensor& tensor) {
    return dispatch(tensor.dtype(), [&]<typename T>() { return to_string_impl<T>(tensor); });
}

// allocate_host_buffer and to_host have been moved to tt-metalium/impl/tensor/tensor_apis.cpp

// to_device, to_device_mesh_buffer, copy_to_host, and copy_to_device have been moved to
// tt-metalium/impl/tensor/tensor_apis.cpp

// ======================================================================================
//     Helpers for converting between logical <-> physical data with full tensor spec
// ======================================================================================
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Useful information about how a shard_shape cuts a 2D shape
// - num_shards_height: Number of shards along the height (including partial last shard, if any)
// - last_shard_height: Height of last partial shard (if None, it will be same as full shard shape height)
// - num_shards_width: Number of shards along the width (including partial last shard, if any)
// - last_shard_width: Width of last partial shard (if None, it will be same as full shard shape width)
struct ShardDivisionSpec {
    size_t num_shards_height = 0;
    size_t last_shard_height = 0;
    size_t num_shards_width = 0;
    size_t last_shard_width = 0;
};

ShardDivisionSpec compute_shard_division_spec(const Shape2D& shape, const Shape2D& shard_shape) {
    const auto num_shards_height = tt::div_up(shape.height(), shard_shape.height());
    const auto last_shard_height =
        shape.height() % shard_shape.height() > 0 ? shape.height() % shard_shape.height() : shard_shape.height();
    const auto num_shards_width = tt::div_up(shape.width(), shard_shape.width());
    const auto last_shard_width =
        shape.width() % shard_shape.width() > 0 ? shape.width() % shard_shape.width() : shard_shape.width();

    return ShardDivisionSpec{num_shards_height, last_shard_height, num_shards_width, last_shard_width};
};

// TODO: Remove when we get rid of physical sharding and generalize interleaved and sharded; when we do, directly get
// from TensorLayout
std::array<Shape2D, 2> get_logical_and_physical_shard_shapes(const TensorSpec& tensor_spec) {
    const auto& logical_shape = tensor_spec.logical_shape();
    const auto& padded_shape = tensor_spec.padded_shape();

    Shape2D logical_shard_shape{logical_shape[-2], logical_shape[-1]};
    Shape2D physical_shard_shape = {padded_shape[-2], padded_shape[-1]};
    return {logical_shard_shape, physical_shard_shape};
}

using LogicalPhysicalIdxPairs = std::vector<std::pair<size_t, size_t>>;
using LogicalPhysicalMapping = std::pair<LogicalPhysicalIdxPairs, size_t>;
std::vector<LogicalPhysicalMapping> compute_logical_to_physical_shards_mapping(
    const Shape2D& logical_2d_shape,
    const Shape2D& logical_shard_shape,
    const Shape2D& physical_shard_shape,
    const size_t physical_stride) {
    const auto logical_stride = logical_2d_shape.width();

    const auto [num_shards_height, last_shard_height, num_shards_width, last_shard_width] =
        compute_shard_division_spec(logical_2d_shape, logical_shard_shape);

    std::vector<LogicalPhysicalMapping> logical_physical_mapping{};
    logical_physical_mapping.reserve(num_shards_height * num_shards_width);

    for (size_t shard_height_idx = 0; shard_height_idx < num_shards_height; shard_height_idx++) {
        for (size_t shard_width_idx = 0; shard_width_idx < num_shards_width; shard_width_idx++) {
            const auto num_shard_rows =
                shard_height_idx == num_shards_height - 1 ? last_shard_height : logical_shard_shape.height();
            const auto num_shard_cols =
                shard_width_idx == num_shards_width - 1 ? last_shard_width : logical_shard_shape.width();

            auto indices = LogicalPhysicalIdxPairs(num_shard_rows);
            const auto logical_start_idx = (shard_height_idx * logical_shard_shape.height() * logical_stride) +
                                           (shard_width_idx * logical_shard_shape.width());
            const auto physical_start_idx = (shard_height_idx * physical_shard_shape.height() * physical_stride) +
                                            (shard_width_idx * physical_shard_shape.width());
            for (size_t i = 0; i < num_shard_rows; i++) {
                indices[i] = {(i * logical_stride) + logical_start_idx, (i * physical_stride) + physical_start_idx};
            }

            logical_physical_mapping.emplace_back(indices, num_shard_cols);
        }
    }
    return logical_physical_mapping;
};

// Converts a span of logical data to row major physical data.
template <typename T>
std::vector<T> convert_to_row_major_physical_data(
    tt::stl::Span<const T> logical_data, const TensorSpec& tensor_spec, T pad_value) {
    const auto& physical_shape = tensor_spec.physical_shape();
    const size_t physical_stride = physical_shape.width();
    auto [logical_shard_shape, physical_shard_shape] =
        CMAKE_UNIQUE_NAMESPACE::get_logical_and_physical_shard_shapes(tensor_spec);

    std::vector<T> row_major_physical_data(physical_shape.height() * physical_shape.width(), pad_value);

    const auto logical_physical_mapping = CMAKE_UNIQUE_NAMESPACE::compute_logical_to_physical_shards_mapping(
        tensor_spec.logical_2d_shape(), logical_shard_shape, physical_shard_shape, physical_stride);

    for (const auto& [indices, cols] : logical_physical_mapping) {
        for (const auto& [logical_idx_start, physical_idx_start] : indices) {
            for (size_t col = 0; col < cols; col++) {
                row_major_physical_data[physical_idx_start + col] = logical_data[logical_idx_start + col];
            }
        }
    }
    return row_major_physical_data;
}

// Converts a span of row major physical data to logical data.
template <typename T>
std::vector<T> convert_to_logical_data(tt::stl::Span<const T> row_major_physical_data, const TensorSpec& tensor_spec) {
    const auto& logical_2d_shape = tensor_spec.logical_2d_shape();
    const size_t physical_stride = tensor_spec.physical_shape().width();
    auto [logical_shard_shape, physical_shard_shape] =
        CMAKE_UNIQUE_NAMESPACE::get_logical_and_physical_shard_shapes(tensor_spec);

    std::vector<T> logical_data(logical_2d_shape.height() * logical_2d_shape.width(), 0);

    const auto logical_physical_mapping = CMAKE_UNIQUE_NAMESPACE::compute_logical_to_physical_shards_mapping(
        logical_2d_shape, logical_shard_shape, physical_shard_shape, physical_stride);

    for (const auto& [indices, cols] : logical_physical_mapping) {
        for (const auto& [logical_idx_start, physical_idx_start] : indices) {
            for (size_t col = 0; col < cols; col++) {
                logical_data[logical_idx_start + col] = row_major_physical_data[physical_idx_start + col];
            }
        }
    }
    return logical_data;
}

template <typename T>
std::vector<T> convert_layout_row_major_to_tile(
    const Shape2D& shape, const Tile& tile, tt::stl::Span<const T> data_to_convert) {
    if (shape.width() * shape.height() == 0) {
        return std::vector<T>();
    }
    TT_FATAL(
        (shape.height() % tile.get_tile_shape()[0] == 0 && shape.width() % tile.get_tile_shape()[1] == 0),
        "Unsupported shape for tensor conversion from row-major to tile layout. The tensor shape height and width must "
        "be a multiple of tile height ({}) and width ({}), but the provided shape is {}",
        tile.get_tile_shape()[0],
        tile.get_tile_shape()[1],
        shape);

    auto tile_shape = tile.get_tile_shape();
    auto face_shape = tile.get_face_shape();
    auto transpose_within_face = tile.get_transpose_within_face();
    auto transpose_of_faces = tile.get_transpose_of_faces();

    return convert_layout(
        data_to_convert,
        shape,
        TensorLayoutType::LIN_ROW_MAJOR,
        TensorLayoutType::TILED_NFACES,
        tile_shape,
        face_shape,
        transpose_within_face,
        transpose_of_faces);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

template <typename T>
std::vector<T> encode_tensor_data(tt::stl::Span<const T> logical_data, const TensorSpec& tensor_spec, T pad_value) {
    if (logical_data.size() == 0) {
        return {};
    }

    const auto& logical_shape = tensor_spec.logical_shape();
    const auto& physical_shape = tensor_spec.physical_shape();

    TT_FATAL(
        logical_data.size() == logical_shape.volume(),
        "Logical data size {} should be same as volume indicated by logical shape {}",
        logical_data.size(),
        logical_shape);

    // If needed, convert logical data to row major physical data.
    // `row_major_physical_data_span` stores span unconditionally (cheap), while `row_major_physical_data` stores the
    // converted vector only when needed (expensive).
    std::vector<T> row_major_physical_data;
    tt::stl::Span<const T> row_major_physical_data_span;
    if (tensor_spec.logical_2d_shape() != physical_shape) {
        row_major_physical_data =
            CMAKE_UNIQUE_NAMESPACE::convert_to_row_major_physical_data(logical_data, tensor_spec, pad_value);
        row_major_physical_data_span = tt::stl::make_const_span(row_major_physical_data);
    } else {
        row_major_physical_data_span = logical_data;
    }

    TT_FATAL(
        row_major_physical_data_span.size() == physical_shape.height() * physical_shape.width(),
        "Physical data size {} should be same as volume indicated by physical shape {}",
        row_major_physical_data_span.size(),
        physical_shape);

    if (tensor_spec.layout() == Layout::TILE) {
        return CMAKE_UNIQUE_NAMESPACE::convert_layout_row_major_to_tile(
            physical_shape, tensor_spec.tile(), row_major_physical_data_span);
    }
    if (!row_major_physical_data.empty()) {
        // If conversion to physical data was performed, return the row major physical data to avoid extra copy.
        return row_major_physical_data;
    }  // Otherwise, copy the `row_major_physical_data_span`.
    return std::vector<T>(row_major_physical_data_span.begin(), row_major_physical_data_span.end());
}

template std::vector<bfloat16> encode_tensor_data<bfloat16>(
    tt::stl::Span<const bfloat16> logical_data, const TensorSpec& tensor_spec, bfloat16 pad_value);
template std::vector<float> encode_tensor_data<float>(
    tt::stl::Span<const float> logical_data, const TensorSpec& tensor_spec, float pad_value);
template std::vector<int32_t> encode_tensor_data<int32_t>(
    tt::stl::Span<const int32_t> logical_data, const TensorSpec& tensor_spec, int32_t pad_value);
template std::vector<uint32_t> encode_tensor_data<uint32_t>(
    tt::stl::Span<const uint32_t> logical_data, const TensorSpec& tensor_spec, uint32_t pad_value);
template std::vector<uint16_t> encode_tensor_data<uint16_t>(
    tt::stl::Span<const uint16_t> logical_data, const TensorSpec& tensor_spec, uint16_t pad_value);
template std::vector<uint8_t> encode_tensor_data<uint8_t>(
    tt::stl::Span<const uint8_t> logical_data, const TensorSpec& tensor_spec, uint8_t pad_value);

template <typename T>
std::vector<T> decode_tensor_data(tt::stl::Span<const T> physical_data, const TensorSpec& tensor_spec) {
    if (physical_data.size() == 0) {
        return {};
    }

    const auto& physical_shape = tensor_spec.physical_shape();
    TT_FATAL(
        physical_data.size() == physical_shape.height() * physical_shape.width(),
        "Physical data size {} should be same as volume indicated by physical shape {}",
        physical_data.size(),
        physical_shape);

    // If needed, convert physical data to row major physical data.
    // `row_major_physical_data_span` stores span unconditionally (cheap), while `row_major_physical_data` stores the
    // converted vector only when needed (expensive).
    std::vector<T> row_major_physical_data;
    tt::stl::Span<const T> row_major_physical_data_span;
    if (tensor_spec.layout() == Layout::TILE) {
        row_major_physical_data =
            tensor_impl::convert_layout_tile_to_row_major(physical_shape, tensor_spec.tile(), physical_data);
        row_major_physical_data_span = tt::stl::make_const_span(row_major_physical_data);
    } else {
        row_major_physical_data_span = physical_data;
    }

    // Same pattern as the above - `logical_data` is non empty only when the conversion to logical data was performed.
    std::vector<T> logical_data;
    tt::stl::Span<const T> logical_data_span;
    if (const auto& logical_2d_shape = tensor_spec.logical_2d_shape(); logical_2d_shape != physical_shape) {
        logical_data = CMAKE_UNIQUE_NAMESPACE::convert_to_logical_data(row_major_physical_data_span, tensor_spec);
        logical_data_span = tt::stl::make_const_span(logical_data);
    } else {
        logical_data_span = row_major_physical_data_span;
    }

    const auto& logical_shape = tensor_spec.logical_shape();
    TT_FATAL(
        logical_data_span.size() == logical_shape.volume(),
        "Logical data size {} should be same as volume indicated by logical shape {}",
        logical_data_span.size(),
        logical_shape);

    // Check if conversion to logical data was performed, to avoid extra copy upon return.
    if (!logical_data.empty()) {
        return logical_data;
    }
    if (!row_major_physical_data.empty()) {
        return row_major_physical_data;
    }
    return std::vector<T>(logical_data_span.begin(), logical_data_span.end());
}

template std::vector<bfloat16> decode_tensor_data<bfloat16>(
    tt::stl::Span<const bfloat16> physical_data, const TensorSpec& tensor_spec);
template std::vector<float> decode_tensor_data<float>(
    tt::stl::Span<const float> physical_data, const TensorSpec& tensor_spec);
template std::vector<int32_t> decode_tensor_data<int32_t>(
    tt::stl::Span<const int32_t> physical_data, const TensorSpec& tensor_spec);
template std::vector<uint32_t> decode_tensor_data<uint32_t>(
    tt::stl::Span<const uint32_t> physical_data, const TensorSpec& tensor_spec);
template std::vector<uint16_t> decode_tensor_data<uint16_t>(
    tt::stl::Span<const uint16_t> physical_data, const TensorSpec& tensor_spec);
template std::vector<uint8_t> decode_tensor_data<uint8_t>(
    tt::stl::Span<const uint8_t> physical_data, const TensorSpec& tensor_spec);

// to_layout has been moved to tt-metalium/impl/tensor/tensor_apis.cpp

// ======================================================================================
//                                  .view()
// ======================================================================================

// TODO (#25340): Review tensor topology logic for reshape
HostTensor view(const HostTensor& tensor, const Shape& new_logical_shape, const Shape& new_padded_shape) {
    // Just edit shape if shape has a 0 dimension
    if (tensor.logical_volume() == 0) {
        TT_FATAL(new_logical_shape.volume() == 0, "Tensor volume is 0, but shape's volume is not");
    }
    bool is_row_major = tensor.layout() == Layout::ROW_MAJOR;
    bool changing_last_dim = new_padded_shape[-1] != tensor.padded_shape()[-1];
    const auto& input_memory_config = tensor.memory_config();
    TT_FATAL(
        !input_memory_config.is_sharded() || !changing_last_dim ||
            input_memory_config.shard_spec()->shape[1] == tensor.padded_shape()[-1],
        "Changing the last dimension of a sharded tensor is not supported unless the shard width matches the input "
        "last dimension. "
        "Input shape: {}, New shape: {}, Shard width: {}",
        tensor.padded_shape(),
        new_padded_shape,
        input_memory_config.shard_spec()->shape[1]);

    auto output_memory_config = input_memory_config;
    if (is_row_major && input_memory_config.is_sharded() && changing_last_dim) {
        auto shard_spec = input_memory_config.shard_spec().value();
        auto shard_volume = shard_spec.numel();
        shard_spec.shape[1] = new_padded_shape[-1];  // update output shard to match new shard width
        shard_spec.shape[0] = shard_volume / shard_spec.shape[1];
        output_memory_config =
            MemoryConfig{input_memory_config.memory_layout(), input_memory_config.buffer_type(), shard_spec};
    }

    auto new_spec = tt::tt_metal::TensorSpec(
        new_logical_shape,
        TensorLayout::fromPaddedShape(
            tensor.dtype(),
            tensor.tensor_spec().page_config(),
            output_memory_config,
            new_logical_shape,
            new_padded_shape));

    return HostTensor(tensor.get_legacy_host_storage(), new_spec, tensor.tensor_topology());
}

// TODO (#25340): Review tensor topology logic for reshape
// TODO: We should force MeshTensor to be moved in. This essentially copies the MeshTensor.
MeshTensor view(const MeshTensor& tensor, const Shape& new_logical_shape, const Shape& new_padded_shape) {
    // Just edit shape if shape has a 0 dimension
    if (tensor.logical_volume() == 0) {
        TT_FATAL(new_logical_shape.volume() == 0, "Tensor volume is 0, but shape's volume is not");
    }
    bool is_row_major = tensor.layout() == Layout::ROW_MAJOR;
    bool changing_last_dim = new_padded_shape[-1] != tensor.padded_shape()[-1];
    const auto& input_memory_config = tensor.memory_config();
    TT_FATAL(
        !input_memory_config.is_sharded() || !changing_last_dim ||
            input_memory_config.shard_spec()->shape[1] == tensor.padded_shape()[-1],
        "Changing the last dimension of a sharded tensor is not supported unless the shard width matches the input "
        "last dimension. "
        "Input shape: {}, New shape: {}, Shard width: {}",
        tensor.padded_shape(),
        new_padded_shape,
        input_memory_config.shard_spec()->shape[1]);

    auto output_memory_config = input_memory_config;
    if (is_row_major && input_memory_config.is_sharded() && changing_last_dim) {
        auto shard_spec = input_memory_config.shard_spec().value();
        auto shard_volume = shard_spec.numel();
        shard_spec.shape[1] = new_padded_shape[-1];  // update output shard to match new shard width
        shard_spec.shape[0] = shard_volume / shard_spec.shape[1];
        output_memory_config =
            MemoryConfig{input_memory_config.memory_layout(), input_memory_config.buffer_type(), shard_spec};
    }

    auto new_spec = tt::tt_metal::TensorSpec(
        new_logical_shape,
        TensorLayout::fromPaddedShape(
            tensor.dtype(),
            tensor.tensor_spec().page_config(),
            output_memory_config,
            new_logical_shape,
            new_padded_shape));

    auto device_storage = tensor.get_legacy_device_storage();
    if (tensor.layout() != Layout::ROW_MAJOR || !changing_last_dim) {
        return MeshTensor(std::move(device_storage), new_spec, tensor.tensor_topology());
    }
    if (!tensor.memory_config().is_sharded()) {
        auto* device_buffer = device_storage.get_buffer();
        auto page_size_bytes = new_spec.compute_page_size_bytes();
        device_buffer->set_page_size(page_size_bytes);
        return MeshTensor(std::move(device_storage), new_spec, tensor.tensor_topology());
    }

    ShardSpec new_shard_spec = output_memory_config.shard_spec().value();
    std::array<uint32_t, 2> shard_page_shape = {1, new_shard_spec.shape[1]};
    std::array<uint32_t, 2> tensor2d_shape_in_pages = {
        new_spec.physical_shape().height() / shard_page_shape[0],
        new_spec.physical_shape().width() / shard_page_shape[1]};
    ShardSpecBuffer new_shard_spec_buffer =
        ShardSpecBuffer(new_shard_spec, shard_page_shape, tensor2d_shape_in_pages);

    Shape tensor_shape_pages(tensor2d_shape_in_pages);
    Shape shard_shape_pages(new_shard_spec_buffer.shape_in_pages());
    BufferDistributionSpec new_buffer_dist_spec = BufferDistributionSpec(
        tensor_shape_pages, shard_shape_pages, new_shard_spec.grid, new_shard_spec.orientation);

    auto device_local_config = device_storage.mesh_buffer->device_local_config();
    auto& sharding_args = device_local_config.sharding_args;
    BufferShardingArgs new_sharding_args(
        new_buffer_dist_spec, new_shard_spec_buffer, sharding_args.buffer_layout());

    distributed::DeviceLocalBufferConfig new_device_config = {
        .page_size = new_spec.compute_page_size_bytes(),
        .buffer_type = device_local_config.buffer_type,
        .sharding_args = new_sharding_args,
        .bottom_up = device_local_config.bottom_up};

    auto view_mesh_buffer = distributed::MeshBuffer::create(
        device_storage.mesh_buffer->global_config(),
        new_device_config,
        device_storage.mesh_buffer->device(),
        device_storage.mesh_buffer->address());
    DeviceStorage view_storage(
        view_mesh_buffer, device_storage.coords, device_storage.get_root_mesh_buffer());

    return MeshTensor(view_storage, new_spec, tensor.tensor_topology());
}

// pad, unpad, pad_impl, unpad_impl have been moved to tt-metalium/impl/tensor/tensor_apis.cpp

// ======================================================================================
//                                  .extract_shard()
// ======================================================================================

template <typename T>
Tensor extract_shard_impl(const Tensor& tensor, const uint32_t& core_id) {
    auto* buffer = tensor.buffer();
    auto buffer_shard_shape = buffer->shard_spec().shape();
    tt::tt_metal::Shape shard_shape({1, 1, buffer_shard_shape[0], buffer_shard_shape[1]});
    std::vector<T> device_data;
    ::detail::ReadShard(*buffer, device_data, core_id);

    auto output_buffer = std::vector<T>(std::move(device_data));
    return Tensor(
        HostBuffer(std::move(output_buffer)),
        shard_shape,
        tensor.dtype(),
        tensor.layout(),
        tensor.tensor_spec().tile());
}

template <>
Tensor extract_shard_impl<bfloat8_b>(const Tensor& tensor, const uint32_t& core_id) {
    return extract_shard_impl<uint32_t>(tensor, core_id);
}

template <>
Tensor extract_shard_impl<bfloat4_b>(const Tensor& tensor, const uint32_t& core_id) {
    return extract_shard_impl<uint32_t>(tensor, core_id);
}

Tensor extract_shard(const Tensor& tensor, const uint32_t& core_id) {
    return dispatch(tensor.dtype(), [&]<typename T>() { return extract_shard_impl<T>(tensor, core_id); });
}

// to_dtype, pad_to_tile, unpad_from_tile have been moved to tt-metalium/impl/tensor/tensor_apis.cpp

// ======================================================================================
//                                  Runtime Tensor Creation Functions
// ======================================================================================

MeshTensor allocate_tensor_on_device(const TensorSpec& tensor_spec, distributed::MeshDevice* device) {
    auto mesh_buffer = allocate_device_buffer(device, tensor_spec);
    std::vector<distributed::MeshCoordinate> coords;
    coords.reserve(device->shape().mesh_size());
    for (const auto& coord : distributed::MeshCoordinateRange(device->shape())) {
        coords.push_back(coord);
    }
    DeviceStorage device_storage(std::move(mesh_buffer), coords);
    // TODO (#25340): Implement correct logic and add test for this
    ttsl::SmallVector<distributed::MeshMapperConfig::Placement> placements(device->shape().dims());
    for (size_t i = 0; i < device->shape().dims(); i++) {
        placements[i] = distributed::MeshMapperConfig::Replicate{};
    }

    auto tensor_topology = TensorTopology{device->shape(), placements, coords};
    return MeshTensor(std::move(device_storage), tensor_spec, tensor_topology);
}

// ======================================================================================
//                                  HostTensor Factory Functions
// ======================================================================================

namespace host_tensor {

template <typename T>
HostTensor from_vector(std::vector<T>&& buffer, const TensorSpec& spec, T pad_value) {
    size_t volume = spec.logical_shape().volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    if (spec.data_type() == DataType::BFLOAT8_B || spec.data_type() == DataType::BFLOAT4_B) {
        TT_FATAL(spec.layout() == Layout::TILE, "Block float types are only supported in TILE layout");
    }

    auto buffer_dtype = convert_to_data_type<T>();
    auto buffer_spec =
        TensorSpec(spec.logical_shape(), TensorLayout(buffer_dtype, spec.page_config(), spec.memory_config()));

    auto host_buffer =
        logical_matches_physical(buffer_spec)
            ? HostBuffer(std::move(buffer))
            : HostBuffer(tensor_impl::encode_tensor_data(tt::stl::make_const_span(buffer), spec, pad_value));

    auto result = HostTensor(std::move(host_buffer), buffer_spec, TensorTopology{});
    return to_dtype(result, spec.data_type());
}

template HostTensor from_vector<bfloat16>(std::vector<bfloat16>&& buffer, const TensorSpec& spec, bfloat16 pad_value);
template HostTensor from_vector<float>(std::vector<float>&& buffer, const TensorSpec& spec, float pad_value);
template HostTensor from_vector<int32_t>(std::vector<int32_t>&& buffer, const TensorSpec& spec, int32_t pad_value);
template HostTensor from_vector<uint8_t>(std::vector<uint8_t>&& buffer, const TensorSpec& spec, uint8_t pad_value);
template HostTensor from_vector<uint16_t>(std::vector<uint16_t>&& buffer, const TensorSpec& spec, uint16_t pad_value);
template HostTensor from_vector<uint32_t>(std::vector<uint32_t>&& buffer, const TensorSpec& spec, uint32_t pad_value);

template <typename T>
HostTensor from_span(ttsl::Span<const T> buffer, const TensorSpec& spec, T pad_value) {
    return from_vector(std::vector<T>(buffer.begin(), buffer.end()), spec, pad_value);
}

template HostTensor from_span<bfloat16>(ttsl::Span<const bfloat16> buffer, const TensorSpec& spec, bfloat16 pad_value);
template HostTensor from_span<float>(ttsl::Span<const float> buffer, const TensorSpec& spec, float pad_value);
template HostTensor from_span<int32_t>(ttsl::Span<const int32_t> buffer, const TensorSpec& spec, int32_t pad_value);
template HostTensor from_span<uint8_t>(ttsl::Span<const uint8_t> buffer, const TensorSpec& spec, uint8_t pad_value);
template HostTensor from_span<uint16_t>(ttsl::Span<const uint16_t> buffer, const TensorSpec& spec, uint16_t pad_value);
template HostTensor from_span<uint32_t>(ttsl::Span<const uint32_t> buffer, const TensorSpec& spec, uint32_t pad_value);

template <typename T>
HostTensor from_borrowed_data(
    ttsl::Span<T> buffer, const Shape& shape, MemoryPin buffer_pin, const std::optional<Tile>& tile) {
    size_t volume = shape.volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);

    auto tensor_spec = TensorSpec(
        shape,
        TensorLayout::fromPaddedShape(
            convert_to_data_type<T>(), PageConfig(Layout::ROW_MAJOR, tile), MemoryConfig{}, shape, shape));

    return HostTensor(HostBuffer(buffer, std::move(buffer_pin)), tensor_spec, TensorTopology{});
}

template HostTensor from_borrowed_data<bfloat16>(
    ttsl::Span<bfloat16> buffer, const Shape& shape, MemoryPin buffer_pin, const std::optional<Tile>& tile);
template HostTensor from_borrowed_data<float>(
    ttsl::Span<float> buffer, const Shape& shape, MemoryPin buffer_pin, const std::optional<Tile>& tile);
template HostTensor from_borrowed_data<int32_t>(
    ttsl::Span<int32_t> buffer, const Shape& shape, MemoryPin buffer_pin, const std::optional<Tile>& tile);
template HostTensor from_borrowed_data<uint8_t>(
    ttsl::Span<uint8_t> buffer, const Shape& shape, MemoryPin buffer_pin, const std::optional<Tile>& tile);
template HostTensor from_borrowed_data<uint16_t>(
    ttsl::Span<uint16_t> buffer, const Shape& shape, MemoryPin buffer_pin, const std::optional<Tile>& tile);
template HostTensor from_borrowed_data<uint32_t>(
    ttsl::Span<uint32_t> buffer, const Shape& shape, MemoryPin buffer_pin, const std::optional<Tile>& tile);

// ======================================================================================
//                                  HostTensor to_vector()
// ======================================================================================

/*
 * Special case of to_vector for float,
 * This handles the conversion from bfloat16, bfloat8_b, bfloat4_b, and float32 to float.
 */
std::vector<float> to_vector_float(const HostTensor& tensor) {
    switch (tensor.dtype()) {
        case DataType::BFLOAT16: {
            auto buffer = tt::tt_metal::host_buffer::get_as<bfloat16>(tensor);
            std::vector<float> physical_data;
            physical_data.reserve(buffer.size());
            std::transform(buffer.begin(), buffer.end(), std::back_inserter(physical_data), [](bfloat16 val) {
                return static_cast<float>(val);
            });
            if (logical_matches_physical(tensor.tensor_spec())) {
                return physical_data;
            }
            return tensor_impl::decode_tensor_data(tt::stl::make_const_span(physical_data), tensor.tensor_spec());
        }
        case DataType::FLOAT32: {
            auto buffer = tt::tt_metal::host_buffer::get_as<const float>(tensor);
            return tensor_impl::decode_tensor_data(buffer, tensor.tensor_spec());
        }
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: {
            const auto& tile = tensor.tensor_spec().tile();
            auto buffer = tt::tt_metal::host_buffer::get_as<const uint32_t>(tensor);
            std::vector<float> unpacked_data =
                tensor.tensor_spec().data_type() == DataType::BFLOAT8_B
                    ? unpack_bfp8_tiles_into_float_vec(buffer, /*row_major_output=*/false, /*is_exp_a=*/false, tile)
                    : unpack_bfp4_tiles_into_float_vec(buffer, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
            return tensor_impl::decode_tensor_data(tt::stl::make_const_span(unpacked_data), tensor.tensor_spec());
        }
        default: {
            TT_THROW("Cannot convert tensor to vector for data type: {}", tensor.dtype());
        }
    }
}

template <typename T>
std::vector<T> to_vector(const HostTensor& tensor) {
    if constexpr (std::is_same_v<T, float>) {
        return to_vector_float(tensor);
    }
    TT_FATAL(
        tensor.dtype() == convert_to_data_type<T>(),
        "Unsupported data type for to_vector: got {}, expected: {}",
        tensor.dtype(),
        convert_to_data_type<T>());
    auto data = tt::tt_metal::host_buffer::get_as<const T>(tensor);
    if (logical_matches_physical(tensor.tensor_spec())) {
        return std::vector<T>(data.begin(), data.end());
    }
    return tensor_impl::decode_tensor_data(data, tensor.tensor_spec());
}

template std::vector<float> to_vector<float>(const HostTensor& tensor);
template std::vector<bfloat16> to_vector<bfloat16>(const HostTensor& tensor);
template std::vector<int32_t> to_vector<int32_t>(const HostTensor& tensor);
template std::vector<uint8_t> to_vector<uint8_t>(const HostTensor& tensor);
template std::vector<uint16_t> to_vector<uint16_t>(const HostTensor& tensor);
template std::vector<uint32_t> to_vector<uint32_t>(const HostTensor& tensor);

}  // namespace host_tensor

}  // namespace tt::tt_metal::tensor_impl
