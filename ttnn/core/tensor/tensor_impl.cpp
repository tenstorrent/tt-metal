// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include <fmt/format.h>
#include <optional>
#include <span>

#include <sys/mman.h>
#include <unistd.h>

#include "tt-metalium/experimental/tensor/host_tensor.hpp"
#include "tt-metalium/experimental/tensor/tensor_apis.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
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
bool should_use_scientific_notation(ttsl::Span<const T> buffer) {
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
    ttsl::Span<const T> buffer,
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
    ttsl::Span<const T> buffer,
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
            return to_layout(tt::tt_metal::to_dtype(tensor, DataType::FLOAT32), Layout::ROW_MAJOR);
        }
        return to_layout(tensor, Layout::ROW_MAJOR);
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
    if (!storage.is_allocated()) {
        // Use owned buffer path above.
        return to_string_impl<T>(cpu_tensor);
    }

    auto& mesh_device = storage.get_mesh_tensor().device();
    // TODO: Uncomment after the distributed tensors migration to tt-metal is complete.
    // if (mesh_device->num_devices() == 1) {
    //     return to_string<T>(ttnn::distributed::get_device_tensors(cpu_tensor).at(0));
    // }

    const Tensor row_major_tensor = get_row_major_tensor(cpu_tensor);
    const auto strides = row_major_tensor.tensor_spec().compute_strides();
    const auto coords = storage.get_coords();
    auto coords_it = coords.begin();
    const std::vector<HostBuffer> buffers = get_host_buffers(row_major_tensor.host_storage());
    std::stringstream ss;
    for (size_t i = 0; i < buffers.size(); i++) {
        const distributed::MeshCoordinate coord = *coords_it++;
        if (mesh_device.is_local(coord)) {
            ss << "device_id: " << mesh_device.get_device(coord)->id() << ", " << coord << std::endl;
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

// ======================================================================================
//                                  .view()
// ======================================================================================

HostTensor view(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& new_logical_shape,
    const tt::tt_metal::Shape& new_padded_shape) {
    // Just edit shape if shape has a 0 dimension
    if (tensor.logical_volume() == 0) {
        TT_FATAL(new_logical_shape.volume() == 0, "Tensor volume is 0, but shape's volume is not");
    }
    const auto& input_memory_config = tensor.memory_config();
    auto output_memory_config = input_memory_config;

    if (input_memory_config.memory_layout() == TensorMemoryLayout::ND_SHARDED) {
        const auto old_rank = tensor.padded_shape().rank();
        const auto& old_nd_spec = input_memory_config.nd_shard_spec().value();

        // Rank-expansion of a 0D/1D tensor into a 2D shape — the original allow-listed
        // case. Requires the expanded 2D shape to still match the input's logical footprint.
        bool is_rank_expansion_to_2d = old_rank < 2 && new_padded_shape.rank() == 2 && new_padded_shape[0] == 1 &&
                                       (old_rank == 0 || new_padded_shape[1] == tensor.padded_shape()[-1]);

        // Logical-shape-only update: same rank and same padded shape as the input. No bytes
        // move, the physical tile layout is unchanged, and the per-core shard location and
        // size are unchanged. The caller is just reinterpreting which logical elements live
        // in which physical positions (e.g. a reduction op trimming logical dim after keeping
        // the padded tile intact). Safe for ND-sharded tensors.
        bool is_same_physical_shape = new_padded_shape.rank() == old_rank && new_padded_shape == tensor.padded_shape();

        TT_FATAL(
            is_rank_expansion_to_2d || is_same_physical_shape,
            "View is not supported for ND sharded tensors except for rank expansion to 2D "
            "or same-physical-shape (logical-only) metadata updates. Input shape: {}, New shape: {}",
            tensor.padded_shape(),
            new_padded_shape);

        if (is_same_physical_shape) {
            // Keep the input's MemoryConfig as-is (including nd_shard_spec + flags) — no
            // physical layout change means no metadata adjustment needed.
            output_memory_config = input_memory_config;
        } else {
            // Rank-expansion-to-2D path: synthesize a new nd_shard_spec for the expanded shape.
            ttsl::SmallVector<uint32_t> new_shard_shape =
                old_rank == 0 ? ttsl::SmallVector<uint32_t>{1, 1}
                              : ttsl::SmallVector<uint32_t>{1, old_nd_spec.shard_shape[-1]};
            output_memory_config =
                MemoryConfig(input_memory_config.buffer_type(), old_nd_spec.with_shard_shape(Shape(new_shard_shape)));
        }
    } else {
        bool is_row_major = tensor.layout() == Layout::ROW_MAJOR;
        bool changing_last_dim = new_padded_shape[-1] != tensor.padded_shape()[-1];
        TT_FATAL(
            !input_memory_config.is_sharded() || !changing_last_dim ||
                input_memory_config.shard_spec()->shape[1] == tensor.padded_shape()[-1],
            "Changing the last dimension of a sharded tensor is not supported unless the shard width matches the "
            "input last dimension. "
            "Input shape: {}, New shape: {}, Shard width: {}",
            tensor.padded_shape(),
            new_padded_shape,
            input_memory_config.shard_spec()->shape[1]);

        if (is_row_major && input_memory_config.is_sharded() && changing_last_dim) {
            auto shard_spec = input_memory_config.shard_spec().value();
            auto shard_volume = shard_spec.numel();
            shard_spec.shape[1] = new_padded_shape[-1];
            shard_spec.shape[0] = shard_volume / shard_spec.shape[1];
            output_memory_config =
                MemoryConfig{input_memory_config.memory_layout(), input_memory_config.buffer_type(), shard_spec};
        }
    }

    auto new_spec = tt::tt_metal::TensorSpec(
        new_logical_shape,
        TensorLayout::fromPaddedShape(
            tensor.dtype(),
            tensor.tensor_spec().page_config(),
            output_memory_config,
            new_logical_shape,
            new_padded_shape));

    // TODO (#25340): Review tensor topology logic for reshape
    const auto& buffer = tensor.buffer();
    return HostTensor(buffer, new_spec, tensor.tensor_topology());
}

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

}  // namespace tt::tt_metal::tensor_impl
