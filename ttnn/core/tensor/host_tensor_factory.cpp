// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor_impl.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <ttnn/tensor/layout/tensor_layout.hpp>
#include <ttnn/tensor/tensor_spec.hpp>
#include <ttnn/tensor/types.hpp>
#include <ttnn/distributed/tensor_topology.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include <tt_stl/span.hpp>
#include <tt_stl/fmt.hpp>

#include <algorithm>

// ============================================================================
// Forward declarations of helper functions to be lowered to tt_metal later
// ============================================================================

namespace tt::tt_metal {
bool logical_matches_physical(const TensorSpec& tensor_spec);  // NOLINT(readability-redundant-declaration)
}  // namespace tt::tt_metal

namespace tt::tt_metal::tensor_impl {
template <typename T>
std::vector<T> encode_tensor_data(  // NOLINT(readability-redundant-declaration)
    ttsl::Span<const T> logical_data,
    const TensorSpec& tensor_spec,
    T pad_value);

template <typename T>
std::vector<T> decode_tensor_data(  // NOLINT(readability-redundant-declaration)
    ttsl::Span<const T> physical_data,
    const TensorSpec& tensor_spec);
}  // namespace tt::tt_metal::tensor_impl

// ============================================================================
// Implementation
// ============================================================================

namespace tt::tt_metal::tensor_impl::host_tensor {

namespace {
// TODO(#40348):
// Will be host_buffer::get_host_buffer
HostBuffer get_single_host_buffer(const HostTensor& tensor) {
    const auto& distributed_buffer = tensor.buffer();
    std::vector<HostBuffer> buffers;
    distributed_buffer.apply([&buffers](const HostBuffer& shard) { buffers.push_back(shard); });
    TT_FATAL(
        buffers.size() == 1,
        "Can't get a single buffer from HostTensor distributed over mesh shape {}",
        distributed_buffer.shape());
    return buffers.front();
}
}  // namespace

template <typename T>
HostTensor from_span(ttsl::Span<const T> buffer, const TensorSpec& spec, T pad_value) {
    return from_vector(std::vector<T>(buffer.begin(), buffer.end()), spec, pad_value);
}

template <typename T>
HostTensor from_borrowed_data(
    ttsl::Span<T> buffer, const Shape& shape, MemoryPin pin, const std::optional<Tile>& tile) {
    size_t volume = shape.volume();
    TT_FATAL(buffer.size() == volume, "Buffer size {} differs from shape volume {}", buffer.size(), volume);

    auto host_buffer = HostBuffer(buffer, std::move(pin));
    auto buffer_dtype = convert_to_data_type<T>();
    TensorSpec tensor_spec(shape, TensorLayout(buffer_dtype, PageConfig(Layout::ROW_MAJOR, tile), MemoryConfig{}));

    return HostTensor(std::move(host_buffer), std::move(tensor_spec), TensorTopology{});
}

template <typename T>
HostTensor from_vector(const std::vector<T>& buffer, const TensorSpec& spec, T pad_value) {
    return from_span(ttsl::make_const_span(buffer), spec, pad_value);
}

template <typename T>
HostTensor from_vector(std::vector<T>&& buffer, const TensorSpec& spec, T pad_value) {
    size_t volume = spec.logical_shape().volume();
    TT_FATAL(buffer.size() == volume, "Buffer size {} differs from shape volume {}", buffer.size(), volume);

    if (spec.data_type() == DataType::BFLOAT8_B || spec.data_type() == DataType::BFLOAT4_B) {
        TT_FATAL(spec.layout() == Layout::TILE, "Block float types only supported in TILE layout");
    }

    auto buffer_dtype = convert_to_data_type<T>();
    auto buffer_spec =
        TensorSpec(spec.logical_shape(), TensorLayout(buffer_dtype, spec.page_config(), spec.memory_config()));

    auto host_buffer = logical_matches_physical(buffer_spec)
                           ? HostBuffer(std::move(buffer))
                           : HostBuffer(encode_tensor_data(ttsl::make_const_span(buffer), spec, pad_value));

    return HostTensor(std::move(host_buffer), buffer_spec, TensorTopology{});
}

template <typename T>
std::vector<T> to_vector_generic(const HostTensor& tensor) {
    TT_FATAL(
        tensor.dtype() == convert_to_data_type<T>(),
        "Unsupported data type for to_vector: got {}, expected: {}",
        tensor.dtype(),
        convert_to_data_type<T>());

    HostBuffer host_buffer = get_single_host_buffer(tensor);

    auto data = host_buffer.view_as<const T>();
    if (logical_matches_physical(tensor.tensor_spec())) {
        return std::vector<T>(data.begin(), data.end());
    }
    return decode_tensor_data(data, tensor.tensor_spec());
}

std::vector<float> to_vector_float(const HostTensor& tensor) {
    HostBuffer host_buffer = get_single_host_buffer(tensor);

    switch (tensor.dtype()) {
        case DataType::BFLOAT16: {
            auto buffer = host_buffer.view_as<const bfloat16>();
            std::vector<float> physical_data;
            physical_data.reserve(buffer.size());
            std::transform(buffer.begin(), buffer.end(), std::back_inserter(physical_data), [](bfloat16 val) {
                return static_cast<float>(val);
            });
            if (logical_matches_physical(tensor.tensor_spec())) {
                return physical_data;
            }
            return decode_tensor_data(ttsl::make_const_span(physical_data), tensor.tensor_spec());
        }
        case DataType::FLOAT32: {
            auto buffer = host_buffer.view_as<const float>();
            if (logical_matches_physical(tensor.tensor_spec())) {
                return std::vector<float>(buffer.begin(), buffer.end());
            }
            return decode_tensor_data(buffer, tensor.tensor_spec());
        }
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: {
            const auto& tile = tensor.tensor_spec().tile();
            auto buffer = host_buffer.view_as<const uint32_t>();
            std::vector<float> unpacked_data =
                tensor.dtype() == DataType::BFLOAT8_B
                    ? unpack_bfp8_tiles_into_float_vec(buffer, /*row_major_output=*/false, /*is_exp_a=*/false, tile)
                    : unpack_bfp4_tiles_into_float_vec(buffer, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
            return decode_tensor_data(ttsl::make_const_span(unpacked_data), tensor.tensor_spec());
        }
        default: {
            TT_THROW("Cannot convert HostTensor to vector<float> for data type: {}", tensor.dtype());
        }
    }
}

template <typename T>
std::vector<T> to_vector(const HostTensor& tensor) {
    if constexpr (std::is_same_v<T, float>) {
        return to_vector_float(tensor);
    }
    return to_vector_generic<T>(tensor);
}

// ============================================================================
// Explicit template instantiations
// ============================================================================

template HostTensor from_span<bfloat16>(ttsl::Span<const bfloat16>, const TensorSpec&, bfloat16);
template HostTensor from_span<float>(ttsl::Span<const float>, const TensorSpec&, float);
template HostTensor from_span<int32_t>(ttsl::Span<const int32_t>, const TensorSpec&, int32_t);
template HostTensor from_span<uint32_t>(ttsl::Span<const uint32_t>, const TensorSpec&, uint32_t);
template HostTensor from_span<uint16_t>(ttsl::Span<const uint16_t>, const TensorSpec&, uint16_t);
template HostTensor from_span<uint8_t>(ttsl::Span<const uint8_t>, const TensorSpec&, uint8_t);

template HostTensor from_borrowed_data<bfloat16>(
    ttsl::Span<bfloat16>, const Shape&, MemoryPin, const std::optional<Tile>&);
template HostTensor from_borrowed_data<float>(ttsl::Span<float>, const Shape&, MemoryPin, const std::optional<Tile>&);
template HostTensor from_borrowed_data<int32_t>(
    ttsl::Span<int32_t>, const Shape&, MemoryPin, const std::optional<Tile>&);
template HostTensor from_borrowed_data<uint32_t>(
    ttsl::Span<uint32_t>, const Shape&, MemoryPin, const std::optional<Tile>&);
template HostTensor from_borrowed_data<uint16_t>(
    ttsl::Span<uint16_t>, const Shape&, MemoryPin, const std::optional<Tile>&);
template HostTensor from_borrowed_data<uint8_t>(
    ttsl::Span<uint8_t>, const Shape&, MemoryPin, const std::optional<Tile>&);

template HostTensor from_vector<bfloat16>(const std::vector<bfloat16>&, const TensorSpec&, bfloat16);
template HostTensor from_vector<float>(const std::vector<float>&, const TensorSpec&, float);
template HostTensor from_vector<int32_t>(const std::vector<int32_t>&, const TensorSpec&, int32_t);
template HostTensor from_vector<uint32_t>(const std::vector<uint32_t>&, const TensorSpec&, uint32_t);
template HostTensor from_vector<uint16_t>(const std::vector<uint16_t>&, const TensorSpec&, uint16_t);
template HostTensor from_vector<uint8_t>(const std::vector<uint8_t>&, const TensorSpec&, uint8_t);

template HostTensor from_vector<bfloat16>(std::vector<bfloat16>&&, const TensorSpec&, bfloat16);
template HostTensor from_vector<float>(std::vector<float>&&, const TensorSpec&, float);
template HostTensor from_vector<int32_t>(std::vector<int32_t>&&, const TensorSpec&, int32_t);
template HostTensor from_vector<uint32_t>(std::vector<uint32_t>&&, const TensorSpec&, uint32_t);
template HostTensor from_vector<uint16_t>(std::vector<uint16_t>&&, const TensorSpec&, uint16_t);
template HostTensor from_vector<uint8_t>(std::vector<uint8_t>&&, const TensorSpec&, uint8_t);

template std::vector<float> to_vector<float>(const HostTensor&);
template std::vector<bfloat16> to_vector<bfloat16>(const HostTensor&);
template std::vector<int32_t> to_vector<int32_t>(const HostTensor&);
template std::vector<uint32_t> to_vector<uint32_t>(const HostTensor&);
template std::vector<uint16_t> to_vector<uint16_t>(const HostTensor&);
template std::vector<uint8_t> to_vector<uint8_t>(const HostTensor&);

}  // namespace tt::tt_metal::tensor_impl::host_tensor
