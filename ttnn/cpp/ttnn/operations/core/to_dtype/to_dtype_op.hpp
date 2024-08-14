// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace operations {

namespace core {

namespace detail {

inline Tensor convert_to_cpp_supported_dtype(const Tensor& input_tensor) {
    auto input_dtype = input_tensor.get_dtype();

    auto buffer = std::visit(
        [](auto&& storage) -> std::variant<OwnedBuffer, BorrowedBuffer> {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, OwnedStorage>) {
                return storage.buffer;
            } else if constexpr (std::is_same_v<T, DeviceStorage>) {
                TT_THROW("Device input_tensor cannot be converted to torch");
            } else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                return storage.buffer;
            } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                TT_THROW("Tensor with MultiDeviceStorage cannot be converted to torch");
            } else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                TT_THROW(
                    "Tensor MultiDeviceHostStorage cannot be converted to torch directly. Use composer(..) "
                    "functionality.");
            } else {
                raise_unsupported_storage<T>();
            }
        },
        input_tensor.get_storage());

    if (input_dtype == DataType::BFLOAT8_B) {
        TT_ASSERT(std::holds_alternative<OwnedBuffer>(buffer), fmt::format("Unexpected type {} in {}:{} ",tt::stl::get_active_type_name_in_variant(buffer),__FILE__, __LINE__));
        auto uint32_data = std::get<owned_buffer::Buffer<std::uint32_t>>(std::get<OwnedBuffer>(buffer)).get();
        auto float_unpacked_data =
            unpack_bfp8_tiles_into_float_vec(uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false);
        buffer = owned_buffer::create<float>(std::move(float_unpacked_data));
        input_dtype = DataType::FLOAT32;
    } else if (input_dtype == DataType::BFLOAT4_B) {
        TT_ASSERT(std::holds_alternative<OwnedBuffer>(buffer), fmt::format("Unexpected type {} in {}:{} ",tt::stl::get_active_type_name_in_variant(buffer),__FILE__, __LINE__));
        auto uint32_data = std::get<owned_buffer::Buffer<std::uint32_t>>(std::get<OwnedBuffer>(buffer)).get();
        auto float_unpacked_data =
            unpack_bfp4_tiles_into_float_vec(uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false);
        buffer = owned_buffer::create<float>(std::move(float_unpacked_data));
        input_dtype = DataType::FLOAT32;
    }

    return std::visit(
        [&](auto&& buffer) -> Tensor {
            using T = std::decay_t<decltype(buffer)>;
            if constexpr (std::is_same_v<T, OwnedBuffer>) {
                return Tensor{OwnedStorage{buffer}, input_tensor.get_shape(), input_dtype, input_tensor.get_layout()};
            } else if constexpr (std::is_same_v<T, BorrowedBuffer>) {
                return Tensor{
                    BorrowedStorage{buffer, []() {}, []() {}},
                    input_tensor.get_shape(),
                    input_dtype,
                    input_tensor.get_layout()};
            } else {
                TT_THROW("Unsupported buffer type");
            }
        },
        buffer);
}

template <typename NewT, typename OldT>
inline std::vector<NewT> cast(const borrowed_buffer::Buffer<OldT>& input_buffer) {
    std::vector<NewT> output_vector(input_buffer.size());
    for (auto index = 0; index < input_buffer.size(); ++index) {
        auto convert_value = [](auto&& value) {
            if constexpr (std::is_same_v<OldT, ::bfloat16>) {
                return value.to_float();
            } else if constexpr (std::is_same_v<NewT, ::bfloat16>) {
                return static_cast<float>(value);
            } else {
                return value;
            }
        };
        auto value = input_buffer[index];
        output_vector[index] = static_cast<NewT>(convert_value(value));
    }
    return output_vector;
}

template <typename T>
Tensor create_owned_tensor(std::vector<T>&& data, const Shape& shape, DataType data_type, Layout layout) {
    auto buffer = owned_buffer::create(std::move(data));
    auto storage = OwnedStorage{std::move(buffer)};
    return Tensor(std::move(storage), shape, data_type, layout);
}

template <typename T>
inline Tensor create_tensor_from_buffer(
    const borrowed_buffer::Buffer<T>& input_buffer,
    const Shape& shape,
    const Layout& input_layout,
    const DataType& dtype) {
    switch (dtype) {
        case DataType::UINT16: {
            auto data = cast<uint16_t, T>(input_buffer);
            return create_owned_tensor(std::move(data), shape, dtype, Layout::ROW_MAJOR).to(input_layout);
        }
        case DataType::INT32: {
            auto data = cast<int32_t, T>(input_buffer);
            return create_owned_tensor(std::move(data), shape, dtype, Layout::ROW_MAJOR).to(input_layout);
        }
        case DataType::UINT32: {
            auto data = cast<uint32_t, T>(input_buffer);
            return create_owned_tensor(std::move(data), shape, dtype, Layout::ROW_MAJOR).to(input_layout);
        }
        case DataType::FLOAT32: {
            auto data = cast<float, T>(input_buffer);
            return create_owned_tensor(std::move(data), shape, dtype, Layout::ROW_MAJOR).to(input_layout);
        }
        case DataType::BFLOAT16: {
            auto data = cast<::bfloat16, T>(input_buffer);
            return create_owned_tensor(std::move(data), shape, dtype, Layout::ROW_MAJOR).to(input_layout);
        }
        case DataType::BFLOAT8_B: {
            auto data = cast<float, T>(input_buffer);
            auto uint32_vector = pack_fp32_vec_as_bfp8_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
            auto buffer = owned_buffer::create<uint32_t>(std::move(uint32_vector));
            auto storage = OwnedStorage{std::move(buffer)};
            return Tensor(std::move(storage), shape, dtype, Layout::ROW_MAJOR)
                .to(ttnn::TILE_LAYOUT);  // has to be in tile layout
        }
        case DataType::BFLOAT4_B: {
            auto data = cast<float, T>(input_buffer);
            auto uint32_vector = pack_fp32_vec_as_bfp4_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
            auto buffer = owned_buffer::create<uint32_t>(std::move(uint32_vector));
            auto storage = OwnedStorage{std::move(buffer)};
            return Tensor(std::move(storage), shape, dtype, Layout::ROW_MAJOR)
                .to(ttnn::TILE_LAYOUT);  // has to be in tile layout
        }
        default: {
            TT_THROW(fmt::format("Unsupported DataType: {}", dtype));
            break;
        }
    }
}

inline Tensor convert_to_dtype(const Tensor& input_tensor, const Layout& input_layout, const DataType& dtype) {
    auto input_dtype = input_tensor.get_dtype();

    auto convert_dtype = [&input_layout, &input_dtype, &dtype](const Tensor& input_tensor) {
        switch (input_dtype) {
            case DataType::UINT16: {
                auto buffer = host_buffer::get_as<uint16_t>(input_tensor);
                return create_tensor_from_buffer(buffer, input_tensor.get_shape(), input_layout, dtype);
            }
            case DataType::INT32: {
                auto buffer = host_buffer::get_as<int32_t>(input_tensor);
                return create_tensor_from_buffer(buffer, input_tensor.get_shape(), input_layout, dtype);
            }
            case DataType::UINT32: {
                auto buffer = host_buffer::get_as<uint32_t>(input_tensor);
                return create_tensor_from_buffer(buffer, input_tensor.get_shape(), input_layout, dtype);
            }
            case DataType::FLOAT32: {
                auto buffer = host_buffer::get_as<float>(input_tensor);
                return create_tensor_from_buffer(buffer, input_tensor.get_shape(), input_layout, dtype);
            }
            case DataType::BFLOAT16: {
                auto buffer = host_buffer::get_as<::bfloat16>(input_tensor);
                return create_tensor_from_buffer(buffer, input_tensor.get_shape(), input_layout, dtype);
            }
            default: TT_THROW(fmt::format("Unsupported DataType: {}", input_dtype)); break;
        }
    };
    return is_multi_device_tensor(input_tensor) ? transform(input_tensor, convert_dtype) : convert_dtype(input_tensor);
}

}  // namespace detail

struct ToDtype {
    // TODO: Move to cpp once we merge with tt_eager
    static Tensor invoke(const ttnn::Tensor& input_tensor, const ttnn::DataType& dtype) {
        auto input_layout = input_tensor.get_layout();
        auto input_dtype = input_tensor.get_dtype();

        if (input_dtype == dtype) {
            return input_tensor;
        }

        auto row_major_input_tensor = input_tensor.to(ttnn::ROW_MAJOR_LAYOUT);
        auto intermediate_tensor = is_multi_device_tensor(row_major_input_tensor) ? transform(row_major_input_tensor, detail::convert_to_cpp_supported_dtype) : detail::convert_to_cpp_supported_dtype(row_major_input_tensor);
        return detail::convert_to_dtype(intermediate_tensor, input_layout, dtype);
    };
};

}  // namespace core
}  // namespace operations
}  // namespace ttnn
