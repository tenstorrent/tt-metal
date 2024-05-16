// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "tensor/tensor.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace operations {

namespace core {

namespace detail {

inline Tensor convert_to_cpp_dtypes(const Tensor& input_tensor) {
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
        auto uint32_data = std::get<owned_buffer::Buffer<std::uint32_t>>(std::get<OwnedBuffer>(buffer)).get();
        auto float_unpacked_data =
            unpack_bfp8_tiles_into_float_vec(uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false);
        buffer = owned_buffer::create<float>(std::move(float_unpacked_data));
        input_dtype = DataType::FLOAT32;
    } else if (input_dtype == DataType::BFLOAT4_B) {
        auto uint32_data = std::get<owned_buffer::Buffer<std::uint32_t>>(std::get<OwnedBuffer>(buffer)).get();
        auto float_unpacked_data =
            unpack_bfp4_tiles_into_float_vec(uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false);
        buffer = owned_buffer::create<float>(std::move(float_unpacked_data));
        input_dtype = DataType::FLOAT32;
    } else {
        TT_THROW("Unsupported input data type");
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

}  // namespace detail

struct ToDtype {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            1,
            8,
            {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::float32, ttnn::uint16, ttnn::uint32, ttnn::int32},
            {ttnn::ROW_MAJOR_LAYOUT, ttnn::TILE_LAYOUT},
            true,
            true,
            false,
            false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& tensor_arg, Args&&... args) {
        return std::make_tuple(tensor_arg);
    };

    // TODO: Move to cpp once we merge with tt_eager
    static Tensor execute(const ttnn::Tensor& input_tensor, const ttnn::DataType& dtype) {
        auto input_layout = input_tensor.get_layout();
        auto input_dtype = input_tensor.get_dtype();

        if (input_dtype == dtype) {
            return input_tensor;
        }

        if (input_layout != ttnn::ROW_MAJOR_LAYOUT) {
            TT_THROW("Only ROW_MAJOR_LAYOUT is supported");
        }

        auto output_tensor = input_tensor;
        return output_tensor;
    };
};

}  // namespace core
}  // namespace operations
}  // namespace ttnn
