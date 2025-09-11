// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/core/to_dtype/to_dtype_op.hpp"

#include "tt_stl/concepts.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

#include "tt-metalium/host_buffer.hpp"
#include "tt-metalium/bfloat16.hpp"

#include <algorithm>
#include <type_traits>

namespace ttnn::operations::core {
namespace detail {

struct bfloat4_tag {};
struct bfloat8_tag {};

// Preprocess the storage to unpack the bfloat8/4 tiles into float32.
tt::tt_metal::HostStorage preprocess_storage(
    const tt::tt_metal::HostStorage& input_storage, const DataType input_dtype) {
    constexpr bool row_major_output = false;
    constexpr bool is_exp_a = false;

    if (input_dtype == DataType::BFLOAT8_B) {
        return input_storage.transform([&](const tt::tt_metal::HostBuffer& buffer) {
            tt::stl::Span<const uint32_t> uint32_data = buffer.view_as<const uint32_t>();
            auto float_unpacked_data = unpack_bfp8_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a);
            return tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
        });
    } else if (input_dtype == DataType::BFLOAT4_B) {
        return input_storage.transform([&](const tt::tt_metal::HostBuffer& buffer) {
            tt::stl::Span<const uint32_t> uint32_data = buffer.view_as<const uint32_t>();
            auto float_unpacked_data = unpack_bfp4_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a);
            return tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
        });
    } else {
        return input_storage;
    }
}

template <typename SrcType, typename DstType>
tt::tt_metal::HostStorage transform_storage(
    const tt::tt_metal::TensorSpec& input_tensor_spec, const tt::tt_metal::HostStorage& input_storage) {
    if constexpr (std::is_same_v<SrcType, DstType>) {
        return input_storage;
    } else if constexpr (std::is_same_v<DstType, bfloat4_tag> || std::is_same_v<DstType, bfloat8_tag>) {
        auto transform_fn = [&](const tt::tt_metal::HostBuffer& buffer) {
            ttsl::Span<const SrcType> data = buffer.view_as<const SrcType>();
            std::vector<SrcType> tilized_data;  // empty if `data` is already in tile layout.
            if (input_tensor_spec.layout() == Layout::ROW_MAJOR) {
                tilized_data = tt::tt_metal::tensor_impl::convert_layout_row_major_to_tile(
                    input_tensor_spec.physical_shape(), input_tensor_spec.tile(), data);
                data = ttsl::make_const_span(tilized_data);
            }

            auto float_packed_data = [&]() {
                constexpr bool row_major_input = false;
                constexpr bool is_exp_a = false;
                if constexpr (std::is_same_v<DstType, bfloat8_tag>) {
                    return pack_as_bfp8_tiles(data, row_major_input, is_exp_a);
                } else if constexpr (std::is_same_v<DstType, bfloat4_tag>) {
                    return pack_as_bfp4_tiles(data, row_major_input, is_exp_a);
                } else {
                    static_assert(ttsl::concepts::always_false_v<DstType>, "Unsupported data type");
                }
            }();
            return tt::tt_metal::HostBuffer(std::move(float_packed_data));
        };

        return input_storage.transform(transform_fn);
    } else {
        auto transform_fn = [&](const tt::tt_metal::HostBuffer& buffer) {
            auto data = buffer.view_as<const SrcType>();
            std::vector<DstType> output_vector(data.size());
            std::transform(data.begin(), data.end(), output_vector.begin(), [](SrcType value) {
                if constexpr (std::is_same_v<DstType, bfloat16>) {
                    return bfloat16(static_cast<float>(value));
                } else {
                    return static_cast<DstType>(value);
                }
            });
            return tt::tt_metal::HostBuffer(std::move(output_vector));
        };

        return input_storage.transform(transform_fn);
    }
}

}  // namespace detail

Tensor ToDtype::invoke(const ttnn::Tensor& input_tensor, const ttnn::DataType& dtype) {
    const auto src_type = input_tensor.dtype();
    if (src_type == dtype) {
        return input_tensor;
    }

    TT_FATAL(is_cpu_tensor(input_tensor), "to_dtype(...) function only supports host tensors!");

    auto input_storage = detail::preprocess_storage(input_tensor.host_storage(), src_type);

    auto output_storage = [src_type, dst_type = dtype, &input_tensor, &input_storage]() {
        auto with_src_and_dst = [&]<typename SrcType, typename DstType>() {
            return detail::transform_storage<SrcType, DstType>(input_tensor.tensor_spec(), input_storage);
        };

        auto with_src = [dst_type, &with_src_and_dst]<typename SrcType>() {
            switch (dst_type) {
                case DataType::BFLOAT4_B: return with_src_and_dst.operator()<SrcType, detail::bfloat4_tag>();
                case DataType::BFLOAT8_B: return with_src_and_dst.operator()<SrcType, detail::bfloat8_tag>();
                case DataType::FLOAT32: return with_src_and_dst.operator()<SrcType, float>();
                case DataType::BFLOAT16: return with_src_and_dst.operator()<SrcType, bfloat16>();
                case DataType::UINT8: return with_src_and_dst.operator()<SrcType, uint8_t>();
                case DataType::UINT16: return with_src_and_dst.operator()<SrcType, uint16_t>();
                case DataType::UINT32: return with_src_and_dst.operator()<SrcType, uint32_t>();
                case DataType::INT32: return with_src_and_dst.operator()<SrcType, int32_t>();
                case DataType::INVALID: TT_THROW("Unsupported data type conversion requested. Source type is invalid!");
            }
            TT_THROW("Unreachable");
        };

        switch (src_type) {
            case DataType::BFLOAT4_B:
            case DataType::BFLOAT8_B:
            case DataType::FLOAT32: return with_src.operator()<float>();
            case DataType::BFLOAT16: return with_src.operator()<bfloat16>();
            case DataType::UINT8: return with_src.operator()<uint8_t>();
            case DataType::UINT16: return with_src.operator()<uint16_t>();
            case DataType::UINT32: return with_src.operator()<uint32_t>();
            case DataType::INT32: return with_src.operator()<int32_t>();
            case DataType::INVALID: TT_THROW("Unsupported data type conversion requested. Source type is invalid!");
        }
        TT_THROW("Unreachable");
    }();

    const auto layout =
        (dtype == DataType::BFLOAT4_B || dtype == DataType::BFLOAT8_B) ? Layout::TILE : input_tensor.layout();

    auto output_spec = TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout::fromPaddedShape(
            dtype,
            tt::tt_metal::PageConfig(layout),
            MemoryConfig{},
            input_tensor.logical_shape(),
            input_tensor.padded_shape()));

    return Tensor(tt::tt_metal::HostStorage(std::move(output_storage)), output_spec, input_tensor.tensor_topology());
};

}  // namespace ttnn::operations::core
