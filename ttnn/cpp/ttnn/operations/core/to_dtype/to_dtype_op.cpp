// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/core/to_dtype/to_dtype_op.hpp"

#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/types.hpp"

#include "tt-metalium/host_buffer.hpp"
#include "tt-metalium/bfloat16.hpp"

#include <algorithm>
#include <type_traits>

namespace ttnn::operations::core {
namespace detail {

struct bfloat4_tag {};
struct bfloat8_tag {};

template <typename SrcType>
auto preprocess_tensor(const tt::tt_metal::Tensor& input_tensor) {
    using namespace tt::tt_metal;
    constexpr bool row_major_output = false;
    constexpr bool is_exp_a = false;

    if constexpr (std::is_same_v<SrcType, bfloat8_tag>) {
        return input_tensor.host_storage().transform([&](const tt::tt_metal::HostBuffer& buffer) {
            tt::stl::Span<const uint32_t> uint32_data = tt::tt_metal::host_buffer::get_as<uint32_t>(buffer);
            auto float_unpacked_data = unpack_bfp8_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a);
            return tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
        });
    } else if constexpr (std::is_same_v<SrcType, bfloat4_tag>) {
        return input_tensor.host_storage().transform([&](const tt::tt_metal::HostBuffer& buffer) {
            tt::stl::Span<const uint32_t> uint32_data = tt::tt_metal::host_buffer::get_as<uint32_t>(buffer);
            auto float_unpacked_data = unpack_bfp4_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a);
            return tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
        });
    } else {
        return input_tensor.host_storage();
    }
}

template <typename DstType, typename U>
auto postprocess_vector(std::vector<U>&& data, const tt::tt_metal::Tensor& input_tensor) {
    using namespace tt::tt_metal;
    constexpr bool row_major_input = false;
    constexpr bool is_exp_a = false;

    if constexpr (std::is_same_v<DstType, bfloat8_tag> || std::is_same_v<DstType, bfloat4_tag>) {
        if (input_tensor.layout() == Layout::ROW_MAJOR) {
            data = tensor_impl::convert_layout_row_major_to_tile(
                input_tensor.tensor_spec().physical_shape(),
                input_tensor.tensor_spec().tile(),
                tt::stl::make_const_span(data));
        }
        return tt::tt_metal::HostBuffer(
            std::is_same_v<DstType, bfloat8_tag> ? pack_fp32_vec_as_bfp8_tiles(data, row_major_input, is_exp_a)
                                                 : pack_fp32_vec_as_bfp4_tiles(data, row_major_input, is_exp_a));
    }
    return tt::tt_metal::HostBuffer(std::move(data));
}

template <typename SrcType, typename DstType>
tt::tt_metal::Tensor transform_type(const tt::tt_metal::Tensor& input_tensor, const DataType dst_type) {
    using namespace tt::tt_metal;

    TT_FATAL(is_cpu_tensor(input_tensor), "transform_type(...) function only supports host tensors!");
    auto input_dtype = input_tensor.dtype();
    auto storage = preprocess_tensor<SrcType>(input_tensor);

    // For SrcType or DstType that are bfloat4_tag or bfloat8_tag, use 'float' as the
    // intermediate type because these compressed formats require (un)packing from/to float
    // for conversion.
    //
    // Otherwise, use the original SrcType or DstType directly as the intermediate type.
    using IntermediateSrcType = std::
        conditional_t<std::is_same_v<SrcType, bfloat4_tag> || std::is_same_v<SrcType, bfloat8_tag>, float, SrcType>;
    using IntermediateDstType = std::
        conditional_t<std::is_same_v<DstType, bfloat4_tag> || std::is_same_v<DstType, bfloat8_tag>, float, DstType>;

    auto output_storage = storage.transform([&](const tt::tt_metal::HostBuffer& host_buffer) {
        auto buffer = host_buffer.view_as<IntermediateSrcType>();
        std::vector<IntermediateDstType> output_vector(buffer.size());
        std::transform(buffer.begin(), buffer.end(), output_vector.begin(), [](const auto& value) {
            if constexpr (std::is_same_v<SrcType, ::bfloat16>) {
                return IntermediateDstType(value.to_float());
            } else if constexpr (std::is_same_v<IntermediateDstType, ::bfloat16>) {
                return IntermediateDstType(static_cast<float>(value));
            } else {
                return IntermediateDstType(value);
            }
        });
        return postprocess_vector<DstType>(std::move(output_vector), input_tensor);
    });

    auto layout = std::is_same_v<DstType, bfloat4_tag> || std::is_same_v<DstType, bfloat8_tag> ? Layout::TILE
                                                                                               : input_tensor.layout();

    auto spec = TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout::fromPaddedShape(
            dst_type,
            tt::tt_metal::PageConfig(layout),
            MemoryConfig{},
            input_tensor.logical_shape(),
            input_tensor.padded_shape()));

    return Tensor(std::move(output_storage), spec, input_tensor.distributed_tensor_config());
}

}  // namespace detail

Tensor ToDtype::invoke(const ttnn::Tensor& input_tensor, const ttnn::DataType& dtype) {
    using namespace detail;
    const auto src_type = input_tensor.dtype();
    if (src_type == dtype) {
        return input_tensor;
    }

    auto transform_tensor = [src_type, dtype]() {
        auto get_dest_func = [&]<typename SrcType>() {
            switch (dtype) {
                case DataType::BFLOAT4_B: return &transform_type<SrcType, bfloat4_tag>;
                case DataType::BFLOAT8_B: return &transform_type<SrcType, bfloat8_tag>;
                case DataType::BFLOAT16: return &transform_type<SrcType, bfloat16>;
                case DataType::FLOAT32: return &transform_type<SrcType, float>;
                case DataType::UINT8: return &transform_type<SrcType, uint8_t>;
                case DataType::UINT16: return &transform_type<SrcType, uint16_t>;
                case DataType::UINT32: return &transform_type<SrcType, uint32_t>;
                case DataType::INT32: return &transform_type<SrcType, int32_t>;
                case DataType::INVALID:
                    TT_THROW("Unsupported data type conversion requested. Destination type is invalid!");
            }
            TT_THROW("Unreachable");
        };

        switch (src_type) {
            case DataType::BFLOAT4_B: return get_dest_func.operator()<bfloat4_tag>();
            case DataType::BFLOAT8_B: return get_dest_func.operator()<bfloat8_tag>();
            case DataType::BFLOAT16: return get_dest_func.operator()<bfloat16>();
            case DataType::FLOAT32: return get_dest_func.operator()<float>();
            case DataType::UINT8: return get_dest_func.operator()<uint8_t>();
            case DataType::UINT16: return get_dest_func.operator()<uint16_t>();
            case DataType::UINT32: return get_dest_func.operator()<uint32_t>();
            case DataType::INT32: return get_dest_func.operator()<int32_t>();
            case DataType::INVALID: TT_THROW("Unsupported data type conversion requested. Source type is invalid!");
        }
        TT_THROW("Unreachable");
    }();

    return transform_tensor(input_tensor, dtype);
};

}  // namespace ttnn::operations::core
