// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/core/to_dtype/to_dtype_op.hpp"
#include "ttnn/operations/core/to_dtype/transform_type.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::core {
namespace detail {

using DTypeConversionFunc =
    std::function<tt::tt_metal::Tensor(const tt::tt_metal::Tensor&, const tt::tt_metal::DataType dst_type)>;

/**
 * @brief Gets a conversion function for a specific source and destination type.
 *
 * This function acts as a factory, returning a callable std::function that
 * wraps the appropriate templated transform_type function.
 *
 * @param src_type The source DataType.
 * @param dst_type The destination DataType.
 * @return A std::function for the conversion, or throw an exception if unsupported.
 */
static DTypeConversionFunc get_dtype_conversion_function(DataType src_type, DataType dst_type) {
    auto get_dest_func = [&]<typename SrcCppType>() -> DTypeConversionFunc {
        switch (dst_type) {
            case DataType::BFLOAT4_B: return &transform_type<SrcCppType, bfloat4_b>;
            case DataType::BFLOAT8_B: return &transform_type<SrcCppType, bfloat8_b>;
            case DataType::BFLOAT16: return &transform_type<SrcCppType, bfloat16>;
            case DataType::FLOAT32: return &transform_type<SrcCppType, float>;
            case DataType::UINT8: return &transform_type<SrcCppType, uint8_t>;
            case DataType::UINT16: return &transform_type<SrcCppType, uint16_t>;
            case DataType::UINT32: return &transform_type<SrcCppType, uint32_t>;
            case DataType::INT32: return &transform_type<SrcCppType, int32_t>;
            default: TT_FATAL(false, "Unsupported data type conversion requested. Destination type is invalid!");
        }
    };

    switch (src_type) {
        case DataType::BFLOAT4_B: return get_dest_func.operator()<bfloat4_b>();
        case DataType::BFLOAT8_B: return get_dest_func.operator()<bfloat8_b>();
        case DataType::BFLOAT16: return get_dest_func.operator()<bfloat16>();
        case DataType::FLOAT32: return get_dest_func.operator()<float>();
        case DataType::UINT8: return get_dest_func.operator()<uint8_t>();
        case DataType::UINT16: return get_dest_func.operator()<uint16_t>();
        case DataType::UINT32: return get_dest_func.operator()<uint32_t>();
        case DataType::INT32: return get_dest_func.operator()<int32_t>();
        default: TT_FATAL(false, "Unsupported data type conversion requested. Source type is invalid!");
    }
}

}  // namespace detail

Tensor ToDtype::invoke(const ttnn::Tensor& input_tensor, const ttnn::DataType& dtype) {
    if (input_tensor.dtype() == dtype) {
        return input_tensor;
    }

    const auto& transform_type = detail::get_dtype_conversion_function(input_tensor.dtype(), dtype);
    return tt::tt_metal::is_multi_device_host_tensor(input_tensor)
               ? transform(input_tensor, [&](const ttnn::Tensor& tensor) { return transform_type(tensor, dtype); })
               : transform_type(input_tensor, dtype);
};

}  // namespace ttnn::operations::core
