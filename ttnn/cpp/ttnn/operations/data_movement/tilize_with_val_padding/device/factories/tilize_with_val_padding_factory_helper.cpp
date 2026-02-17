// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
using namespace tt::tt_metal;

namespace ttnn::prim::detail {

uint32_t get_packed_value(const Tensor& tensor, const PadValue& pad_value) {
    return std::visit(
        [&tensor](auto&& pad_value) {
            using T = std::decay_t<decltype(pad_value)>;
            if constexpr (std::is_same_v<T, float>) {
                if (tensor.dtype() == DataType::BFLOAT16) {
                    bfloat16 bfloat_pad_value = bfloat16((pad_value));
                    return pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});
                }
                if (tensor.dtype() == DataType::UINT16) {
                    uint16_t uint16_pad_value = static_cast<uint16_t>(pad_value);
                    return ttnn::operations::data_movement::pack_two_uint16_into_uint32(
                        {uint16_pad_value, uint16_pad_value});
                }
                TT_FATAL(
                    tensor.dtype() == DataType::FLOAT32 or tensor.dtype() == DataType::UINT32 or
                        tensor.dtype() == DataType::INT32,
                    "only supporting bfloat16, float32, and uint32/int32/uint16");
                return (uint32_t)((pad_value));
            }
            if constexpr (std::is_same_v<T, uint32_t>) {
                if (tensor.dtype() == DataType::BFLOAT16) {
                    bfloat16 bfloat_pad_value = bfloat16((float)(pad_value));
                    return pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});
                }
                if (tensor.dtype() == DataType::UINT16) {
                    uint16_t uint16_pad_value = static_cast<uint16_t>(pad_value);
                    return ttnn::operations::data_movement::pack_two_uint16_into_uint32(
                        {uint16_pad_value, uint16_pad_value});
                }
                TT_FATAL(
                    tensor.dtype() == DataType::FLOAT32 or tensor.dtype() == DataType::INT32 or
                        tensor.dtype() == DataType::UINT32,
                    "only supporting bfloat16, float32, and int32/uint32/uint16");
                return ((pad_value));
            }
            TT_THROW("type not supported");
        },
        pad_value);
}

}  // namespace ttnn::prim::detail
