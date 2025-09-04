// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_impl.hpp"

namespace tt::tt_metal {

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::DataType& data_type) {
    switch (data_type) {
        case DataType::BFLOAT16: return os << "DataType::BFLOAT16";
        case DataType::FLOAT32: return os << "DataType::FLOAT32";
        case DataType::UINT32: return os << "DataType::UINT32";
        case DataType::BFLOAT8_B: return os << "DataType::BFLOAT8_B";
        case DataType::BFLOAT4_B: return os << "DataType::BFLOAT4_B";
        case DataType::UINT8: return os << "DataType::UINT8";
        case DataType::UINT16: return os << "DataType::UINT16";
        case DataType::INT32: return os << "DataType::INT32";
        case DataType::INVALID:
        default: return os << "Invalid";
    }
}

bool is_floating_point(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16:
        case DataType::FLOAT32:
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: return true;
        default: return false;
    }
}

bool is_block_float(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: return true;
        default: return false;
    }
}

tt::DataFormat datatype_to_dataformat_converter(tt::tt_metal::DataType datatype) {
    switch (datatype) {
        case tt::tt_metal::DataType::BFLOAT16: return tt::DataFormat::Float16_b;
        case tt::tt_metal::DataType::BFLOAT8_B: return tt::DataFormat::Bfp8_b;
        case tt::tt_metal::DataType::BFLOAT4_B: return tt::DataFormat::Bfp4_b;
        case tt::tt_metal::DataType::FLOAT32: return tt::DataFormat::Float32;
        case tt::tt_metal::DataType::INT32: return tt::DataFormat::Int32;
        case tt::tt_metal::DataType::UINT32: return tt::DataFormat::UInt32;
        case tt::tt_metal::DataType::UINT16: return tt::DataFormat::UInt16;
        case tt::tt_metal::DataType::UINT8: return tt::DataFormat::UInt8;
        default: TT_ASSERT(false, "Unsupported DataType"); return tt::DataFormat::Float16_b;
    }
}

}  // namespace tt::tt_metal
