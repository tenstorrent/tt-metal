// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/tensor/types.hpp"

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

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::NdShardSpec& spec) {
    os << "{";
    os << "\"shard_shape\":[";

    // Format shard_shape as array
    const auto& shape = spec.shard_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        os << shape[i];
        if (i < shape.size() - 1) {
            os << ", ";
        }
    }
    os << "],";

    os << "\"grid\":[";

    const auto& ranges = spec.grid.ranges();
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto& range = ranges[i];
        os << R"({"start":{"x":)" << range.start_coord.x << ",\"y\":" << range.start_coord.y << "},";
        os << R"("end":{"x":)" << range.end_coord.x << ",\"y\":" << range.end_coord.y << "}}";
        if (i < ranges.size() - 1) {
            os << ", ";
        }
    }
    os << "],";

    os << R"("orientation":")";
    switch (spec.orientation) {
        case ShardOrientation::ROW_MAJOR: os << "ShardOrientation::ROW_MAJOR"; break;
        case ShardOrientation::COL_MAJOR: os << "ShardOrientation::COL_MAJOR"; break;
    }
    os << "\",";

    os << R"("shard_distribution_strategy":")";
    switch (spec.shard_distribution_strategy) {
        case ShardDistributionStrategy::ROUND_ROBIN_1D: os << "ShardDistributionStrategy::ROUND_ROBIN_1D"; break;
        case ShardDistributionStrategy::GRID_2D: os << "ShardDistributionStrategy::GRID_2D"; break;
    }
    os << "\"";

    os << "}";
    return os;
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
        default: TT_THROW("Unsupported DataType"); return tt::DataFormat::Float16_b;  // for clang-tidy
    }
}

tt::tt_metal::DataType dataformat_to_datatype_converter(tt::DataFormat dataformat) {
    switch (dataformat) {
        case tt::DataFormat::Float16_b: return tt::tt_metal::DataType::BFLOAT16;
        case tt::DataFormat::Bfp8_b: return tt::tt_metal::DataType::BFLOAT8_B;
        case tt::DataFormat::Bfp4_b: return tt::tt_metal::DataType::BFLOAT4_B;
        case tt::DataFormat::Float32: return tt::tt_metal::DataType::FLOAT32;
        case tt::DataFormat::Int32: return tt::tt_metal::DataType::INT32;
        case tt::DataFormat::UInt32: return tt::tt_metal::DataType::UINT32;
        case tt::DataFormat::UInt16: return tt::tt_metal::DataType::UINT16;
        case tt::DataFormat::UInt8: return tt::tt_metal::DataType::UINT8;
        default: TT_THROW("Unsupported DataFormat"); return tt::tt_metal::DataType::BFLOAT16;  // for clang-tidy
    }
}

}  // namespace tt::tt_metal
