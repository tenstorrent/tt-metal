// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "types_wrapper.hpp"
#include <optional>

#include "operations/eltwise/unary/common/unary_op_types.hpp"
#include "tensor/types.hpp" // DataType, Layout, StorageType, MemoryConfig
#include "tt_metal/impl/buffers/buffer_constants.hpp" // TensorMemoryLayout, ShardOrientation
#include "tt_metal/impl/buffers/buffer.hpp" // BufferType, ShardSpec
#include "common/core_coord.h" // CoreRangeSet
#include "ttnn/operations/matmul/device/matmul_types.hpp" // MatmulMultiCoreReuseProgramConfig et al.

namespace ttnn::str_wrapper
{

// no MLIR equivalent for storage type
std::optional<tt::tt_metal::StorageType> to_storage_type(const std::string& storage_type_str)
{
    if (storage_type_str == "owned") return tt::tt_metal::StorageType::OWNED;
    if (storage_type_str == "device") return tt::tt_metal::StorageType::DEVICE;
    if (storage_type_str == "borrowed") return tt::tt_metal::StorageType::BORROWED;
    if (storage_type_str == "multi_device") return tt::tt_metal::StorageType::MULTI_DEVICE;
    if (storage_type_str == "multi_device_host") return tt::tt_metal::StorageType::MULTI_DEVICE_HOST;
    return std::nullopt;
}

std::optional<tt::tt_metal::Layout> to_layout(const std::string& layout_str)
{
    // Replicating MLIR defined strings
    // def TT_OperandConstraintScalar : I32BitEnumAttrCaseBit<"Scalar", 3, "scalar">;
    // def TT_OperandConstraintTile : I32BitEnumAttrCaseBit<"Tile", 4, "tile">;
    if (layout_str == "scalar") return tt::tt_metal::Layout::ROW_MAJOR;
    if (layout_str == "tile") return tt::tt_metal::Layout::TILE;
    return std::nullopt;
}

std::optional<tt::tt_metal::TensorMemoryLayout> to_tensor_memory_layout(const std::string& tensor_memory_layout_str)
{
    // Replicating MLIR defined strings
    // def TT_OperandConstraintInterleaved : I32BitEnumAttrCaseBit<"Interleaved", 6, "interleaved">;
    // def TT_OperandConstraintSingleBank : I32BitEnumAttrCaseBit<"SingleBank", 7, "single_bank">;
    // def TT_OperandConstraintHeightSharded : I32BitEnumAttrCaseBit<"HeightSharded", 8, "height_sharded">;
    // def TT_OperandConstraintWidthSharded : I32BitEnumAttrCaseBit<"WidthSharded", 9, "width_sharded">;
    // def TT_OperandConstraintBlockSharded : I32BitEnumAttrCaseBit<"BlockSharded", 10, "block_sharded">;
    if (tensor_memory_layout_str == "interleaved") return tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
    if (tensor_memory_layout_str == "single_bank") return tt::tt_metal::TensorMemoryLayout::SINGLE_BANK;
    if (tensor_memory_layout_str == "height_sharded") return tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED;
    if (tensor_memory_layout_str == "width_sharded") return tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
    if (tensor_memory_layout_str == "block_sharded") return tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
    return std::nullopt;
}

std::optional<tt::tt_metal::DataType> to_data_type(const std::string& data_type_str)
{
    // Replicate MLIR defined strings
    // def TT_Float32 : I32EnumAttrCase<"Float32", 0, "f32">;
    // def TT_Float16 : I32EnumAttrCase<"Float16", 1, "f16">;
    // def TT_BFloat16 : I32EnumAttrCase<"BFloat16", 2, "bf16">;
    // def TT_BFP_Float8 : I32EnumAttrCase<"BFP_Float8", 3, "bfp_f8">;
    // def TT_BFP_BFloat8 : I32EnumAttrCase<"BFP_BFloat8", 4, "bfp_bf8">;
    // def TT_BFP_Float4 : I32EnumAttrCase<"BFP_Float4", 5, "bfp_f4">;
    // def TT_BFP_BFloat4 : I32EnumAttrCase<"BFP_BFloat4", 6, "bfp_bf4">;
    // def TT_BFP_Float2 : I32EnumAttrCase<"BFP_Float2", 7, "bfp_f2">;
    // def TT_BFP_BFloat2 : I32EnumAttrCase<"BFP_BFloat2", 8, "bfp_bf2">;
    // def TT_UInt32 : I32EnumAttrCase<"UInt32", 9, "u32">;
    // def TT_UInt16 : I32EnumAttrCase<"UInt16", 10, "u16">;
    // def TT_UInt8 : I32EnumAttrCase<"UInt8", 11, "u8">;

    if (data_type_str == "f32") return tt::tt_metal::DataType::FLOAT32;
    if (data_type_str == "bf16") return tt::tt_metal::DataType::BFLOAT16;
    if (data_type_str == "bfp_bf8") return tt::tt_metal::DataType::BFLOAT8_B;
    if (data_type_str == "bfp_bf4") return tt::tt_metal::DataType::BFLOAT4_B;
    if (data_type_str == "u32") return tt::tt_metal::DataType::UINT32;
    if (data_type_str == "u16") return tt::tt_metal::DataType::UINT16;
    if (data_type_str == "u8") return tt::tt_metal::DataType::UINT8;
    // if (data_type_str == "INT32") return tt::tt_metal::DataType::INT32;
    return std::nullopt;
}

std::optional<tt::tt_metal::BufferType> to_buffer_type(const std::string& buffer_type_str)
{
    // Replicate MLIR defined strings
    // def TT_OperandConstraintSystem : I32BitEnumAttrCaseBit<"System", 0, "system">;
    // def TT_OperandConstraintDRAM : I32BitEnumAttrCaseBit<"DRAM", 1, "dram">;
    // def TT_OperandConstraintL1 : I32BitEnumAttrCaseBit<"L1", 2, "l1">;
    if (buffer_type_str == "dram") return tt::tt_metal::BufferType::DRAM;
    if (buffer_type_str == "l1") return tt::tt_metal::BufferType::L1;
    if (buffer_type_str == "system") return tt::tt_metal::BufferType::SYSTEM_MEMORY;
    // if (buffer_type_str == "L1_SMALL") return tt::tt_metal::BufferType::L1_SMALL;
    // if (buffer_type_str == "TRACE") return tt::tt_metal::BufferType::TRACE;
    return std::nullopt;
}

std::optional<tt::tt_metal::ShardOrientation> to_shard_orientation(const std::string& shard_str)
{
    // no MLIR equivalent yet
    if (shard_str == "row_major") return tt::tt_metal::ShardOrientation::ROW_MAJOR;
    if (shard_str == "col_major") return tt::tt_metal::ShardOrientation::COL_MAJOR;
    return std::nullopt;
}

std::optional<ttnn::operations::unary::UnaryOpType> to_unary_op_type(const std::string& unary_op_type_str)
{
    if (unary_op_type_str == "EXP") return ttnn::operations::unary::UnaryOpType::EXP;
    if (unary_op_type_str == "RECIP") return ttnn::operations::unary::UnaryOpType::RECIP;


    if (unary_op_type_str == "GELU") return ttnn::operations::unary::UnaryOpType::GELU;
    if (unary_op_type_str == "RELU") return ttnn::operations::unary::UnaryOpType::RELU;
    if (unary_op_type_str == "SQRT") return ttnn::operations::unary::UnaryOpType::SQRT;
    if (unary_op_type_str == "SIGMOID") return ttnn::operations::unary::UnaryOpType::SIGMOID;
    if (unary_op_type_str == "LOG") return ttnn::operations::unary::UnaryOpType::LOG;
    if (unary_op_type_str == "TANH") return ttnn::operations::unary::UnaryOpType::TANH;
    if (unary_op_type_str == "LOG2") return ttnn::operations::unary::UnaryOpType::LOG2;
    if (unary_op_type_str == "LOG10") return ttnn::operations::unary::UnaryOpType::LOG10;
    if (unary_op_type_str == "SIN") return ttnn::operations::unary::UnaryOpType::SIN;
    if (unary_op_type_str == "COS") return ttnn::operations::unary::UnaryOpType::COS;
    if (unary_op_type_str == "ABS") return ttnn::operations::unary::UnaryOpType::ABS;
    if (unary_op_type_str == "SIGN") return ttnn::operations::unary::UnaryOpType::SIGN;
    if (unary_op_type_str == "SQUARE") return ttnn::operations::unary::UnaryOpType::SQUARE;
    if (unary_op_type_str == "EQZ") return ttnn::operations::unary::UnaryOpType::EQZ;
    if (unary_op_type_str == "NEZ") return ttnn::operations::unary::UnaryOpType::NEZ;
    if (unary_op_type_str == "GTZ") return ttnn::operations::unary::UnaryOpType::GTZ;
    if (unary_op_type_str == "LTZ") return ttnn::operations::unary::UnaryOpType::LTZ;
    if (unary_op_type_str == "GEZ") return ttnn::operations::unary::UnaryOpType::GEZ;
    if (unary_op_type_str == "LEZ") return ttnn::operations::unary::UnaryOpType::LEZ;
    if (unary_op_type_str == "RELU_MAX") return ttnn::operations::unary::UnaryOpType::RELU_MAX;
    if (unary_op_type_str == "RELU_MIN") return ttnn::operations::unary::UnaryOpType::RELU_MIN;
    if (unary_op_type_str == "POWER") return ttnn::operations::unary::UnaryOpType::POWER;
    if (unary_op_type_str == "LEAKY_RELU") return ttnn::operations::unary::UnaryOpType::LEAKY_RELU;
    if (unary_op_type_str == "ELU") return ttnn::operations::unary::UnaryOpType::ELU;
    if (unary_op_type_str == "EXP2") return ttnn::operations::unary::UnaryOpType::EXP2;
    if (unary_op_type_str == "HEAVISIDE") return ttnn::operations::unary::UnaryOpType::HEAVISIDE;
    if (unary_op_type_str == "EXPM1") return ttnn::operations::unary::UnaryOpType::EXPM1;
    if (unary_op_type_str == "SIGNBIT") return ttnn::operations::unary::UnaryOpType::SIGNBIT;
    if (unary_op_type_str == "ASIN") return ttnn::operations::unary::UnaryOpType::ASIN;
    if (unary_op_type_str == "ACOS") return ttnn::operations::unary::UnaryOpType::ACOS;
    if (unary_op_type_str == "RSQRT") return ttnn::operations::unary::UnaryOpType::RSQRT;
    if (unary_op_type_str == "RELU6") return ttnn::operations::unary::UnaryOpType::RELU6;
    if (unary_op_type_str == "ATAN") return ttnn::operations::unary::UnaryOpType::ATAN;
    if (unary_op_type_str == "ERF") return ttnn::operations::unary::UnaryOpType::ERF;
    if (unary_op_type_str == "ERFC") return ttnn::operations::unary::UnaryOpType::ERFC;
    if (unary_op_type_str == "ISINF") return ttnn::operations::unary::UnaryOpType::ISINF;
    if (unary_op_type_str == "ISPOSINF") return ttnn::operations::unary::UnaryOpType::ISPOSINF;
    if (unary_op_type_str == "ISNEGINF") return ttnn::operations::unary::UnaryOpType::ISNEGINF;
    if (unary_op_type_str == "ISNAN") return ttnn::operations::unary::UnaryOpType::ISNAN;
    if (unary_op_type_str == "LOGICAL_NOT_UNARY") return ttnn::operations::unary::UnaryOpType::LOGICAL_NOT_UNARY;
    if (unary_op_type_str == "ISFINITE") return ttnn::operations::unary::UnaryOpType::ISFINITE;
    if (unary_op_type_str == "ERFINV") return ttnn::operations::unary::UnaryOpType::ERFINV;
    if (unary_op_type_str == "I0") return ttnn::operations::unary::UnaryOpType::I0;
    if (unary_op_type_str == "TAN") return ttnn::operations::unary::UnaryOpType::TAN;
    if (unary_op_type_str == "RSUB") return ttnn::operations::unary::UnaryOpType::RSUB;
    if (unary_op_type_str == "RDIV") return ttnn::operations::unary::UnaryOpType::RDIV;
    if (unary_op_type_str == "SILU") return ttnn::operations::unary::UnaryOpType::SILU;
    if (unary_op_type_str == "SOFTPLUS") return ttnn::operations::unary::UnaryOpType::SOFTPLUS;
    if (unary_op_type_str == "IDENTITY") return ttnn::operations::unary::UnaryOpType::IDENTITY;
    if (unary_op_type_str == "NEG") return ttnn::operations::unary::UnaryOpType::NEG;
    if (unary_op_type_str == "ADD_UNARY_SFPU") return ttnn::operations::unary::UnaryOpType::ADD_UNARY_SFPU;
    if (unary_op_type_str == "SUB_UNARY_SFPU") return ttnn::operations::unary::UnaryOpType::SUB_UNARY_SFPU;
    if (unary_op_type_str == "MUL_UNARY_SFPU") return ttnn::operations::unary::UnaryOpType::MUL_UNARY_SFPU;
    if (unary_op_type_str == "DIV_UNARY_SFPU") return ttnn::operations::unary::UnaryOpType::DIV_UNARY_SFPU;
    if (unary_op_type_str == "IDENTITY_UINT32") return ttnn::operations::unary::UnaryOpType::IDENTITY_UINT32;
    if (unary_op_type_str == "UNARY_NE") return ttnn::operations::unary::UnaryOpType::UNARY_NE;
    if (unary_op_type_str == "UNARY_GT") return ttnn::operations::unary::UnaryOpType::UNARY_GT;
    if (unary_op_type_str == "UNARY_LT") return ttnn::operations::unary::UnaryOpType::UNARY_LT;
    if (unary_op_type_str == "TILED_PROD") return ttnn::operations::unary::UnaryOpType::TILED_PROD;
    if (unary_op_type_str == "TYPECAST") return ttnn::operations::unary::UnaryOpType::TYPECAST;
    if (unary_op_type_str == "BITWISE_XOR") return ttnn::operations::unary::UnaryOpType::BITWISE_XOR;
    if (unary_op_type_str == "BITWISE_NOT") return ttnn::operations::unary::UnaryOpType::BITWISE_NOT;
    if (unary_op_type_str == "BITWISE_AND") return ttnn::operations::unary::UnaryOpType::BITWISE_AND;
    if (unary_op_type_str == "BITWISE_OR") return ttnn::operations::unary::UnaryOpType::BITWISE_OR;
    if (unary_op_type_str == "RIGHT_SHIFT") return ttnn::operations::unary::UnaryOpType::RIGHT_SHIFT;
    if (unary_op_type_str == "FLOOR") return ttnn::operations::unary::UnaryOpType::FLOOR;
    if (unary_op_type_str == "CEIL") return ttnn::operations::unary::UnaryOpType::CEIL;
    if (unary_op_type_str == "LEFT_SHIFT") return ttnn::operations::unary::UnaryOpType::LEFT_SHIFT;
    if (unary_op_type_str == "REMAINDER") return ttnn::operations::unary::UnaryOpType::REMAINDER;
    if (unary_op_type_str == "FMOD") return ttnn::operations::unary::UnaryOpType::FMOD;

    return std::nullopt;
}

// std::optional<ttnn::Shape> vector_to_shard_shape(const std::vector<uint32_t>& shard_shape_vector)
// {
//     return ttnn::Shape(shard_shape_vector);
// }

// Layout layout_by_index(const int index, const std::vector<std::string>& layouts_str)
// {
//     return to_layout(layouts_str.at(index)).value();
// }

// DataType datatype_by_index(const int index, const std::vector<std::string>& data_types_str)
// {
//     return to_data_type(data_types_str.at(index)).value();
// }

// tt::tt_metal::Shape shape_by_index(const int index, const std::vector<std::vector<uint32_t>>& shapes_vectors)
// {
//     return vector_to_shape(shapes_vectors.at(index));
// }

// bool is_sharded_by_index(const int index, const std::vector<bool>& shareded_vector)
// {
//     return shareded_vector.at(0);
// }

// tt::tt_metal::TensorMemoryLayout memory_layout_by_index(const int index, const std::vector<std::string>& memory_layouts)
// {
//     return to_memory_layout(memory_layouts.at(index)).value();
// }

// ShardOrientation shard_orientation_by_index(const int index, const std::vector<std::string>& shards_str)
// {
//     return to_shard_orientation(shards_str.at(index)).value();
// }

// tt::tt_metal::BufferType buffer_type_by_index(const int index, const std::vector<std::string>& buffer_types_str)
// {
//     return to_buffer_type(buffer_types_str.at(index)).value();
// }

// const uint32_t volume(tt::tt_metal::Shape& shape)
// {
//     auto rank = shape.rank();
//     auto volume = 1;
//     for (auto index = 0; index < rank; index++) {
//         volume *= shape.operator[](index);
//     }
//     return volume;
// }

// ttnn::Shape shard_shape_by_index(const int index, const std::vector<std::vector<uint32_t>>& shard_shapes)
// {
//     return vector_to_shard_shape(shard_shapes.at(index)).value();
// }

// CoreRangeSet get_core_range_set_by_index(const int index, const std::vector<CoreRangeSet>& core_range_set)
// {
//     return core_range_set.at(index);
// }

} // namespace ttnn::str_wrapper

namespace ttnn::vector_wrapper
{
    tt::tt_metal::Shape to_shape(const std::vector<uint32_t>& shape_vector)
    {
        return tt::tt_metal::Shape(shape_vector);
    }
} // namespace ttnn::vector_wrapper

namespace ttnn::tuple_wrapper {
    std::optional<tt::tt_metal::ShardSpec> to_shard_spec(
        const std::vector<std::array<uint32_t, 4>>& core_range_set,
        const std::array<uint32_t, 2>& shard_shape,
        const std::string& shard_orientation,
        const bool& halo = false) {

        if (core_range_set.size() == 0)
        {
            return std::nullopt;
        }

        std::set<CoreRange> core_ranges;
        for (const auto& core_range : core_range_set)
        {
            core_ranges.insert(CoreRange{CoreCoord{core_range[0], core_range[1]}, CoreCoord{core_range[2], core_range[3]}});
        }

        return std::make_optional(tt::tt_metal::ShardSpec{
            CoreRangeSet{core_ranges},
            shard_shape,
            str_wrapper::to_shard_orientation(shard_orientation).value(),
            halo
        });
    }

    std::optional<tt::tt_metal::ShardSpec> to_shard_spec(const mlir_interface::shard_spec_tuple& shard_spec_tuple) {
        return to_shard_spec(
            std::get<std::vector<std::array<uint32_t, 4>>>(shard_spec_tuple),
            std::get<std::array<uint32_t, 2>>(shard_spec_tuple),
            std::get<std::string>(shard_spec_tuple),
            std::get<bool>(shard_spec_tuple));
    }

    std::optional<tt::tt_metal::MemoryConfig> to_memory_config(
        const std::string& tensor_memory_layout,
        const std::string& buffer_type,
        const std::optional<mlir_interface::shard_spec_tuple>& shard_spec) {

        auto deserialized_tensor_memory_layout = str_wrapper::to_tensor_memory_layout(tensor_memory_layout);
        if (!deserialized_tensor_memory_layout.has_value()) {
            return std::nullopt;
        }

        auto deserialized_buffer_type = str_wrapper::to_buffer_type(buffer_type);
        if (!deserialized_buffer_type.has_value()) {
            return std::nullopt;
        }

        auto deserialized_shard_spec = shard_spec.has_value() ? to_shard_spec(shard_spec.value()) : std::nullopt;
        return std::make_optional(tt::tt_metal::MemoryConfig{
            deserialized_tensor_memory_layout.value(),
            deserialized_buffer_type.value(),
            deserialized_shard_spec
        });
    }

    std::optional<tt::tt_metal::MemoryConfig> to_memory_config(const mlir_interface::memory_config_tuple& memory_config_tuple) {
        return to_memory_config(
            std::get<0>(memory_config_tuple),
            std::get<1>(memory_config_tuple),
            std::get<std::optional<mlir_interface::shard_spec_tuple>>(memory_config_tuple));
    }

    // MatmulMultiCoreReuseProgramConfig wrapper
    std::optional<ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig> to_program_config(
        const std::array<uint32_t, 2>& compute_with_storage_grid_size,
        const size_t in0_block_w,
        const size_t out_subblock_h,
        const size_t out_subblock_w,
        const size_t per_core_M,
        const size_t per_core_N) {
            return std::make_optional(ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig{
                CoreCoord{compute_with_storage_grid_size[0], compute_with_storage_grid_size[1]},
                in0_block_w,
                out_subblock_h,
                out_subblock_w,
                per_core_M,
                per_core_N
            });
    }

    std::optional<ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig> to_matmul_program_config(const mlir_interface::matmul_multicore_reuse_config_tuple& program_config_tuple) {
        return to_program_config(
            std::get<0>(program_config_tuple),
            std::get<1>(program_config_tuple),
            std::get<2>(program_config_tuple),
            std::get<3>(program_config_tuple),
            std::get<4>(program_config_tuple),
            std::get<5>(program_config_tuple));
    }

    // MatmulMultiCoreReuseMultiCastProgramConfig wrapper
    std::optional<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig> to_multicast_matmul_program_config(
        const std::array<uint32_t, 2>& compute_with_storage_grid_size,
        const size_t in0_block_w,
        const size_t out_subblock_h,
        const size_t out_subblock_w,
        const size_t per_core_M,
        const size_t per_core_N,
        const bool transpose_mcast,
        const bool fuse_batch) {

        return std::make_optional(ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
            CoreCoord{compute_with_storage_grid_size[0], compute_with_storage_grid_size[1]},
            in0_block_w,
            out_subblock_h,
            out_subblock_w,
            per_core_M,
            per_core_N,
            transpose_mcast,
            std::nullopt,   //  fused_activation
            fuse_batch
        });
    }

    std::optional<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig> to_multicast_matmul_program_config(const mlir_interface::matmul_multicore_reuse_config_tuple &program_config_tuple, bool transpose_mcast, bool fuse_batch)
    {
        return to_multicast_matmul_program_config(
            std::get<0>(program_config_tuple),
            std::get<1>(program_config_tuple),
            std::get<2>(program_config_tuple),
            std::get<3>(program_config_tuple),
            std::get<4>(program_config_tuple),
            std::get<5>(program_config_tuple),
            transpose_mcast,
            fuse_batch);
    }

    // MatmulMultiCoreReuseMultiCast1DProgramConfig wrapper
    std::optional<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig> to_multicast_1d_matmul_program_config(
        const std::array<uint32_t, 2>& compute_with_storage_grid_size,
        const size_t in0_block_w,
        const size_t out_subblock_h,
        const size_t out_subblock_w,
        const size_t per_core_M,
        const size_t per_core_N,
        const bool fuse_batch,
        const bool mcast_in0) {
            return std::make_optional(ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig{
                CoreCoord{compute_with_storage_grid_size[0], compute_with_storage_grid_size[1]},
                in0_block_w,
                out_subblock_h,
                out_subblock_w,
                per_core_M,
                per_core_N,
                fuse_batch,
                std::nullopt,   //  fused_activation
                mcast_in0
            });
        }
    std::optional<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig> to_multicast_1d_matmul_program_config(const mlir_interface::matmul_multicore_reuse_config_tuple &program_config_tuple, bool fuse_batch, bool mcast_in0)
    {
        return to_multicast_1d_matmul_program_config(
            std::get<0>(program_config_tuple),
            std::get<1>(program_config_tuple),
            std::get<2>(program_config_tuple),
            std::get<3>(program_config_tuple),
            std::get<4>(program_config_tuple),
            std::get<5>(program_config_tuple),
            fuse_batch,
            mcast_in0);
    }

} // namespace ttnn::tuple_wrapper
