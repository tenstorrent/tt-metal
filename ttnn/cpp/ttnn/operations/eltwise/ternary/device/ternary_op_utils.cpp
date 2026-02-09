// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ternary_op_utils.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt_stl/assert.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <unordered_map>

namespace ttnn::operations::ternary {

const std::optional<tt::tt_metal::ShardSpec>& get_shard_spec(const TensorSpec& tensor_spec) {
    return tensor_spec.memory_config().shard_spec();
}

bool is_uneven(const TensorSpec& t) {
    if (not t.memory_config().is_sharded()) {
        return false;
    }
    const auto& shape = t.padded_shape();
    const auto& shard = get_shard_spec(t)->shape;
    const auto rank = shape.rank();
    uint64_t volume_except_last = 1;
    for (int i = 0; i < static_cast<int>(rank) - 1; ++i) {
        volume_except_last *= shape[i];
    }
    return (volume_except_last % shard[0]) != 0 or (shape[-1] % shard[1]) != 0;
}

bool is_native_L1_sharding(
    const TensorSpec& predicate_spec,
    const std::optional<TensorSpec>& true_spec,
    const std::optional<TensorSpec>& false_spec,
    const tt::tt_metal::MemoryConfig& output_memory_config) {
    using namespace tt::tt_metal;
    if (!output_memory_config.is_sharded()) {
        return false;
    }
    // TTS or TST: only one of true_spec/false_spec present
    if (!true_spec.has_value() && false_spec.has_value() && predicate_spec.memory_config().is_sharded()) {
        return !is_uneven(predicate_spec) && !is_uneven(*false_spec) &&
               predicate_spec.logical_shape() == false_spec->logical_shape() &&
               predicate_spec.memory_config() == false_spec->memory_config() &&
               predicate_spec.memory_config().buffer_type() != BufferType::DRAM &&
               false_spec->memory_config().buffer_type() != BufferType::DRAM &&
               output_memory_config.buffer_type() != BufferType::DRAM &&
               (predicate_spec.memory_config().buffer_type() == BufferType::L1 ||
                false_spec->memory_config().buffer_type() == BufferType::L1 ||
                output_memory_config.buffer_type() == BufferType::L1);
    }
    if (true_spec.has_value() && !false_spec.has_value() && predicate_spec.memory_config().is_sharded()) {
        return !is_uneven(predicate_spec) && !is_uneven(*true_spec) &&
               predicate_spec.logical_shape() == true_spec->logical_shape() &&
               predicate_spec.memory_config() == true_spec->memory_config() &&
               predicate_spec.memory_config().buffer_type() != BufferType::DRAM &&
               true_spec->memory_config().buffer_type() != BufferType::DRAM &&
               output_memory_config.buffer_type() != BufferType::DRAM &&
               (predicate_spec.memory_config().buffer_type() == BufferType::L1 ||
                true_spec->memory_config().buffer_type() == BufferType::L1 ||
                output_memory_config.buffer_type() == BufferType::L1);
    }
    // TTT: all three specs present
    if (!true_spec.has_value() || !false_spec.has_value()) {
        return false;
    }
    if (predicate_spec.logical_shape() == true_spec->logical_shape() &&
        predicate_spec.logical_shape() == false_spec->logical_shape() &&
        predicate_spec.memory_config() == true_spec->memory_config() &&
        predicate_spec.memory_config() == false_spec->memory_config()) {
        if (is_uneven(predicate_spec) || is_uneven(*true_spec) || is_uneven(*false_spec)) {
            return false;
        }
        if (predicate_spec.memory_config().buffer_type() == BufferType::DRAM ||
            true_spec->memory_config().buffer_type() == BufferType::DRAM ||
            false_spec->memory_config().buffer_type() == BufferType::DRAM ||
            output_memory_config.buffer_type() == BufferType::DRAM) {
            return false;
        }
        if (output_memory_config.is_sharded() && output_memory_config.shard_spec().has_value()) {
            const auto& out_grid = output_memory_config.shard_spec()->grid;
            if (predicate_spec.memory_config().is_sharded() &&
                predicate_spec.memory_config().shard_spec().has_value() &&
                predicate_spec.memory_config().shard_spec()->grid != out_grid) {
                return false;
            }
            if (true_spec->memory_config().is_sharded() && true_spec->memory_config().shard_spec().has_value() &&
                true_spec->memory_config().shard_spec()->grid != out_grid) {
                return false;
            }
            if (false_spec->memory_config().is_sharded() && false_spec->memory_config().shard_spec().has_value() &&
                false_spec->memory_config().shard_spec()->grid != out_grid) {
                return false;
            }
        }
        if ((predicate_spec.memory_config().is_sharded() &&
             predicate_spec.memory_config().buffer_type() == BufferType::L1)) {
            return true;
        }
        if ((true_spec->memory_config().is_sharded() && true_spec->memory_config().buffer_type() == BufferType::L1)) {
            return true;
        }
        if ((false_spec->memory_config().is_sharded() && false_spec->memory_config().buffer_type() == BufferType::L1)) {
            return true;
        }
        if ((output_memory_config.is_sharded() && output_memory_config.buffer_type() == BufferType::L1)) {
            return true;
        }
    }
    return false;
}

// Composite key for kernel lookup
struct KernelLookupKey {
    TernaryOpType op_type;
    TernaryVariant variant;
    TernaryBroadcastType broadcast_type;

    bool operator==(const KernelLookupKey& other) const {
        return op_type == other.op_type && variant == other.variant && broadcast_type == other.broadcast_type;
    }
};

// Hash function for KernelLookupKey
struct KernelLookupKeyHash {
    std::size_t operator()(const KernelLookupKey& key) const {
        // TernaryOpType (0-N) << 8 | TernaryVariant (0-3) << 4 | TernaryBroadcastType (0-7)
        return (static_cast<size_t>(key.op_type) << 8) | (static_cast<size_t>(key.variant) << 4) |
               static_cast<size_t>(key.broadcast_type);
    }
};

// Kernel configuration entry
struct KernelConfigEntry {
    KernelName reader_kernel;
    KernelName compute_kernel;
    KernelName writer_kernel;
};

// Common kernel configuration map for all ternary operations
// WHERE and LERP will use the same kernel structure, just with different compute defines
static const std::unordered_map<KernelLookupKey, KernelConfigEntry, KernelLookupKeyHash> kernel_config_map = {
    // TTT configurations for WHERE
    {{TernaryOpType::WHERE, TernaryVariant::TTT, TernaryBroadcastType::COL_BCAST},
     {KernelName::ReaderColBcastTTT, KernelName::ComputeBcastTTT, KernelName::WriterNoBcastTernary}},
    {{TernaryOpType::WHERE, TernaryVariant::TTT, TernaryBroadcastType::OUTER_BCAST},
     {KernelName::ReaderOuterBcastTTT, KernelName::ComputeNoBcastTTT, KernelName::WriterNoBcastTernary}},
    {{TernaryOpType::WHERE, TernaryVariant::TTT, TernaryBroadcastType::ROW_BCAST},
     {KernelName::ReaderRowBcastTTT, KernelName::ComputeNoBcastTTT, KernelName::WriterNoBcastTernary}},
    {{TernaryOpType::WHERE, TernaryVariant::TTT, TernaryBroadcastType::SCALAR_BCAST},
     {KernelName::ReaderScalarBcastTTT, KernelName::ComputeBcastTTT, KernelName::WriterNoBcastTernary}},
    {{TernaryOpType::WHERE, TernaryVariant::TTT, TernaryBroadcastType::NONE},
     {KernelName::ReaderNoBcastTTT, KernelName::ComputeNoBcastTTT, KernelName::WriterNoBcastTernary}},

    // TTS configurations for WHERE
    {{TernaryOpType::WHERE, TernaryVariant::TTS, TernaryBroadcastType::COL_BCAST},
     {KernelName::ReaderColBcastTTS, KernelName::ComputeBcastTTS_TST, KernelName::WriterNoBcast}},
    {{TernaryOpType::WHERE, TernaryVariant::TTS, TernaryBroadcastType::ROW_BCAST},
     {KernelName::ReaderRowBcastTTS, KernelName::ComputeNoBcastTTS_TST, KernelName::WriterNoBcast}},
    {{TernaryOpType::WHERE, TernaryVariant::TTS, TernaryBroadcastType::OUTER_BCAST},
     {KernelName::ReaderOuterBcastTTS, KernelName::ComputeNoBcastTTS_TST, KernelName::WriterNoBcast}},
    {{TernaryOpType::WHERE, TernaryVariant::TTS, TernaryBroadcastType::SCALAR_A_BCAST},
     {KernelName::ReaderScalarBcastTTS, KernelName::ComputeBcastTTS_TST, KernelName::WriterNoBcast}},
    {{TernaryOpType::WHERE, TernaryVariant::TTS, TernaryBroadcastType::SCALAR_B_BCAST},
     {KernelName::ReaderScalarBcastTTS, KernelName::ComputeBcastTTS_TST, KernelName::WriterNoBcast}},
    {{TernaryOpType::WHERE, TernaryVariant::TTS, TernaryBroadcastType::NONE},
     {KernelName::ReaderNoBcastTTS, KernelName::ComputeNoBcastTTS_TST, KernelName::WriterNoBcast}},

    // TST configurations for WHERE
    {{TernaryOpType::WHERE, TernaryVariant::TST, TernaryBroadcastType::COL_BCAST},
     {KernelName::ReaderColBcastTST, KernelName::ComputeBcastTTS_TST, KernelName::WriterNoBcast}},
    {{TernaryOpType::WHERE, TernaryVariant::TST, TernaryBroadcastType::ROW_BCAST},
     {KernelName::ReaderRowBcastTST, KernelName::ComputeNoBcastTTS_TST, KernelName::WriterNoBcast}},
    {{TernaryOpType::WHERE, TernaryVariant::TST, TernaryBroadcastType::OUTER_BCAST},
     {KernelName::ReaderOuterBcastTST, KernelName::ComputeNoBcastTTS_TST, KernelName::WriterNoBcast}},
    {{TernaryOpType::WHERE, TernaryVariant::TST, TernaryBroadcastType::SCALAR_A_BCAST},
     {KernelName::ReaderScalarBcastTST, KernelName::ComputeBcastTTS_TST, KernelName::WriterNoBcast}},
    {{TernaryOpType::WHERE, TernaryVariant::TST, TernaryBroadcastType::SCALAR_B_BCAST},
     {KernelName::ReaderScalarBcastTST, KernelName::ComputeBcastTTS_TST, KernelName::WriterNoBcast}},
    {{TernaryOpType::WHERE, TernaryVariant::TST, TernaryBroadcastType::NONE},
     {KernelName::ReaderNoBcastTST, KernelName::ComputeNoBcastTTS_TST, KernelName::WriterNoBcast}},

    // TTT configurations for LERP - same kernels, different compute operation
    {{TernaryOpType::LERP, TernaryVariant::TTT, TernaryBroadcastType::COL_BCAST},
     {KernelName::ReaderColBcastTTT, KernelName::ComputeBcastTTT, KernelName::WriterNoBcastTernary}},
    {{TernaryOpType::LERP, TernaryVariant::TTT, TernaryBroadcastType::OUTER_BCAST},
     {KernelName::ReaderOuterBcastTTT, KernelName::ComputeNoBcastTTT, KernelName::WriterNoBcastTernary}},
    {{TernaryOpType::LERP, TernaryVariant::TTT, TernaryBroadcastType::ROW_BCAST},
     {KernelName::ReaderRowBcastTTT, KernelName::ComputeNoBcastTTT, KernelName::WriterNoBcastTernary}},
    {{TernaryOpType::LERP, TernaryVariant::TTT, TernaryBroadcastType::SCALAR_BCAST},
     {KernelName::ReaderScalarBcastTTT, KernelName::ComputeBcastTTT, KernelName::WriterNoBcastTernary}},
    {{TernaryOpType::LERP, TernaryVariant::TTT, TernaryBroadcastType::NONE},
     {KernelName::ReaderNoBcastTTT, KernelName::ComputeNoBcastTTT, KernelName::WriterNoBcastTernary}},

    // TTT configurations for ADDCMUL (shared addc_ops kernels)
    {{TernaryOpType::ADDCMUL, TernaryVariant::TTT, TernaryBroadcastType::NONE},
     {KernelName::ReaderNoBcastTTT, KernelName::ComputeNoBcastAddcOp, KernelName::WriterNoBcastTernary}},
    {{TernaryOpType::ADDCMUL, TernaryVariant::TTT, TernaryBroadcastType::OUTER_BCAST},
     {KernelName::ReaderOuterBcastTTT, KernelName::ComputeNoBcastAddcOp, KernelName::WriterNoBcastTernary}},
    {{TernaryOpType::ADDCMUL, TernaryVariant::TTT, TernaryBroadcastType::ROW_BCAST},
     {KernelName::ReaderRowBcastTTT, KernelName::ComputeNoBcastAddcOp, KernelName::WriterNoBcastTernary}},
    {{TernaryOpType::ADDCMUL, TernaryVariant::TTT, TernaryBroadcastType::SCALAR_BCAST},
     {KernelName::ReaderScalarBcastTTT, KernelName::ComputeBcastAddcOp, KernelName::WriterNoBcastTernary}},
    {{TernaryOpType::ADDCMUL, TernaryVariant::TTT, TernaryBroadcastType::COL_BCAST},
     {KernelName::ReaderColBcastTTT, KernelName::ComputeBcastAddcOp, KernelName::WriterNoBcastTernary}},

    // TTT configurations for ADDCDIV (shared addc_ops kernels)
    {{TernaryOpType::ADDCDIV, TernaryVariant::TTT, TernaryBroadcastType::NONE},
     {KernelName::ReaderNoBcastTTT, KernelName::ComputeNoBcastAddcOp, KernelName::WriterNoBcastTernary}},
    {{TernaryOpType::ADDCDIV, TernaryVariant::TTT, TernaryBroadcastType::OUTER_BCAST},
     {KernelName::ReaderOuterBcastTTT, KernelName::ComputeNoBcastAddcOp, KernelName::WriterNoBcastTernary}},
    {{TernaryOpType::ADDCDIV, TernaryVariant::TTT, TernaryBroadcastType::ROW_BCAST},
     {KernelName::ReaderRowBcastTTT, KernelName::ComputeNoBcastAddcOp, KernelName::WriterNoBcastTernary}},
    {{TernaryOpType::ADDCDIV, TernaryVariant::TTT, TernaryBroadcastType::SCALAR_BCAST},
     {KernelName::ReaderScalarBcastTTT, KernelName::ComputeBcastAddcOp, KernelName::WriterNoBcastTernary}},
    {{TernaryOpType::ADDCDIV, TernaryVariant::TTT, TernaryBroadcastType::COL_BCAST},
     {KernelName::ReaderColBcastTTT, KernelName::ComputeBcastAddcOp, KernelName::WriterNoBcastTernary}},

    // TTS configurations for LERP
    {{TernaryOpType::LERP, TernaryVariant::TTS, TernaryBroadcastType::COL_BCAST},
     {KernelName::ReaderColBcastTTS, KernelName::ComputeBcastTTS_TST, KernelName::WriterNoBcast}},
    {{TernaryOpType::LERP, TernaryVariant::TTS, TernaryBroadcastType::ROW_BCAST},
     {KernelName::ReaderRowBcastTTS, KernelName::ComputeNoBcastTTS_TST, KernelName::WriterNoBcast}},
    {{TernaryOpType::LERP, TernaryVariant::TTS, TernaryBroadcastType::OUTER_BCAST},
     {KernelName::ReaderOuterBcastTTS, KernelName::ComputeNoBcastTTS_TST, KernelName::WriterNoBcast}},
    {{TernaryOpType::LERP, TernaryVariant::TTS, TernaryBroadcastType::SCALAR_A_BCAST},
     {KernelName::ReaderScalarBcastTTS, KernelName::ComputeBcastTTS_TST, KernelName::WriterNoBcast}},
    {{TernaryOpType::LERP, TernaryVariant::TTS, TernaryBroadcastType::SCALAR_B_BCAST},
     {KernelName::ReaderScalarBcastTTS, KernelName::ComputeBcastTTS_TST, KernelName::WriterNoBcast}},
    {{TernaryOpType::LERP, TernaryVariant::TTS, TernaryBroadcastType::NONE},
     {KernelName::ReaderNoBcastTTS, KernelName::ComputeNoBcastTTS_TST, KernelName::WriterNoBcast}},
};

TernaryKernelConfig::TernaryKernelConfig(
    TernaryOpType op_type, TernaryVariant ternary_variant, TernaryBroadcastType broadcast_type) {
    // Check for unsupported TSS variant
    if (ternary_variant == TernaryVariant::TSS) {
        TT_FATAL(false, "TSS variant is yet to be moved into Ternary Device Operation");
    }

    // Find matching configuration using O(1) hash map lookup
    KernelLookupKey key{op_type, ternary_variant, broadcast_type};
    auto it = kernel_config_map.find(key);
    if (it != kernel_config_map.end()) {
        reader_kernel = it->second.reader_kernel;
        compute_kernel = it->second.compute_kernel;
        writer_kernel = it->second.writer_kernel;
        return;
    }

    TT_FATAL(false, "Invalid ternary operation type, variant or broadcast type combination");
}

std::string get_kernel_file_path(KernelName kernel_name, bool is_fpu) {
    constexpr std::string_view root = "ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels";
    constexpr std::string_view dataflow = "{}/dataflow/{}";
    constexpr std::string_view compute = "{}/compute/{}";
    switch (kernel_name) {
        case KernelName::ReaderNoBcastTTT:
        case KernelName::ReaderOuterBcastTTT:
            return fmt::format(dataflow, root, "ternary_reader_nosubtilebcast_ttt.cpp");
        case KernelName::ReaderNoBcastTST:
        case KernelName::ReaderNoBcastTTS: return fmt::format(dataflow, root, "ternary_reader_nobcast_tst_tts.cpp");
        case KernelName::ReaderOuterBcastTTS:
        case KernelName::ReaderOuterBcastTST: return fmt::format(dataflow, root, "tst_tts_reader_outer_bcast.cpp");
        case KernelName::ReaderScalarBcastTTS:
        case KernelName::ReaderScalarBcastTST: return fmt::format(dataflow, root, "tst_tts_reader_scalar_bcast.cpp");
        case KernelName::ReaderScalarBcastTTT: return fmt::format(dataflow, root, "ternary_reader_scalar_ttt.cpp");
        case KernelName::ReaderColBcastTTT: return fmt::format(dataflow, root, "ternary_reader_colbcast_ttt.cpp");
        case KernelName::ReaderColBcastTTS:
        case KernelName::ReaderColBcastTST: return fmt::format(dataflow, root, "tts_tst_reader_col_bcast.cpp");
        case KernelName::ReaderRowBcastTTT: return fmt::format(dataflow, root, "ternary_reader_rowbcast_ttt.cpp");
        case KernelName::ReaderRowBcastTST:
        case KernelName::ReaderRowBcastTTS: return fmt::format(dataflow, root, "tts_tst_reader_row_bcast.cpp");
        case KernelName::WriterNoBcastTernary:
        case KernelName::WriterNoBcast:
        case KernelName::WriterColBcastTTT: return fmt::format(dataflow, root, "ternary_writer_nobcast.cpp");
        case KernelName::ComputeNoBcastTTT: return fmt::format(compute, root, "ternary_sfpu_no_bcast_ttt.cpp");
        case KernelName::ComputeBcastTTT: return fmt::format(compute, root, "ternary_sfpu_col_scalar_bcast_ttt.cpp");
        case KernelName::ComputeRowBcastTTT: return fmt::format(compute, root, "ternary_sfpu_row_bcast_ttt.cpp");
        case KernelName::ComputeBcastTTS_TST:
            return fmt::format(compute, root, "ternary_sfpu_col_scalar_bcast_tts_tst.cpp");
        case KernelName::ComputeNoBcastTTS_TST: return fmt::format(compute, root, "ternary_sfpu_no_bcast_tts_tst.cpp");
        case KernelName::ComputeNoBcastAddcOp: return fmt::format(compute, root, "ternary_addc_ops_sfpu.cpp");
        case KernelName::ComputeBcastAddcOp: return fmt::format(compute, root, "ternary_addc_ops_sfpu_bcast.cpp");
        case KernelName::ComputeRowBcastAddcOp:
            return fmt::format(
                compute, root, is_fpu ? "ternary_addc_ops_fpu_rowbcast.cpp" : "ternary_addc_ops_sfpu.cpp");
        default: __builtin_unreachable();
    }
}

uint32_t pack_scalar_runtime_arg(const float scalar, const DataType dtype) {
    if (dtype == DataType::INT32) {
        return std::bit_cast<uint32_t>(static_cast<int32_t>(scalar));
    }
    return std::bit_cast<uint32_t>(scalar);
}

std::map<std::string, std::string> make_dataflow_defines(
    const DataType dtype, const DataType b_dtype, std::optional<DataType> c_dtype) {
    std::map<std::string, std::string> defines;
    // Exact copy of binary_ng make_dataflow_defines for compatibility
    if (dtype == DataType::FLOAT32) {
        defines["FILL_TILE_WITH_FIRST_COLUMN"] = "fill_tile_with_first_column";
        defines["FILL_TILE_WITH_FIRST_ROW"] = "fill_tile_with_first_row";
        defines["FILL_TILE_WITH_FIRST_ELEMENT"] = "fill_tile_with_first_element<float>";
        defines["FILL_WITH_VALUE_FLOAT"] = "fill_with_val<1024, float>";
    } else if (dtype == DataType::INT32) {
        defines["FILL_TILE_WITH_FIRST_COLUMN"] = "fill_tile_with_first_column";
        defines["FILL_TILE_WITH_FIRST_ROW"] = "fill_tile_with_first_row";
        defines["FILL_TILE_WITH_FIRST_ELEMENT"] = "fill_tile_with_first_element<int32_t>";
        defines["FILL_WITH_VALUE"] = "fill_with_val<1024, int32_t>";
    } else if (dtype == DataType::UINT32) {
        defines["FILL_TILE_WITH_FIRST_COLUMN"] = "fill_tile_with_first_column";
        defines["FILL_TILE_WITH_FIRST_ROW"] = "fill_tile_with_first_row";
        defines["FILL_TILE_WITH_FIRST_ELEMENT"] = "fill_tile_with_first_element<uint32_t>";
        defines["FILL_WITH_VALUE"] = "fill_with_val<1024, uint32_t>";
    } else {
        defines["FILL_TILE_WITH_FIRST_COLUMN"] = "fill_tile_with_first_column_bfloat16";
        defines["FILL_TILE_WITH_FIRST_ROW"] = "fill_tile_with_first_row_bfloat16";
        defines["FILL_TILE_WITH_FIRST_ELEMENT"] = "fill_tile_with_first_element_bfloat16";
        defines["FILL_WITH_VALUE"] = "fill_with_val_bfloat16";
    }

    // Add defines for second tensor (true tensor)
    if (b_dtype == DataType::FLOAT32) {
        defines["FILL_TILE_WITH_FIRST_COLUMN_B"] = "fill_tile_with_first_column";
        defines["FILL_TILE_WITH_FIRST_ROW_B"] = "fill_tile_with_first_row";
        defines["FILL_TILE_WITH_FIRST_ELEMENT_B"] = "fill_tile_with_first_element<float>";
        defines["FILL_WITH_VALUE_FLOAT_B"] = "fill_with_val<1024, float>";
    } else if (b_dtype == DataType::INT32) {
        defines["FILL_TILE_WITH_FIRST_COLUMN_B"] = "fill_tile_with_first_column";
        defines["FILL_TILE_WITH_FIRST_ROW_B"] = "fill_tile_with_first_row";
        defines["FILL_TILE_WITH_FIRST_ELEMENT_B"] = "fill_tile_with_first_element<int32_t>";
        defines["FILL_WITH_VALUE_B"] = "fill_with_val<1024, int32_t>";
    } else if (b_dtype == DataType::UINT32) {
        defines["FILL_TILE_WITH_FIRST_COLUMN_B"] = "fill_tile_with_first_column";
        defines["FILL_TILE_WITH_FIRST_ROW_B"] = "fill_tile_with_first_row";
        defines["FILL_TILE_WITH_FIRST_ELEMENT_B"] = "fill_tile_with_first_element<uint32_t>";
        defines["FILL_WITH_VALUE_B"] = "fill_with_val<1024, uint32_t>";
    } else {
        defines["FILL_TILE_WITH_FIRST_COLUMN_B"] = "fill_tile_with_first_column_bfloat16";
        defines["FILL_TILE_WITH_FIRST_ROW_B"] = "fill_tile_with_first_row_bfloat16";
        defines["FILL_TILE_WITH_FIRST_ELEMENT_B"] = "fill_tile_with_first_element_bfloat16";
        defines["FILL_WITH_VALUE_B"] = "fill_with_val_bfloat16";
    }
    // For TTT variant, Add defines for third tensor (false tensor)
    if (c_dtype.has_value()) {
        if (c_dtype == DataType::FLOAT32) {
            defines["FILL_TILE_WITH_FIRST_COLUMN_C"] = "fill_tile_with_first_column";
            defines["FILL_TILE_WITH_FIRST_ROW_C"] = "fill_tile_with_first_row";
            defines["FILL_TILE_WITH_FIRST_ELEMENT_C"] = "fill_tile_with_first_element<float>";
            defines["FILL_WITH_VALUE_FLOAT_C"] = "fill_with_val<1024, float>";
        } else if (c_dtype == DataType::INT32) {
            defines["FILL_TILE_WITH_FIRST_COLUMN_C"] = "fill_tile_with_first_column";
            defines["FILL_TILE_WITH_FIRST_ROW_C"] = "fill_tile_with_first_row";
            defines["FILL_TILE_WITH_FIRST_ELEMENT_C"] = "fill_tile_with_first_element<int32_t>";
            defines["FILL_WITH_VALUE_C"] = "fill_with_val<1024, int32_t>";
        } else if (c_dtype == DataType::UINT32) {
            defines["FILL_TILE_WITH_FIRST_COLUMN_C"] = "fill_tile_with_first_column";
            defines["FILL_TILE_WITH_FIRST_ROW_C"] = "fill_tile_with_first_row";
            defines["FILL_TILE_WITH_FIRST_ELEMENT_C"] = "fill_tile_with_first_element<uint32_t>";
            defines["FILL_WITH_VALUE_C"] = "fill_with_val<1024, uint32_t>";
        } else {
            defines["FILL_TILE_WITH_FIRST_COLUMN_C"] = "fill_tile_with_first_column_bfloat16";
            defines["FILL_TILE_WITH_FIRST_ROW_C"] = "fill_tile_with_first_row_bfloat16";
            defines["FILL_TILE_WITH_FIRST_ELEMENT_C"] = "fill_tile_with_first_element_bfloat16";
            defines["FILL_WITH_VALUE_C"] = "fill_with_val_bfloat16";
        }
    }

    return defines;
}

std::map<std::string, std::string> get_compute_defines(TernaryOpType op_type, DataType dtype) {
    std::map<std::string, std::string> defines;

    switch (op_type) {
        case TernaryOpType::WHERE:
            defines["TERNARY_SFPU_OP_INIT"] = "where_tile_init";
            if (dtype == DataType::FLOAT32) {
                defines["TERNARY_SFPU_OP_FUNC"] = "where_tile<DataFormat::Float32>";
            } else if (dtype == DataType::INT32) {
                defines["TERNARY_SFPU_OP_FUNC"] = "where_tile<DataFormat::Int32>";
            } else {
                defines["TERNARY_SFPU_OP_FUNC"] = "where_tile<DataFormat::Float16_b>";
            }
            break;
        case TernaryOpType::LERP:
            defines["TERNARY_SFPU_OP_INIT"] = "lerp_tile_init";
            defines["TERNARY_SFPU_OP_FUNC"] =
                (dtype == DataType::FLOAT32) ? "lerp_tile<DataFormat::Float32>" : "lerp_tile<DataFormat::Float16_b>";
            break;
        case TernaryOpType::ADDCMUL:
            defines["TERNARY_SFPU_OP_INIT"] = "addcmul_tile_init";
            defines["TERNARY_SFPU_OP_FUNC"] = (dtype == DataType::FLOAT32) ? "addcmul_tile<DataFormat::Float32>"
                                                                           : "addcmul_tile<DataFormat::Float16_b>";
            break;
        case TernaryOpType::ADDCDIV:
            defines["TERNARY_SFPU_OP_INIT"] = "addcdiv_tile_init";
            defines["TERNARY_SFPU_OP_FUNC"] = (dtype == DataType::FLOAT32) ? "addcdiv_tile<DataFormat::Float32>"
                                                                           : "addcdiv_tile<DataFormat::Float16_b>";
            break;
        default: TT_FATAL(false, "Unsupported ternary operation type");
    }

    return defines;
}

// 2-tensor broadcast compatibility (used by both TTS and TST variants)
// For TTS: checks predicate vs true tensor (false is scalar)
// For TST: checks predicate vs false tensor (true is scalar)
TernaryBroadcastType get_broadcast_type(const ttnn::Shape& predicate_shape, const ttnn::Shape& tensor_shape) {
    // Check for exact match
    if (predicate_shape == tensor_shape) {
        return TernaryBroadcastType::NONE;
    }

    bool same_width = (predicate_shape[-1] == tensor_shape[-1]);
    bool same_height = (predicate_shape[-2] == tensor_shape[-2]);

    log_debug(
        tt::LogOp,
        "2-tensor broadcast detection - predicate shape: {}, tensor shape: {}",
        predicate_shape,
        tensor_shape);
    log_debug(tt::LogOp, "same_width: {}, same_height: {}", same_width, same_height);

    // Check for outer broadcast first: if last two dimensions match exactly,
    // it's outer broadcast (broadcasting in dimensions beyond -2)
    if (same_height && same_width) {
        log_debug(tt::LogOp, "Detected OUTER_BCAST for 2-tensor case");
        return TernaryBroadcastType::OUTER_BCAST;
    }

    // Multi-dimensional ROW and COL broadcast is not supported for now
    if (!same_height && !same_width) {
        // Get last dimension sizes
        auto pred_w = predicate_shape[-1];
        auto b_w = tensor_shape[-1];

        auto pred_h = predicate_shape[-2];  // height (second-to-last dimension)
        auto b_h = tensor_shape[-2];

        auto max_h = std::max({pred_h, b_h});
        auto max_w = std::max({pred_w, b_w});
        bool pred_row_bcast = (pred_h == 1 && max_h > 1);
        bool b_row_bcast = (b_h == 1 && max_h > 1);
        bool pred_col_bcast = (pred_w == 1 && max_w > 1);
        bool b_col_bcast = (b_w == 1 && max_w > 1);

        if (b_row_bcast && b_col_bcast) {
            return TernaryBroadcastType::SCALAR_B_BCAST;
        }

        if (pred_row_bcast && pred_col_bcast) {
            return TernaryBroadcastType::SCALAR_A_BCAST;
        }

        return TernaryBroadcastType::INVALID_BCAST;
    }

    // Get dimension sizes
    auto pred_w = predicate_shape[-1];
    auto tensor_w = tensor_shape[-1];
    auto pred_h = predicate_shape[-2];
    auto tensor_h = tensor_shape[-2];

    if (!same_width) {
        // Check for column broadcast patterns (width dimension differs, heights same)
        auto max_w = std::max(pred_w, tensor_w);
        bool pred_col_broadcasted = (pred_w == 1 && max_w > 1);
        bool tensor_col_broadcasted = (tensor_w == 1 && max_w > 1);

        // Column broadcast case: at least one tensor is broadcasting in width and heights are same
        if ((pred_col_broadcasted || tensor_col_broadcasted) && same_height) {
            // TTS and TST support column broadcast
            log_debug(tt::LogOp, "2-tensor case detected column broadcast");
            return TernaryBroadcastType::COL_BCAST;
        }
    }

    // Check for row broadcast patterns (height dimension differs)
    if (!same_height) {
        // Check for row broadcast patterns: at least one tensor is broadcasting in height and widths are same
        auto max_h = std::max(pred_h, tensor_h);
        bool pred_row_broadcasted = (pred_h == 1 && max_h > 1);
        bool tensor_row_broadcasted = (tensor_h == 1 && max_h > 1);

        log_debug(
            tt::LogOp,
            "2-tensor row broadcast check: pred_h={}, tensor_h={}, same_width={}, pred_row_broadcasted={}, "
            "tensor_row_broadcasted={}",
            pred_h,
            tensor_h,
            same_width,
            pred_row_broadcasted,
            tensor_row_broadcasted);

        // Row broadcast case: at least one tensor is broadcasting in height and widths are same
        if ((pred_row_broadcasted || tensor_row_broadcasted) && same_width) {
            // TTS and TST now support row broadcast
            log_debug(tt::LogOp, "2-tensor row broadcast detected for TTS/TST");
            return TernaryBroadcastType::ROW_BCAST;
        }
    }

    // If we reach here, no valid broadcast pattern was found
    return TernaryBroadcastType::INVALID_BCAST;
}

TernaryBroadcastType get_broadcast_type(
    const ttnn::Shape& predicate_shape, const ttnn::Shape& true_shape, const ttnn::Shape& false_shape) {
    // Check for column broadcast pattern:
    // Examples: (1,1,32,32), (1,1,32,1), (1,1,32,32) or (1,1,32,1), (1,1,32,1), (1,1,32,32)
    // Column broadcast means one or more tensors have last dimension = 1 while at least one has full width
    if ((predicate_shape == true_shape) && (predicate_shape == false_shape)) {
        return TernaryBroadcastType::NONE;
    }

    bool same_width = (predicate_shape[-1] == true_shape[-1]) && (predicate_shape[-1] == false_shape[-1]);
    bool same_height = (predicate_shape[-2] == true_shape[-2]) && (predicate_shape[-2] == false_shape[-2]);

    // Check for outer broadcast: same height and width
    if (same_height && same_width) {
        return TernaryBroadcastType::OUTER_BCAST;
    }

    // Multi-dimensional mixed ROW and COL broadcast (i.e., cases where both height and width require broadcasting in
    // different tensors)
    if (!same_height && !same_width) {
        // Get last dimension sizes
        auto pred_w = predicate_shape[-1];
        auto true_w = true_shape[-1];
        auto false_w = false_shape[-1];

        auto pred_h = predicate_shape[-2];
        auto true_h = true_shape[-2];
        auto false_h = false_shape[-2];

        auto max_h = std::max({pred_h, true_h, false_h});
        auto max_w = std::max({pred_w, true_w, false_w});

        bool pred_row_bcast = (pred_h == 1 && max_h > 1);
        bool true_row_bcast = (true_h == 1 && max_h > 1);
        bool false_row_bcast = (false_h == 1 && max_h > 1);

        bool pred_col_bcast = (pred_w == 1 && max_w > 1);
        bool true_col_bcast = (true_w == 1 && max_w > 1);
        bool false_col_bcast = (false_w == 1 && max_w > 1);

        bool is_pred_scalar = (pred_row_bcast && pred_col_bcast);
        bool is_true_scalar = (true_row_bcast && true_col_bcast);
        bool is_false_scalar = (false_row_bcast && false_col_bcast);

        bool is_pred_non_bcast = (pred_h == max_h && pred_w == max_w);
        bool is_true_non_bcast = (true_h == max_h && true_w == max_w);
        bool is_false_non_bcast = (false_h == max_h && false_w == max_w);

        if ((is_pred_scalar || is_pred_non_bcast) && (is_true_scalar || is_true_non_bcast) &&
            (is_false_scalar || is_false_non_bcast)) {
            return TernaryBroadcastType::SCALAR_BCAST;
        }

        return TernaryBroadcastType::INVALID_BCAST;
    }

    // Get last dimension sizes
    auto pred_w = predicate_shape[-1];
    auto true_w = true_shape[-1];
    auto false_w = false_shape[-1];

    auto pred_h = predicate_shape[-2];  // height (second-to-last dimension)
    auto true_h = true_shape[-2];
    auto false_h = false_shape[-2];

    if (!same_height) {
        // Check for row broadcast patterns first (height dimension differs)
        auto max_h = std::max({pred_h, true_h, false_h});
        bool pred_row_broadcasted = (pred_h == 1 && max_h > 1);
        bool true_row_broadcasted = (true_h == 1 && max_h > 1);
        bool false_row_broadcasted = (false_h == 1 && max_h > 1);

        // Row broadcast case: at least one tensor is broadcasting in height and widths are same
        if ((pred_row_broadcasted || true_row_broadcasted || false_row_broadcasted) &&
            (pred_w == true_w && pred_w == false_w)) {
            return TernaryBroadcastType::ROW_BCAST;
        }
    }

    if (!same_width) {
        // Check for column broadcast patterns (width dimension differs, heights same)
        auto max_w = std::max({pred_w, true_w, false_w});
        bool pred_col_broadcasted = (pred_w == 1 && max_w > 1);
        bool true_col_broadcasted = (true_w == 1 && max_w > 1);
        bool false_col_broadcasted = (false_w == 1 && max_w > 1);

        // Column broadcast case: at least one tensor is broadcasting in width and heights are same
        if ((pred_col_broadcasted || true_col_broadcasted || false_col_broadcasted) &&
            (pred_h == true_h && pred_h == false_h)) {
            return TernaryBroadcastType::COL_BCAST;
        }
    }

    return TernaryBroadcastType::INVALID_BCAST;
}

tt::tt_metal::ShardSpec adjust_to_shape(
    const tt::tt_metal::ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape) {
    auto ret = shard_spec;

    // Calculate volume of all dimensions EXCEPT the last (width)
    // This is the "collapsed height" for sharding purposes
    uint32_t from_volume_except_width = 1;
    uint32_t to_volume_except_width = 1;

    const int rank = std::max(from_shape.rank(), to_shape.rank());

    // Accumulate all dimensions except the last
    for (int i = 0; i < rank - 1; ++i) {
        uint32_t from_dim = (i < from_shape.rank()) ? from_shape[i] : 1;
        uint32_t to_dim = (i < to_shape.rank()) ? to_shape[i] : 1;
        from_volume_except_width *= from_dim;
        to_volume_except_width *= to_dim;
    }

    // Get width dimensions
    uint32_t from_width = from_shape[-1];
    uint32_t to_width = to_shape[-1];

    // Adjust shard shape based on full volume ratios
    TT_FATAL(from_volume_except_width > 0, "Invalid from_shape: volume is zero");
    TT_FATAL(from_width > 0, "Invalid from_shape: width dimension is zero");
    ret.shape[0] = std::max((ret.shape[0] * to_volume_except_width) / from_volume_except_width, 32u);
    ret.shape[1] = std::max((ret.shape[1] * to_width) / from_width, 32u);
    return ret;
}

tt::tt_metal::MemoryConfig compute_mem_config_actual(
    const Tensor& input_tensor, const ttnn::Shape& output_logical_shape) {
    const auto& padded_a_shape = input_tensor.padded_shape();
    const auto& padded_out_shape =
        input_tensor.tensor_spec().tensor_layout().compute_padded_shape(output_logical_shape);
    auto adjusted_shard_spec =
        adjust_to_shape(*input_tensor.memory_config().shard_spec(), padded_a_shape, padded_out_shape);
    return tt::tt_metal::MemoryConfig(
        input_tensor.memory_config().memory_layout(), input_tensor.memory_config().buffer_type(), adjusted_shard_spec);
}

}  // namespace ttnn::operations::ternary
