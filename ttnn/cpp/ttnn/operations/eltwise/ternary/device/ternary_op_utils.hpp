// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ternary_device_operation.hpp"
#include "ttnn/tensor/types.hpp"

#include <map>
#include <optional>
#include <string>

namespace ttnn::operations::ternary {

enum class KernelName {
    ReaderNoBcastTTT,
    ReaderNoBcastTST,
    ReaderNoBcastTTS,
    ReaderColBcastTTT,
    ReaderColBcastTTS,
    ReaderColBcastTST,
    ReaderOuterBcastTTT,
    ReaderOuterBcastTTS,
    ReaderOuterBcastTST,
    ReaderScalarBcastTTS,
    ReaderScalarBcastTST,
    ReaderScalarBcastTTT,
    WriterNoBcast,
    ReaderRowBcastTTT,
    ReaderRowBcastTST,
    ReaderRowBcastTTS,
    WriterNoBcastTernary,
    WriterColBcastTTT,
    ComputeNoBcastTTT,      // TTT: no bcast, outer dim and row bcast cases
    ComputeBcastTTT,        // TTT : column and scalar bcast cases
    ComputeRowBcastTTT,     // TTT : row bcast cases : bfloat16 only
    ComputeBcastTTS_TST,    // TTS, TST: column and scalar bcast cases
    ComputeNoBcastTTS_TST,  // TTS, TST: no bcast, outer dim and row bcast cases
    // Shared by ADDCMUL and ADDCDIV (same kernel files, op-specific defines)
    ComputeNoBcastAddcOp,   // no bcast: ternary_addc_ops_sfpu.cpp
    ComputeBcastAddcOp,     // column/scalar bcast: ternary_addc_ops_sfpu_bcast.cpp
    ComputeRowBcastAddcOp,  // row bcast: ternary_addc_ops_fpu_rowbcast.cpp or ternary_addc_ops_sfpu.cpp
};

struct TernaryKernelConfig {
    TernaryKernelConfig(TernaryOpType op_type, TernaryVariant ternary_variant, TernaryBroadcastType broadcast_type);

    KernelName reader_kernel;
    KernelName compute_kernel;
    KernelName writer_kernel;
};

std::string get_kernel_file_path(KernelName kernel_name, bool is_fpu = false);

uint32_t pack_scalar_runtime_arg(float scalar, DataType dtype);

std::map<std::string, std::string> make_dataflow_defines(
    DataType dtype, DataType b_dtype, std::optional<DataType> c_dtype = std::nullopt);  // for binary & ternary variant

// Get compute kernel defines based on operation type
std::map<std::string, std::string> get_compute_defines(TernaryOpType op_type, DataType dtype);

// TTT variant (tensor-tensor-tensor)
TernaryBroadcastType get_broadcast_type(
    const ttnn::Shape& predicate_shape, const ttnn::Shape& value_true_shape, const ttnn::Shape& value_false_shape);

// 2-tensor broadcast compatibility (used by both TTS and TST)
TernaryBroadcastType get_broadcast_type(const ttnn::Shape& predicate_shape, const ttnn::Shape& tensor_shape);

// AllShardSpecs structure for TensorSpecs
struct AllShardSpecs {
    tt::tt_metal::ShardSpec predicate_shard_spec;
    tt::tt_metal::ShardSpec true_shard_spec;
    tt::tt_metal::ShardSpec false_shard_spec;
    tt::tt_metal::ShardSpec output_shard_spec;
};

// AllShardVolumes structure for TensorSpecs
struct AllShardVolumes {
    std::optional<std::uint32_t> predicate_shard_volume;
    std::optional<std::uint32_t> true_shard_volume;
    std::optional<std::uint32_t> false_shard_volume;
    std::optional<std::uint32_t> output_shard_volume;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "predicate_shard_volume", "true_shard_volume", "false_shard_volume", "output_shard_volume");
    auto attribute_values() const {
        return std::forward_as_tuple(
            predicate_shard_volume, true_shard_volume, false_shard_volume, output_shard_volume);
    }
};

tt::tt_metal::ShardSpec adjust_to_shape(
    const tt::tt_metal::ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape);

std::optional<AllShardVolumes> get_shard_volumes(
    const TensorSpec& predicate_spec,
    const std::optional<TensorSpec>& true_spec,
    const std::optional<TensorSpec>& false_spec,
    const TensorSpec& output);
}  // namespace ttnn::operations::ternary
