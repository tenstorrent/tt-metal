// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "where_device_operation.hpp"
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
    WriterColBcastTTT,
    ComputeNoBcastTTT,  // TTT: no bcast, outer dim and row bcast cases
    ComputeNoBcastTST,
    ComputeNoBcastTTS,
    ComputeBcastTTT,  // TTT : column and scalar bcast cases
    ComputeColBcastTTS,
    ComputeColBcastTST,
    ComputeScalarBcastTST,
    ComputeScalarBcastTTS,
};

struct WhereKernelConfig {
    WhereKernelConfig(WhereVariant where_variant, WhereBroadcastType broadcast_type);

    KernelName reader_kernel;
    KernelName compute_kernel;
    KernelName writer_kernel;
};

std::string get_kernel_file_path(KernelName kernel_name);

uint32_t pack_scalar_runtime_arg(float scalar, DataType dtype);

std::map<std::string, std::string> make_dataflow_defines(
    DataType dtype, DataType b_dtype, std::optional<DataType> c_dtype = std::nullopt);  // for binary & ternary variant

// TTT variant (tensor-tensor-tensor)
WhereBroadcastType get_broadcast_type(
    const ttnn::Shape& predicate_shape, const ttnn::Shape& value_true_shape, const ttnn::Shape& value_false_shape);

// 2-tensor broadcast compatibility (used by both TTS and TST)
WhereBroadcastType get_broadcast_type(const ttnn::Shape& predicate_shape, const ttnn::Shape& tensor_shape);

}  // namespace ttnn::operations::ternary
