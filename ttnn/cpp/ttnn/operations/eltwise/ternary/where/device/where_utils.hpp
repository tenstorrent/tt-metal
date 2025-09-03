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
    ReaderNoBcastTSS,
    ReaderColBcastTTT,
    ReaderOuterBcastTTT,
    ReaderOuterBcastTTS,
    ReaderOuterBcastTST,
    WriterNoBcast,
    ReaderRowBcastTTT,
    WriterColBcastTTT,
    ComputeNoBcastTTT,
    ComputeNoBcastTST,
    ComputeNoBcastTTS,
    ComputeNoBcastTSS,
    ComputeColBcastTTT,
};

struct WhereKernelConfig {
    WhereKernelConfig(WhereVariant where_variant, WhereBroadcastType broadcast_type);

    KernelName reader_kernel;
    KernelName compute_kernel;
    KernelName writer_kernel;
};

std::string get_kernel_file_path(KernelName kernel_name);

uint32_t pack_scalar_runtime_arg(float scalar, DataType dtype);

std::map<std::string, std::string> make_dataflow_defines(DataType dtype, DataType b_dtype);  // for binary variant
std::map<std::string, std::string> make_dataflow_defines(
    DataType dtype, DataType b_dtype, DataType c_dtype);  // for ternary variant

WhereBroadcastType get_broadcast_type(
    const ttnn::Shape& predicate_shape, const ttnn::Shape& value_true_shape, const ttnn::Shape& value_false_shape);
WhereBroadcastType get_broadcast_type(const ttnn::Shape& predicate_shape, const ttnn::Shape& b_shape);

}  // namespace ttnn::operations::ternary
