// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "where_device_operation.hpp"
#include "ttnn/tensor/types.hpp"

#include <optional>
#include <string>

namespace ttnn::operations::ternary {

enum class KernelName {
    ReaderNoBcastTTT,
    ReaderNoBcastTST,
    ReaderNoBcastTTS,
    ReaderNoBcastTSS,
    ReaderColBcastTTT,  // New broadcast version for TTT
    WriterNoBcastTTT,
    WriterNoBcastTST,
    WriterNoBcastTTS,
    WriterNoBcastTSS,
    ComputeNoBcastTTT,
    ComputeNoBcastTST,
    ComputeNoBcastTTS,
    ComputeNoBcastTSS,
};

struct WhereKernelConfig {
    WhereKernelConfig(WhereVariant where_variant, WhereBroadcastType broadcast_type = WhereBroadcastType::NONE);

    KernelName reader_kernel;
    KernelName compute_kernel;
    KernelName writer_kernel;
};

std::string get_kernel_file_path(KernelName kernel_name);

uint32_t pack_scalar_runtime_arg(float scalar, DataType dtype);

// Create dataflow defines for ternary where operation
std::map<std::string, std::string> make_ternary_dataflow_defines(
    const DataType predicate_dtype, const DataType value_true_dtype, const DataType value_false_dtype);

// Forward declarations for broadcast utilities
using Tensor = tt::tt_metal::Tensor;

// Broadcast detection and validation utilities
struct WhereBroadcastInfo {
    WhereBroadcastType type = WhereBroadcastType::NONE;
    bool predicate_broadcast = false;
    bool value_true_broadcast = false;
    bool value_false_broadcast = false;
};

// Check if two tensors are broadcastable for general validation (permissive)
bool are_tensors_broadcastable_general(const Tensor& a, const Tensor& b);

// Check if two tensors are broadcastable for LLK path (restrictive - only column broadcast)
bool are_tensors_broadcastable_llk(const Tensor& a, const Tensor& b);

// Comprehensive broadcast type detection for WHERE TTT operation
WhereBroadcastInfo get_where_broadcast_info(
    const Tensor& predicate, const Tensor& value_true, const Tensor& value_false);

// Check if LLK can be used with the given broadcast pattern
bool can_use_llk_with_broadcast(const WhereBroadcastInfo& broadcast_info);

}  // namespace ttnn::operations::ternary
