// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "binary_ng_device_operation.hpp"
#include "ttnn/operations/eltwise/binary_ng/types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

#include <optional>
#include <string>

namespace ttnn::operations::binary_ng {

enum class KernelName {
    ReaderNoBcast,
    ReaderRowBcast,
    ReaderColBcast,
    ReaderScalarBcast,
    WriterNoBcast,
    WriterRowBcast,
    WriterColBcast,
    WriterScalarBcast,
    WriterScalar,
    ComputeNoBcast,
    ComputeBcast,
    ComputeScalar
};

struct BinaryNgKernelConfig {
    BinaryNgKernelConfig(SubtileBroadcastType subtile_broadcast_type);

    std::string bcast_input_str() const;

    KernelName reader_kernel;
    KernelName compute_kernel;
    KernelName writer_kernel;
    std::optional<uint32_t> bcast_input;
};

std::string get_kernel_file_path(KernelName kernel_name);
std::string get_sfpu_kernel_file_path(KernelName kernel_name);

struct OpConfig {
    struct SfpuConfig {
        SfpuConfig() = default;
        constexpr SfpuConfig(
            std::string_view init, std::string_view apply, std::string_view include = "compute_kernel_api.h") :
            init{init}, apply{apply}, include{include} {}
        std::string_view init{};
        std::string_view apply{};
        std::string_view include{};

        std::map<std::string, std::string> as_defines(std::string_view prefix) const;
    };

    enum class FpuBinaryOp { ADD, SUB, MUL };

    OpConfig(BinaryOpType binary_op_type);

    std::map<std::string, std::string> as_defines() const;

    SfpuConfig preprocess_a{};
    SfpuConfig preprocess_b{};
    SfpuConfig postprocess{};
    FpuBinaryOp fpu_binary_op;
};

struct Lowercase {
    std::string_view view;
};

std::map<std::string, std::string> get_defines_fp32(
    BinaryOpType op_type,
    const std::optional<tt::tt_metal::DataType> in_a_dtype = std::nullopt,
    const std::optional<tt::tt_metal::DataType> in_b_dtype = std::nullopt,
    const std::optional<ttnn::operations::unary::FusedActivations>& fused_activations = std::nullopt,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& input_tensor_a_activation = std::nullopt);

}  // namespace ttnn::operations::binary_ng
