// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "binary_ng_device_operation.hpp"
#include "ttnn/operations/eltwise/binary_ng/types.hpp"
#include "ttnn/tensor/types.hpp"

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
    ComputeScalar,
    ReaderNoBcastNg,
    WriterNoBcastNg,
    ReaderRowBcastNg,
    ReaderColBcastNg,
    ReaderRowBColABcastNg,
    ReaderScalarBcastNg,
};

struct BinaryNgKernelConfig {
    BinaryNgKernelConfig(SubtileBroadcastType subtile_broadcast_type);

    std::string bcast_input_str() const;

    KernelName reader_kernel;
    KernelName compute_kernel;
    KernelName writer_kernel;
    std::optional<uint32_t> bcast_input;
};

std::string get_kernel_file_path(KernelName kernel_name, bool is_sfpu);

struct OpConfig {
    enum class FpuBinaryOp { ADD, SUB, MUL };
    enum class SfpuBinaryOp {
        ADD,
        SUB,
        MUL,
        DIV,
        POWER,
        RSUB,
        GCD,
        LCM,
        LEFT_SHIFT,
        RIGHT_SHIFT,
        LOGICAL_RIGHT_SHIFT,
        BITWISE_AND,
        BITWISE_OR,
        BITWISE_XOR,
        QUANT,
        REQUANT,
        DEQUANT,
        MAXIMUM,
        MINIMUM,
        XLOGY
    };

    template <class EnumT>
    OpConfig(BinaryOpType binary_op_type, std::in_place_type_t<EnumT>);

    std::map<std::string, std::string> as_defines(DataType dtype) const;

    std::optional<unary::UnaryOpType> process_lhs{};
    std::optional<unary::UnaryOpType> process_rhs{};
    std::optional<unary::UnaryOpType> postprocess{};
    std::variant<FpuBinaryOp, SfpuBinaryOp> binary_op;
    bool is_sfpu_op() const;
};

void add_activation_defines(
    std::map<std::string, std::string>& defines,
    tt::stl::Span<const unary::UnaryWithParam> activations,
    std::string_view operand,
    std::optional<DataType> dtype = std::nullopt);

uint32_t pack_scalar_runtime_arg(float scalar, DataType dtype, bool is_quant_op);

std::map<std::string, std::string> make_dataflow_defines(DataType dtype, DataType b_dtype);

}  // namespace ttnn::operations::binary_ng
