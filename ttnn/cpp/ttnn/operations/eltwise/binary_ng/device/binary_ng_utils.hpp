// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "binary_ng_device_operation.hpp"
#include <tt_stl/span.hpp>
#include "ttnn/operations/eltwise/binary_ng/types.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations {
namespace binary_ng {
enum class BinaryOpType;
enum class SubtileBroadcastType;
}  // namespace binary_ng
namespace unary {
enum class UnaryOpType;
struct UnaryWithParam;
}  // namespace unary
}  // namespace operations
}  // namespace ttnn
namespace tt {
namespace tt_metal {
enum class DataType;
}  // namespace tt_metal
}  // namespace tt

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
        LEFT_SHIFT,
        RIGHT_SHIFT,
        BITWISE_AND,
        BITWISE_OR,
        BITWISE_XOR,
        QUANT,
        REQUANT,
        DEQUANT
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
    std::string_view operand);

uint32_t pack_scalar_runtime_arg(const float scalar, const DataType dtype, const bool is_quant_op);

std::map<std::string, std::string> make_dataflow_defines(const DataType dtype, const bool is_sfpu_op);

}  // namespace ttnn::operations::binary_ng
