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
    ComputeRowBcastNg,
    ComputeRowColBcastNg,
};

struct BinaryNgKernelConfig {
    BinaryNgKernelConfig(SubtileBroadcastType subtile_broadcast_type);

    std::string bcast_input_str() const;

    KernelName reader_kernel;
    KernelName compute_kernel;
    KernelName writer_kernel;
    std::optional<uint32_t> bcast_input;
};

std::string get_kernel_file_path(KernelName kernel_name, bool is_sfpu, bool is_where_op);

struct OpConfig {
    enum class FpuBinaryOp { ADD, SUB, MUL };
    enum class SfpuBinaryOp {
        ADD,
        SUB,
        MUL,
        DIV,
        DIV_FLOOR,
        DIV_TRUNC,
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
        XLOGY,
        LT,
        GT,
        GE,
        LE,
        HYPOT,
        WHERE,
    };

    template <class EnumT>
    OpConfig(BinaryOpType binary_op_type, std::in_place_type_t<EnumT>, std::optional<DataType> dtype = std::nullopt);

    std::map<std::string, std::string> as_defines(DataType dtype) const;

    std::optional<unary::UnaryOpType> process_lhs;
    std::optional<unary::UnaryOpType> process_rhs;
    std::optional<unary::UnaryOpType> postprocess;
    std::variant<FpuBinaryOp, SfpuBinaryOp> binary_op;
    bool is_sfpu_op() const;
};

void add_activation_defines(
    std::map<std::string, std::string>& defines,
    tt::stl::Span<const unary::EltwiseUnaryWithParam> activations,
    std::string_view operand,
    std::optional<DataType> dtype = std::nullopt);

uint32_t pack_scalar_runtime_arg(unary::ScalarVariant scalar, DataType dtype, bool is_quant_op);

std::map<std::string, std::string> make_dataflow_defines(
    DataType dtype, std::optional<DataType> b_dtype = std::nullopt);

struct AllShardSpecs {
    tt::tt_metal::ShardSpec a_shard_spec;
    tt::tt_metal::ShardSpec b_shard_spec;
    tt::tt_metal::ShardSpec c_shard_spec;
};

tt::tt_metal::ShardSpec adjust_to_shape(
    const tt::tt_metal::ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape);

struct AllShardVolumes {
    std::optional<std::uint32_t> a_shard_volume;
    std::optional<std::uint32_t> b_shard_volume;
    std::optional<std::uint32_t> c_shard_volume;
};

std::optional<AllShardVolumes> get_shard_volumes(
    const TensorSpec& a, const std::optional<TensorSpec>& b, const TensorSpec& c);

const std::optional<tt::tt_metal::ShardSpec>& get_shard_spec(const TensorSpec& tensor_spec);

bool is_native_L1_sharding(const TensorSpec& a, const std::optional<TensorSpec>& b, const MemoryConfig& c);

ttnn::Shape compute_broadcasted_output(const ttnn::Shape& shape_a, const ttnn::Shape& shape_b);

MemoryConfig compute_mem_config_actual(const ttnn::Tensor& input_tensor_a, const ttnn::Shape& shape_b);
}  // namespace ttnn::operations::binary_ng
