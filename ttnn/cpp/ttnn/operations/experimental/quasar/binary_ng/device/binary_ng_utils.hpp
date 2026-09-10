// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "binary_ng_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/binary_ng/types.hpp"
#include "ttnn/tensor/types.hpp"

#include <optional>
#include <string>

namespace ttnn::operations::experimental::quasar::binary_ng {

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
    ReaderRmNoBcastNg,
    ReaderRmRowBcastNg,
    ReaderRmColBcastNg,
    ReaderRmRowBColABcastNg,
    ReaderRmScalarBcastNg,
    ReaderRmScalarOpNg,
    WriterRmNoBcastNg,
    ComputeRowBcastNg,
    ComputeColBcastNg,
    ComputeScalarBcastNg,
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
        REMAINDER,
        FMOD,
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
        ATAN2,
        LT,
        GT,
        GE,
        LE,
        HYPOT,
        WHERE,
        EQ,
        NE,
        ISCLOSE,
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
    ttsl::Span<const unary::EltwiseUnaryWithParam> activations,
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
    const tt::tt_metal::TensorSpec& a,
    const std::optional<tt::tt_metal::TensorSpec>& b,
    const tt::tt_metal::TensorSpec& c);

const std::optional<tt::tt_metal::ShardSpec>& get_shard_spec(const tt::tt_metal::TensorSpec& tensor_spec);

bool is_uneven(const tt::tt_metal::TensorSpec& t);

bool is_native_L1_sharding(
    const tt::tt_metal::TensorSpec& a, const std::optional<tt::tt_metal::TensorSpec>& b, const MemoryConfig& c);

ttnn::Shape compute_broadcasted_output(const ttnn::Shape& shape_a, const ttnn::Shape& shape_b);

MemoryConfig compute_mem_config_actual(const ttnn::Tensor& input_tensor_a, const ttnn::Shape& shape_b);

// Env-driven tuning for ProgramFactoryQuasarNative, read once per process. R/C/W set KernelSpec
// num_threads AND gate admission: matches_quasar_native_slice rejects shapes whose per-core tile count
// does not divide by lcm(R,C,W), because the kernels' strided share assumes an exact split.
struct NativeTuning {
    bool implicit_sync = false;       // parsed and logged; NOT consumed -- the factory hardcodes
                                      // explicit sync, so setting the env var changes nothing
    uint32_t entries_per_thread = 2;  // per-thread ring depth; num_entries = this x max(producers, consumers)
    uint32_t reader_threads = 1;      // R
    uint32_t compute_threads = 1;     // C -- must be 1, 2 or 4
    uint32_t writer_threads = 1;      // W
    bool enabled = false;             // TTNN_QSR_NATIVE; 0 and unset both mean OFF
};

// Parsed once into a function-local static. Knobs are TTNN_QSR_{NATIVE, IMPLICIT_SYNC,
// ENTRIES_PER_THREAD, READER_THREADS, COMPUTE_THREADS, WRITER_THREADS}. Topology invariants are
// asserted only when `enabled`, so a bad knob cannot take down the fallback reference arm.
const NativeTuning& native_tuning();
}  // namespace ttnn::operations::experimental::quasar::binary_ng
