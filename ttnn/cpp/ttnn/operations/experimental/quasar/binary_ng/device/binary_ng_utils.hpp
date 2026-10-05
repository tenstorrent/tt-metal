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
    OpConfig(
        BinaryOpType binary_op_type,
        std::in_place_type_t<EnumT>,
        std::optional<DataType> dtype = std::nullopt,
        const std::optional<binary::BinaryOpParams>& op_params = std::nullopt);

    std::map<std::string, std::string> as_defines(DataType dtype) const;

    std::optional<unary::UnaryOpType> process_lhs;
    std::optional<unary::UnaryOpType> process_rhs;
    // Carries a parameter: a bare UnaryOpType reaches get_op_init_and_func_default, which emits the
    // paramless form and so inherits the compute API's default template argument.
    std::optional<unary::EltwiseUnaryWithParam> postprocess;
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

// A TTNN_QSR_* tuning knob for ProgramFactoryQuasarNative: an override for experiments. Unset, the rule
// decides its value (resolve_native_config).
struct NativeKnob {
    const char* name = nullptr;   // the environment variable
    std::optional<uint32_t> env;  // the value it holds, if it is set
};

// Env-driven tuning for ProgramFactoryQuasarNative, read once per process. The resolved R/C/W set KernelSpec
// num_threads. They do not restrict which shapes are admitted: each kernel derives its own share from
// thread_id and num_threads, so any tile count works and a thread may draw zero tiles. The one R/C/W
// admission rule is the per-DFB STRIDED ratio, max(p,c) % min(p,c) == 0, which the rule's values always meet.
// A borrowed shard or slice runs one reader and one writer thread at the compute count; a shard's tiles past
// the largest multiple of that count go through small owned rings, and a slice is borrowed only when that
// count divides it.
struct NativeTuning {
    bool enabled = false;        // TTNN_QSR_NATIVE; 0 and unset both mean OFF
    bool implicit_sync = false;  // NOT consumed, and native_tuning() throws if set: enabling it needs the
                                 // guarantee that no thread draws zero tiles, which uneven tile counts removed
    NativeKnob reader_threads{"TTNN_QSR_READER_THREADS"};    // R
    NativeKnob compute_threads{"TTNN_QSR_COMPUTE_THREADS"};  // C -- must be 1, 2 or 4
    NativeKnob writer_threads{"TTNN_QSR_WRITER_THREADS"};    // W
    // Per-thread ring depth; num_entries = this x max(producers, consumers). It must hold two batches of
    // either kind, for double buffering, and be a multiple of each, so that no batch straddles the ring end.
    NativeKnob entries_per_thread{"TTNN_QSR_ENTRIES_PER_THREAD"};
    // Tiles per barrier in the reader and the writer. Independent of tiles_per_cycle: a ring lets producer
    // and consumer transact at different granularities.
    NativeKnob dm_batch{"TTNN_QSR_DM_BATCH"};
    NativeKnob tiles_per_cycle{"TTNN_QSR_TILES_PER_CYCLE"};  // tiles per tile_regs_acquire in the compute
};

// The user DM cores of a Quasar cluster, which the reader and the writer threads share.
inline constexpr uint32_t kNativeUserDmCores = 6;

// What a program moves over the NoC. A borrowed operand moves nothing.
struct NativeMoves {
    bool inputs = false;  // a or b is read over the NoC
    bool output = false;  // c is written over the NoC
};

// The values one program runs with.
struct NativeConfig {
    uint32_t reader_threads = 0;
    uint32_t compute_threads = 0;
    uint32_t writer_threads = 0;
    uint32_t entries_per_thread = 0;
    uint32_t dm_batch = 0;
    uint32_t tiles_per_cycle = 0;
};

// The compute count of every program: the knob, else the rule's 4.
uint32_t native_compute_threads(const NativeTuning& tuning);

// Each set knob, else the rule's value for what the program moves. The rule takes the most threads and the
// largest batches the hardware allows: C = 4; R = C when an input moves, else 1; W = min(C, 6 - R) when the
// output moves, else 1, so R + W fits the 6 user DM cores; 16 ring entries per thread; DM batch 8; compute
// batch 8 where R <= C and W <= C, else 1. A borrowed program runs R = W = 1 whatever is set. Above ring
// stride 1 a compute batch packs wrong until the pack path applies the ring stride per tile.
NativeConfig resolve_native_config(const NativeTuning& tuning, NativeMoves moves);

// "8 (rule)" or "8 (TTNN_QSR_DM_BATCH)": a value the knob or the rule gave, with its source, for messages.
std::string describe_native_value(const NativeKnob& knob, uint32_t value);

// Parsed once into a function-local static. Knobs are TTNN_QSR_{NATIVE, IMPLICIT_SYNC, READER_THREADS,
// COMPUTE_THREADS, WRITER_THREADS, ENTRIES_PER_THREAD, DM_BATCH, TILES_PER_CYCLE}. Topology invariants are
// asserted only when `enabled`, so a bad knob cannot take down the fallback reference arm.
const NativeTuning& native_tuning();
}  // namespace ttnn::operations::experimental::quasar::binary_ng
