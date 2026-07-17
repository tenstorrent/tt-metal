// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) program factory for binary_ng's no-broadcast (SubtileBroadcastType::
// NONE), tiled path.
//
// This is a generic, faithful port of the descriptor factory's NONE-tiled path
// (binary_ng_program_factory.cpp::create_descriptor). It inherits that factory's logic — per-operand
// layout, FPU/SFPU OpConfig, the full lhs/rhs/post activation machinery, fp32_dest_acc_en,
// unpack_modes, num_tiles_per_cycle, and the per-core arg derivation — and differs from it ONLY
// in the CB->DFB translation:
//   - CBDescriptor -> DataflowBufferSpec (one per CB index the NONE path allocates: c_0->in0/pre_lhs,
//     c_1->in1/pre_rhs, c_2->out, c_3->post_lhs, c_4->post_rhs).
//   - The decisive property is BORROWED vs NoC-READ, all-or-nothing. An operand is BORROWED only when the
//     config is is_native_L1_sharding (output L1-sharded, a and b sharing that memory config, all sharded
//     grids == the output grid, all buffers L1) AND every operand is itself L1-sharded with a shard spec.
//     is_native_L1_sharding can hold with an L1-interleaved input (a single sharded operand satisfies it),
//     and an interleaved operand has no shard spec to back a DFB, so any interleaved operand forces the
//     whole op to the NoC path. When borrowed, all three are co-resident L1 shards on one grid with
//     identical per-core tile partitions: each CB `.buffer = tensor.buffer()` ->
//     DataflowBufferSpec::borrowed_from (reader/writer do no NoC work). Otherwise NONE are borrowed:
//     every operand's CB -> a real ring + a KernelSpec TensorBinding (TensorAccessor(tensor::name),
//     sharding-aware), and the buffer-address runtime arg the descriptor passed is dropped (the binding
//     injects the address). Placement and has_sharding follow the same all-or-nothing borrow decision.
//     The descriptor borrows per-operand and gates placement on get_shard_volumes().has_value()
//     (is_native_L1_sharding plus an uneven-shard all-specs-match guard); this factory's all-or-nothing
//     borrow is the stricter subset.
//   - Positional get_arg_val<>(idx) / get_compile_time_arg_val(idx) -> named get_arg(args::name). The
//     reader's 19 named args / writer's 9 named args are the descriptor's 21 / 11 arg vectors minus
//     the buffer-address args (reader src0@0 / src1@15, writer dst@0, writer trailing pad).
//   - KernelDescriptor::core_ranges + per-core dummy args on unused cores -> WorkUnitSpec::target_nodes
//     scoped to exactly the active cores (DFB places only on target_nodes, so no dummy args needed).
//   - The descriptor's `has_sharding` TensorAccessor CTA -> a HAS_SHARDING #define (the DFB kernels
//     read it as a macro).
//
// Deferred to the descriptor (rejected by matches_metal_v2_slice, NOT handled here): row-major (non-tile)
// layout, tensor-scalar (no input_tensor_b), where-op, quantization, and mixed lhs/rhs dtype. Mixed
// sharded/interleaved layouts AND width sharding ARE handled: the borrow path is taken only when all
// three operands are co-resident L1 shards on one matching grid; everything else (interleaved output OR
// input, mixed strategies, divergent grids) takes the NoC path via sharding-aware TensorAccessors. A
// borrowed operand is L1-sharded-tiled (height/block/width).

#include "binary_ng_device_operation.hpp"
#include "binary_ng_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/experimental/quasar/binary/common/binary_op_utils.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::operations::experimental::quasar::binary_ng {

namespace {

using ttnn::device_operation::ProgramArtifacts;

// Kernel sources for the no-broadcast DFB path (dual-mode: a sharded operand publishes its borrowed
// shard, an interleaved operand reads/writes over the NoC via a tensor binding).
constexpr const char* kReaderDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_dfb/dataflow/reader_no_bcast_dfb.cpp";
constexpr const char* kWriterDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_dfb/dataflow/writer_no_bcast_dfb.cpp";
constexpr const char* kComputeFpuDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_dfb/compute/eltwise_binary_no_bcast_dfb.cpp";
constexpr const char* kComputeSfpuDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_dfb/compute/"
    "eltwise_binary_sfpu_no_bcast_dfb.cpp";

// The compute kernel includes eltwise_utils_common.hpp (in kernels/compute) and its DFB preprocess
// helper (in kernels_dfb/compute) by bare name; both directories go on the compute include path.
constexpr const char* kComputeIncludeCommon =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels/compute";
constexpr const char* kComputeIncludeDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_dfb/compute";

// Per-broadcast-type kernel-source seam (the DFB analogue of the descriptor's BinaryNgKernelConfig):
// it maps a SubtileBroadcastType to the reader/writer/compute kernel sources. Only the no-broadcast
// (NONE) triple is wired; the broadcast variants follow when their DFB kernels are added.
struct DfbKernelSources {
    const char* reader = nullptr;
    const char* writer = nullptr;
    const char* compute = nullptr;
};

DfbKernelSources select_dfb_kernel_sources(SubtileBroadcastType subtile_broadcast_type, bool is_sfpu) {
    TT_FATAL(
        subtile_broadcast_type == SubtileBroadcastType::NONE,
        "binary_ng Metal 2.0 factory only wires the no-broadcast (NONE) kernel sources");
    return DfbKernelSources{
        .reader = kReaderDfb,
        .writer = kWriterDfb,
        .compute = is_sfpu ? kComputeSfpuDfb : kComputeFpuDfb,
    };
}

// --- Small pure helpers mirrored from the descriptor factory's anonymous namespace. They live in
// that .cpp's anonymous namespace (not the shared header), so they are reimplemented here to avoid
// touching the proven descriptor path. Keep behavior identical to binary_ng_program_factory.cpp. ---

// For rank > 5, dims are collapsed into a single dim. Mirrors extract_nD_dims in the descriptor factory.
uint32_t extract_nD_dims(const Tensor& x, int out_rank) {
    const auto& shape = x.logical_shape();
    uint32_t nD_dim = 1;
    if (out_rank >= 6 && shape.rank() >= 6) {
        for (int i = -6; i >= -out_rank; --i) {
            nD_dim *= shape[i];
        }
    }
    return nD_dim;
}

// (D, N, C, Ht, Wt) with Ht/Wt in tiles. Mirrors get_shape_dims in the descriptor factory.
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> get_shape_dims(const Tensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {
        shape.rank() >= 5 ? shape[-5] : 1,
        shape[-4],
        shape[-3],
        tt::div_up(shape[-2], tile.get_height()),
        tt::div_up(shape[-1], tile.get_width())};
}

// Number of shards spanning the output width. Mirrors get_shards_per_width in the descriptor factory.
uint32_t get_shards_per_width(const ShardSpec& shard_spec, TensorMemoryLayout memory_layout) {
    const auto num_cores = shard_spec.grid.num_cores();
    if (memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        return 1;
    }
    if (memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        return num_cores;
    }
    const auto& bbox = shard_spec.grid.bounding_box();
    const auto& start = bbox.start_coord;
    const auto& end = bbox.end_coord;
    return (shard_spec.orientation == ShardOrientation::ROW_MAJOR ? end.x - start.x : end.y - start.y) + 1;
}

// Tile count for one full (tile-rounded) shard. Every core processes this same count, including a
// partial end core under an uneven shard: a sharded buffer allocates the full rounded-up shard on
// every core, so the trailing tiles on a partial core are allocated L1 with no host page mapped —
// computing on them is in-bounds and never reaches the logical output. The gate pins equal shard
// specs across a/b/c, so the three tensors over-allocate identically (mirrors the descriptor's
// all_same_shard_spec path, which uses the uniform full shard tile count).
uint32_t full_shard_tiles(const Tensor& tensor, const ShardSpec& shard_spec) {
    const uint32_t tile_h = tensor.tensor_spec().tile().get_height();
    const uint32_t tile_w = tensor.tensor_spec().tile().get_width();
    const uint32_t shard_ht = tt::round_up(shard_spec.shape[0], tile_h) / tile_h;
    const uint32_t shard_wt = tt::round_up(shard_spec.shape[1], tile_w) / tile_w;
    return shard_ht * shard_wt;
}

m2::DataflowBufferSpec make_dfb(
    const m2::DFBSpecName& name,
    uint32_t entry_size,
    uint32_t num_entries,
    tt::DataFormat df,
    const tt::tt_metal::Tile& tile,
    std::optional<m2::TensorParamName> borrowed) {
    return m2::DataflowBufferSpec{
        .unique_id = name,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = df,
        .tile_format_metadata = tile,
        .borrowed_from = std::move(borrowed),
    };
}

ProgramArtifacts create_no_bcast_artifacts(
    const BinaryNgDeviceOperation::operation_attributes_t& op,
    const BinaryNgDeviceOperation::tensor_args_t& tensor_args,
    Tensor& c) {
    const Tensor& a = tensor_args.input_tensor_a;
    TT_FATAL(
        tensor_args.input_tensor_b.has_value(),
        "binary_ng Metal 2.0 no-bcast factory requires a second input tensor (tensor-scalar routes to the "
        "descriptor)");
    const Tensor& b = *tensor_args.input_tensor_b;

    const bool is_sfpu = op.is_sfpu;
    const DataType input_dtype = op.input_dtype;
    const auto op_type = op.binary_op_type;

    Buffer* a_buffer = a.buffer();
    Buffer* b_buffer = b.buffer();
    Buffer* c_buffer = c.buffer();
    TT_FATAL(
        a_buffer != nullptr && b_buffer != nullptr && c_buffer != nullptr,
        "binary_ng Metal 2.0 no-bcast factory requires allocated device buffers");

    // --- Borrow vs NoC routing. The borrow facts come from the shared get_shard_volumes helper -- the
    // same one the descriptor factory and compute_program_hash use -- so this factory and the descriptor
    // agree on what "native L1 sharding" means: get_shard_volumes is is_native_L1_sharding plus an
    // uneven-shard guard (an unevenly-sharded output additionally requires all three operands to share one
    // shard spec), so it returns nullopt both for a non-native config and for an uneven output the
    // all-or-nothing borrow cannot serve; it reports a per-operand shard volume only for an operand that is
    // itself L1-sharded with a shard spec. The per-operand a/b/c_sharded flags are derived
    // from it, but this no-broadcast slice collapses borrow to all-or-nothing: an operand is BORROWED (its
    // resident L1 shard backs the DFB; reader/writer do no NoC work) only when EVERY operand is sharded, so
    // all three are co-resident L1 shards on one grid with identical per-core tile partitions. Otherwise --
    // an interleaved output OR input, mixed strategies, or a divergent grid -- NONE are borrowed and every
    // operand is read/written through its own sharding-aware TensorAccessor over a linear page-id walk (the
    // get_shard_volumes == nullopt / single-or-partial-sharded case). The per-operand a/b/c_borrowed flags
    // are kept separate (the DFB specs, SRC_SHARDED defines, tensor bindings and placement already branch
    // on them), so enabling per-operand borrow for broadcast later is a one-line change here. The all-NoC
    // path is correct for any layout mix; borrowing is a throughput optimization for the fully co-resident
    // case (a block-sharded residual add). ---
    const auto shard_volumes =
        get_shard_volumes(a.tensor_spec(), std::optional<TensorSpec>{b.tensor_spec()}, c.tensor_spec());
    const bool native = shard_volumes.has_value();
    const bool a_sharded = native && shard_volumes->a_shard_volume.has_value();
    const bool b_sharded = native && shard_volumes->b_shard_volume.has_value();
    const bool c_sharded = native && shard_volumes->c_shard_volume.has_value();
    const bool borrow_shards = a_sharded && b_sharded && c_sharded;

    const bool a_borrowed = borrow_shards;
    const bool b_borrowed = borrow_shards;
    const bool c_borrowed = borrow_shards;

    // HAS_SHARDING gates the borrow-path one-shard-per-core placement + per-shard tile-walk wrap. Off the
    // borrow path every operand (including a sharded output) is NoC-read/written via TensorAccessor over a
    // linear page-id walk. The reader/writer read it as a #define.
    const bool has_sharding = borrow_shards;

    // --- Dtypes / data formats / tile sizes (mirrors the descriptor factory). b is a tensor here. ---
    const DataType a_dtype = a.dtype();
    const DataType b_dtype = b.dtype();
    const DataType c_dtype = c.dtype();
    const tt::DataFormat a_df = datatype_to_dataformat_converter(a_dtype);
    const tt::DataFormat b_df = datatype_to_dataformat_converter(b_dtype);
    const tt::DataFormat c_df = datatype_to_dataformat_converter(c_dtype);
    const tt::tt_metal::Tile a_tile = a.tensor_spec().tile();
    const tt::tt_metal::Tile b_tile = b.tensor_spec().tile();
    const tt::tt_metal::Tile c_tile = c.tensor_spec().tile();
    const uint32_t a_tile_bytes = static_cast<uint32_t>(a_tile.get_tile_size(a_df));
    const uint32_t b_tile_bytes = static_cast<uint32_t>(b_tile.get_tile_size(b_df));
    const uint32_t c_tile_bytes = static_cast<uint32_t>(c_tile.get_tile_size(c_df));

    // --- OpConfig + compute defines (mirrors the descriptor factory). ---
    OpConfig op_config = is_sfpu ? OpConfig(op_type, std::in_place_type<OpConfig::SfpuBinaryOp>, a_dtype)
                                 : OpConfig(op_type, std::in_place_type<OpConfig::FpuBinaryOp>, a_dtype);
    std::map<std::string, std::string> compute_defines = op_config.as_defines(a_dtype);

    // ISCLOSE: the DFB SFPU kernel reads rtol/atol as named args (args::rtol_bits/atol_bits), so the
    // positional ISCLOSE_*_RT_ARG_IDX defines the descriptor emits are not needed here (mirrors the
    // descriptor factory's ISCLOSE handling, minus the positional-index defines).
    if (op_type == BinaryOpType::ISCLOSE) {
        compute_defines["ISCLOSE_OP"] = "1";
        compute_defines["ISCLOSE_EQUAL_NAN"] = op.equal_nan ? "1" : "0";
    }

    // --- Activation assembly (faithful copy of the descriptor factory). ---
    {
        ttnn::SmallVector<unary::EltwiseUnaryWithParam> lhs_activations = op.lhs_activations;
        ttnn::SmallVector<unary::EltwiseUnaryWithParam> rhs_activations = op.rhs_activations;
        ttnn::SmallVector<unary::EltwiseUnaryWithParam> post_activations = op.post_activations;

        if (op_config.process_lhs.has_value()) {
            lhs_activations.push_back(*op_config.process_lhs);
        }
        if (op_config.process_rhs.has_value()) {
            rhs_activations.push_back(*op_config.process_rhs);
        }

        // LDEXP on the FPU path leaves LHS in its block-float format while RHS becomes Float16_b; force
        // LHS through a Float16_b intermediate so both operands share a format (as in the descriptor factory).
        if (!is_sfpu && lhs_activations.empty() && !rhs_activations.empty() && op_type == BinaryOpType::LDEXP &&
            (a_dtype == DataType::BFLOAT8_B || a_dtype == DataType::BFLOAT4_B)) {
            lhs_activations.push_back({
                unary::UnaryOpType::TYPECAST,
                {static_cast<int>(a_dtype), static_cast<int>(DataType::BFLOAT16)},
            });
        }

        if (op_config.postprocess.has_value()) {
            post_activations.insert(post_activations.begin(), *op_config.postprocess);
        }

        const bool is_integer_division =
            (op_type == BinaryOpType::DIV && a_dtype == DataType::INT32 && b_dtype == DataType::INT32);
        if (binary::utils::is_typecast(a_dtype, c_dtype) && !is_integer_division) {
            post_activations.push_back({
                unary::UnaryOpType::TYPECAST,
                {static_cast<int>(a_dtype), static_cast<int>(c_dtype)},
            });
        }

        add_activation_defines(compute_defines, lhs_activations, "LHS", a_dtype);
        add_activation_defines(compute_defines, rhs_activations, "RHS", b_dtype);

        // The descriptor disables the PACK_RELU fast path under subtile broadcast: a broadcast compute
        // kernel does a per-iteration intermediate pack + pack_reconfig that clears the packer's ZERO_RELU
        // state, silently dropping the RELU clip on the final pack. This factory mirrors that guard; the
        // no-broadcast slice admits only NONE, so is_subtile_broadcast is always false and the guard is a
        // safe no-op unless broadcast types are added.
        const bool is_subtile_broadcast = op.subtile_broadcast_type != SubtileBroadcastType::NONE;

        // PACK_RELU fast path (no-broadcast only). Other single post-acts expand to the SFPU post chain
        // (as in the descriptor factory; ZERO_POINT/quant routes to the descriptor).
        if (lhs_activations.empty() && rhs_activations.empty() && post_activations.size() == 1) {
            compute_defines["PROCESS_POST_ACTIVATIONS(i)"] = "";
            if (post_activations[0].type() == unary::UnaryOpType::RELU && !is_subtile_broadcast) {
                compute_defines["PACK_RELU"] = "1";
                unary::utils::update_macro_defines(unary::UnaryOpType::RELU, compute_defines);
            } else {
                add_activation_defines(compute_defines, post_activations, "POST", input_dtype);
            }
        } else {
            add_activation_defines(compute_defines, post_activations, "POST", input_dtype);
        }
    }

    const bool has_lhs_act = !compute_defines["PROCESS_LHS_ACTIVATIONS(i)"].empty();
    const bool has_rhs_act = !compute_defines["PROCESS_RHS_ACTIVATIONS(i)"].empty();
    const bool op_has_exp =
        op_type == BinaryOpType::LOGADDEXP || op_type == BinaryOpType::LDEXP || op_type == BinaryOpType::LOGADDEXP2;

    // --- num_tiles_per_cycle: DST register capacity per tile_regs_acquire (mirrors the descriptor
    // factory + the shipped factory's min(.,shard_tiles) cap). Multi-tile only when EVERY operand is borrowed
    // (the whole pipeline is a single bulk reserve/push of full shards); any NoC-read operand uses a
    // 2-entry ring, so the chunk size must be 1. SFPU uses 2 (double-DST stride), FPU uses 8. ---
    const bool all_borrowed = a_borrowed && b_borrowed && c_borrowed;
    const uint32_t c_full_shard_tiles = c_borrowed ? full_shard_tiles(c, *c.shard_spec()) : 0u;
    uint32_t num_tiles_per_cycle = 1;
    if (all_borrowed) {
        num_tiles_per_cycle = std::min<uint32_t>(is_sfpu ? 2u : 8u, c_full_shard_tiles);
    }

    // --- DFB names. compute uses pre_lhs/pre_rhs/out (+post_lhs/post_rhs when activations); reader/
    // writer use in0/in1/out. The SAME DFBSpecs (c_0/c_1/c_2) are bound under per-kernel accessor
    // names. ---
    const m2::TensorParamName T_A{"binary_ng_a"};
    const m2::TensorParamName T_B{"binary_ng_b"};
    const m2::TensorParamName T_C{"binary_ng_c"};
    const m2::DFBSpecName IN0{"binary_ng_in0_dfb"};            // CBIndex::c_0
    const m2::DFBSpecName IN1{"binary_ng_in1_dfb"};            // CBIndex::c_1
    const m2::DFBSpecName OUT{"binary_ng_out_dfb"};            // CBIndex::c_2
    const m2::DFBSpecName POST_LHS{"binary_ng_post_lhs_dfb"};  // CBIndex::c_3 (LHS activations)
    const m2::DFBSpecName POST_RHS{"binary_ng_post_rhs_dfb"};  // CBIndex::c_4 (RHS activations)
    const m2::KernelSpecName READER{"binary_ng_reader"};
    const m2::KernelSpecName WRITER{"binary_ng_writer"};
    const m2::KernelSpecName COMPUTE{"binary_ng_compute"};

    // Per-broadcast-type kernel-source selection (the descriptor's BinaryNgKernelConfig analogue); only
    // the no-broadcast (NONE) triple is wired.
    const DfbKernelSources kernel_sources = select_dfb_kernel_sources(op.subtile_broadcast_type, is_sfpu);

    // --- DataflowBuffers (mirrors the descriptor factory's CB block). A BORROWED operand backs the DFB
    // with its resident L1 shard (num_entries == full shard, borrowed_from set); any NoC-read operand
    // (interleaved, or sharded on a non-matching grid) is a 2-entry ring filled over the NoC.
    // post_lhs/post_rhs exist only when that operand has activations; their format is the op_has_exp
    // Float16_b intermediate on the FPU path, else the operand's own format. ---
    const uint32_t a_entries = a_borrowed ? full_shard_tiles(a, *a.shard_spec()) : 2u;
    const uint32_t b_entries = b_borrowed ? full_shard_tiles(b, *b.shard_spec()) : 2u;
    const uint32_t c_entries = c_borrowed ? full_shard_tiles(c, *c.shard_spec()) : 2u;

    std::vector<m2::DataflowBufferSpec> dfbs;
    dfbs.push_back(
        make_dfb(IN0, a_tile_bytes, a_entries, a_df, a_tile, a_borrowed ? std::optional{T_A} : std::nullopt));
    dfbs.push_back(
        make_dfb(IN1, b_tile_bytes, b_entries, b_df, b_tile, b_borrowed ? std::optional{T_B} : std::nullopt));
    dfbs.push_back(
        make_dfb(OUT, c_tile_bytes, c_entries, c_df, c_tile, c_borrowed ? std::optional{T_C} : std::nullopt));

    const tt::DataFormat a_inter_df = is_sfpu ? a_df : (op_has_exp ? tt::DataFormat::Float16_b : a_df);
    const tt::DataFormat b_inter_df = is_sfpu ? b_df : (op_has_exp ? tt::DataFormat::Float16_b : b_df);
    // post_lhs/post_rhs intermediate rings (c_3/c_4): num_tiles_per_cycle entries, matching the
    // descriptor factory's intermediate CBs. Allocated only when that operand has an activation chain.
    // The compute kernel both produces (PREPROCESS) and consumes (binary op) these in strict program
    // order on a single thread, so one chunk's worth of entries is sufficient.
    if (has_lhs_act) {
        dfbs.push_back(make_dfb(
            POST_LHS,
            static_cast<uint32_t>(a_tile.get_tile_size(a_inter_df)),
            num_tiles_per_cycle,
            a_inter_df,
            a_tile,
            std::nullopt));
    }
    if (has_rhs_act) {
        dfbs.push_back(make_dfb(
            POST_RHS,
            static_cast<uint32_t>(b_tile.get_tile_size(b_inter_df)),
            num_tiles_per_cycle,
            b_inter_df,
            b_tile,
            std::nullopt));
    }

    // --- Dataflow defines. SRC_SHARDED[_B]/DST_SHARDED select the BORROWED (publish/drain a resident
    // shard, no NoC) vs NoC-read code path per operand. HAS_SHARDING follows the OUTPUT layout: it tells
    // a NoC-read operand's tile walk to wrap each row onto one output-shard width (the descriptor passed
    // has_sharding as a TensorAccessor CTA; the DFB kernels read the macro). ---
    std::map<std::string, std::string> reader_defines = make_dataflow_defines(a_dtype, b_dtype);
    reader_defines["SRC_SHARDED"] = a_borrowed ? "1" : "0";
    reader_defines["SRC_SHARDED_B"] = b_borrowed ? "1" : "0";
    reader_defines["HAS_SHARDING"] = has_sharding ? "1" : "0";

    std::map<std::string, std::string> writer_defines = make_dataflow_defines(b_dtype);
    writer_defines["DST_SHARDED"] = c_borrowed ? "1" : "0";
    writer_defines["HAS_SHARDING"] = has_sharding ? "1" : "0";

    // --- Compute config (mirrors the descriptor factory). ---
    const bool fp32_dest_acc_en = c_df == tt::DataFormat::UInt32 || c_df == tt::DataFormat::Int32 ||
                                  c_df == tt::DataFormat::Float32 || a_df == tt::DataFormat::Float32 ||
                                  b_df == tt::DataFormat::Float32 ||
                                  (a_df == tt::DataFormat::Int32 && b_df == tt::DataFormat::Int32) ||
                                  (a_df == tt::DataFormat::UInt32 && b_df == tt::DataFormat::UInt32);

    // unpack_modes: the descriptor sets UnpackToDest on all SFPU consumer CBs, but that is inert for
    // non-Float32 entries (and the DFB validator rejects UnpackToDest on a 32-bit format unless
    // enable_32_bit_dest is true). So set it only on compute-consumer DFBs whose format is Float32
    // (which forces fp32_dest_acc_en true). A Float32 FPU consumer must still carry an explicit entry,
    // which is UnpackToSrc. compute consumers: in0(pre_lhs), in1(pre_rhs), post_lhs, post_rhs.
    m2::ComputeUnpackModes unpack_modes;
    auto set_unpack_mode = [&](const m2::DFBSpecName& dfb, tt::DataFormat df) {
        if (df == tt::DataFormat::Float32) {
            unpack_modes.emplace(dfb, is_sfpu ? UnpackMode::UnpackToDest : UnpackMode::UnpackToSrc);
        }
    };
    set_unpack_mode(IN0, a_df);
    set_unpack_mode(IN1, b_df);
    if (has_lhs_act) {
        set_unpack_mode(POST_LHS, a_inter_df);
    }
    if (has_rhs_act) {
        set_unpack_mode(POST_RHS, b_inter_df);
    }

    // --- Kernels. ---
    m2::KernelSpec::CompilerOptions::Defines reader_defines_tbl;
    for (const auto& [k, v] : reader_defines) {
        reader_defines_tbl.emplace(k, v);
    }
    m2::KernelSpec::CompilerOptions::Defines writer_defines_tbl;
    for (const auto& [k, v] : writer_defines) {
        writer_defines_tbl.emplace(k, v);
    }
    m2::KernelSpec::CompilerOptions::Defines compute_defines_tbl;
    for (const auto& [k, v] : compute_defines) {
        compute_defines_tbl.emplace(k, v);
    }

    // Reader: publishes a borrowed shard (borrowed operand) or reads it over the NoC (via
    // TensorAccessor(tensor::in0/in1)). A borrowed operand needs no tensor binding (its tensor::
    // reference is compiled out under SRC_SHARDED) and borrows via the DFB; every NoC-read operand is
    // bound — including a sharded input on a non-matching grid, whose accessor is sharding-aware.
    m2::Group<m2::TensorBinding> reader_tensor_bindings;
    if (!a_borrowed) {
        reader_tensor_bindings.push_back(m2::TensorBinding{T_A, "in0"});
    }
    if (!b_borrowed) {
        reader_tensor_bindings.push_back(m2::TensorBinding{T_B, "in1"});
    }
    m2::KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path(kernel_sources.reader),
        .num_threads = 1,
        .compiler_options = {.defines = reader_defines_tbl},
        .dfb_bindings = {m2::ProducerOf(IN0, "in0"), m2::ProducerOf(IN1, "in1")},
        .tensor_bindings = reader_tensor_bindings,
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"start_tile_id",
                  "src_num_tiles",
                  "dst_num_tiles",
                  "dst_shard_width",
                  "nD_stride",
                  "d_stride",
                  "n_stride",
                  "c_stride",
                  "D",
                  "N",
                  "C",
                  "Ht",
                  "Wt",
                  "cND",
                  "nD_stride_b",
                  "d_stride_b",
                  "n_stride_b",
                  "c_stride_b",
                  "src_num_tiles_b"}},
        .hw_config = ttnn::create_reader_datamovement_config(a.device()->arch()),
    };

    m2::Group<m2::TensorBinding> writer_tensor_bindings;
    if (!c_borrowed) {
        writer_tensor_bindings.push_back(m2::TensorBinding{T_C, "out"});
    }
    m2::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path(kernel_sources.writer),
        .num_threads = 1,
        .compiler_options = {.defines = writer_defines_tbl},
        .dfb_bindings = {m2::ConsumerOf(OUT, "out")},
        .tensor_bindings = writer_tensor_bindings,
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"start_tile_id", "dst_num_tiles", "dst_shard_width", "D", "N", "C", "Ht", "Wt", "cND"}},
        .hw_config = ttnn::create_writer_datamovement_config(a.device()->arch()),
    };

    // Compute: consumes pre_lhs/pre_rhs, produces out. When an operand has activations, the kernel both
    // produces (PREPROCESS writes) and consumes (binary op reads) its post DFB — a self-loop pair
    // (one PRODUCER + one CONSUMER binding on the same DFB, resolved intra-thread by default).
    m2::Group<m2::DFBBinding> compute_dfb_bindings = {
        m2::ConsumerOf(IN0, "pre_lhs"), m2::ConsumerOf(IN1, "pre_rhs"), m2::ProducerOf(OUT, "out")};
    if (has_lhs_act) {
        compute_dfb_bindings.push_back(m2::ProducerOf(POST_LHS, "post_lhs"));
        compute_dfb_bindings.push_back(m2::ConsumerOf(POST_LHS, "post_lhs"));
    }
    if (has_rhs_act) {
        compute_dfb_bindings.push_back(m2::ProducerOf(POST_RHS, "post_rhs"));
        compute_dfb_bindings.push_back(m2::ConsumerOf(POST_RHS, "post_rhs"));
    }

    m2::Group<std::string> compute_rt_names = {"num_tiles"};
    if (op_type == BinaryOpType::ISCLOSE) {
        compute_rt_names.push_back("rtol_bits");
        compute_rt_names.push_back("atol_bits");
    }

    // to_compute_hardware_config maps the common knobs and picks the arch's variant (ComputeGen1Config on
    // Wormhole, ComputeGen2Config on Quasar); it deliberately leaves the per-DFB unpack_modes
    // default, so set it here via std::visit — the arch-agnostic pattern main's quasar untilize factories use.
    auto compute_hw = ttnn::to_compute_hardware_config(
        a.device()->arch(),
        ttnn::ComputeKernelConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = false,
            .fp32_dest_acc_en = fp32_dest_acc_en,
        });
    std::visit([&](auto& cfg) { cfg.unpack_modes = unpack_modes; }, compute_hw);

    m2::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(kernel_sources.compute),
        .num_threads = 1,
        .compiler_options =
            {.include_paths = {std::filesystem::path(kComputeIncludeCommon), std::filesystem::path(kComputeIncludeDfb)},
             .defines = compute_defines_tbl},
        .dfb_bindings = compute_dfb_bindings,
        .compile_time_args = {{"num_tiles_per_cycle", num_tiles_per_cycle}},
        .runtime_arg_schema = {.runtime_arg_names = compute_rt_names},
        .hw_config = compute_hw,
    };

    // --- Placement + per-core runtime args (mirrors the descriptor factory's per-core loop). Sharded:
    // place on the shard grid, uniform full-shard counts. Interleaved: split_work_to_cores over the
    // worker grid; per-node num_tiles by group. Either way target_nodes is exactly the active cores
    // (no dummy args). ---
    const int out_rank = c.logical_shape().rank();
    const uint32_t aND = extract_nD_dims(a, out_rank);
    const uint32_t bND = extract_nD_dims(b, out_rank);
    const uint32_t cND = extract_nD_dims(c, out_rank);
    const auto [aD, aN, aC, aHt, aWt] = get_shape_dims(a);
    const auto [bD, bN, bC, bHt, bWt] = get_shape_dims(b);
    const auto [cD, cN, cC, cHt, cWt] = get_shape_dims(c);

    // a/b input strides (the (dim>1) gate zeroes a stride for a unit dim). Same on every core.
    const uint32_t nD_stride = aHt * aWt * aC * aN * aD * (aND > 1);
    const uint32_t d_stride = aHt * aWt * aC * aN * (aD > 1);
    const uint32_t n_stride = aHt * aWt * aC * (aN > 1);
    const uint32_t c_stride = aHt * aWt * (aC > 1);
    const uint32_t nD_stride_b = bHt * bWt * bC * bN * bD * (bND > 1);
    const uint32_t d_stride_b = bHt * bWt * bC * bN * (bD > 1);
    const uint32_t n_stride_b = bHt * bWt * bC * (bN > 1);
    const uint32_t c_stride_b = bHt * bWt * (bC > 1);

    std::vector<CoreCoord> cores;
    CoreRangeSet core_group_1, core_group_2;
    uint32_t num_tiles_per_core_group_1 = 0, num_tiles_per_core_group_2 = 0;
    uint32_t c_shard_height = 0, c_shard_width = 0, num_shards_per_width = 1;
    bool row_major = true;

    // Placement is driven by borrow_shards. The borrow path places one output shard per core on the
    // output shard grid (each core's output shard is its work), with the per-shard tile counts and the
    // per-core output start tile derived from the output shard geometry. Off the borrow path,
    // split_work_to_cores spreads the output tiles linearly across all of op.worker_grid (the full
    // sub-device TENSIX worker set returned by get_worker_grid, NOT the output shard grid); for a
    // NoC-written sharded output the sharding-aware TensorAccessor maps each global output tile id to its
    // shard core, so the linear split need not be shard-aligned. (The descriptor instead places one output
    // shard per core on the shard grid; the all-NoC sharded-output path is unique to this factory.)
    // Equal-shape no-broadcast means each core processes the same tile count of all three operands -- its
    // output-shard count on the borrow path, its split_work_to_cores share otherwise.
    if (borrow_shards) {
        const ShardSpec c_shard = *c.shard_spec();
        row_major = c_shard.orientation == ShardOrientation::ROW_MAJOR;
        cores = corerange_to_cores(c_shard.grid, std::nullopt, row_major);
        c_shard_height = tt::round_up(c_shard.shape[0], c_tile.get_height()) / c_tile.get_height();
        c_shard_width = tt::round_up(c_shard.shape[1], c_tile.get_width()) / c_tile.get_width();
        const TensorMemoryLayout memory_layout = c.memory_config().memory_layout();
        num_shards_per_width = get_shards_per_width(c_shard, memory_layout);
    } else {
        row_major = true;
        const uint32_t tile_hw = c.tensor_spec().tile().get_height() * c.tensor_spec().tile().get_width();
        const uint32_t rt_c_num_tiles = c.physical_volume() / tile_hw;
        CoreRangeSet all_cores;
        std::tie(
            std::ignore,
            all_cores,
            core_group_1,
            core_group_2,
            num_tiles_per_core_group_1,
            num_tiles_per_core_group_2) = tt::tt_metal::split_work_to_cores(op.worker_grid, rt_c_num_tiles, row_major);
        cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    }
    TT_FATAL(!cores.empty(), "binary_ng Metal 2.0 no-bcast factory: empty core set");

    m2::KernelRunArgs::RuntimeArgValues reader_args, writer_args, compute_args;

    const uint32_t isclose_rtol_bits = std::bit_cast<uint32_t>(op.rtol);
    const uint32_t isclose_atol_bits = std::bit_cast<uint32_t>(op.atol);

    std::set<CoreRange> target_ranges;
    uint32_t start_tile_id = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        target_ranges.insert(CoreRange(core, core));
        const m2::NodeCoord node{static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y)};

        uint32_t a_num_tiles = 0, b_num_tiles = 0, c_num_tiles_core = 0;
        uint32_t c_start_id = 0, c_current_shard_width = 0;
        if (borrow_shards) {
            // Borrow path: each core owns one output shard. Equal-shape no-broadcast => a/b per-core
            // counts equal the output-shard tile count (every operand is borrowed and publishes its own
            // matching shard). c_start_id is the global tile id of this output shard's top-left tile.
            c_num_tiles_core = c_shard_height * c_shard_width;
            a_num_tiles = c_num_tiles_core;
            b_num_tiles = c_num_tiles_core;
            c_current_shard_width = c_shard_width;
            c_start_id =
                (i / num_shards_per_width) * (c_shard_height * cWt) + (i % num_shards_per_width) * c_shard_width;
        } else {
            // Off the borrow path: split_work_to_cores gives this core's output tile count, walked
            // linearly from start_tile_id. Nothing is borrowed here, so every operand is NoC-read/written
            // and ignores src_num_tiles, walking dst_num_tiles instead; src_num_tiles is still set to the
            // per-core count for arg-shape parity. (c_current_shard_width stays 0, which the kernels read
            // only under HAS_SHARDING, i.e. never off the borrow path.)
            c_num_tiles_core = core_group_1.contains(core) ? num_tiles_per_core_group_1 : num_tiles_per_core_group_2;
            a_num_tiles = c_num_tiles_core;
            b_num_tiles = c_num_tiles_core;
            c_start_id = start_tile_id;
            start_tile_id += c_num_tiles_core;
        }

        reader_args["start_tile_id"][node] = c_start_id;
        reader_args["src_num_tiles"][node] = a_num_tiles;
        reader_args["dst_num_tiles"][node] = c_num_tiles_core;
        reader_args["dst_shard_width"][node] = c_current_shard_width;
        reader_args["nD_stride"][node] = nD_stride;
        reader_args["d_stride"][node] = d_stride;
        reader_args["n_stride"][node] = n_stride;
        reader_args["c_stride"][node] = c_stride;
        reader_args["D"][node] = cD;
        reader_args["N"][node] = cN;
        reader_args["C"][node] = cC;
        reader_args["Ht"][node] = cHt;
        reader_args["Wt"][node] = cWt;
        reader_args["cND"][node] = cND;
        reader_args["nD_stride_b"][node] = nD_stride_b;
        reader_args["d_stride_b"][node] = d_stride_b;
        reader_args["n_stride_b"][node] = n_stride_b;
        reader_args["c_stride_b"][node] = c_stride_b;
        reader_args["src_num_tiles_b"][node] = b_num_tiles;

        writer_args["start_tile_id"][node] = c_start_id;
        writer_args["dst_num_tiles"][node] = c_num_tiles_core;
        writer_args["dst_shard_width"][node] = c_current_shard_width;
        writer_args["D"][node] = cD;
        writer_args["N"][node] = cN;
        writer_args["C"][node] = cC;
        writer_args["Ht"][node] = cHt;
        writer_args["Wt"][node] = cWt;
        writer_args["cND"][node] = cND;

        compute_args["num_tiles"][node] = c_num_tiles_core;
        if (op_type == BinaryOpType::ISCLOSE) {
            compute_args["rtol_bits"][node] = isclose_rtol_bits;
            compute_args["atol_bits"][node] = isclose_atol_bits;
        }
    }
    m2::NodeRangeSet target_nodes(target_ranges);

    // --- Assemble the spec. ---
    m2::WorkUnitSpec wu{
        .name = "binary_ng_metal_v2_no_bcast",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = target_nodes,
    };

    m2::ProgramSpec spec{
        .name = "binary_ng_metal_v2_no_bcast",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = dfbs,
        .tensor_parameters =
            {{.unique_id = T_A, .spec = a.tensor_spec()},
             {.unique_id = T_B, .spec = b.tensor_spec()},
             {.unique_id = T_C, .spec = c.tensor_spec()}},
        .work_units = {wu},
    };

    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{.kernel = READER, .runtime_arg_values = std::move(reader_args)},
        m2::ProgramRunArgs::KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_args)},
        m2::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE, .runtime_arg_values = std::move(compute_args)},
    };
    // Bind each TensorParameter to its MeshTensor. For in-place add_ (a aliases c) both bindings
    // resolve to the same MeshTensor; the borrowed DFBs alias the same shard (permitted).
    run_params.tensor_args.emplace(T_A, m2::ProgramRunArgs::TensorArgument{a.mesh_tensor()});
    run_params.tensor_args.emplace(T_B, m2::ProgramRunArgs::TensorArgument{b.mesh_tensor()});
    run_params.tensor_args.emplace(T_C, m2::ProgramRunArgs::TensorArgument{c.mesh_tensor()});

    return ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
        .op_owned_tensors = {},
    };
}

}  // namespace

ttnn::device_operation::ProgramArtifacts BinaryNgDeviceOperation::ProgramFactoryMetalV2::create_program_artifacts(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, tensor_return_value_t& c) {
    return create_no_bcast_artifacts(operation_attributes, tensor_args, c);
}

}  // namespace ttnn::operations::experimental::quasar::binary_ng
