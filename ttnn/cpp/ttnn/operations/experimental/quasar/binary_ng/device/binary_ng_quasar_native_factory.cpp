// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar-native Metal 2.0 / DataflowBuffer program factory for binary_ng.
//
// Starts as a mechanical copy of binary_ng_metal_v2_factory.cpp so later commits read as a diff of
// what "Quasar-native" means (>2 of 6 user DM cores, >1 of 4 Tensix). Four intentional deviations at
// copy time: the out-of-class definition names ProgramFactoryQuasarNative (external linkage, else a
// duplicate symbol); the anonymous namespace is wrapped in CMAKE_UNIQUE_NAMESPACE (unity build);
// kernel paths point at kernels_qsr/ except kComputeIncludeCommon (a shared third tree); and a local
// ProgramSpecFactoryConcept static_assert so a declaration slip blames this file.
//
// Reachable only via matches_quasar_native_slice (on Quasar unless TTNN_QSR_NATIVE=0). Only that slice's kernels were
// copied (the no-broadcast reader, writer, and FPU and SFPU compute), so the bcast/scalar sources named below
// point at files absent from kernels_qsr/ -- deliberate; see the note above select_dfb_kernel_sources.
//
// --- inherited description of the original follows ---
//
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
//   - The decisive property is BORROWED vs NoC-READ, per operand. The anchor (select_borrow_anchor) is an L1
//     shard config or the L1 banks, on at least as many clusters as the legacy binary_ng uses, and it borrows
//     every operand that sits in it: the operands with exactly that shard config, or the L1-interleaved ones. An
//     operand that is not borrowed moves over the NoC in the anchor's tile order. A borrowed CB
//     `.buffer = tensor.buffer()` -> DataflowBufferSpec::borrowed_from, and the reader and writer move only a
//     shard's tail tiles. Every other operand's CB -> a real ring + a KernelSpec TensorBinding
//     (TensorAccessor(tensor::name), sharding-aware), and the buffer-address runtime arg the descriptor passed is
//     dropped (the binding injects the address). Placement follows the anchor. The descriptor borrows shards per
//     operand and never borrows an interleaved operand.
//   - Positional get_arg_val<>(idx) / get_compile_time_arg_val(idx) -> named get_arg(args::name). The
//     reader's 19 named args / writer's 9 named args are the descriptor's 21 / 11 arg vectors minus
//     the buffer-address args (reader src0@0 / src1@15, writer dst@0, writer trailing pad).
//   - KernelDescriptor::core_ranges + per-core dummy args on unused cores -> WorkUnitSpec::target_nodes
//     scoped to exactly the active cores (DFB places only on target_nodes, so no dummy args needed).
//   - The descriptor's `has_sharding` TensorAccessor CTA -> a HAS_SHARDING #define, set for a borrowed
//     shard; kernels_qsr/ does not read it.
//
// Deferred to the descriptor (rejected by the ORIGINAL's matches_metal_v2_slice; the native gate
// matches_quasar_native_slice is narrower still in op, dtype and broadcast, and it also admits DRAM
// shards, which the original sends to the descriptor -- see below): row-major (non-tile)
// layout, tensor-scalar (no input_tensor_b), where-op, quantization, and mixed lhs/rhs dtype. Every mix of
// interleaved, L1-sharded and DRAM-sharded operands IS handled: each operand is borrowed where the anchor
// holds it, and moves over the NoC through its sharding-aware TensorAccessor otherwise. A borrowed operand is
// tiled: L1-sharded (height/block/width) or L1-interleaved.

#include "binary_ng_device_operation.hpp"
#include "binary_ng_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/experimental/quasar/binary/common/binary_op_utils.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
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
// log_debug and log_warning. Included explicitly: this target is a unity build, so a missing include
// here still compiles.
#include <tt-logger/tt-logger.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::operations::experimental::quasar::binary_ng {

// Assert the concept HERE rather than relying on AllFactoriesValid: that concept is checked against
// the whole program_factory_t variant, so any slip in ProgramFactoryQuasarNative's declaration
// surfaces as a failure pointing at BinaryNgDeviceOperation (the operation) instead of at this
// factory. See ttnn/api/ttnn/operation_concepts.hpp.
static_assert(
    ttnn::device_operation::ProgramSpecFactoryConcept<BinaryNgDeviceOperation::ProgramFactoryQuasarNative>,
    "ProgramFactoryQuasarNative does not satisfy ProgramSpecFactoryConcept");

namespace {
// Most of this file's anonymous namespace is a name-for-name copy of
// binary_ng_metal_v2_factory.cpp's. This target is a unity build (TT_ENABLE_UNITY_BUILD in
// quasar/CMakeLists.txt), which concatenates translation units, so the shared names would
// collide. CMAKE_UNIQUE_NAMESPACE expands to a per-source-file unique id under a unity build and to
// a plain (harmless) identifier otherwise -- the same idiom binary_ng_program_factory.cpp uses.
namespace CMAKE_UNIQUE_NAMESPACE {

using ttnn::device_operation::ProgramArtifacts;

// Kernel sources for the no-broadcast DFB path (dual-mode: a borrowed operand publishes its resident shard
// or slice; any other operand reads/writes over the NoC via a tensor binding).
constexpr const char* kReaderDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/dataflow/reader_no_bcast_dfb.cpp";
constexpr const char* kWriterDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/dataflow/writer_no_bcast_dfb.cpp";
constexpr const char* kComputeFpuDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/compute/eltwise_binary_no_bcast_dfb.cpp";
constexpr const char* kComputeSfpuDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/compute/"
    "eltwise_binary_sfpu_no_bcast_dfb.cpp";
// Tensor-SCALAR DFB kernels. The writer becomes the producer of the RHS input DFB (in1) and fills it
// once with the packed scalar (coherent uncached-L1-alias store on Quasar DM cores); the reader
// produces in0 only; the compute waits on in1 once and reuses tile index 0. FPU handles bf16
// add/subtract; the SFPU compute file is added in a later task (referenced but not yet exercised).
constexpr const char* kReaderScalarDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/dataflow/reader_scalar_op_dfb.cpp";
constexpr const char* kWriterScalarDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/dataflow/writer_scalar_dfb.cpp";
constexpr const char* kComputeFpuScalarDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/compute/eltwise_binary_scalar_dfb.cpp";
constexpr const char* kComputeSfpuScalarDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/compute/"
    "eltwise_binary_sfpu_scalar_dfb.cpp";  // added in a later task; only the FPU path runs here
// Subtile-broadcast DFB kernels. The bcast reader delivers the partial tile (BCAST_LLK path, no fill);
// the row-bcast compute expands it via unary_bcast<ROW> through the intermediate llk_post DFB.
constexpr const char* kReaderBcastDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/dataflow/reader_bcast_dfb.cpp";
constexpr const char* kComputeFpuRowBcastDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/compute/"
    "eltwise_binary_row_bcast_dfb.cpp";
constexpr const char* kComputeSfpuRowBcastDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/compute/"
    "eltwise_binary_sfpu_row_bcast_dfb.cpp";
// COL subtile broadcast: a dedicated reader (the broadcast operand's column tile is read once per
// tile-row and reused across the row) + the col-bcast compute (unary_bcast<COL>, which lowers to the
// same MOVB2D LLK datacopy as ROW/SCALAR, keyed by COL's broadcast constants; freq=Wt reuse loop).
constexpr const char* kReaderColBcastDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/dataflow/reader_col_bcast_dfb.cpp";
constexpr const char* kComputeFpuColBcastDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/compute/"
    "eltwise_binary_col_bcast_dfb.cpp";
constexpr const char* kComputeSfpuColBcastDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/compute/"
    "eltwise_binary_sfpu_col_bcast_dfb.cpp";
// SCALAR subtile broadcast: a dedicated reader (the broadcast operand's single-element tile is read once
// per (N,C) slab and reused across the whole Ht * Wt block) + the scalar-bcast compute (unary_bcast<
// SCALAR>, which lowers to the same MOVB2D LLK datacopy as ROW/COL, keyed by SCALAR's broadcast constants;
// freq=Ht*Wt reuse loop).
constexpr const char* kReaderScalarBcastDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/dataflow/"
    "reader_scalar_bcast_dfb.cpp";
constexpr const char* kComputeFpuScalarBcastDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/compute/"
    "eltwise_binary_scalar_bcast_dfb.cpp";
constexpr const char* kComputeSfpuScalarBcastDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/compute/"
    "eltwise_binary_sfpu_scalar_bcast_dfb.cpp";
// MIXED subtile broadcast (ROW_A_COL_B / ROW_B_COL_A): a dedicated reader that software-fills the COL
// operand (FILL_TILE_WITH_FIRST_COLUMN, once per tile-row) and delivers the ROW operand as a raw partial
// tile (BCAST_LLK, per column) + the mixed compute (a HYBRID: unary_bcast<ROW> in compute for the ROW
// operand, reader software-fill for the COL operand; freq=Wt reuse loop). The COL-via-reader-fill split
// keeps compute at 2 LLK passes -- a deliberate reader/compute load-balance, NOT a fallback.
constexpr const char* kReaderRowColMixedBcastDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/dataflow/"
    "reader_row_col_mixed_bcast_dfb.cpp";
constexpr const char* kComputeFpuRowColBcastDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/compute/"
    "eltwise_binary_row_col_bcast_dfb.cpp";
constexpr const char* kComputeSfpuRowColBcastDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/compute/"
    "eltwise_binary_sfpu_row_col_bcast_dfb.cpp";

// The compute kernel includes eltwise_utils_common.hpp (in kernels/compute) and its DFB preprocess
// helper (in kernels_qsr/compute) by bare name; both directories go on the compute include path.
constexpr const char* kComputeIncludeCommon =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels/compute";
constexpr const char* kComputeIncludeDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/compute";

// Per-broadcast-type kernel-source seam (the DFB analogue of the descriptor's BinaryNgKernelConfig):
// it maps a SubtileBroadcastType to the reader/writer/compute kernel sources. NONE, ROW_A/ROW_B,
// COL_A/COL_B, and SCALAR_A/SCALAR_B are wired; the remaining (mixed ROW/COL) broadcast variants follow
// when their DFB kernels are added.
struct DfbKernelSources {
    const char* reader = nullptr;
    const char* writer = nullptr;
    const char* compute = nullptr;
};

// IMPORTANT for this (native) copy: only the NONE arm is reachable, with the FPU or the SFPU compute.
// matches_quasar_native_slice admits no subtile broadcast, and only the no-broadcast kernels were copied
// into kernels_qsr/, so every literal returned by the other arms names a file that does not exist in
// that tree. That is DELIBERATE: widening the gate without first copying those kernels then fails loudly
// at JIT naming exactly the file that was forgotten, instead of silently succeeding on non-native
// kernels from kernels_dfb/ under a factory called "native". create_no_bcast_artifacts asserts the same
// precondition on the host, before any JIT work. Hardening this file means deleting these branches.
DfbKernelSources select_dfb_kernel_sources(SubtileBroadcastType subtile_broadcast_type, bool is_sfpu) {
    switch (subtile_broadcast_type) {
        case SubtileBroadcastType::NONE:
            return DfbKernelSources{
                .reader = kReaderDfb,
                .writer = kWriterDfb,
                .compute = is_sfpu ? kComputeSfpuDfb : kComputeFpuDfb,
            };
        case SubtileBroadcastType::ROW_A:
        case SubtileBroadcastType::ROW_B:
            // ROW subtile broadcast: the bcast reader delivers the partial tile and the row-bcast
            // compute expands it via unary_bcast<ROW>, then runs the binary op -- FPU (add/subtract) or
            // SFPU (multiply/divide). UNREACHABLE here (the native gate rejects every broadcast type);
            // in the metal_v2 original its gate restricts this path to bf16.
            return DfbKernelSources{
                .reader = kReaderBcastDfb,
                .writer = kWriterDfb,
                .compute = is_sfpu ? kComputeSfpuRowBcastDfb : kComputeFpuRowBcastDfb,
            };
        case SubtileBroadcastType::COL_A:
        case SubtileBroadcastType::COL_B:
            // COL subtile broadcast: the col reader delivers the partial column tile (once per tile-row)
            // and the col-bcast compute expands it via unary_bcast<COL> (MOVB2D LLK datacopy, same as
            // ROW/SCALAR), reusing it across the row (freq=Wt), then runs the binary op -- FPU
            // (add/subtract) or SFPU (multiply/divide/maximum). UNREACHABLE here (the native gate rejects
            // every broadcast type); in the metal_v2 original its gate restricts this to bf16.
            return DfbKernelSources{
                .reader = kReaderColBcastDfb,
                .writer = kWriterDfb,
                .compute = is_sfpu ? kComputeSfpuColBcastDfb : kComputeFpuColBcastDfb,
            };
        case SubtileBroadcastType::SCALAR_A:
        case SubtileBroadcastType::SCALAR_B:
            // SCALAR subtile broadcast: the scalar reader delivers the single-element tile (once per
            // (N,C) slab) and the scalar-bcast compute expands it via unary_bcast<SCALAR> (MOVB2D LLK
            // datacopy, same as ROW/COL), reusing it across the whole slab (freq=Ht*Wt), then runs the
            // binary op -- FPU (add/subtract) or SFPU (multiply/divide/maximum). UNREACHABLE here (the
            // native gate rejects every broadcast type); in the metal_v2 original its gate restricts this
            // to bf16.
            return DfbKernelSources{
                .reader = kReaderScalarBcastDfb,
                .writer = kWriterDfb,
                .compute = is_sfpu ? kComputeSfpuScalarBcastDfb : kComputeFpuScalarBcastDfb,
            };
        case SubtileBroadcastType::ROW_A_COL_B:
        case SubtileBroadcastType::ROW_B_COL_A:
            // MIXED subtile broadcast: the mixed reader software-fills the COL operand and delivers the ROW
            // operand as a raw partial tile; the mixed compute expands the ROW operand via unary_bcast<ROW>
            // (through llk_post, freq=Wt reuse loop) and reads the reader-filled COL operand directly, then
            // runs the binary op -- FPU (add/subtract) or SFPU (multiply/divide/maximum/minimum). The COL
            // operand uses reader software-fill (NOT a compute unary_bcast<COL>) as a deliberate reader/compute
            // load-balance keeping compute at 2 LLK passes. On Quasar the reader fill uses a COHERENT store
            // (non-cacheable L1 alias) because the DM core's write-back L1 D$ is incoherent with the TL1 SRAM
            // the compute consumer reads -- a plain cacheable fill is invisible to the consumer and corrupts
            // the neighbor llk_post DFB (see reader_row_col_mixed_bcast_dfb.cpp). UNREACHABLE here (the
            // native gate rejects every broadcast type); in the metal_v2 original its gate restricts this
            // to bf16, and both orientations reach it (llk_post is the binary srcA for ROW_A_COL_B, srcB
            // for ROW_B_COL_A).
            return DfbKernelSources{
                .reader = kReaderRowColMixedBcastDfb,
                .writer = kWriterDfb,
                .compute = is_sfpu ? kComputeSfpuRowColBcastDfb : kComputeFpuRowColBcastDfb,
            };
    }
    TT_THROW(
        "binary_ng Quasar-native factory: unsupported subtile broadcast type {}",
        static_cast<int>(subtile_broadcast_type));
}

// --- Small pure helpers mirrored from the descriptor factory's anonymous namespace. They live in
// that .cpp's anonymous namespace (not the shared header), so they are reimplemented here to avoid
// touching the proven descriptor path. Keep behavior identical to binary_ng_program_factory.cpp. ---

// For rank > 5, dims are collapsed into a single dim. Mirrors extract_nD_dims in the descriptor factory.
uint32_t extract_nD_dims(const Tensor& x, int out_rank) {
    const auto& shape = x.logical_shape();
    uint32_t nD_dim = 1;
    if (out_rank >= 6 && shape.rank() >= 6) {
        // A lower-rank operand has no dims beyond its own rank; they broadcast as 1.
        const int rank = std::min<int>(out_rank, shape.rank());
        for (int i = -6; i >= -rank; --i) {
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

// Compute-kernel {freq, tile_start} for the subtile-broadcast reuse loop. Mirrors
// calculate_compute_kernel_args in binary_ng_program_factory.cpp (anonymous namespace there, so
// reimplemented here like the other small helpers). For COL it returns {Wt, start_tw}: the compute
// broadcasts the column tile once and reuses it across a full tile-row (freq = Wt), starting at this
// core's column offset (start_tw). ROW/NONE need no reuse ({1, 0}); SCALAR reuses across the whole
// HtWt block. Kept as a full switch so a later SCALAR/mixed wiring reuses it unchanged.
std::tuple<uint32_t, uint32_t> calculate_compute_kernel_args(
    SubtileBroadcastType broadcast_type, uint32_t start_tile_id, uint32_t Ht, uint32_t Wt) {
    const uint32_t start_t = start_tile_id % (Ht * Wt);
    const uint32_t start_tw = start_t % Wt;
    switch (broadcast_type) {
        case SubtileBroadcastType::NONE:
        case SubtileBroadcastType::ROW_A:
        case SubtileBroadcastType::ROW_B: return {1, 0};
        case SubtileBroadcastType::SCALAR_A:
        case SubtileBroadcastType::SCALAR_B: return {Ht * Wt, start_t};
        case SubtileBroadcastType::COL_A:
        case SubtileBroadcastType::COL_B:
        case SubtileBroadcastType::ROW_A_COL_B:
        case SubtileBroadcastType::ROW_B_COL_A: return {Wt, start_tw};
    }
    TT_THROW("binary_ng Quasar-native factory: unreachable subtile broadcast type in calculate_compute_kernel_args");
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

// The anchor of a program: a placement of the output tiles over cores in which some operands already sit, so
// that a core borrows them in place. A shard anchor is the memory config of an L1-sharded operand: core i of the
// allocator's core list computes shard i, and every operand with exactly that memory config is borrowed. The
// slice anchor is the L1 banks: the core of bank i computes the pages i, i + B, i + 2B, ... (B banks), and every
// L1-interleaved operand is borrowed. Without an anchor every operand moves over the NoC. An operand that is
// not borrowed moves in the anchor's tile order.
enum class AnchorKind { kNone, kShard, kSlice };

// Whether a program may send a borrowed shard's last tiles through the tail rings. The tail rings carry no
// activation chain, so a program with activations runs none.
enum class TailRings { kAllowed, kNone };

struct BorrowAnchor {
    AnchorKind kind = AnchorKind::kNone;
    std::array<bool, 3> borrowed{};  // a, b, c
    // Tiles per core, pad slots included: a shard, or the reservation of one bank.
    uint32_t tiles = 0;
    // In anchor order: shard i or bank i is on cores[i].
    std::vector<CoreCoord> cores;
    // The tensor in tiles, its height collapsed over every dim but the last; the operands share it.
    uint32_t height_tiles = 0;
    uint32_t width_tiles = 0;
    // A shard anchor's shard in tiles, and the shards across the tensor's width.
    uint32_t shard_height = 0;
    uint32_t shard_width = 0;
    uint32_t shards_per_width = 1;
    // The bank count of a slice anchor: the page step between a core's tiles.
    uint32_t page_stride = 0;
    // Some core computes a slot outside the tensor: an uneven shard, or a bank with one page fewer.
    bool has_pad_slots = false;

    uint32_t num_borrowed() const { return static_cast<uint32_t>(std::count(borrowed.begin(), borrowed.end(), true)); }
};

// The anchor of `holder`'s memory config when it is an L1 2D shard with one shard per core, on cores of the
// worker grid: a sub-device grid may leave a shard's core to another sub-device.
std::optional<BorrowAnchor> shard_anchor(
    const Tensor& holder, const std::array<const Tensor*, 3>& operands, const CoreRangeSet& worker_grid) {
    const MemoryConfig& memory_config = holder.memory_config();
    if (!memory_config.is_sharded() || memory_config.buffer_type() != BufferType::L1 ||
        !memory_config.shard_spec().has_value()) {
        return std::nullopt;
    }
    const std::optional<BufferDistributionSpec>& distribution = holder.buffer()->buffer_distribution_spec();
    if (!distribution.has_value() || distribution->max_num_shards_per_core() != 1) {
        return std::nullopt;
    }
    const std::vector<CoreCoord>& cores = distribution->cores_with_data();
    if (!std::all_of(cores.begin(), cores.end(), [&](const CoreCoord& core) { return worker_grid.contains(core); })) {
        return std::nullopt;
    }
    BorrowAnchor result{.kind = AnchorKind::kShard};
    // Full memory-config equality: one ShardSpec in another layout or orientation puts shards on other cores.
    for (size_t i = 0; i < operands.size(); ++i) {
        result.borrowed[i] = operands[i]->memory_config() == memory_config;
    }
    const ShardSpec& shard = *memory_config.shard_spec();
    const tt::tt_metal::Tile& tile = holder.tensor_spec().tile();
    const auto& shape = holder.padded_shape();
    result.width_tiles = static_cast<uint32_t>(shape[-1] / tile.get_width());
    result.height_tiles = static_cast<uint32_t>(shape.volume() / shape[-1] / tile.get_height());
    result.shard_height = tt::div_up(shard.shape[0], tile.get_height());
    result.shard_width = tt::div_up(shard.shape[1], tile.get_width());
    result.tiles = result.shard_height * result.shard_width;
    // The allocator numbers the shards row-major over the tensor and puts shard i on this list's core i.
    result.cores = cores;
    result.shards_per_width = tt::div_up(result.width_tiles, result.shard_width);
    result.has_pad_slots =
        result.height_tiles % result.shard_height != 0 || result.width_tiles % result.shard_width != 0;
    return result;
}

// The anchor of the L1 banks: bank i holds the pages i, i + B, ... of every L1-interleaved operand, back to back
// from the buffer's address. It needs one bank per core at offset 0, on exactly the cores of the worker grid; a
// sub-device grid is not.
std::optional<BorrowAnchor> slice_anchor(
    const std::array<const Tensor*, 3>& operands, const CoreRangeSet& worker_grid) {
    BorrowAnchor result{.kind = AnchorKind::kSlice};
    const Buffer* slice = nullptr;
    for (size_t i = 0; i < operands.size(); ++i) {
        const Tensor& operand = *operands[i];
        const MemoryConfig& memory_config = operand.memory_config();
        if (memory_config.memory_layout() != TensorMemoryLayout::INTERLEAVED ||
            memory_config.buffer_type() != BufferType::L1 || operand.layout() != Layout::TILE) {
            continue;
        }
        // One slot per tile: the page stride must be the tile size.
        const Buffer* buffer = operand.buffer();
        const uint32_t tile_bytes =
            operand.tensor_spec().tile().get_tile_size(datatype_to_dataformat_converter(operand.dtype()));
        if (buffer->aligned_page_size() != tile_bytes) {
            continue;
        }
        // The slices of one core hold the same pages only with one page count and one bank layout.
        if (slice != nullptr &&
            (buffer->num_pages() != slice->num_pages() || buffer->allocator() != slice->allocator())) {
            return std::nullopt;
        }
        slice = buffer;
        result.borrowed[i] = true;
    }
    if (slice == nullptr) {
        return std::nullopt;
    }
    const Allocator* allocator = slice->allocator();
    const uint32_t num_banks = allocator->get_num_banks(BufferType::L1);
    if (worker_grid.num_cores() != num_banks) {
        return std::nullopt;
    }
    std::set<CoreCoord> bank_cores;
    result.cores.reserve(num_banks);
    for (uint32_t bank = 0; bank < num_banks; ++bank) {
        const CoreCoord core = allocator->get_logical_core_from_bank_id(bank);
        if (allocator->get_bank_offset(BufferType::L1, bank) != 0 || !worker_grid.contains(core) ||
            !bank_cores.insert(core).second) {
            return std::nullopt;
        }
        result.cores.push_back(core);
    }
    // The bank's own reservation, which the runtime also checks the borrowed DFB against. A slice anchor reads the
    // tensor as one column of pages.
    result.tiles = static_cast<uint32_t>(slice->aligned_size_per_bank() / slice->aligned_page_size());
    result.height_tiles = static_cast<uint32_t>(slice->num_pages());
    result.width_tiles = 1;
    result.page_stride = num_banks;
    result.has_pad_slots = slice->num_pages() % num_banks != 0;
    return result;
}

// The borrow decision of the legacy binary_ng in eltwise/binary_ng (its is_native_L1_sharding and get_shard_specs, not
// the copies in this op) for two inputs of one shape without a broadcast: a sharded output, one memory config for a and
// b, no DRAM operand, every sharded input on the output's grid, and one shard spec on all three where a shard is
// uneven. The legacy binary_ng then runs on the output's shard grid.
bool legacy_borrows_output_shard(const Tensor& a, const Tensor& b, const Tensor& c) {
    const MemoryConfig& a_config = a.memory_config();
    const MemoryConfig& c_config = c.memory_config();
    if (!c_config.is_sharded() || !c_config.shard_spec().has_value() || a_config != b.memory_config()) {
        return false;
    }
    const auto in_dram = [](const MemoryConfig& config) { return config.buffer_type() == BufferType::DRAM; };
    const bool a_has_spec = a_config.is_sharded() && a_config.shard_spec().has_value();
    const bool one_spec = a_has_spec && *a_config.shard_spec() == *c_config.shard_spec();
    // b's memory config is a's, so b is uneven exactly when a is.
    if (is_uneven(a.tensor_spec())) {
        return !in_dram(a_config) && !in_dram(c_config) && one_spec;
    }
    if (in_dram(a_config) || in_dram(c_config)) {
        return false;
    }
    if (a_has_spec && a_config.shard_spec()->grid != c_config.shard_spec()->grid) {
        return false;
    }
    return !is_uneven(c.tensor_spec()) || one_spec;
}

// The number of clusters that the legacy binary_ng uses for this op: the clusters that hold the output's shards where
// it borrows them, else its split of the output tiles over the worker grid. No anchor runs on fewer. A cluster's cost
// per tile is about the same borrowed or over the NoC, so a program on fewer clusters takes longer.
uint32_t legacy_cluster_count(const Tensor& a, const Tensor& b, const Tensor& c, const CoreRangeSet& worker_grid) {
    if (legacy_borrows_output_shard(a, b, c)) {
        const std::optional<BufferDistributionSpec>& distribution = c.buffer()->buffer_distribution_spec();
        TT_FATAL(distribution.has_value(), "binary_ng Quasar-native: an L1-sharded output has no distribution spec");
        return static_cast<uint32_t>(distribution->cores_with_data().size());
    }
    const uint32_t tiles = static_cast<uint32_t>(c.physical_volume() / c.tensor_spec().tile().get_tile_hw());
    return std::min(tiles, static_cast<uint32_t>(worker_grid.num_cores()));
}

// The anchor that borrows the most operands, among the anchors on at least as many clusters as the legacy binary_ng
// uses (legacy_cluster_count). On a tie, the one that borrows the output, then the one on more clusters, then the
// earlier candidate: the output's shard config, a's, b's, the banks. A borrowed ring must divide by its max(producers,
// consumers); with all three borrowed, a shard's tail tiles go through the tail rings instead where `tail_rings`
// allows them, and a slice has no tail rings. A pad slot is computed and never read back; an operand that moves skips
// it. A program-cache hit keeps this decision. It reads only the hashed specs, activations and worker grid, the tuning,
// and the bank layout of the device, which never changes after the device opens.
BorrowAnchor select_borrow_anchor(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const CoreRangeSet& worker_grid,
    const NativeTuning& tuning,
    TailRings tail_rings) {
    const std::array<const Tensor*, 3> operands{&a, &b, &c};
    const uint32_t compute_threads = native_compute_threads(tuning);
    const uint32_t min_clusters = legacy_cluster_count(a, b, c, worker_grid);
    const auto eligible = [&](const BorrowAnchor& anchor) {
        if (anchor.cores.size() < min_clusters) {
            return false;
        }
        const bool inputs_move = !anchor.borrowed[0] || !anchor.borrowed[1];
        const bool output_moves = !anchor.borrowed[2];
        if (!inputs_move && !output_moves) {
            const bool tails_take_the_rest = anchor.kind == AnchorKind::kShard && tail_rings == TailRings::kAllowed;
            return tails_take_the_rest || anchor.tiles % compute_threads == 0;
        }
        const NativeConfig config = resolve_native_config(tuning, {.inputs = inputs_move, .output = output_moves});
        const uint32_t input_ring = std::max(config.reader_threads, compute_threads);
        const uint32_t output_ring = std::max(compute_threads, config.writer_threads);
        const bool inputs_divide = (!anchor.borrowed[0] && !anchor.borrowed[1]) || anchor.tiles % input_ring == 0;
        return inputs_divide && (output_moves || anchor.tiles % output_ring == 0);
    };
    const auto rank = [](const BorrowAnchor& anchor) {
        return std::tuple{anchor.num_borrowed(), anchor.borrowed[2], anchor.cores.size()};
    };
    std::optional<BorrowAnchor> best;
    for (std::optional<BorrowAnchor> candidate :
         {shard_anchor(c, operands, worker_grid),
          shard_anchor(a, operands, worker_grid),
          shard_anchor(b, operands, worker_grid),
          slice_anchor(operands, worker_grid)}) {
        if (!candidate.has_value() || candidate->num_borrowed() == 0 || !eligible(*candidate)) {
            continue;
        }
        if (!best.has_value() || rank(*candidate) > rank(*best)) {
            best = std::move(candidate);
        }
    }
    return std::move(best).value_or(BorrowAnchor{});
}

// Core i's tiles that lie in the tensor, as (rows, columns) of its anchor tiles: a shard is a block of
// shard_height x shard_width tiles, and a bank slice a column of pages. The last shard row or column, and a bank
// with one page fewer, hold pad slots past them.
std::pair<uint32_t, uint32_t> anchor_valid_bounds(const BorrowAnchor& anchor, uint32_t core_index) {
    switch (anchor.kind) {
        case AnchorKind::kShard: {
            const uint32_t first_row = (core_index / anchor.shards_per_width) * anchor.shard_height;
            const uint32_t first_col = (core_index % anchor.shards_per_width) * anchor.shard_width;
            return {
                std::min(anchor.shard_height, anchor.height_tiles - first_row),
                std::min(anchor.shard_width, anchor.width_tiles - first_col)};
        }
        case AnchorKind::kSlice: {
            const uint32_t pages = anchor.height_tiles;
            return {core_index < pages ? tt::div_up(pages - core_index, anchor.page_stride) : 0u, 1u};
        }
        case AnchorKind::kNone: break;
    }
    TT_THROW("binary_ng Quasar-native: a program without an anchor has no anchor bounds");
}

// The page of core i's tile 0 under the anchor: its shard's top-left tile, or its bank's first page.
uint32_t anchor_first_page(const BorrowAnchor& anchor, uint32_t core_index) {
    switch (anchor.kind) {
        case AnchorKind::kShard:
            return (core_index / anchor.shards_per_width) * anchor.shard_height * anchor.width_tiles +
                   (core_index % anchor.shards_per_width) * anchor.shard_width;
        case AnchorKind::kSlice: return core_index;
        case AnchorKind::kNone: break;
    }
    TT_THROW("binary_ng Quasar-native: a program without an anchor has no anchor order");
}

ProgramArtifacts create_no_bcast_artifacts(
    const BinaryNgDeviceOperation::operation_attributes_t& op,
    const BinaryNgDeviceOperation::tensor_args_t& tensor_args,
    Tensor& c) {
    const Tensor& a = tensor_args.input_tensor_a;
    // Tensor-scalar: no input_tensor_b. The writer fills the RHS DFB (in1) once from the packed scalar,
    // so `b` is absent -- every b-derived value below is guarded on is_scalar (b_dtype is derived, the
    // in1 DFB is a 1-tile non-borrowed ring, and the reader drops its in1 producer/binding).
    const bool is_scalar = !tensor_args.input_tensor_b.has_value();
    // NOTE vs the descriptor factory this was copied from: it also asserted
    // `is_scalar || input_tensor_b.has_value()` here. Since is_scalar IS !has_value(), that reads
    // `!x || x` -- vacuously true, so it was already dead there. The precondition below asserts
    // !is_scalar outright, which subsumes it. Dropped rather than carried, so the diff against the
    // descriptor factory stays a diff of real behaviour.
    TT_FATAL(
        !is_scalar || op.scalar.has_value(),
        "binary_ng Quasar-native no-bcast factory tensor-scalar path requires a scalar value");
    // The kernels_qsr/ tree holds ONLY the no-broadcast tensor-tensor sources, with the FPU and the SFPU compute.
    // Turning the dangling bcast/scalar path literals from a JIT "file not found" into a named host-side
    // precondition: fires before any JIT work and says what to do about it. This mirrors
    // matches_quasar_native_slice, which is the only admitting caller.
    TT_FATAL(
        op.subtile_broadcast_type == SubtileBroadcastType::NONE && !is_scalar,
        "ProgramFactoryQuasarNative supports only the no-broadcast tensor-tensor path; kernels_qsr/ "
        "holds no bcast/scalar sources. Copy them before widening matches_quasar_native_slice. Got "
        "subtile_broadcast_type={}, is_scalar={}",
        static_cast<int>(op.subtile_broadcast_type),
        is_scalar);
    const std::optional<Tensor>& b_opt = tensor_args.input_tensor_b;

    // The knobs: the same cached instance matches_quasar_native_slice gated on. Every knob left unset takes the
    // rule's value for what this program moves (resolve_native_config). The compute count is the same on
    // both paths, and the borrow decision needs it first.
    const NativeTuning& t = native_tuning();
    const uint32_t compute_threads = native_compute_threads(t);

    const bool is_sfpu = op.is_sfpu;
    const DataType input_dtype = op.input_dtype;
    const auto op_type = op.binary_op_type;

    Buffer* a_buffer = a.buffer();
    Buffer* b_buffer = is_scalar ? nullptr : b_opt->buffer();
    Buffer* c_buffer = c.buffer();
    TT_FATAL(
        a_buffer != nullptr && (is_scalar || b_buffer != nullptr) && c_buffer != nullptr,
        "binary_ng Quasar-native no-bcast factory requires allocated device buffers");

    // --- Dtypes / data formats / tile sizes (mirrors the descriptor factory). For a scalar there is no
    // `b`, so b_dtype is derived exactly as the descriptor does (binary_ng_program_factory.cpp): the
    // scalar is packed as bf16 for block-float a, else the sfpu path keeps a_dtype, else bf16. b_tile
    // borrows a's tile geometry (a single bf16 scalar tile). ---
    const DataType a_dtype = a.dtype();
    const DataType b_dtype =
        is_scalar ? ((op.is_sfpu && !is_block_float(a_dtype)) ? a_dtype : DataType::BFLOAT16) : b_opt->dtype();
    const DataType c_dtype = c.dtype();
    const tt::DataFormat a_df = datatype_to_dataformat_converter(a_dtype);
    const tt::DataFormat b_df = datatype_to_dataformat_converter(b_dtype);
    const tt::DataFormat c_df = datatype_to_dataformat_converter(c_dtype);
    // The DST mode, as in the descriptor factory. It sets the compute batch: a 32-bit DST section holds half the tiles.
    const bool fp32_dest_acc_en = c_df == tt::DataFormat::UInt32 || c_df == tt::DataFormat::Int32 ||
                                  c_df == tt::DataFormat::Float32 || a_df == tt::DataFormat::Float32 ||
                                  b_df == tt::DataFormat::Float32 ||
                                  (a_df == tt::DataFormat::Int32 && b_df == tt::DataFormat::Int32) ||
                                  (a_df == tt::DataFormat::UInt32 && b_df == tt::DataFormat::UInt32);
    const tt::tt_metal::Tile a_tile = a.tensor_spec().tile();
    const tt::tt_metal::Tile b_tile = is_scalar ? a_tile : b_opt->tensor_spec().tile();
    const tt::tt_metal::Tile c_tile = c.tensor_spec().tile();
    const uint32_t a_tile_bytes = static_cast<uint32_t>(a_tile.get_tile_size(a_df));
    const uint32_t b_tile_bytes = static_cast<uint32_t>(b_tile.get_tile_size(b_df));
    const uint32_t c_tile_bytes = static_cast<uint32_t>(c_tile.get_tile_size(c_df));

    // --- OpConfig + compute defines (mirrors the descriptor factory). ---
    const auto& op_params = op.op_params;
    OpConfig op_config = is_sfpu ? OpConfig(op_type, std::in_place_type<OpConfig::SfpuBinaryOp>, a_dtype, op_params)
                                 : OpConfig(op_type, std::in_place_type<OpConfig::FpuBinaryOp>, a_dtype, op_params);
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
        ttsl::SmallVector<unary::EltwiseUnaryWithParam> lhs_activations = op.lhs_activations;
        ttsl::SmallVector<unary::EltwiseUnaryWithParam> rhs_activations = op.rhs_activations;
        ttsl::SmallVector<unary::EltwiseUnaryWithParam> post_activations = op.post_activations;

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

    // --- Borrow vs NoC routing: the anchor (select_borrow_anchor) borrows what sits in its placement, and every
    // other operand moves through its own sharding-aware TensorAccessor in the anchor's tile order, or, without
    // an anchor, over a linear page-id walk, which is correct for any mix. ---
    // An activation as the compute kernels' tail blocks see it: a chain on a, on b or on the result, or PACK_RELU.
    const bool has_activations = has_lhs_act || has_rhs_act ||
                                 !compute_defines["PROCESS_POST_ACTIVATIONS(i)"].empty() ||
                                 compute_defines.contains("PACK_RELU");
    const BorrowAnchor anchor =
        select_borrow_anchor(a, *b_opt, c, op.worker_grid, t, has_activations ? TailRings::kNone : TailRings::kAllowed);
    const bool a_borrowed = anchor.borrowed[0];
    const bool b_borrowed = anchor.borrowed[1];
    const bool c_borrowed = anchor.borrowed[2];
    const bool borrow_all = a_borrowed && b_borrowed && c_borrowed;
    // Tiles of every core under the anchor, pad slots included: a shard, or a bank's slice.
    const uint32_t borrowed_tiles = anchor.tiles;
    const uint32_t c_num_tiles = c.physical_volume() / c.tensor_spec().tile().get_tile_hw();

    // The values this program runs with. Where both inputs are borrowed the reader runs one thread, and where
    // the output is borrowed the writer does; a borrowed ring's stride is then max(R, C) or max(C, W), which the
    // anchor's tile count divides. The rule does not batch beside a DRAM operand.
    const auto in_dram = [](const Tensor& tensor) { return tensor.memory_config().buffer_type() == BufferType::DRAM; };
    const bool any_dram = in_dram(a) || (!is_scalar && in_dram(*b_opt)) || in_dram(c);
    const NativeConfig cfg = resolve_native_config(
        t,
        {.inputs = !a_borrowed || !b_borrowed,
         .output = !c_borrowed,
         .dram = any_dram,
         .sfpu = is_sfpu,
         .dst_32bit = fp32_dest_acc_en});
    const uint32_t reader_threads = cfg.reader_threads;
    const uint32_t writer_threads = cfg.writer_threads;
    // A borrowed ring must divide by the compute count, so, until the DFB can give each tile counter its own
    // capacity, the tiles past its largest multiple go through three owned tail rings of one entry per compute
    // thread. A shard smaller than the count has no borrowed ring: all of it uses the tail rings. The tail rings
    // run only with all three operands borrowed and no activation; otherwise the anchor is borrowed only when the
    // count divides.
    // The tail rings are temporary. When the DFB gives each tile counter its own capacity, remove them: the tail DFBs,
    // TAIL_TILES, HAS_MAIN_RING and the tail blocks of the reader, the writer and both compute kernels. The reader's
    // publish and the compute already take uneven shares. The divisibility checks in select_borrow_anchor come from the
    // same DFB rule.
    const uint32_t tail_tiles = borrow_all ? borrowed_tiles % compute_threads : 0u;
    const bool has_main_ring = !borrow_all || borrowed_tiles > tail_tiles;
    if (anchor.kind != AnchorKind::kNone) {
        log_debug(
            tt::LogOp,
            "binary_ng Quasar-native: the {} anchor borrows a={} b={} c={}, {} tiles per core on {} cores, R={} C={} "
            "W={}, {} in the tail rings",
            anchor.kind == AnchorKind::kShard ? "shard" : "slice",
            a_borrowed,
            b_borrowed,
            c_borrowed,
            borrowed_tiles,
            anchor.cores.size(),
            reader_threads,
            compute_threads,
            writer_threads,
            tail_tiles);
    }

    // HAS_SHARDING mirrors the descriptor's define for a borrowed shard; kernels_qsr/ does not read it.
    const bool has_sharding = anchor.kind == AnchorKind::kShard;

    // --- num_tiles_per_cycle: tiles per tile_regs_acquire, which the DST capacity bounds; the rule takes all
    // of it where every tile of a batch comes out right. Neither batch has to divide a thread's share: the kernels
    // run a short last batch. A borrowed ring is the shard or slice, walked once, so a batch there never wraps, and
    // the ring holds the thread's whole share. Above ring stride 1 an FPU-kernel compute batch gives a wrong output;
    // see the warning below. ---
    // in0/in1 endpoints are (R, C); out endpoints are (C, W).
    const uint32_t in_producers_consumers = std::max(reader_threads, compute_threads);
    const uint32_t out_producers_consumers = std::max(compute_threads, writer_threads);
    // Every core processes borrowed_tiles, including a partial end core under an uneven shard and a bank
    // with one real page fewer: the buffer allocates the full count on every core, so computing on the pad
    // slots is in bounds and never reaches the logical output. The borrowed rings hold the first
    // c_main_tiles of it, and the tail rings the rest.
    const uint32_t c_main_tiles = borrowed_tiles - tail_tiles;
    const uint32_t num_tiles_per_cycle = cfg.tiles_per_cycle;
    // A batch must fit one DST section, so a larger set batch is refused.
    const uint32_t max_tiles_per_cycle = native_max_tiles_per_cycle(is_sfpu, fp32_dest_acc_en);
    TT_FATAL(
        num_tiles_per_cycle <= max_tiles_per_cycle,
        "binary_ng Quasar-native: tiles_per_cycle={} exceeds {}, the most the {} kernel runs in a {}-bit DST",
        describe_native_value(t.tiles_per_cycle, num_tiles_per_cycle),
        max_tiles_per_cycle,
        is_sfpu ? "SFPU" : "FPU",
        fp32_dest_acc_en ? 32 : 16);
    // Capacity rules for the NoC rings; the borrowed path builds none. reserve_back(n) and wait_front(n)
    // cannot complete if n exceeds a thread's capacity, entries_per_thread, and the ring must hold two batches
    // for double buffering, so check here rather than let the op hang. A batch occupies n consecutive slots
    // from the write pointer and the ring wraps only AFTER the batch, so a batch that straddles the counter's
    // end writes past it: the depth must be a multiple of the batch. The rule's values pass; a set knob may
    // not, so each message names every value's source.
    if (!borrow_all) {
        const auto check_batch = [&](std::string_view label, const NativeKnob& knob, uint32_t batch) {
            TT_FATAL(
                cfg.entries_per_thread >= 2 * batch,
                "binary_ng Quasar-native: {}={} needs entries_per_thread >= {} for double buffering, got "
                "entries_per_thread={}",
                label,
                describe_native_value(knob, batch),
                2 * batch,
                describe_native_value(t.entries_per_thread, cfg.entries_per_thread));
            TT_FATAL(
                cfg.entries_per_thread % batch == 0,
                "binary_ng Quasar-native: {}={} must divide entries_per_thread={}, or a batch straddles the ring end",
                label,
                describe_native_value(knob, batch),
                describe_native_value(t.entries_per_thread, cfg.entries_per_thread));
        };
        check_batch("tiles_per_cycle", t.tiles_per_cycle, cfg.tiles_per_cycle);
        check_batch("dm_batch", t.dm_batch, cfg.dm_batch);
        // A compute thread beside more reader or writer threads owns several tile counters of that ring, and a
        // compute batch reads or fills one counter: it takes the wrong tiles, or waits for tiles that never
        // come. The rule gives 1 there; a set batch must be refused.
        TT_FATAL(
            cfg.tiles_per_cycle == 1 || (reader_threads <= compute_threads && writer_threads <= compute_threads),
            "binary_ng Quasar-native: tiles_per_cycle={} needs R <= C and W <= C, so that each compute thread "
            "owns one tile counter of each ring, got R={} C={} W={}",
            describe_native_value(t.tiles_per_cycle, cfg.tiles_per_cycle),
            describe_native_value(t.reader_threads, reader_threads),
            describe_native_value(t.compute_threads, compute_threads),
            describe_native_value(t.writer_threads, writer_threads));
    }

    // --- DFB names. compute uses pre_lhs/pre_rhs/out (+post_lhs/post_rhs when activations); reader/
    // writer use in0/in1/out. The SAME DFBSpecs (c_0/c_1/c_2) are bound under per-kernel accessor
    // names. ---
    const m2::TensorParamName T_A{"binary_ng_a"};
    const m2::TensorParamName T_B{"binary_ng_b"};
    const m2::TensorParamName T_C{"binary_ng_c"};
    const m2::DFBSpecName IN0{"binary_ng_in0_dfb"};                    // CBIndex::c_0
    const m2::DFBSpecName IN1{"binary_ng_in1_dfb"};                    // CBIndex::c_1
    const m2::DFBSpecName OUT{"binary_ng_out_dfb"};                    // CBIndex::c_2
    const m2::DFBSpecName POST_LHS{"binary_ng_post_lhs_dfb"};          // CBIndex::c_3 (LHS activations)
    const m2::DFBSpecName POST_RHS{"binary_ng_post_rhs_dfb"};          // CBIndex::c_4 (RHS activations)
    const m2::DFBSpecName LLK_POST_LHS{"binary_ng_llk_post_lhs_dfb"};  // CBIndex::c_5 (unary_bcast out, LHS bcast)
    const m2::DFBSpecName LLK_POST_RHS{"binary_ng_llk_post_rhs_dfb"};  // CBIndex::c_6 (unary_bcast out, RHS bcast)
    const m2::DFBSpecName IN0_TAIL{"binary_ng_in0_tail_dfb"};          // borrowed shard tiles past the ring
    const m2::DFBSpecName IN1_TAIL{"binary_ng_in1_tail_dfb"};
    const m2::DFBSpecName OUT_TAIL{"binary_ng_out_tail_dfb"};
    const m2::KernelSpecName READER{"binary_ng_reader"};
    const m2::KernelSpecName WRITER{"binary_ng_writer"};
    const m2::KernelSpecName COMPUTE{"binary_ng_compute"};

    // Per-broadcast-type kernel-source selection (the descriptor's BinaryNgKernelConfig analogue): the
    // no-broadcast (NONE) triple, the ROW subtile-broadcast reader/compute for ROW_A / ROW_B, the COL
    // subtile-broadcast reader/compute for COL_A / COL_B, and the SCALAR subtile-broadcast reader/compute
    // for SCALAR_A / SCALAR_B.
    // Tensor-scalar selects the in0-only reader + scalar-fill writer + wait-once/reuse-0 compute (FPU
    // for bf16 add/subtract; the SFPU scalar compute file is added in a later task). Tensor-tensor
    // dispatches per subtile-broadcast type as before.
    const DfbKernelSources kernel_sources =
        is_scalar ? DfbKernelSources{
                        .reader = kReaderScalarDfb,
                        .writer = kWriterScalarDfb,
                        .compute = is_sfpu ? kComputeSfpuScalarDfb : kComputeFpuScalarDfb,
                    }
                  : select_dfb_kernel_sources(op.subtile_broadcast_type, is_sfpu);

    // bcast_lhs / bcast_rhs mark which operand owns the compute unary_bcast -> llk_post intermediate
    // (c_5 / c_6). Single-operand: the broadcast operand itself (ROW_A / COL_A / SCALAR_A -> a; ROW_B /
    // COL_B / SCALAR_B -> b). MIXED (ROW_A_COL_B / ROW_B_COL_A): the ROW operand -- the only one expanded
    // in compute (unary_bcast<ROW>); the COL operand is reader-software-filled and owns no llk_post. So
    // ROW_A_COL_B (a is ROW) -> bcast_lhs (c_5); ROW_B_COL_A (b is ROW) -> bcast_rhs (c_6). Exactly one
    // holds on the bcast path; both false on no-broadcast. is_mixed_bcast distinguishes the mixed types
    // (their define set differs -- see below). uses_tile_freq_reuse selects the COL/SCALAR/MIXED
    // freq/tile_start reuse loop (compute args); ROW's compute is a flat per-tile loop (freq always 1).
    const bool is_mixed_bcast = op.subtile_broadcast_type == SubtileBroadcastType::ROW_A_COL_B ||
                                op.subtile_broadcast_type == SubtileBroadcastType::ROW_B_COL_A;
    const bool bcast_lhs = op.subtile_broadcast_type == SubtileBroadcastType::ROW_A ||
                           op.subtile_broadcast_type == SubtileBroadcastType::COL_A ||
                           op.subtile_broadcast_type == SubtileBroadcastType::SCALAR_A ||
                           op.subtile_broadcast_type == SubtileBroadcastType::ROW_A_COL_B;
    const bool bcast_rhs = op.subtile_broadcast_type == SubtileBroadcastType::ROW_B ||
                           op.subtile_broadcast_type == SubtileBroadcastType::COL_B ||
                           op.subtile_broadcast_type == SubtileBroadcastType::SCALAR_B ||
                           op.subtile_broadcast_type == SubtileBroadcastType::ROW_B_COL_A;
    const bool uses_tile_freq_reuse = op.subtile_broadcast_type == SubtileBroadcastType::COL_A ||
                                      op.subtile_broadcast_type == SubtileBroadcastType::COL_B ||
                                      op.subtile_broadcast_type == SubtileBroadcastType::SCALAR_A ||
                                      op.subtile_broadcast_type == SubtileBroadcastType::SCALAR_B || is_mixed_bcast;

    // The bcast compute processes exactly ONE tile per cycle (unary_bcast -> pack -> binary op on DST
    // index 0) while advancing every DFB by num_tiles_per_cycle; that is only correct at
    // num_tiles_per_cycle == 1 (the NoC-read path this task wires). A future borrowed
    // or multi-tile broadcast path MUST revisit the kernel's per-tile loop before relaxing this.
    TT_FATAL(
        !(bcast_lhs || bcast_rhs) || num_tiles_per_cycle == 1,
        "binary_ng Quasar-native factory: subtile broadcast requires num_tiles_per_cycle == 1, got {}",
        num_tiles_per_cycle);

    // --- DataflowBuffers (mirrors the descriptor factory's CB block). A BORROWED operand backs the DFB
    // with its resident L1 shard or slice (num_entries == its borrowed part, borrowed_from set); any
    // NoC-read operand is a derived-depth ring filled over the NoC.
    // post_lhs/post_rhs exist only when that operand has activations; their format is the op_has_exp
    // Float16_b intermediate on the FPU path, else the operand's own format. ---
    // capacity = num_entries / max(producers, consumers) is what reaches the credit register, so the
    // knob is PER-THREAD and num_entries is derived. A global depth is meaningless: depth 2 is illegal
    // at R=4 (must divide by max(R,C)) and depth 4 at R=C=4 gives capacity 1, i.e. no double buffering.
    const uint32_t in_entries = cfg.entries_per_thread * in_producers_consumers;
    const uint32_t out_entries = cfg.entries_per_thread * out_producers_consumers;
    // threshold and num_entries_per_txn_id are uint8_t with NO guard, and the only TT_FATAL there checks
    // divisibility, which 0 % anything == 0 passes. num_txn_ids falls back to 1 when no n in [2,4]
    // divides, so the cliff is at 255 entries -- not 510 as a "num_txn_ids >= 2" assumption would suggest.
    // Only a NoC operand gets a ring of these sizes.
    TT_FATAL(
        ((a_borrowed && b_borrowed) || in_entries <= 255) && (c_borrowed || out_entries <= 255),
        "binary_ng Quasar-native: num_entries must be <= 255 (unguarded uint8_t threshold), got in={} out={} from "
        "entries_per_thread={} x max(R,C)={} and x max(C,W)={}",
        in_entries,
        out_entries,
        describe_native_value(t.entries_per_thread, cfg.entries_per_thread),
        in_producers_consumers,
        out_producers_consumers);
    // Above ring stride 1 the FPU kernel's binary unpack reads tile p of a batch p entries on, not p strides on, so
    // only one tile of each batch is right; the SFPU kernel copies through the single-operand unpack, which strides.
    // The rule never batches there, so only a set TTNN_QSR_TILES_PER_CYCLE reaches this.
    if (num_tiles_per_cycle > 1 && !native_compute_batch_exact(cfg, is_sfpu)) {
        log_warning(
            tt::LogOp,
            "binary_ng Quasar-native: tiles_per_cycle={} above ring stride 1 (in: max(R,C)={}, out: "
            "max(C,W)={}) runs, but its output is WRONG: only one tile of each batch is right",
            describe_native_value(t.tiles_per_cycle, num_tiles_per_cycle),
            in_producers_consumers,
            out_producers_consumers);
    }
    // Tile counters each DM role round-robins, mirroring the DFB's calculate_num_tile_counters for a
    // STRIDED ring: the reader produces into the in DFB and the writer consumes from the out DFB, so
    // each owns the ratio when compute is the wider side. A batch strides within one counter by it.
    const uint32_t reader_num_tcs = compute_threads >= reader_threads ? compute_threads / reader_threads : 1u;
    const uint32_t writer_num_tcs = compute_threads >= writer_threads ? compute_threads / writer_threads : 1u;

    // A borrowed ring is the first tiles of the anchor's shard or slice, which divide by the ring's
    // max(producers, consumers) as the DFB host asserts, so the reader's publish fills each counter exactly.
    // Every borrowed operand holds the anchor's tile count.
    const uint32_t a_entries = a_borrowed ? c_main_tiles : in_entries;
    // Scalar in1 is a single writer-filled tile (never borrowed); otherwise the borrowed shard or slice, or
    // the derived NoC ring.
    const uint32_t b_entries = is_scalar ? 1u : (b_borrowed ? c_main_tiles : in_entries);
    const uint32_t c_entries = c_borrowed ? c_main_tiles : out_entries;

    std::vector<m2::DataflowBufferSpec> dfbs;
    constexpr uint32_t num_main_dfbs = 3;  // in0, in1, out
    constexpr uint32_t num_tail_dfbs = 3;  // in0_tail, in1_tail, out_tail
    dfbs.reserve(
        (has_main_ring ? num_main_dfbs : 0) + (tail_tiles > 0 ? num_tail_dfbs : 0) + (has_lhs_act ? 1 : 0) +
        (has_rhs_act ? 1 : 0));
    if (has_main_ring) {
        dfbs.push_back(
            make_dfb(IN0, a_tile_bytes, a_entries, a_df, a_tile, a_borrowed ? std::optional{T_A} : std::nullopt));
        dfbs.push_back(
            make_dfb(IN1, b_tile_bytes, b_entries, b_df, b_tile, b_borrowed ? std::optional{T_B} : std::nullopt));
        dfbs.push_back(
            make_dfb(OUT, c_tile_bytes, c_entries, c_df, c_tile, c_borrowed ? std::optional{T_C} : std::nullopt));
    }
    // One entry per compute thread, so each thread takes exactly one; the reader copies the shard's tail
    // tiles in, and entries past tail_tiles stay padding that the writer drops.
    if (tail_tiles > 0) {
        dfbs.push_back(make_dfb(IN0_TAIL, a_tile_bytes, compute_threads, a_df, a_tile, std::nullopt));
        dfbs.push_back(make_dfb(IN1_TAIL, b_tile_bytes, compute_threads, b_df, b_tile, std::nullopt));
        dfbs.push_back(make_dfb(OUT_TAIL, c_tile_bytes, compute_threads, c_df, c_tile, std::nullopt));
    }

    const tt::DataFormat a_inter_df = is_sfpu ? a_df : (op_has_exp ? tt::DataFormat::Float16_b : a_df);
    const tt::DataFormat b_inter_df = is_sfpu ? b_df : (op_has_exp ? tt::DataFormat::Float16_b : b_df);
    // post_lhs/post_rhs intermediate rings (c_3/c_4), allocated only when that operand has an activation chain.
    // Each compute thread produces (PREPROCESS) and consumes (binary op) its own tile counter of the ring, in
    // program order, so the ring holds one batch per compute thread.
    if (has_lhs_act) {
        dfbs.push_back(make_dfb(
            POST_LHS,
            static_cast<uint32_t>(a_tile.get_tile_size(a_inter_df)),
            compute_threads * num_tiles_per_cycle,
            a_inter_df,
            a_tile,
            std::nullopt));
    }
    if (has_rhs_act) {
        dfbs.push_back(make_dfb(
            POST_RHS,
            static_cast<uint32_t>(b_tile.get_tile_size(b_inter_df)),
            compute_threads * num_tiles_per_cycle,
            b_inter_df,
            b_tile,
            std::nullopt));
    }

    // llk_post intermediate DFB for the subtile broadcast (c_5 for LHS bcast, c_6 for RHS bcast): each compute
    // thread produces (unary_bcast<ROW|COL|SCALAR> -> pack) and consumes (binary op) its own tile counter of it in
    // program order, so the ring holds one batch per compute thread, as the POST_LHS/POST_RHS rings do. (COL reuses
    // the single expanded tile across a tile-row, SCALAR across the whole (N,C) slab: produced once, read freq
    // times, popped once -- still one live entry.) Format is the broadcast operand's own format (bf16 on this path;
    // op_has_exp never reaches the LLK bcast path). Allocated only for the operand being broadcast.
    if (bcast_lhs) {
        dfbs.push_back(make_dfb(
            LLK_POST_LHS,
            static_cast<uint32_t>(a_tile.get_tile_size(a_inter_df)),
            compute_threads * num_tiles_per_cycle,
            a_inter_df,
            a_tile,
            std::nullopt));
    }
    if (bcast_rhs) {
        dfbs.push_back(make_dfb(
            LLK_POST_RHS,
            static_cast<uint32_t>(b_tile.get_tile_size(b_inter_df)),
            compute_threads * num_tiles_per_cycle,
            b_inter_df,
            b_tile,
            std::nullopt));
    }

    // The output's dims in tiles. The anchor's walk and placement derive from its width.
    const auto [cD, cN, cC, cHt, cWt] = get_shape_dims(c);

    // --- Dataflow defines. SRC_SHARDED[_B]/DST_SHARDED select the BORROWED (publish/drain a resident
    // shard or L1-interleaved slice, no NoC) vs NoC-read code path per operand; for a slice they mean
    // borrowed, not sharded. HAS_SHARDING mirrors the descriptor's define; kernels_qsr/ does not read it. ---
    std::map<std::string, std::string> reader_defines = make_dataflow_defines(a_dtype, b_dtype);
    reader_defines["SRC_SHARDED"] = a_borrowed ? "1" : "0";
    reader_defines["SRC_SHARDED_B"] = b_borrowed ? "1" : "0";
    reader_defines["HAS_SHARDING"] = has_sharding ? "1" : "0";
    std::map<std::string, std::string> writer_defines = make_dataflow_defines(b_dtype);
    writer_defines["DST_SHARDED"] = c_borrowed ? "1" : "0";
    writer_defines["HAS_SHARDING"] = has_sharding ? "1" : "0";

    // An operand that moves beside borrowed ones walks the anchor's tile order. Tile k of a core is page
    // start_tile_id + (k / w) * Wt + k % w in a shard, with w = SHARD_WALK_WIDTH, and
    // start_tile_id + k * BANK_WALK_STRIDE in the bank walk. Otherwise both are 0, and the walk is linear: an
    // even height shard, and the NoC path. With pad slots, WALK_PADS checks each tile against the core's
    // valid_rows and valid_cols, and a height shard takes the shard walk, which holds the check. The kernels
    // take w from this define, not from the dst_shard_width arg.
    const bool anchor_walk = anchor.kind != AnchorKind::kNone && !borrow_all;
    const bool walk_pads = anchor_walk && anchor.has_pad_slots;
    const uint32_t shard_walk_width =
        (anchor_walk && anchor.kind == AnchorKind::kShard && (anchor.shard_width < cWt || walk_pads))
            ? anchor.shard_width
            : 0u;
    const uint32_t bank_walk_stride = (anchor_walk && anchor.kind == AnchorKind::kSlice) ? anchor.page_stride : 0u;
    for (auto* defines : {&reader_defines, &writer_defines}) {
        (*defines)["SHARD_WALK_WIDTH"] = std::to_string(shard_walk_width);
        (*defines)["BANK_WALK_STRIDE"] = std::to_string(bank_walk_stride);
        (*defines)["WALK_PADS"] = walk_pads ? "1" : "0";
    }

    // TAIL_TILES > 0 compiles in the tail-ring code of all three kernels; HAS_MAIN_RING = 0 compiles out
    // every use of the in0/in1/out DFBs, which then do not exist.
    for (auto* defines : {&reader_defines, &writer_defines, &compute_defines}) {
        (*defines)["TAIL_TILES"] = std::to_string(tail_tiles);
        (*defines)["HAS_MAIN_RING"] = has_main_ring ? "1" : "0";
    }

    // Subtile-broadcast defines. Set only on the bcast path so the no-broadcast define set is untouched.
    //
    // Single-operand (ROW / COL / SCALAR): the reader keys the partial-tile walk off SRC_BCAST /
    // SRC_BCAST_B; BCAST_LLK=1 means it delivers the partial (or, for SCALAR, single-element) tile (no
    // FILL_TILE), leaving the broadcast to the compute's unary_bcast<ROW|COL|SCALAR>. The compute selects
    // its c_5/c_6 llk_post mapping off SRC_BCAST[_B]. BCAST_INPUT (0 = LHS bcast, 1 = RHS bcast) mirrors the
    // descriptor's compute define (COL/SCALAR preprocess the broadcast operand once and stream the other).
    //
    // MIXED (ROW_A_COL_B / ROW_B_COL_A) is a HYBRID and uses a DIFFERENT define set: the reader keys off
    // SRC_BCAST_COL (which operand is the reader-software-filled COL operand -- FILL_TILE_WITH_FIRST_COLUMN,
    // UNCONDITIONAL) and SRC_BCAST_ROW_B (which is the raw-partial ROW operand), with BCAST_LLK=1 keeping
    // the ROW operand a partial tile (the compute's unary_bcast<ROW> expands it; only the ROW fill is
    // BCAST_LLK-gated). The mixed compute selects the COL (once-per-row BCAST_OP) vs ROW (streamed
    // OTHER_OP) operand off BCAST_INPUT (1 = ROW_A_COL_B, 0 = ROW_B_COL_A) and reads the ROW operand's
    // llk_post via the dfb::llk_post accessor (bound to c_5 / c_6 by bcast_lhs / bcast_rhs above), so it
    // needs no SRC_BCAST[_B]. SRC_BCAST_COL and SRC_BCAST_ROW_B carry the same value (a is COL <=> b is
    // ROW, and vice versa), matching the descriptor factory (binary_ng_program_factory.cpp).
    if (is_mixed_bcast) {
        const std::string src_bcast_col = op.subtile_broadcast_type == SubtileBroadcastType::ROW_B_COL_A ? "1" : "0";
        reader_defines["SRC_BCAST_COL"] = src_bcast_col;
        reader_defines["SRC_BCAST_ROW_B"] = src_bcast_col;
        reader_defines["BCAST_LLK"] = "1";
        compute_defines["BCAST_INPUT"] = op.subtile_broadcast_type == SubtileBroadcastType::ROW_A_COL_B ? "1" : "0";
    } else if (bcast_lhs || bcast_rhs) {
        const std::string src_bcast = bcast_lhs ? "1" : "0";
        const std::string src_bcast_b = bcast_rhs ? "1" : "0";
        const std::string bcast_input = bcast_lhs ? "0" : "1";
        for (auto* defines : {&reader_defines, &compute_defines}) {
            (*defines)["SRC_BCAST"] = src_bcast;
            (*defines)["SRC_BCAST_B"] = src_bcast_b;
            (*defines)["BCAST_LLK"] = "1";
            (*defines)["BCAST_INPUT"] = bcast_input;
        }
    }

    // --- Compute config (mirrors the descriptor factory). ---

    // unpack_modes: the descriptor sets UnpackToDest on all SFPU consumer CBs, but that is inert for
    // non-Float32 entries (and the DFB validator rejects UnpackToDest on a 32-bit format unless
    // enable_32_bit_dest is true). So set it only on compute-consumer DFBs whose format is Float32
    // (which forces fp32_dest_acc_en true). A Float32 FPU consumer must still carry an explicit entry,
    // which is UnpackToSrc. compute consumers: in0(pre_lhs), in1(pre_rhs), post_lhs, post_rhs.
    m2::ComputeHardwareConfig::ComputeUnpackModes unpack_modes;
    auto set_unpack_mode = [&](const m2::DFBSpecName& dfb, tt::DataFormat df) {
        if (df == tt::DataFormat::Float32) {
            unpack_modes.emplace(dfb, is_sfpu ? UnpackMode::UnpackToDest : UnpackMode::UnpackToSrc);
        }
    };
    if (has_main_ring) {
        set_unpack_mode(IN0, a_df);
        set_unpack_mode(IN1, b_df);
    }
    if (tail_tiles > 0) {
        set_unpack_mode(IN0_TAIL, a_df);
        set_unpack_mode(IN1_TAIL, b_df);
    }
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

    // Reader: publishes a borrowed shard or slice (borrowed operand) or reads it over the NoC (via
    // TensorAccessor(tensor::in0/in1)). A borrowed operand borrows via the DFB and needs a tensor binding
    // only for the tail copy, whose LocalTensorAccessor gives the shard's L1 base; every NoC-read operand is
    // bound — including a sharded input with a different shard spec, whose accessor is sharding-aware.
    m2::Group<m2::TensorBinding> reader_tensor_bindings;
    if (!a_borrowed || tail_tiles > 0) {
        reader_tensor_bindings.push_back(m2::TensorBinding{T_A, "in0"});
    }
    // Scalar: the reader produces in0 only; in1 is produced by the writer (the scalar fill), so it gets
    // neither a "in1" DFB producer binding nor a T_B tensor binding.
    if (!is_scalar && (!b_borrowed || tail_tiles > 0)) {
        reader_tensor_bindings.push_back(m2::TensorBinding{T_B, "in1"});
    }
    m2::Group<m2::DFBBinding> reader_dfb_bindings;
    if (has_main_ring) {
        reader_dfb_bindings.push_back(m2::ProducerOf(IN0, "in0"));
        if (!is_scalar) {
            reader_dfb_bindings.push_back(m2::ProducerOf(IN1, "in1"));
        }
    }
    if (tail_tiles > 0) {
        reader_dfb_bindings.push_back(m2::ProducerOf(IN0_TAIL, "in0_tail"));
        reader_dfb_bindings.push_back(m2::ProducerOf(IN1_TAIL, "in1_tail"));
    }
    m2::Group<std::string> reader_rt_names = {
        "start_tile_id",
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
        "src_num_tiles_b"};
    // The bounds of a walk past pad slots, last, so that the other args keep their places.
    if (walk_pads) {
        reader_rt_names.push_back("valid_rows");
        reader_rt_names.push_back("valid_cols");
    }
    m2::KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path(kernel_sources.reader),
        .num_threads = reader_threads,
        .compiler_options = {.defines = reader_defines_tbl},
        .dfb_bindings = reader_dfb_bindings,
        .tensor_bindings = reader_tensor_bindings,
        .compile_time_args = {{"dm_batch", cfg.dm_batch}, {"num_tcs", reader_num_tcs}},
        .runtime_arg_schema = {.runtime_arg_names = reader_rt_names},
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    m2::Group<m2::TensorBinding> writer_tensor_bindings;
    if (!c_borrowed || tail_tiles > 0) {
        writer_tensor_bindings.push_back(m2::TensorBinding{T_C, "out"});
    }
    // Scalar: the writer is ALSO the producer of the in1 DFB (it fills the single scalar tile), so it
    // gets a "in1" producer binding and a leading "packed_scalar" runtime arg. Tensor-tensor keeps the
    // out-drain-only writer unchanged.
    m2::Group<m2::DFBBinding> writer_dfb_bindings;
    if (has_main_ring) {
        writer_dfb_bindings.push_back(m2::ConsumerOf(OUT, "out"));
    }
    if (is_scalar) {
        writer_dfb_bindings.push_back(m2::ProducerOf(IN1, "in1"));
    }
    if (tail_tiles > 0) {
        writer_dfb_bindings.push_back(m2::ConsumerOf(OUT_TAIL, "out_tail"));
    }
    m2::Group<std::string> writer_rt_names = {
        "start_tile_id", "dst_num_tiles", "dst_shard_width", "D", "N", "C", "Ht", "Wt", "cND"};
    if (is_scalar) {
        writer_rt_names.insert(writer_rt_names.begin(), "packed_scalar");
    }
    if (walk_pads) {
        writer_rt_names.push_back("valid_rows");
        writer_rt_names.push_back("valid_cols");
    }
    m2::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path(kernel_sources.writer),
        .num_threads = writer_threads,
        .compiler_options = {.defines = writer_defines_tbl},
        .dfb_bindings = writer_dfb_bindings,
        .tensor_bindings = writer_tensor_bindings,
        .compile_time_args = {{"dm_batch", cfg.dm_batch}, {"num_tcs", writer_num_tcs}},
        .runtime_arg_schema = {.runtime_arg_names = writer_rt_names},
        .hw_config = ttnn::create_writer_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    // Compute: consumes pre_lhs/pre_rhs, produces out. When an operand has activations, the kernel both
    // produces (PREPROCESS writes) and consumes (binary op reads) its post DFB — a self-loop pair
    // (one PRODUCER + one CONSUMER binding on the same DFB, resolved intra-thread by default).
    m2::Group<m2::DFBBinding> compute_dfb_bindings;
    if (has_main_ring) {
        compute_dfb_bindings.push_back(m2::ConsumerOf(IN0, "pre_lhs"));
        compute_dfb_bindings.push_back(m2::ConsumerOf(IN1, "pre_rhs"));
        compute_dfb_bindings.push_back(m2::ProducerOf(OUT, "out"));
    }
    if (tail_tiles > 0) {
        compute_dfb_bindings.push_back(m2::ConsumerOf(IN0_TAIL, "pre_lhs_tail"));
        compute_dfb_bindings.push_back(m2::ConsumerOf(IN1_TAIL, "pre_rhs_tail"));
        compute_dfb_bindings.push_back(m2::ProducerOf(OUT_TAIL, "out_tail"));
    }
    if (has_lhs_act) {
        compute_dfb_bindings.push_back(m2::ProducerOf(POST_LHS, "post_lhs"));
        compute_dfb_bindings.push_back(m2::ConsumerOf(POST_LHS, "post_lhs"));
    }
    if (has_rhs_act) {
        compute_dfb_bindings.push_back(m2::ProducerOf(POST_RHS, "post_rhs"));
        compute_dfb_bindings.push_back(m2::ConsumerOf(POST_RHS, "post_rhs"));
    }
    // Subtile broadcast (ROW/COL/SCALAR): the compute both produces (unary_bcast<ROW|COL|SCALAR> -> pack)
    // and consumes (binary op) the llk_post intermediate — the same self-loop pair as post_lhs/post_rhs.
    // Exactly one of LLK_POST_LHS (c_5, ROW_A/COL_A/SCALAR_A) / LLK_POST_RHS (c_6, ROW_B/COL_B/SCALAR_B) is
    // allocated per op; whichever it is binds under the accessor name "llk_post", and that accessor-name
    // binding is what lets the single compute kernel read either as dfb::llk_post.
    if (bcast_lhs) {
        compute_dfb_bindings.push_back(m2::ProducerOf(LLK_POST_LHS, "llk_post"));
        compute_dfb_bindings.push_back(m2::ConsumerOf(LLK_POST_LHS, "llk_post"));
    }
    if (bcast_rhs) {
        compute_dfb_bindings.push_back(m2::ProducerOf(LLK_POST_RHS, "llk_post"));
        compute_dfb_bindings.push_back(m2::ConsumerOf(LLK_POST_RHS, "llk_post"));
    }

    m2::Group<std::string> compute_rt_names = {"num_tiles"};
    // COL/SCALAR broadcast reuse the expanded tile across a freq/tile_start loop (COL: freq = Wt, one
    // tile-row; SCALAR: freq = Ht * Wt, the whole (N,C) slab); the compute reads freq and the per-core
    // tile offset as runtime args. ROW/NONE do not (their compute is a flat per-tile loop), so the args
    // are added only on the COL/SCALAR path.
    if (uses_tile_freq_reuse) {
        compute_rt_names.push_back("tile_freq");
        compute_rt_names.push_back("tile_start");
    }
    if (op_type == BinaryOpType::ISCLOSE) {
        compute_rt_names.push_back("rtol_bits");
        compute_rt_names.push_back("atol_bits");
    }

    // to_compute_hardware_config maps the common knobs; it deliberately leaves the per-DFB unpack_modes
    // default, so set it here.
    auto compute_hw = ttnn::to_compute_hardware_config(ttnn::ComputeKernelConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    });
    compute_hw.unpack_modes = unpack_modes;

    m2::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(kernel_sources.compute),
        .num_threads = compute_threads,
        .compiler_options =
            {.include_paths = {std::filesystem::path(kComputeIncludeCommon), std::filesystem::path(kComputeIncludeDfb)},
             .defines = compute_defines_tbl},
        .dfb_bindings = compute_dfb_bindings,
        .compile_time_args = {{"num_tiles_per_cycle", num_tiles_per_cycle}},
        .runtime_arg_schema = {.runtime_arg_names = compute_rt_names},
        .hw_config = compute_hw,
    };

    // --- Placement + per-core runtime args (mirrors the descriptor factory's per-core loop). With an anchor:
    // its cores, each with the anchor's tile count. NoC path: split_work_to_cores over the worker grid, with
    // per-node num_tiles by group. Either way target_nodes is exactly the active cores (no dummy args). ---
    const int out_rank = c.logical_shape().rank();
    const uint32_t aND = extract_nD_dims(a, out_rank);
    // Scalar has no `b`: its dims are all 1 so every b stride below collapses to 0 (unread by the
    // scalar reader, which drops the *_b args, but the reader arg schema still carries them).
    const uint32_t bND = is_scalar ? 1u : extract_nD_dims(*b_opt, out_rank);
    const uint32_t cND = extract_nD_dims(c, out_rank);
    const auto [aD, aN, aC, aHt, aWt] = get_shape_dims(a);
    uint32_t bD = 1, bN = 1, bC = 1, bHt = 1, bWt = 1;
    if (!is_scalar) {
        std::tie(bD, bN, bC, bHt, bWt) = get_shape_dims(*b_opt);
    }

    // a/b input strides (the (dim>1) gate zeroes a stride for a unit dim). Same on every core.
    const uint32_t nD_stride = aHt * aWt * aC * aN * aD * (aND > 1);
    const uint32_t d_stride = aHt * aWt * aC * aN * (aD > 1);
    const uint32_t n_stride = aHt * aWt * aC * (aN > 1);
    const uint32_t c_stride = aHt * aWt * (aC > 1);
    const uint32_t nD_stride_b = bHt * bWt * bC * bN * bD * (bND > 1);
    const uint32_t d_stride_b = bHt * bWt * bC * bN * (bD > 1);
    const uint32_t n_stride_b = bHt * bWt * bC * (bN > 1);
    const uint32_t c_stride_b = bHt * bWt * (bC > 1);

    // These strides are the INPUT's while the loop bounds are the OUTPUT's, so a zeroed stride makes an
    // operand re-read one tile across that dim -- outer-dim broadcast ("no_bcast" means no SUBTILE
    // broadcast). kernels_qsr/ collapses the cascade to page = start_tile_id + k and ignores them, which
    // is the identity only while every input dim equals the output's.
    const bool a_dims_are_output_dims = aND == cND && aD == cD && aN == cN && aC == cC && aHt == cHt && aWt == cWt;
    const bool b_dims_are_output_dims = bND == cND && bD == cD && bN == cN && bC == cC && bHt == cHt && bWt == cWt;
    TT_FATAL(
        a_dims_are_output_dims && b_dims_are_output_dims,
        "Quasar-native binary_ng requires both inputs to have the output's dims; the collapsed kernel "
        "traversal cannot express outer-dim broadcast. a=({},{},{},{},{},{}) b=({},{},{},{},{},{}) "
        "out=({},{},{},{},{},{}).",
        aND,
        aD,
        aN,
        aC,
        aHt,
        aWt,
        bND,
        bD,
        bN,
        bC,
        bHt,
        bWt,
        cND,
        cD,
        cN,
        cC,
        cHt,
        cWt);

    std::vector<CoreCoord> cores;
    CoreRangeSet core_group_1, core_group_2;
    uint32_t num_tiles_per_core_group_1 = 0, num_tiles_per_core_group_2 = 0;

    // Placement follows the anchor. A shard anchor places shard i on core i of the allocator's own core list,
    // which holds only the cores with a shard; the slice anchor places bank i on its core, every core of
    // op.worker_grid. Without an anchor, split_work_to_cores spreads the output tiles linearly across all of
    // op.worker_grid (the full sub-device TENSIX worker set returned by get_worker_grid, NOT a shard grid); for
    // a NoC-written sharded output the sharding-aware TensorAccessor maps each global output tile id to its
    // shard core, so the linear split need not be shard-aligned. Equal-shape no-broadcast means each core
    // processes the same tile count of all three operands: the anchor's, or its split_work_to_cores share.
    if (anchor.kind != AnchorKind::kNone) {
        cores = anchor.cores;
    } else {
        CoreRangeSet all_cores;
        std::tie(
            std::ignore,
            all_cores,
            core_group_1,
            core_group_2,
            num_tiles_per_core_group_1,
            num_tiles_per_core_group_2) = tt::tt_metal::split_work_to_cores(op.worker_grid, c_num_tiles, true);
        cores = corerange_to_cores(all_cores, std::nullopt, true);
    }
    TT_FATAL(!cores.empty(), "binary_ng Quasar-native no-bcast factory: empty core set");

    m2::KernelRunArgs::RuntimeArgValues reader_args, writer_args, compute_args;

    const uint32_t isclose_rtol_bits = std::bit_cast<uint32_t>(op.rtol);
    const uint32_t isclose_atol_bits = std::bit_cast<uint32_t>(op.atol);

    // Scalar packed to the in1 data format for the writer's one-time fill (mirrors the descriptor
    // scalar packing). Same value on every core. Quant never reaches this factory (gated out).
    const uint32_t packed_scalar = is_scalar ? pack_scalar_runtime_arg(*op.scalar, a.dtype(), /*is_quant*/ false) : 0u;

    std::set<CoreRange> target_ranges;
    uint32_t start_tile_id = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        target_ranges.insert(CoreRange(core, core));
        const m2::NodeCoord node{static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y)};

        uint32_t a_num_tiles = 0, b_num_tiles = 0, c_num_tiles_core = 0;
        uint32_t c_start_id = 0, c_current_shard_width = 0;
        if (anchor.kind != AnchorKind::kNone) {
            // Each core computes its anchor's shard or bank slice. The count is the borrowed part, the same for
            // a, b and c; the tail tiles start right after it. c_start_id is the page of the core's tile 0, where
            // the walk of an operand that moves starts; the borrowed kernels do not read it.
            c_num_tiles_core = c_main_tiles;
            a_num_tiles = c_num_tiles_core;
            b_num_tiles = c_num_tiles_core;
            c_current_shard_width = anchor.shard_width;
            c_start_id = anchor_first_page(anchor, i);
            if (walk_pads) {
                const auto [valid_rows, valid_cols] = anchor_valid_bounds(anchor, i);
                for (auto* args : {&reader_args, &writer_args}) {
                    (*args)["valid_rows"][node] = valid_rows;
                    (*args)["valid_cols"][node] = valid_cols;
                }
            }
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

        if (is_scalar) {
            writer_args["packed_scalar"][node] = packed_scalar;
        }
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
        if (uses_tile_freq_reuse) {
            // COL: {freq = Wt, tile_start = this core's column offset} -- broadcasts the column tile once
            // per tile-row and reuses it across the row (freq iterations). SCALAR: {freq = Ht * Wt,
            // tile_start = this core's offset into the (N,C) slab} -- broadcasts the single-element tile
            // once per slab and reuses it across the whole slab. Either way tile_start is where this
            // core's tile range begins within one broadcast period.
            const auto [tile_freq, tile_start] =
                calculate_compute_kernel_args(op.subtile_broadcast_type, c_start_id, cHt, cWt);
            compute_args["tile_freq"][node] = tile_freq;
            compute_args["tile_start"][node] = tile_start;
        }
        if (op_type == BinaryOpType::ISCLOSE) {
            compute_args["rtol_bits"][node] = isclose_rtol_bits;
            compute_args["atol_bits"][node] = isclose_atol_bits;
        }
    }
    m2::NodeRangeSet target_nodes(target_ranges);

    // --- Assemble the spec. ---
    m2::WorkUnitSpec wu{
        .name = "binary_ng_quasar_native_no_bcast",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = target_nodes,
    };

    // Scalar has no `b` tensor, so it declares only T_A / T_C (the in1 DFB is writer-filled, not
    // backed by a TensorParameter).
    m2::Group<m2::TensorParameter> tensor_params = {{.unique_id = T_A, .spec = a.tensor_spec()}};
    if (!is_scalar) {
        tensor_params.push_back({.unique_id = T_B, .spec = b_opt->tensor_spec()});
    }
    tensor_params.push_back({.unique_id = T_C, .spec = c.tensor_spec()});

    m2::ProgramSpec spec{
        .name = "binary_ng_quasar_native_no_bcast",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = dfbs,
        .tensor_parameters = tensor_params,
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
    if (!is_scalar) {
        run_params.tensor_args.emplace(T_B, m2::ProgramRunArgs::TensorArgument{b_opt->mesh_tensor()});
    }
    run_params.tensor_args.emplace(T_C, m2::ProgramRunArgs::TensorArgument{c.mesh_tensor()});

    return ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
        .op_owned_tensors = {},
    };
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts BinaryNgDeviceOperation::ProgramFactoryQuasarNative::create_program_artifacts(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, tensor_return_value_t& c) {
    return CMAKE_UNIQUE_NAMESPACE::create_no_bcast_artifacts(operation_attributes, tensor_args, c);
}

}  // namespace ttnn::operations::experimental::quasar::binary_ng
