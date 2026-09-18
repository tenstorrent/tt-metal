// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified (placement-first) matmul factory: stage A of the Quasar-native matmul (GH#41910).
//
// One Metal 2.0 program for every placement. The config names the cores and the MN chunk per core; this
// file turns that into a MN chunk assignment, four DFB rings (A slice, B slice, MN chunk, C partials),
// one reader, one compute kernel and one writer. Nothing here depends on how the operands are laid out in
// memory: the kernels address tiles by tile index through the tensor accessor.

#include "ttnn/operations/experimental/quasar/matmul/device/factory/matmul_unified_program_factory.hpp"

#include <algorithm>
#include <filesystem>
#include <map>
#include <string>
#include <tuple>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/quasar/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/tensor/shape/shape.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Names the kernels see: dfb::A_slice / B_slice / MN_chunk / C_partials and tensor::A / B / C.
const DFBSpecName A_SLICE_DFB{"A_slice"};
const DFBSpecName B_SLICE_DFB{"B_slice"};
const DFBSpecName MN_CHUNK_DFB{"MN_chunk"};
const DFBSpecName C_PARTIALS_DFB{"C_partials"};

const TensorParamName A_TENSOR{"A"};
const TensorParamName B_TENSOR{"B"};
const TensorParamName C_TENSOR{"C"};

const KernelSpecName READER_KERNEL{"reader"};
const KernelSpecName COMPUTE_KERNEL{"compute"};
const KernelSpecName WRITER_KERNEL{"writer"};

constexpr const char* KERNEL_DIR = "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/";

// A DFB touched by a TRISC keeps its ring extent in a uint16_t of 16-byte units (see
// validate_ring_extent in dataflow_buffer.cpp). Enforced on every arch so a config that is legal on
// Wormhole never becomes a program-creation FATAL on Quasar.
constexpr uint64_t MAX_DFB_RING_BYTES = 65535ull * 16ull;
constexpr uint32_t MAX_AUTO_K_ITERATION_TILES = 8;

uint64_t l1_budget_bytes(tt::tt_metal::IDevice* device) {
    const uint32_t l1_base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const std::optional<tt::tt_metal::DeviceAddr> lowest_occupied = device->lowest_occupied_compute_l1_address();
    const uint32_t l1_ceiling =
        lowest_occupied.has_value() ? static_cast<uint32_t>(lowest_occupied.value()) : device->l1_size_per_core();
    TT_FATAL(l1_ceiling > l1_base, "L1 ceiling ({}) must exceed base ({})", l1_ceiling, l1_base);
    return l1_ceiling - l1_base;
}

// Fills the ring sizing of `plan` for a given K iteration. Returns the total footprint in bytes.
uint64_t size_rings(UnifiedMatmulPlan& plan, uint32_t K_iteration_tiles, bool fp32_dest_acc_en, bool packer_l1_acc) {
    plan.K_iteration_tiles = K_iteration_tiles;
    plan.num_K_iterations = plan.K_tiles / K_iteration_tiles;

    // The packer accumulates partials in L1 only when there are enough K iterations for the reconfig overhead
    // to pay off (the last step spills and reloads either way, so more than two).
    plan.packer_l1_acc_en = packer_l1_acc && plan.num_K_iterations > 2;
    plan.C_partials_format = plan.packer_l1_acc_en
                                 ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                 : (fp32_dest_acc_en ? tt::DataFormat::Float32 : plan.C_format);

    // Interleaved tiles live in DRAM at the DRAM-aligned stride and the reader copies at that stride;
    // a no-op for every 32x32 format.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    plan.A_slot_bytes = tt::align(tt::tile_size(plan.A_format), dram_alignment);
    plan.B_slot_bytes = tt::align(tt::tile_size(plan.B_format), dram_alignment);
    plan.C_slot_bytes = tt::tile_size(plan.C_format);
    plan.C_partials_slot_bytes = tt::tile_size(plan.C_partials_format);

    const uint32_t A_slice_tiles = plan.MN_chunk_M_tiles * K_iteration_tiles;
    const uint32_t B_slice_tiles = K_iteration_tiles * plan.MN_chunk_N_tiles;
    const uint32_t MN_chunk_tiles = plan.MN_chunk_M_tiles * plan.MN_chunk_N_tiles;
    // Double-buffer the slices whenever more than one slice passes through the ring.
    const bool more_than_one_slice =
        (uint64_t)plan.batch_size * plan.max_MN_chunks_per_core * plan.num_K_iterations > 1;
    const uint32_t slice_ring_depth = more_than_one_slice ? 2 : 1;
    plan.A_slice_ring_slots = A_slice_tiles * slice_ring_depth;
    plan.B_slice_ring_slots = B_slice_tiles * slice_ring_depth;
    plan.MN_chunk_ring_slots = MN_chunk_tiles;
    plan.C_partials_ring_slots = MN_chunk_tiles;

    // Aliasing C_partials onto MN_chunk when a core produces more than one MN chunk (counting batches) is a race: the
    // writer may still be draining MN chunk i from MN_chunk while the compute packs MN chunk i+1's first partials into
    // the same bytes. Alias only when the partials can never be live while MN_chunk holds unread data: a single MN
    // chunk per core, or no partials at all (one K iteration).
    const bool partials_ever_written = plan.num_K_iterations > 1;
    const bool one_MN_chunk_per_core = plan.batch_size == 1 && plan.max_MN_chunks_per_core == 1;
    plan.alias_C_partials_onto_MN_chunk =
        (plan.C_partials_format == plan.C_format) && (!partials_ever_written || one_MN_chunk_per_core);

    plan.l1_bytes =
        (uint64_t)plan.A_slice_ring_slots * plan.A_slot_bytes + (uint64_t)plan.B_slice_ring_slots * plan.B_slot_bytes +
        (uint64_t)plan.MN_chunk_ring_slots * plan.C_slot_bytes +
        (plan.alias_C_partials_onto_MN_chunk ? 0 : (uint64_t)plan.C_partials_ring_slots * plan.C_partials_slot_bytes);
    return plan.l1_bytes;
}

bool rings_fit(const UnifiedMatmulPlan& plan, uint64_t l1_budget) {
    const uint64_t rings[] = {
        (uint64_t)plan.A_slice_ring_slots * plan.A_slot_bytes,
        (uint64_t)plan.B_slice_ring_slots * plan.B_slot_bytes,
        (uint64_t)plan.MN_chunk_ring_slots * plan.C_slot_bytes,
        (uint64_t)plan.C_partials_ring_slots * plan.C_partials_slot_bytes};
    for (uint64_t ring_bytes : rings) {
        if (ring_bytes > MAX_DFB_RING_BYTES) {
            return false;
        }
    }
    return plan.l1_bytes <= l1_budget;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

tt::tt_metal::TensorMemoryLayout UnifiedMatmulPlan::sharded_output_layout() const {
    if (MN_chunk_N_tiles >= N_tiles) {  // a MN chunk spans all of N: MN chunks are stacked down M
        return tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED;
    }
    if (MN_chunk_M_tiles >= M_tiles) {  // a MN chunk spans all of M: MN chunks sit side by side across N
        return tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
    }
    return tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
}

UnifiedMatmulPlan plan_unified_matmul(
    const ttnn::Tensor& A,
    const ttnn::Tensor& B,
    const operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig& config,
    const MatmulParams& attributes) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    UnifiedMatmulPlan plan{};

    TT_FATAL(
        !attributes.transpose_a && !attributes.transpose_b,
        "MatmulUnifiedProgramConfig: transposes are applied to the operands before the op, not in the kernels");
    TT_FATAL(!attributes.untilize_out, "MatmulUnifiedProgramConfig does not support untilize_out");
    TT_FATAL(attributes.bcast_batch.has_value(), "bcast_batch should have been populated");
    TT_FATAL(attributes.compute_kernel_config.has_value(), "compute_kernel_config should have been populated");
    TT_FATAL(attributes.output_dtype.has_value(), "output_dtype should have been populated");

    const tt::tt_metal::Shape& A_shape = A.padded_shape();
    const tt::tt_metal::Shape& B_shape = B.padded_shape();
    const tt::tt_metal::Tile A_tile = A.tensor_spec().tile();
    const tt::tt_metal::Tile B_tile = B.tensor_spec().tile();
    TT_FATAL(
        A_tile.get_height() == TILE_HEIGHT && A_tile.get_width() == TILE_WIDTH && B_tile.get_height() == TILE_HEIGHT &&
            B_tile.get_width() == TILE_WIDTH,
        "MatmulUnifiedProgramConfig supports 32x32 tiles only (A {}x{}, B {}x{})",
        A_tile.get_height(),
        A_tile.get_width(),
        B_tile.get_height(),
        B_tile.get_width());
    if (attributes.output_tile.has_value()) {
        TT_FATAL(
            attributes.output_tile->get_tile_shape()[0] == TILE_HEIGHT &&
                attributes.output_tile->get_tile_shape()[1] == TILE_WIDTH,
            "MatmulUnifiedProgramConfig supports a 32x32 output tile only");
    }

    // ---- GEMM size ----
    plan.batch_size = get_batch_size(A_shape);
    plan.M_tiles = A_shape[-2] / TILE_HEIGHT;
    plan.K_tiles = A_shape[-1] / TILE_WIDTH;
    plan.N_tiles = B_shape[-1] / TILE_WIDTH;
    plan.broadcast_B_over_batch = attributes.bcast_batch.value();
    TT_FATAL(
        plan.broadcast_B_over_batch || get_batch_size(B_shape) == plan.batch_size,
        "Batched B must match A's batch ({} vs {})",
        get_batch_size(B_shape),
        plan.batch_size);

    // ---- MN chunk assignment ----
    TT_FATAL(
        config.MN_chunk_M_tiles > 0 && config.MN_chunk_N_tiles > 0,
        "MN_chunk_M_tiles and MN_chunk_N_tiles must be > 0");
    plan.MN_chunk_M_tiles = config.MN_chunk_M_tiles;
    plan.MN_chunk_N_tiles = config.MN_chunk_N_tiles;
    // The chunks of one batch, walked across N then down M; every core produces its chunks for all batches.
    const uint32_t MN_chunks_across_N = tt::div_up(plan.N_tiles, plan.MN_chunk_N_tiles);
    const uint32_t MN_chunks_down_M = tt::div_up(plan.M_tiles, plan.MN_chunk_M_tiles);
    plan.MN_chunks_per_batch = MN_chunks_down_M * MN_chunks_across_N;

    TT_FATAL(config.cores.num_cores() > 0, "MatmulUnifiedProgramConfig.cores is empty");
    const CoreCoord grid = A.device()->compute_with_storage_grid_size();
    const CoreRange bounding_box = config.cores.bounding_box();
    TT_FATAL(
        bounding_box.end_coord.x < grid.x && bounding_box.end_coord.y < grid.y,
        "MatmulUnifiedProgramConfig.cores {} exceed the device compute grid {}x{}",
        config.cores.str(),
        grid.x,
        grid.y);
    plan.row_major_cores = config.row_major_cores;
    const std::vector<CoreCoord> all_cores = corerange_to_cores(config.cores, std::nullopt, config.row_major_cores);
    // Each active core takes a contiguous run of the walk, the first (MN_chunks_per_batch % num_active) cores
    // one chunk longer. A core's start is expressed in tile coordinates so the kernels only ever step by
    // MN_chunk_M_tiles / MN_chunk_N_tiles.
    const uint32_t num_active = std::min<uint32_t>(all_cores.size(), plan.MN_chunks_per_batch);
    plan.cores.assign(all_cores.begin(), all_cores.begin() + num_active);
    const uint32_t MN_chunks_per_core_floor = plan.MN_chunks_per_batch / num_active;
    const uint32_t cores_with_extra_MN_chunk = plan.MN_chunks_per_batch % num_active;
    plan.first_MN_chunk_M_tile.resize(num_active);
    plan.first_MN_chunk_N_tile.resize(num_active);
    plan.num_MN_chunks.resize(num_active);
    uint32_t next_MN_chunk = 0;  // position in the walk of the next unassigned chunk
    for (uint32_t core = 0; core < num_active; ++core) {
        plan.num_MN_chunks[core] = MN_chunks_per_core_floor + (core < cores_with_extra_MN_chunk ? 1 : 0);
        plan.first_MN_chunk_M_tile[core] = (next_MN_chunk / MN_chunks_across_N) * plan.MN_chunk_M_tiles;
        plan.first_MN_chunk_N_tile[core] = (next_MN_chunk % MN_chunks_across_N) * plan.MN_chunk_N_tiles;
        next_MN_chunk += plan.num_MN_chunks[core];
    }
    plan.max_MN_chunks_per_core = plan.num_MN_chunks.front();

    // ---- Subblock: the MN chunk's tiles accumulated in DST at once ----
    const bool fp32_dest_acc_en = get_fp32_dest_acc_en(attributes.compute_kernel_config);
    const bool packer_l1_acc =
        std::get<3>(get_compute_kernel_config_args(A.device()->arch(), attributes.compute_kernel_config.value()));
    if (config.subblock_M_tiles == 0 && config.subblock_N_tiles == 0) {
        // The chooser's (h, w) is (M tiles, N tiles) of the subblock.
        const std::tuple<uint32_t, uint32_t> subblock =
            operations::experimental::quasar::matmul::bmm_op_utils_qsr::get_matmul_subblock_params(
                plan.MN_chunk_M_tiles, plan.MN_chunk_N_tiles, false, false, fp32_dest_acc_en);
        plan.subblock_M_tiles = std::get<0>(subblock);
        plan.subblock_N_tiles = std::get<1>(subblock);
    } else {
        TT_FATAL(
            config.subblock_M_tiles > 0 && config.subblock_N_tiles > 0,
            "subblock_M_tiles and subblock_N_tiles must both be set or both be 0 (auto)");
        plan.subblock_M_tiles = config.subblock_M_tiles;
        plan.subblock_N_tiles = config.subblock_N_tiles;
    }
    TT_FATAL(
        plan.MN_chunk_M_tiles % plan.subblock_M_tiles == 0 && plan.MN_chunk_N_tiles % plan.subblock_N_tiles == 0,
        "subblock {}x{} must divide the per-core MN chunk {}x{}",
        plan.subblock_M_tiles,
        plan.subblock_N_tiles,
        plan.MN_chunk_M_tiles,
        plan.MN_chunk_N_tiles);
    const uint32_t dst_capacity_tiles = fp32_dest_acc_en ? 4 : 8;
    TT_FATAL(
        plan.subblock_M_tiles * plan.subblock_N_tiles <= dst_capacity_tiles,
        "subblock {}x{} holds {} tiles; DST fits {} (fp32 accumulation: {})",
        plan.subblock_M_tiles,
        plan.subblock_N_tiles,
        plan.subblock_M_tiles * plan.subblock_N_tiles,
        dst_capacity_tiles,
        fp32_dest_acc_en);

    // ---- Formats, K iteration and ring sizing ----
    plan.A_format = tt::tt_metal::datatype_to_dataformat_converter(A.dtype());
    plan.B_format = tt::tt_metal::datatype_to_dataformat_converter(B.dtype());
    plan.C_format = tt::tt_metal::datatype_to_dataformat_converter(attributes.output_dtype.value());
    const uint64_t l1_budget = l1_budget_bytes(A.device());
    if (config.K_iteration_tiles == 0) {
        // Largest divisor of K_tiles (capped) whose rings fit; 1 is the floor and must fit.
        uint32_t chosen = 0;
        for (uint32_t K_iteration_tiles = std::min<uint32_t>(plan.K_tiles, MAX_AUTO_K_ITERATION_TILES);
             K_iteration_tiles >= 1;
             --K_iteration_tiles) {
            if (plan.K_tiles % K_iteration_tiles != 0) {
                continue;
            }
            size_rings(plan, K_iteration_tiles, fp32_dest_acc_en, packer_l1_acc);
            if (rings_fit(plan, l1_budget)) {
                chosen = K_iteration_tiles;
                break;
            }
        }
        TT_FATAL(
            chosen > 0,
            "MatmulUnifiedProgramConfig: a {}x{}-tile MN chunk does not fit L1 even with K_iteration_tiles=1 "
            "(needs {} B, budget {} B, max ring {} B); shrink MN_chunk_M_tiles / MN_chunk_N_tiles",
            plan.MN_chunk_M_tiles,
            plan.MN_chunk_N_tiles,
            plan.l1_bytes,
            l1_budget,
            MAX_DFB_RING_BYTES);
    } else {
        TT_FATAL(
            plan.K_tiles % config.K_iteration_tiles == 0,
            "K_iteration_tiles ({}) must divide K_tiles ({})",
            config.K_iteration_tiles,
            plan.K_tiles);
        size_rings(plan, config.K_iteration_tiles, fp32_dest_acc_en, packer_l1_acc);
        TT_FATAL(
            rings_fit(plan, l1_budget),
            "MatmulUnifiedProgramConfig: rings for a {}x{}-tile MN chunk with K_iteration_tiles={} do not fit "
            "(needs {} B, budget {} B, max ring {} B: A slice {} B, B slice {} B, MN chunk {} B, C partials {} B)",
            plan.MN_chunk_M_tiles,
            plan.MN_chunk_N_tiles,
            plan.K_iteration_tiles,
            plan.l1_bytes,
            l1_budget,
            MAX_DFB_RING_BYTES,
            (uint64_t)plan.A_slice_ring_slots * plan.A_slot_bytes,
            (uint64_t)plan.B_slice_ring_slots * plan.B_slot_bytes,
            (uint64_t)plan.MN_chunk_ring_slots * plan.C_slot_bytes,
            (uint64_t)plan.C_partials_ring_slots * plan.C_partials_slot_bytes);
    }

    // ---- Sharded output: one MN chunk per core, batch 1, and a grid the accessor maps the same way ----
    if (attributes.output_mem_config.is_sharded()) {
        TT_FATAL(plan.batch_size == 1, "Sharded output needs batch 1 (a core's chunks would not form one shard)");
        TT_FATAL(
            plan.MN_chunks_per_batch == plan.cores.size(),
            "Sharded output needs exactly one MN chunk per core ({} MN chunks, {} active cores)",
            plan.MN_chunks_per_batch,
            plan.cores.size());
        if (plan.sharded_output_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED) {
            const std::vector<CoreRange>& ranges = config.cores.ranges();
            const bool one_rectangle = ranges.size() == 1;
            const uint32_t rectangle_columns = one_rectangle ? ranges[0].grid_size().x : 0;
            const uint32_t rectangle_rows = one_rectangle ? ranges[0].grid_size().y : 0;
            TT_FATAL(
                one_rectangle && rectangle_columns == MN_chunks_across_N && rectangle_rows == MN_chunks_down_M,
                "Block-sharded output needs cores to be one {}x{} rectangle (one core per MN chunk, laid out as "
                "the chunks tile C), got {}",
                MN_chunks_across_N,
                MN_chunks_down_M,
                config.cores.str());
        }
    }
    return plan;
}

ttnn::device_operation::ProgramArtifacts MatmulUnifiedProgramFactory::create_program_artifacts(
    const MatmulParams& operation_attributes,
    const MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    TT_FATAL(
        tensor_args.optional_input_tensors.empty() || !tensor_args.optional_input_tensors[0].has_value(),
        "MatmulUnifiedProgramConfig does not fuse bias; the op applies it as a separate add");
    TT_FATAL(
        operation_attributes.program_config.has_value() &&
            std::holds_alternative<operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig>(
                operation_attributes.program_config.value()),
        "MatmulUnifiedProgramFactory needs a MatmulUnifiedProgramConfig");
    const operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig& config =
        std::get<operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig>(
            operation_attributes.program_config.value());

    const ttnn::Tensor& A_tensor = tensor_args.input_tensors.at(0);
    const ttnn::Tensor& B_tensor = tensor_args.input_tensors.at(1);
    const tt::tt_metal::MeshTensor& A = A_tensor.mesh_tensor();
    const tt::tt_metal::MeshTensor& B = B_tensor.mesh_tensor();
    const tt::tt_metal::MeshTensor& C = tensor_return_value.at(0).mesh_tensor();
    tt::tt_metal::IDevice* device = &A.mutable_device();

    const UnifiedMatmulPlan plan = plan_unified_matmul(A_tensor, B_tensor, config, operation_attributes);

    // ---- Tensor parameters: the kernels' tensor accessors are generated from these specs ----
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = A_TENSOR, .spec = A.tensor_spec()},
        TensorParameter{.unique_id = B_TENSOR, .spec = B.tensor_spec()},
        TensorParameter{.unique_id = C_TENSOR, .spec = C.tensor_spec()},
    };

    // ---- Dataflow-buffer rings ----
    const tt::tt_metal::Tile C_tile = C.tensor_spec().tile();
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = A_SLICE_DFB,
            .entry_size = plan.A_slot_bytes,
            .num_entries = plan.A_slice_ring_slots,
            .data_format_metadata = plan.A_format,
            .tile_format_metadata = A.tensor_spec().tile(),
        },
        DataflowBufferSpec{
            .unique_id = B_SLICE_DFB,
            .entry_size = plan.B_slot_bytes,
            .num_entries = plan.B_slice_ring_slots,
            .data_format_metadata = plan.B_format,
            .tile_format_metadata = B.tensor_spec().tile(),
        },
    };
    {
        DataflowBufferSpec MN_chunk_dfb{
            .unique_id = MN_CHUNK_DFB,
            .entry_size = plan.C_slot_bytes,
            .num_entries = plan.MN_chunk_ring_slots,
            .data_format_metadata = plan.C_format,
            .tile_format_metadata = C_tile,
        };
        DataflowBufferSpec C_partials_dfb{
            .unique_id = C_PARTIALS_DFB,
            .entry_size = plan.C_partials_slot_bytes,
            .num_entries = plan.C_partials_ring_slots,
            .data_format_metadata = plan.C_partials_format,
            .tile_format_metadata = C_tile,
        };
        if (plan.alias_C_partials_onto_MN_chunk) {
            MN_chunk_dfb.advanced_options.alias_with = {C_PARTIALS_DFB};
            C_partials_dfb.advanced_options.alias_with = {MN_CHUNK_DFB};
        }
        dataflow_buffers.push_back(std::move(MN_chunk_dfb));
        dataflow_buffers.push_back(std::move(C_partials_dfb));
    }

    // ---- Reader ----
    const uint32_t A_last_K_tile_valid_columns = A.logical_shape()[-1] % TILE_WIDTH;
    KernelSpec reader{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path(std::string(KERNEL_DIR) + "dataflow/unified_matmul_reader.cpp"),
        .compiler_options = {},
        .dfb_bindings = {ProducerOf(A_SLICE_DFB, "A_slice"), ProducerOf(B_SLICE_DFB, "B_slice")},
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = A_TENSOR, .accessor_name = "A"},
                TensorBinding{.tensor_parameter_name = B_TENSOR, .accessor_name = "B"},
            },
        .compile_time_args =
            {
                {"M_tiles", plan.M_tiles},
                {"K_tiles", plan.K_tiles},
                {"N_tiles", plan.N_tiles},
                {"batch_size", plan.batch_size},
                {"broadcast_B_over_batch", plan.broadcast_B_over_batch ? 1u : 0u},
                {"MN_chunk_M_tiles", plan.MN_chunk_M_tiles},
                {"MN_chunk_N_tiles", plan.MN_chunk_N_tiles},
                {"K_iteration_tiles", plan.K_iteration_tiles},
                {"num_K_iterations", plan.num_K_iterations},
                {"A_last_K_tile_valid_columns", A_last_K_tile_valid_columns},
            },
        .runtime_arg_schema =
            {.runtime_arg_names = {"first_MN_chunk_M_tile", "first_MN_chunk_N_tile", "num_MN_chunks"}},
        .hw_config =
            ttnn::create_reader_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // ---- Writer ----
    KernelSpec writer{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path(std::string(KERNEL_DIR) + "dataflow/unified_matmul_writer.cpp"),
        .compiler_options = {},
        .dfb_bindings = {ConsumerOf(MN_CHUNK_DFB, "MN_chunk")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = C_TENSOR, .accessor_name = "C"}},
        .compile_time_args =
            {
                {"M_tiles", plan.M_tiles},
                {"N_tiles", plan.N_tiles},
                {"batch_size", plan.batch_size},
                {"MN_chunk_M_tiles", plan.MN_chunk_M_tiles},
                {"MN_chunk_N_tiles", plan.MN_chunk_N_tiles},
                {"subblock_M_tiles", plan.subblock_M_tiles},
                {"subblock_N_tiles", plan.subblock_N_tiles},
            },
        .runtime_arg_schema =
            {.runtime_arg_names = {"first_MN_chunk_M_tile", "first_MN_chunk_N_tile", "num_MN_chunks"}},
        .hw_config =
            ttnn::create_writer_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // ---- Compute ----
    const bool fp32_dest_acc_en = get_fp32_dest_acc_en(operation_attributes.compute_kernel_config);
    std::map<std::string, std::string> compute_defines_map;
    if (plan.packer_l1_acc_en) {
        compute_defines_map["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        compute_defines_map["FP32_DEST_ACC_EN"] = "1";
    }
    const ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level =
        ttnn::get_throttle_level(operation_attributes.compute_kernel_config);
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), plan.cores.size(), compute_defines_map);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), plan.cores.size(), compute_defines_map, throttle_level);
    KernelSpec::CompilerOptions::Defines compute_defines(compute_defines_map);

    ComputeHardwareConfig compute_hw_config =
        ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config.value());
    if (fp32_dest_acc_en) {
        // With a 32-bit DST every consumed 32-bit DFB needs an explicit unpack mode. The partials are
        // reloaded with a data copy into DST, so unpack them straight to DST and keep fp32 precision;
        // fp32 operands feed the FPU and go through SrcA/SrcB.
        auto is_32bit = [](tt::DataFormat format) {
            return format == tt::DataFormat::Float32 || format == tt::DataFormat::Int32 ||
                   format == tt::DataFormat::UInt32;
        };
        ComputeUnpackModes& modes = unpack_modes(compute_hw_config);
        if (is_32bit(plan.C_partials_format)) {
            modes.emplace(C_PARTIALS_DFB, tt::tt_metal::UnpackMode::UnpackToDest);
        }
        if (is_32bit(plan.A_format)) {
            modes.emplace(A_SLICE_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
        if (is_32bit(plan.B_format)) {
            modes.emplace(B_SLICE_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
    }

    KernelSpec compute{
        .unique_id = COMPUTE_KERNEL,
        .source = std::filesystem::path(std::string(KERNEL_DIR) + "compute/unified_matmul_compute.cpp"),
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings =
            {
                ConsumerOf(A_SLICE_DFB, "A_slice"),
                ConsumerOf(B_SLICE_DFB, "B_slice"),
                ProducerOf(MN_CHUNK_DFB, "MN_chunk"),
                ProducerOf(C_PARTIALS_DFB, "C_partials"),
                ConsumerOf(C_PARTIALS_DFB, "C_partials"),
            },
        .compile_time_args =
            {
                {"batch_size", plan.batch_size},
                {"K_iteration_tiles", plan.K_iteration_tiles},
                {"num_K_iterations", plan.num_K_iterations},
                {"MN_chunk_M_tiles", plan.MN_chunk_M_tiles},
                {"MN_chunk_N_tiles", plan.MN_chunk_N_tiles},
                {"subblock_M_tiles", plan.subblock_M_tiles},
                {"subblock_N_tiles", plan.subblock_N_tiles},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_MN_chunks"}},
        .hw_config = compute_hw_config,
    };

    // ---- One work unit over the active cores ----
    const CoreRangeSet active_cores(ttsl::Span<const CoreCoord>(plan.cores));
    Group<WorkUnitSpec> work_units = {WorkUnitSpec{
        .name = "unified_matmul",
        .kernels = {READER_KERNEL, COMPUTE_KERNEL, WRITER_KERNEL},
        .target_nodes = active_cores,
    }};

    // ---- Per-core runtime args: where each core's run of MN chunks starts and how long it is ----
    ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = READER_KERNEL};
    ProgramRunArgs::KernelRunArgs compute_run_args{.kernel = COMPUTE_KERNEL};
    ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = WRITER_KERNEL};
    for (uint32_t core = 0; core < plan.cores.size(); ++core) {
        const std::initializer_list<std::pair<std::string, uint32_t>> run_start = {
            {"first_MN_chunk_M_tile", plan.first_MN_chunk_M_tile[core]},
            {"first_MN_chunk_N_tile", plan.first_MN_chunk_N_tile[core]},
            {"num_MN_chunks", plan.num_MN_chunks[core]}};
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, plan.cores[core], run_start);
        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, plan.cores[core], run_start);
        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values, plan.cores[core], {{"num_MN_chunks", plan.num_MN_chunks[core]}});
    }

    ProgramSpec spec{
        .name = "matmul_unified",
        .kernels = {reader, compute, writer},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };
    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run_args), std::move(compute_run_args), std::move(writer_run_args)},
        .tensor_args = {{A_TENSOR, A}, {B_TENSOR, B}, {C_TENSOR, C}},
    };
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
