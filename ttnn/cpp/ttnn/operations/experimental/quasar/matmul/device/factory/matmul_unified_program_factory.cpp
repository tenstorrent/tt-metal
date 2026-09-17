// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified (placement-first) matmul factory: stage A of the Quasar-native matmul (GH#41910).
//
// One Metal 2.0 program for every placement. The config names the cores and the C subblock per core; this
// file turns that into a work-item assignment, four DFB rings (A slice, B slice, C subblock, C partials),
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

// Names the kernels see: dfb::A_slice / B_slice / C_subblock / C_partials and tensor::A / B / C.
const DFBSpecName A_SLICE_DFB{"A_slice"};
const DFBSpecName B_SLICE_DFB{"B_slice"};
const DFBSpecName C_SUBBLOCK_DFB{"C_subblock"};
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

    const uint32_t A_slice_tiles = plan.per_core_M * K_iteration_tiles;
    const uint32_t B_slice_tiles = K_iteration_tiles * plan.per_core_N;
    const uint32_t C_subblock_tiles = plan.per_core_M * plan.per_core_N;
    // Double-buffer the slices whenever more than one slice passes through the ring.
    const bool more_than_one_slice = (uint64_t)plan.max_work_items_per_core * plan.num_K_iterations > 1;
    const uint32_t slice_ring_depth = more_than_one_slice ? 2 : 1;
    plan.A_slice_ring_slots = A_slice_tiles * slice_ring_depth;
    plan.B_slice_ring_slots = B_slice_tiles * slice_ring_depth;
    plan.C_subblock_ring_slots = C_subblock_tiles;
    plan.C_partials_ring_slots = C_subblock_tiles;

    // Aliasing C_partials onto C_subblock with more than one work item per core is a race: the writer may
    // still be draining item i from C_subblock while the compute packs item i+1's first partials into the
    // same bytes. Alias only when the partials can never be live while C_subblock holds unread data: a
    // single work item per core, or no partials at all (one K iteration).
    const bool partials_ever_written = plan.num_K_iterations > 1;
    const bool one_work_item_per_core = plan.max_work_items_per_core == 1;
    plan.alias_C_partials_onto_C_subblock =
        (plan.C_partials_format == plan.C_format) && (!partials_ever_written || one_work_item_per_core);

    plan.l1_bytes =
        (uint64_t)plan.A_slice_ring_slots * plan.A_slot_bytes + (uint64_t)plan.B_slice_ring_slots * plan.B_slot_bytes +
        (uint64_t)plan.C_subblock_ring_slots * plan.C_slot_bytes +
        (plan.alias_C_partials_onto_C_subblock ? 0 : (uint64_t)plan.C_partials_ring_slots * plan.C_partials_slot_bytes);
    return plan.l1_bytes;
}

bool rings_fit(const UnifiedMatmulPlan& plan, uint64_t l1_budget) {
    const uint64_t rings[] = {
        (uint64_t)plan.A_slice_ring_slots * plan.A_slot_bytes,
        (uint64_t)plan.B_slice_ring_slots * plan.B_slot_bytes,
        (uint64_t)plan.C_subblock_ring_slots * plan.C_slot_bytes,
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
    if (C_subblock_grid_columns == 1) {
        return tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED;
    }
    if (C_subblock_grid_rows == 1) {
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

    // ---- C subblock grid and work-item assignment ----
    TT_FATAL(config.per_core_M > 0 && config.per_core_N > 0, "per_core_M and per_core_N must be > 0");
    plan.per_core_M = config.per_core_M;
    plan.per_core_N = config.per_core_N;
    plan.C_subblock_grid_rows = tt::div_up(plan.M_tiles, plan.per_core_M);
    plan.C_subblock_grid_columns = tt::div_up(plan.N_tiles, plan.per_core_N);
    plan.C_subblocks_per_batch = plan.C_subblock_grid_rows * plan.C_subblock_grid_columns;
    plan.num_work_items = plan.batch_size * plan.C_subblocks_per_batch;

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
    // Contiguous runs of work items, the first (num_work_items % num_active) cores taking one extra.
    const uint32_t num_active = std::min<uint32_t>(all_cores.size(), plan.num_work_items);
    plan.cores.assign(all_cores.begin(), all_cores.begin() + num_active);
    const uint32_t items_per_core_floor = plan.num_work_items / num_active;
    const uint32_t cores_with_extra_item = plan.num_work_items % num_active;
    plan.first_work_item.resize(num_active);
    plan.work_items_per_core.resize(num_active);
    for (uint32_t core = 0; core < num_active; ++core) {
        plan.work_items_per_core[core] = items_per_core_floor + (core < cores_with_extra_item ? 1 : 0);
        plan.first_work_item[core] = core * items_per_core_floor + std::min(core, cores_with_extra_item);
    }
    plan.max_work_items_per_core = plan.work_items_per_core.front();

    // ---- C tiles accumulated in DST at once ----
    const bool fp32_dest_acc_en = get_fp32_dest_acc_en(attributes.compute_kernel_config);
    const bool packer_l1_acc =
        std::get<3>(get_compute_kernel_config_args(A.device()->arch(), attributes.compute_kernel_config.value()));
    if (config.dst_M_tiles == 0 && config.dst_N_tiles == 0) {
        // The chooser's (h, w) is (M tiles, N tiles) of the DST group.
        const std::tuple<uint32_t, uint32_t> dst_tiles =
            operations::experimental::quasar::matmul::bmm_op_utils_qsr::get_matmul_subblock_params(
                plan.per_core_M, plan.per_core_N, false, false, fp32_dest_acc_en);
        plan.dst_M_tiles = std::get<0>(dst_tiles);
        plan.dst_N_tiles = std::get<1>(dst_tiles);
    } else {
        TT_FATAL(
            config.dst_M_tiles > 0 && config.dst_N_tiles > 0,
            "dst_M_tiles and dst_N_tiles must both be set or both be 0 (auto)");
        plan.dst_M_tiles = config.dst_M_tiles;
        plan.dst_N_tiles = config.dst_N_tiles;
    }
    TT_FATAL(
        plan.per_core_M % plan.dst_M_tiles == 0 && plan.per_core_N % plan.dst_N_tiles == 0,
        "DST group {}x{} tiles must divide the per-core C subblock {}x{}",
        plan.dst_M_tiles,
        plan.dst_N_tiles,
        plan.per_core_M,
        plan.per_core_N);
    const uint32_t dst_capacity_tiles = fp32_dest_acc_en ? 4 : 8;
    TT_FATAL(
        plan.dst_M_tiles * plan.dst_N_tiles <= dst_capacity_tiles,
        "DST group {}x{} holds {} tiles; DST fits {} (fp32 accumulation: {})",
        plan.dst_M_tiles,
        plan.dst_N_tiles,
        plan.dst_M_tiles * plan.dst_N_tiles,
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
            "MatmulUnifiedProgramConfig: a {}x{}-tile C subblock does not fit L1 even with K_iteration_tiles=1 "
            "(needs {} B, budget {} B, max ring {} B); shrink per_core_M / per_core_N",
            plan.per_core_M,
            plan.per_core_N,
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
            "MatmulUnifiedProgramConfig: rings for a {}x{}-tile C subblock with K_iteration_tiles={} do not fit "
            "(needs {} B, budget {} B, max ring {} B: A slice {} B, B slice {} B, C subblock {} B, C partials {} B)",
            plan.per_core_M,
            plan.per_core_N,
            plan.K_iteration_tiles,
            plan.l1_bytes,
            l1_budget,
            MAX_DFB_RING_BYTES,
            (uint64_t)plan.A_slice_ring_slots * plan.A_slot_bytes,
            (uint64_t)plan.B_slice_ring_slots * plan.B_slot_bytes,
            (uint64_t)plan.C_subblock_ring_slots * plan.C_slot_bytes,
            (uint64_t)plan.C_partials_ring_slots * plan.C_partials_slot_bytes);
    }

    // ---- Sharded output: one C subblock per core, batch 1, and a grid the accessor maps the same way ----
    if (attributes.output_mem_config.is_sharded()) {
        TT_FATAL(plan.batch_size == 1, "Sharded output needs batch 1 (a core's C subblocks would not form one shard)");
        TT_FATAL(
            plan.num_work_items == plan.cores.size(),
            "Sharded output needs exactly one C subblock per core ({} C subblocks, {} active cores)",
            plan.num_work_items,
            plan.cores.size());
        if (plan.sharded_output_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED) {
            const std::vector<CoreRange>& ranges = config.cores.ranges();
            const bool one_rectangle = ranges.size() == 1;
            const uint32_t rectangle_columns = one_rectangle ? ranges[0].grid_size().x : 0;
            const uint32_t rectangle_rows = one_rectangle ? ranges[0].grid_size().y : 0;
            TT_FATAL(
                one_rectangle && rectangle_columns == plan.C_subblock_grid_columns &&
                    rectangle_rows == plan.C_subblock_grid_rows,
                "Block-sharded output needs cores to be one {}x{} rectangle (the C subblock grid), got {}",
                plan.C_subblock_grid_columns,
                plan.C_subblock_grid_rows,
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
        DataflowBufferSpec C_subblock_dfb{
            .unique_id = C_SUBBLOCK_DFB,
            .entry_size = plan.C_slot_bytes,
            .num_entries = plan.C_subblock_ring_slots,
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
        if (plan.alias_C_partials_onto_C_subblock) {
            C_subblock_dfb.advanced_options.alias_with = {C_PARTIALS_DFB};
            C_partials_dfb.advanced_options.alias_with = {C_SUBBLOCK_DFB};
        }
        dataflow_buffers.push_back(std::move(C_subblock_dfb));
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
                {"C_subblocks_per_batch", plan.C_subblocks_per_batch},
                {"C_subblock_grid_columns", plan.C_subblock_grid_columns},
                {"broadcast_B_over_batch", plan.broadcast_B_over_batch ? 1u : 0u},
                {"per_core_M", plan.per_core_M},
                {"per_core_N", plan.per_core_N},
                {"K_iteration_tiles", plan.K_iteration_tiles},
                {"num_K_iterations", plan.num_K_iterations},
                {"A_last_K_tile_valid_columns", A_last_K_tile_valid_columns},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"first_work_item", "num_work_items"}},
        .hw_config =
            ttnn::create_reader_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // ---- Writer ----
    KernelSpec writer{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path(std::string(KERNEL_DIR) + "dataflow/unified_matmul_writer.cpp"),
        .compiler_options = {},
        .dfb_bindings = {ConsumerOf(C_SUBBLOCK_DFB, "C_subblock")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = C_TENSOR, .accessor_name = "C"}},
        .compile_time_args =
            {
                {"M_tiles", plan.M_tiles},
                {"N_tiles", plan.N_tiles},
                {"C_subblocks_per_batch", plan.C_subblocks_per_batch},
                {"C_subblock_grid_columns", plan.C_subblock_grid_columns},
                {"per_core_M", plan.per_core_M},
                {"per_core_N", plan.per_core_N},
                {"dst_M_tiles", plan.dst_M_tiles},
                {"dst_N_tiles", plan.dst_N_tiles},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"first_work_item", "num_work_items"}},
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
                ProducerOf(C_SUBBLOCK_DFB, "C_subblock"),
                ProducerOf(C_PARTIALS_DFB, "C_partials"),
                ConsumerOf(C_PARTIALS_DFB, "C_partials"),
            },
        .compile_time_args =
            {
                {"K_iteration_tiles", plan.K_iteration_tiles},
                {"num_K_iterations", plan.num_K_iterations},
                {"per_core_M", plan.per_core_M},
                {"per_core_N", plan.per_core_N},
                {"dst_M_tiles", plan.dst_M_tiles},
                {"dst_N_tiles", plan.dst_N_tiles},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_work_items"}},
        .hw_config = compute_hw_config,
    };

    // ---- One work unit over the active cores ----
    const CoreRangeSet active_cores(ttsl::Span<const CoreCoord>(plan.cores));
    Group<WorkUnitSpec> work_units = {WorkUnitSpec{
        .name = "unified_matmul",
        .kernels = {READER_KERNEL, COMPUTE_KERNEL, WRITER_KERNEL},
        .target_nodes = active_cores,
    }};

    // ---- Per-core runtime args: each core's run of work items ----
    ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = READER_KERNEL};
    ProgramRunArgs::KernelRunArgs compute_run_args{.kernel = COMPUTE_KERNEL};
    ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = WRITER_KERNEL};
    for (uint32_t core = 0; core < plan.cores.size(); ++core) {
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            plan.cores[core],
            {{"first_work_item", plan.first_work_item[core]}, {"num_work_items", plan.work_items_per_core[core]}});
        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values,
            plan.cores[core],
            {{"num_work_items", plan.work_items_per_core[core]}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            plan.cores[core],
            {{"first_work_item", plan.first_work_item[core]}, {"num_work_items", plan.work_items_per_core[core]}});
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
