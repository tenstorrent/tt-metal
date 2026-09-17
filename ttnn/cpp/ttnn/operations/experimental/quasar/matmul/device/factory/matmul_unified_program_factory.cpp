// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified (placement-first) matmul factory: stage A of the Quasar-native matmul (GH#41910).
//
// One Metal 2.0 program for every placement. The config names the cores and the C block per core; this
// file turns that into a work-item assignment, four DFB rings (A panel, B panel, C block, C partials),
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

// Names the kernels see: dfb::A_panel / B_panel / C_block / C_partials and tensor::A / B / C.
const DFBSpecName A_PANEL_DFB{"A_panel"};
const DFBSpecName B_PANEL_DFB{"B_panel"};
const DFBSpecName C_BLOCK_DFB{"C_block"};
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
constexpr uint32_t MAX_AUTO_K_STEP_TILES = 8;

uint64_t l1_budget_bytes(tt::tt_metal::IDevice* device) {
    const uint32_t l1_base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const std::optional<tt::tt_metal::DeviceAddr> lowest_occupied = device->lowest_occupied_compute_l1_address();
    const uint32_t l1_ceiling =
        lowest_occupied.has_value() ? static_cast<uint32_t>(lowest_occupied.value()) : device->l1_size_per_core();
    TT_FATAL(l1_ceiling > l1_base, "L1 ceiling ({}) must exceed base ({})", l1_ceiling, l1_base);
    return l1_ceiling - l1_base;
}

// Fills the ring sizing of `plan` for a given K step. Returns the total footprint in bytes.
uint64_t size_rings(UnifiedMatmulPlan& plan, uint32_t K_step_tiles, bool fp32_dest_acc_en, bool packer_l1_acc) {
    plan.K_step_tiles = K_step_tiles;
    plan.num_K_steps = plan.K_tiles / K_step_tiles;

    // The packer accumulates partials in L1 only when there are enough K steps for the reconfig overhead
    // to pay off (the last step spills and reloads either way, so more than two).
    plan.packer_l1_acc_en = packer_l1_acc && plan.num_K_steps > 2;
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

    const uint32_t A_panel_tiles = plan.per_core_M * K_step_tiles;
    const uint32_t B_panel_tiles = K_step_tiles * plan.per_core_N;
    const uint32_t C_block_tiles = plan.per_core_M * plan.per_core_N;
    // Double-buffer the panels whenever more than one panel passes through the ring.
    const bool more_than_one_panel = (uint64_t)plan.max_work_items_per_core * plan.num_K_steps > 1;
    const uint32_t panel_ring_depth = more_than_one_panel ? 2 : 1;
    plan.A_panel_ring_slots = A_panel_tiles * panel_ring_depth;
    plan.B_panel_ring_slots = B_panel_tiles * panel_ring_depth;
    plan.C_block_ring_slots = C_block_tiles;
    plan.C_partials_ring_slots = C_block_tiles;

    // Aliasing C_partials onto C_block with more than one work item per core is a race: the writer may
    // still be draining item i from C_block while the compute packs item i+1's first partials into the
    // same bytes. Alias only when the partials can never be live while C_block holds unread data: a
    // single work item per core, or no partials at all (one K step).
    const bool partials_ever_written = plan.num_K_steps > 1;
    const bool one_work_item_per_core = plan.max_work_items_per_core == 1;
    plan.alias_C_partials_onto_C_block =
        (plan.C_partials_format == plan.C_format) && (!partials_ever_written || one_work_item_per_core);

    plan.l1_bytes =
        (uint64_t)plan.A_panel_ring_slots * plan.A_slot_bytes + (uint64_t)plan.B_panel_ring_slots * plan.B_slot_bytes +
        (uint64_t)plan.C_block_ring_slots * plan.C_slot_bytes +
        (plan.alias_C_partials_onto_C_block ? 0 : (uint64_t)plan.C_partials_ring_slots * plan.C_partials_slot_bytes);
    return plan.l1_bytes;
}

bool rings_fit(const UnifiedMatmulPlan& plan, uint64_t l1_budget) {
    const uint64_t rings[] = {
        (uint64_t)plan.A_panel_ring_slots * plan.A_slot_bytes,
        (uint64_t)plan.B_panel_ring_slots * plan.B_slot_bytes,
        (uint64_t)plan.C_block_ring_slots * plan.C_slot_bytes,
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
    if (num_C_block_columns == 1) {
        return tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED;
    }
    if (num_C_block_rows == 1) {
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

    // ---- C block grid and work-item assignment ----
    TT_FATAL(config.per_core_M > 0 && config.per_core_N > 0, "per_core_M and per_core_N must be > 0");
    plan.per_core_M = config.per_core_M;
    plan.per_core_N = config.per_core_N;
    plan.num_C_block_rows = tt::div_up(plan.M_tiles, plan.per_core_M);
    plan.num_C_block_columns = tt::div_up(plan.N_tiles, plan.per_core_N);
    plan.num_C_blocks = plan.num_C_block_rows * plan.num_C_block_columns;
    plan.num_work_items = plan.batch_size * plan.num_C_blocks;

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

    // ---- DST subblock ----
    const bool fp32_dest_acc_en = get_fp32_dest_acc_en(attributes.compute_kernel_config);
    const bool packer_l1_acc =
        std::get<3>(get_compute_kernel_config_args(A.device()->arch(), attributes.compute_kernel_config.value()));
    if (config.subblock_M_tiles == 0 && config.subblock_N_tiles == 0) {
        // The chooser's (h, w) is (M tiles, N tiles) of the subblock.
        const std::tuple<uint32_t, uint32_t> subblock =
            operations::experimental::quasar::matmul::bmm_op_utils_qsr::get_matmul_subblock_params(
                plan.per_core_M, plan.per_core_N, false, false, fp32_dest_acc_en);
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
        plan.per_core_M % plan.subblock_M_tiles == 0 && plan.per_core_N % plan.subblock_N_tiles == 0,
        "subblock {}x{} must divide the per-core C block {}x{}",
        plan.subblock_M_tiles,
        plan.subblock_N_tiles,
        plan.per_core_M,
        plan.per_core_N);
    const uint32_t dst_capacity_tiles = fp32_dest_acc_en ? 4 : 8;
    TT_FATAL(
        plan.subblock_M_tiles * plan.subblock_N_tiles <= dst_capacity_tiles,
        "subblock {}x{} holds {} tiles; DST fits {} (fp32 accumulation: {})",
        plan.subblock_M_tiles,
        plan.subblock_N_tiles,
        plan.subblock_M_tiles * plan.subblock_N_tiles,
        dst_capacity_tiles,
        fp32_dest_acc_en);

    // ---- Formats, K step and ring sizing ----
    plan.A_format = tt::tt_metal::datatype_to_dataformat_converter(A.dtype());
    plan.B_format = tt::tt_metal::datatype_to_dataformat_converter(B.dtype());
    plan.C_format = tt::tt_metal::datatype_to_dataformat_converter(attributes.output_dtype.value());
    const uint64_t l1_budget = l1_budget_bytes(A.device());
    if (config.K_step_tiles == 0) {
        // Largest divisor of K_tiles (capped) whose rings fit; 1 is the floor and must fit.
        uint32_t chosen = 0;
        for (uint32_t K_step_tiles = std::min<uint32_t>(plan.K_tiles, MAX_AUTO_K_STEP_TILES); K_step_tiles >= 1;
             --K_step_tiles) {
            if (plan.K_tiles % K_step_tiles != 0) {
                continue;
            }
            size_rings(plan, K_step_tiles, fp32_dest_acc_en, packer_l1_acc);
            if (rings_fit(plan, l1_budget)) {
                chosen = K_step_tiles;
                break;
            }
        }
        TT_FATAL(
            chosen > 0,
            "MatmulUnifiedProgramConfig: a {}x{}-tile C block does not fit L1 even with K_step_tiles=1 "
            "(needs {} B, budget {} B, max ring {} B); shrink per_core_M / per_core_N",
            plan.per_core_M,
            plan.per_core_N,
            plan.l1_bytes,
            l1_budget,
            MAX_DFB_RING_BYTES);
    } else {
        TT_FATAL(
            plan.K_tiles % config.K_step_tiles == 0,
            "K_step_tiles ({}) must divide K_tiles ({})",
            config.K_step_tiles,
            plan.K_tiles);
        size_rings(plan, config.K_step_tiles, fp32_dest_acc_en, packer_l1_acc);
        TT_FATAL(
            rings_fit(plan, l1_budget),
            "MatmulUnifiedProgramConfig: rings for a {}x{}-tile C block with K_step_tiles={} do not fit "
            "(needs {} B, budget {} B, max ring {} B: A panel {} B, B panel {} B, C block {} B, C partials {} B)",
            plan.per_core_M,
            plan.per_core_N,
            plan.K_step_tiles,
            plan.l1_bytes,
            l1_budget,
            MAX_DFB_RING_BYTES,
            (uint64_t)plan.A_panel_ring_slots * plan.A_slot_bytes,
            (uint64_t)plan.B_panel_ring_slots * plan.B_slot_bytes,
            (uint64_t)plan.C_block_ring_slots * plan.C_slot_bytes,
            (uint64_t)plan.C_partials_ring_slots * plan.C_partials_slot_bytes);
    }

    // ---- Sharded output: one C block per core, batch 1, and a grid the accessor maps the same way ----
    if (attributes.output_mem_config.is_sharded()) {
        TT_FATAL(plan.batch_size == 1, "Sharded output needs batch 1 (a core's C blocks would not form one shard)");
        TT_FATAL(
            plan.num_work_items == plan.cores.size(),
            "Sharded output needs exactly one C block per core ({} C blocks, {} active cores)",
            plan.num_work_items,
            plan.cores.size());
        if (plan.sharded_output_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED) {
            const std::vector<CoreRange>& ranges = config.cores.ranges();
            const bool one_rectangle = ranges.size() == 1;
            const uint32_t rectangle_columns = one_rectangle ? ranges[0].grid_size().x : 0;
            const uint32_t rectangle_rows = one_rectangle ? ranges[0].grid_size().y : 0;
            TT_FATAL(
                one_rectangle && rectangle_columns == plan.num_C_block_columns &&
                    rectangle_rows == plan.num_C_block_rows,
                "Block-sharded output needs cores to be one {}x{} rectangle (the C block grid), got {}",
                plan.num_C_block_columns,
                plan.num_C_block_rows,
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
            .unique_id = A_PANEL_DFB,
            .entry_size = plan.A_slot_bytes,
            .num_entries = plan.A_panel_ring_slots,
            .data_format_metadata = plan.A_format,
            .tile_format_metadata = A.tensor_spec().tile(),
        },
        DataflowBufferSpec{
            .unique_id = B_PANEL_DFB,
            .entry_size = plan.B_slot_bytes,
            .num_entries = plan.B_panel_ring_slots,
            .data_format_metadata = plan.B_format,
            .tile_format_metadata = B.tensor_spec().tile(),
        },
    };
    {
        DataflowBufferSpec C_block_dfb{
            .unique_id = C_BLOCK_DFB,
            .entry_size = plan.C_slot_bytes,
            .num_entries = plan.C_block_ring_slots,
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
        if (plan.alias_C_partials_onto_C_block) {
            C_block_dfb.advanced_options.alias_with = {C_PARTIALS_DFB};
            C_partials_dfb.advanced_options.alias_with = {C_BLOCK_DFB};
        }
        dataflow_buffers.push_back(std::move(C_block_dfb));
        dataflow_buffers.push_back(std::move(C_partials_dfb));
    }

    // ---- Reader ----
    const uint32_t A_last_K_tile_valid_columns = A.logical_shape()[-1] % TILE_WIDTH;
    KernelSpec reader{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path(std::string(KERNEL_DIR) + "dataflow/unified_matmul_reader.cpp"),
        .compiler_options = {},
        .dfb_bindings = {ProducerOf(A_PANEL_DFB, "A_panel"), ProducerOf(B_PANEL_DFB, "B_panel")},
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
                {"num_C_blocks", plan.num_C_blocks},
                {"num_C_block_columns", plan.num_C_block_columns},
                {"broadcast_B_over_batch", plan.broadcast_B_over_batch ? 1u : 0u},
                {"per_core_M", plan.per_core_M},
                {"per_core_N", plan.per_core_N},
                {"K_step_tiles", plan.K_step_tiles},
                {"num_K_steps", plan.num_K_steps},
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
        .dfb_bindings = {ConsumerOf(C_BLOCK_DFB, "C_block")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = C_TENSOR, .accessor_name = "C"}},
        .compile_time_args =
            {
                {"M_tiles", plan.M_tiles},
                {"N_tiles", plan.N_tiles},
                {"num_C_blocks", plan.num_C_blocks},
                {"num_C_block_columns", plan.num_C_block_columns},
                {"per_core_M", plan.per_core_M},
                {"per_core_N", plan.per_core_N},
                {"subblock_M_tiles", plan.subblock_M_tiles},
                {"subblock_N_tiles", plan.subblock_N_tiles},
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
            modes.emplace(A_PANEL_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
        if (is_32bit(plan.B_format)) {
            modes.emplace(B_PANEL_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
    }

    KernelSpec compute{
        .unique_id = COMPUTE_KERNEL,
        .source = std::filesystem::path(std::string(KERNEL_DIR) + "compute/unified_matmul_compute.cpp"),
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings =
            {
                ConsumerOf(A_PANEL_DFB, "A_panel"),
                ConsumerOf(B_PANEL_DFB, "B_panel"),
                ProducerOf(C_BLOCK_DFB, "C_block"),
                ProducerOf(C_PARTIALS_DFB, "C_partials"),
                ConsumerOf(C_PARTIALS_DFB, "C_partials"),
            },
        .compile_time_args =
            {
                {"K_step_tiles", plan.K_step_tiles},
                {"num_K_steps", plan.num_K_steps},
                {"per_core_M", plan.per_core_M},
                {"per_core_N", plan.per_core_N},
                {"subblock_M_tiles", plan.subblock_M_tiles},
                {"subblock_N_tiles", plan.subblock_N_tiles},
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
