// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified (placement-first) matmul factory: stage A of the Quasar-native matmul (GH#41910).
//
// One Metal 2.0 program for every placement. The config names the cores and the output block per
// core; this file turns that into a block assignment, three DFB rings (+ a partials ring), one reader,
// one writer and one compute kernel per distinct block count. Nothing here depends on how the operands
// are laid out in memory: the kernels address tiles by page id through the tensor accessor.

#include "ttnn/operations/experimental/quasar/matmul/device/factory/matmul_unified_program_factory.hpp"

#include <algorithm>
#include <filesystem>
#include <map>
#include <string>

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

// DFB names surface kernel-side as dfb::cb_in0 / cb_in1 / cb_out / cb_intermed0 -- the names the
// shared compute kernel (bmm_large_block_zm_fused_bias_activation_metal2.cpp) is written against.
const DFBSpecName IN0_DFB{"cb_in0"};
const DFBSpecName IN1_DFB{"cb_in1"};
const DFBSpecName OUT_DFB{"cb_out"};
const DFBSpecName INTERM0_DFB{"cb_intermed0"};

const TensorParamName IN0_TENSOR{"in0"};
const TensorParamName IN1_TENSOR{"in1"};
const TensorParamName OUT_TENSOR{"out"};

const KernelSpecName READER_KERNEL{"reader"};
const KernelSpecName WRITER_KERNEL{"writer"};
const KernelSpecName COMPUTE_KERNEL_G1{"compute_g1"};
const KernelSpecName COMPUTE_KERNEL_G2{"compute_g2"};

constexpr const char* KERNEL_DIR = "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/";

// A DFB touched by a TRISC keeps its ring extent in a uint16_t of 16-byte units (see
// validate_ring_extent in dataflow_buffer.cpp). Enforced on every arch so a config that is legal on
// Wormhole never becomes a program-creation FATAL on Quasar.
constexpr uint64_t MAX_DFB_RING_BYTES = 65535ull * 16ull;
constexpr uint32_t MAX_AUTO_IN0_BLOCK_W = 8;

uint64_t l1_budget_bytes(tt::tt_metal::IDevice* device) {
    const uint32_t l1_base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const std::optional<tt::tt_metal::DeviceAddr> lowest_occupied = device->lowest_occupied_compute_l1_address();
    const uint32_t l1_ceiling =
        lowest_occupied.has_value() ? static_cast<uint32_t>(lowest_occupied.value()) : device->l1_size_per_core();
    TT_FATAL(l1_ceiling > l1_base, "L1 ceiling ({}) must exceed base ({})", l1_ceiling, l1_base);
    return l1_ceiling - l1_base;
}

// Fills the DFB sizing of `p` for a given in0_block_w. Returns the total footprint.
uint64_t size_buffers(UnifiedMatmulPlan& p, uint32_t in0_block_w, bool fp32_dest_acc_en, bool packer_l1_acc) {
    p.in0_block_w = in0_block_w;
    p.num_blocks_inner_dim = p.Kt / in0_block_w;

    // Same rule as the mcast factories: the packer accumulates in L1 only when there are enough K
    // blocks for the reconfig overhead to pay off (last block spills and reloads, so > 2).
    p.packer_l1_acc_en = packer_l1_acc && p.num_blocks_inner_dim > 2;
    p.interm_format = p.packer_l1_acc_en ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                         : (fp32_dest_acc_en ? tt::DataFormat::Float32 : p.out_format);

    // Interleaved tiles live in DRAM at the DRAM-aligned stride; the reader copies at that stride
    // too. No-op for every 32x32 format.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    p.in0_entry_size = tt::align(tt::tile_size(p.in0_format), dram_alignment);
    p.in1_entry_size = tt::align(tt::tile_size(p.in1_format), dram_alignment);
    p.out_entry_size = tt::tile_size(p.out_format);
    p.interm_entry_size = tt::tile_size(p.interm_format);

    const uint32_t in0_block_tiles = p.per_core_M * in0_block_w;
    const uint32_t in1_block_tiles = in0_block_w * p.per_core_N;
    const uint32_t out_block_tiles = p.per_core_M * p.per_core_N;
    // Double-buffer the inputs whenever more than one block passes through the ring.
    const bool more_than_one_input_block = (uint64_t)p.B * p.max_blocks_per_core * p.num_blocks_inner_dim > 1;
    const uint32_t depth = more_than_one_input_block ? 2 : 1;
    p.in0_entries = in0_block_tiles * depth;
    p.in1_entries = in1_block_tiles * depth;
    p.out_entries = out_block_tiles;
    p.interm_entries = out_block_tiles;

    // The legacy factories alias interm onto out whenever the formats match. With more than one
    // output block per core that is a race: the writer may still be draining block i from the out
    // ring while the compute packs block i+1's first partials into the same bytes through the interm
    // ring. Alias only when interm can never be live while out holds unread data: a single output
    // block per core, or no spill at all (one K block).
    const bool interm_ever_written = p.num_blocks_inner_dim > 1;
    const bool one_output_block_per_core = p.B == 1 && p.max_blocks_per_core == 1;
    p.alias_out_interm = (p.interm_format == p.out_format) && (!interm_ever_written || one_output_block_per_core);

    p.l1_bytes = (uint64_t)p.in0_entries * p.in0_entry_size + (uint64_t)p.in1_entries * p.in1_entry_size +
                 (uint64_t)p.out_entries * p.out_entry_size +
                 (p.alias_out_interm ? 0 : (uint64_t)p.interm_entries * p.interm_entry_size);
    return p.l1_bytes;
}

bool rings_fit(const UnifiedMatmulPlan& p, uint64_t l1_budget) {
    const uint64_t rings[] = {
        (uint64_t)p.in0_entries * p.in0_entry_size,
        (uint64_t)p.in1_entries * p.in1_entry_size,
        (uint64_t)p.out_entries * p.out_entry_size,
        (uint64_t)p.interm_entries * p.interm_entry_size};
    for (uint64_t r : rings) {
        if (r > MAX_DFB_RING_BYTES) {
            return false;
        }
    }
    return p.l1_bytes <= l1_budget;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

tt::tt_metal::TensorMemoryLayout UnifiedMatmulPlan::sharded_output_layout() const {
    if (num_block_cols == 1) {
        return tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED;
    }
    if (num_block_rows == 1) {
        return tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
    }
    return tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
}

UnifiedMatmulPlan plan_unified_matmul(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig& config,
    const MatmulParams& attributes) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    UnifiedMatmulPlan p{};

    TT_FATAL(
        !attributes.transpose_a && !attributes.transpose_b,
        "MatmulUnifiedProgramConfig: transposes are applied to the operands before the op, not in the kernels");
    TT_FATAL(!attributes.untilize_out, "MatmulUnifiedProgramConfig does not support untilize_out");
    TT_FATAL(attributes.bcast_batch.has_value(), "bcast_batch should have been populated");
    TT_FATAL(attributes.compute_kernel_config.has_value(), "compute_kernel_config should have been populated");
    TT_FATAL(attributes.output_dtype.has_value(), "output_dtype should have been populated");

    const tt::tt_metal::Shape& ashape = a.padded_shape();
    const tt::tt_metal::Shape& bshape = b.padded_shape();
    const tt::tt_metal::Tile in0_tile = a.tensor_spec().tile();
    const tt::tt_metal::Tile in1_tile = b.tensor_spec().tile();
    TT_FATAL(
        in0_tile.get_height() == TILE_HEIGHT && in0_tile.get_width() == TILE_WIDTH &&
            in1_tile.get_height() == TILE_HEIGHT && in1_tile.get_width() == TILE_WIDTH,
        "MatmulUnifiedProgramConfig supports 32x32 tiles only (in0 {}x{}, in1 {}x{})",
        in0_tile.get_height(),
        in0_tile.get_width(),
        in1_tile.get_height(),
        in1_tile.get_width());
    if (attributes.output_tile.has_value()) {
        TT_FATAL(
            attributes.output_tile->get_tile_shape()[0] == TILE_HEIGHT &&
                attributes.output_tile->get_tile_shape()[1] == TILE_WIDTH,
            "MatmulUnifiedProgramConfig supports a 32x32 output tile only");
    }

    p.B = get_batch_size(ashape);
    p.Mt = ashape[-2] / TILE_HEIGHT;
    p.Kt = ashape[-1] / TILE_WIDTH;
    p.Nt = bshape[-1] / TILE_WIDTH;
    p.bcast_batch = attributes.bcast_batch.value();
    TT_FATAL(
        p.bcast_batch || get_batch_size(bshape) == p.B,
        "Batched in1 must match in0's batch ({} vs {})",
        get_batch_size(bshape),
        p.B);

    // ---- Output block grid and core assignment ----
    TT_FATAL(config.per_core_M > 0 && config.per_core_N > 0, "per_core_M and per_core_N must be > 0");
    p.per_core_M = config.per_core_M;
    p.per_core_N = config.per_core_N;
    p.num_block_rows = tt::div_up(p.Mt, p.per_core_M);
    p.num_block_cols = tt::div_up(p.Nt, p.per_core_N);
    p.num_blocks = p.num_block_rows * p.num_block_cols;

    TT_FATAL(config.cores.num_cores() > 0, "MatmulUnifiedProgramConfig.cores is empty");
    const CoreCoord grid = a.device()->compute_with_storage_grid_size();
    const CoreRange bbox = config.cores.bounding_box();
    TT_FATAL(
        bbox.end_coord.x < grid.x && bbox.end_coord.y < grid.y,
        "MatmulUnifiedProgramConfig.cores {} exceed the device compute grid {}x{}",
        config.cores.str(),
        grid.x,
        grid.y);
    p.row_major_cores = config.row_major_cores;
    const std::vector<CoreCoord> all_cores = corerange_to_cores(config.cores, std::nullopt, config.row_major_cores);
    const uint32_t num_active = std::min<uint32_t>(all_cores.size(), p.num_blocks);
    p.cores.assign(all_cores.begin(), all_cores.begin() + num_active);
    const uint32_t q = p.num_blocks / num_active;
    const uint32_t r = p.num_blocks % num_active;
    p.block_start.resize(num_active);
    p.blocks_per_core.resize(num_active);
    for (uint32_t i = 0; i < num_active; ++i) {
        p.blocks_per_core[i] = q + (i < r ? 1 : 0);
        p.block_start[i] = i * q + std::min(i, r);
    }
    p.max_blocks_per_core = p.blocks_per_core.front();

    // ---- DST subblock ----
    const bool fp32_dest_acc_en = get_fp32_dest_acc_en(attributes.compute_kernel_config);
    const bool packer_l1_acc =
        std::get<3>(get_compute_kernel_config_args(a.device()->arch(), attributes.compute_kernel_config.value()));
    if (config.out_subblock_h == 0 && config.out_subblock_w == 0) {
        const std::tuple<uint32_t, uint32_t> subblock_hw =
            operations::experimental::quasar::matmul::bmm_op_utils_qsr::get_matmul_subblock_params(
                p.per_core_M, p.per_core_N, false, false, fp32_dest_acc_en);
        p.out_subblock_h = std::get<0>(subblock_hw);
        p.out_subblock_w = std::get<1>(subblock_hw);
    } else {
        TT_FATAL(
            config.out_subblock_h > 0 && config.out_subblock_w > 0,
            "out_subblock_h and out_subblock_w must both be set or both be 0 (auto)");
        p.out_subblock_h = config.out_subblock_h;
        p.out_subblock_w = config.out_subblock_w;
    }
    TT_FATAL(
        p.per_core_M % p.out_subblock_h == 0 && p.per_core_N % p.out_subblock_w == 0,
        "out_subblock {}x{} must divide the per-core block {}x{}",
        p.out_subblock_h,
        p.out_subblock_w,
        p.per_core_M,
        p.per_core_N);
    const uint32_t dst_capacity = fp32_dest_acc_en ? 4 : 8;
    TT_FATAL(
        p.out_subblock_h * p.out_subblock_w <= dst_capacity,
        "out_subblock {}x{} holds {} tiles; DST fits {} (fp32 accumulation: {})",
        p.out_subblock_h,
        p.out_subblock_w,
        p.out_subblock_h * p.out_subblock_w,
        dst_capacity,
        fp32_dest_acc_en);

    // ---- Formats and K blocking / buffer sizing ----
    p.in0_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    p.in1_format = tt::tt_metal::datatype_to_dataformat_converter(b.dtype());
    p.out_format = tt::tt_metal::datatype_to_dataformat_converter(attributes.output_dtype.value());
    const uint64_t l1_budget = l1_budget_bytes(a.device());
    if (config.in0_block_w == 0) {
        // Largest divisor of Kt (capped) whose rings fit; bw = 1 is the floor and must fit.
        uint32_t chosen = 0;
        for (uint32_t bw = std::min<uint32_t>(p.Kt, MAX_AUTO_IN0_BLOCK_W); bw >= 1; --bw) {
            if (p.Kt % bw != 0) {
                continue;
            }
            size_buffers(p, bw, fp32_dest_acc_en, packer_l1_acc);
            if (rings_fit(p, l1_budget)) {
                chosen = bw;
                break;
            }
        }
        TT_FATAL(
            chosen > 0,
            "MatmulUnifiedProgramConfig: a {}x{}-tile output block does not fit L1 even with in0_block_w=1 "
            "(needs {} B, budget {} B, max ring {} B); shrink per_core_M / per_core_N",
            p.per_core_M,
            p.per_core_N,
            p.l1_bytes,
            l1_budget,
            MAX_DFB_RING_BYTES);
    } else {
        TT_FATAL(p.Kt % config.in0_block_w == 0, "in0_block_w ({}) must divide Kt ({})", config.in0_block_w, p.Kt);
        size_buffers(p, config.in0_block_w, fp32_dest_acc_en, packer_l1_acc);
        TT_FATAL(
            rings_fit(p, l1_budget),
            "MatmulUnifiedProgramConfig: buffers for a {}x{}-tile block with in0_block_w={} do not fit "
            "(needs {} B, budget {} B, max ring {} B: in0 {} B, in1 {} B, out {} B, interm {} B)",
            p.per_core_M,
            p.per_core_N,
            p.in0_block_w,
            p.l1_bytes,
            l1_budget,
            MAX_DFB_RING_BYTES,
            (uint64_t)p.in0_entries * p.in0_entry_size,
            (uint64_t)p.in1_entries * p.in1_entry_size,
            (uint64_t)p.out_entries * p.out_entry_size,
            (uint64_t)p.interm_entries * p.interm_entry_size);
    }

    // ---- Sharded output: one block per core, batch 1, and a grid the accessor maps the same way ----
    if (attributes.output_mem_config.is_sharded()) {
        TT_FATAL(p.B == 1, "Sharded output needs batch 1 (a core's blocks would not form one shard)");
        TT_FATAL(
            p.num_blocks == p.cores.size(),
            "Sharded output needs exactly one output block per core ({} blocks, {} active cores)",
            p.num_blocks,
            p.cores.size());
        if (p.sharded_output_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED) {
            const std::vector<CoreRange>& ranges = config.cores.ranges();
            const bool one_rect = ranges.size() == 1;
            const uint32_t rect_w = one_rect ? ranges[0].grid_size().x : 0;
            const uint32_t rect_h = one_rect ? ranges[0].grid_size().y : 0;
            const bool matches =
                one_rect && (config.row_major_cores ? (rect_w == p.num_block_cols && rect_h == p.num_block_rows)
                                                    : (rect_h == p.num_block_rows && rect_w == p.num_block_cols));
            TT_FATAL(
                matches,
                "Block-sharded output needs cores to be one {}x{} rectangle (block grid), got {}",
                p.num_block_cols,
                p.num_block_rows,
                config.cores.str());
        }
    }
    return p;
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

    const ttnn::Tensor& a_tensor = tensor_args.input_tensors.at(0);
    const ttnn::Tensor& b_tensor = tensor_args.input_tensors.at(1);
    const tt::tt_metal::MeshTensor& a = a_tensor.mesh_tensor();
    const tt::tt_metal::MeshTensor& b = b_tensor.mesh_tensor();
    const tt::tt_metal::MeshTensor& output = tensor_return_value.at(0).mesh_tensor();
    tt::tt_metal::IDevice* device = &a.mutable_device();

    const UnifiedMatmulPlan p = plan_unified_matmul(a_tensor, b_tensor, config, operation_attributes);

    // ---- Tensor parameters ----
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = IN0_TENSOR, .spec = a.tensor_spec()},
        TensorParameter{.unique_id = IN1_TENSOR, .spec = b.tensor_spec()},
        TensorParameter{.unique_id = OUT_TENSOR, .spec = output.tensor_spec()},
    };

    // ---- Dataflow buffers ----
    const tt::tt_metal::Tile out_tile = output.tensor_spec().tile();
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = IN0_DFB,
            .entry_size = p.in0_entry_size,
            .num_entries = p.in0_entries,
            .data_format_metadata = p.in0_format,
            .tile_format_metadata = a.tensor_spec().tile(),
        },
        DataflowBufferSpec{
            .unique_id = IN1_DFB,
            .entry_size = p.in1_entry_size,
            .num_entries = p.in1_entries,
            .data_format_metadata = p.in1_format,
            .tile_format_metadata = b.tensor_spec().tile(),
        },
    };
    {
        DataflowBufferSpec out_dfb{
            .unique_id = OUT_DFB,
            .entry_size = p.out_entry_size,
            .num_entries = p.out_entries,
            .data_format_metadata = p.out_format,
            .tile_format_metadata = out_tile,
        };
        DataflowBufferSpec interm_dfb{
            .unique_id = INTERM0_DFB,
            .entry_size = p.interm_entry_size,
            .num_entries = p.interm_entries,
            .data_format_metadata = p.interm_format,
            .tile_format_metadata = out_tile,
        };
        if (p.alias_out_interm) {
            out_dfb.advanced_options.alias_with = {INTERM0_DFB};
            interm_dfb.advanced_options.alias_with = {OUT_DFB};
        }
        dataflow_buffers.push_back(std::move(out_dfb));
        dataflow_buffers.push_back(std::move(interm_dfb));
    }

    // ---- Reader ----
    const uint32_t in0_last_ktile_w = a.logical_shape()[-1] % TILE_WIDTH;
    KernelSpec reader{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path(std::string(KERNEL_DIR) + "dataflow/reader_bmm_unified.cpp"),
        .compiler_options = {},
        .dfb_bindings = {ProducerOf(IN0_DFB, "cb_in0"), ProducerOf(IN1_DFB, "cb_in1")},
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = IN0_TENSOR, .accessor_name = "in0"},
                TensorBinding{.tensor_parameter_name = IN1_TENSOR, .accessor_name = "in1"},
            },
        .compile_time_args =
            {
                {"Mt", p.Mt},
                {"Kt", p.Kt},
                {"Nt", p.Nt},
                {"batch", p.B},
                {"bcast_batch", p.bcast_batch ? 1u : 0u},
                {"per_core_M", p.per_core_M},
                {"per_core_N", p.per_core_N},
                {"in0_block_w", p.in0_block_w},
                {"num_blocks_inner_dim", p.num_blocks_inner_dim},
                {"num_block_cols", p.num_block_cols},
                {"in0_last_ktile_w", in0_last_ktile_w},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"block_start", "num_blocks"}},
        .hw_config =
            ttnn::create_reader_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // ---- Writer ----
    KernelSpec writer{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path(std::string(KERNEL_DIR) + "dataflow/writer_bmm_unified.cpp"),
        .compiler_options = {},
        .dfb_bindings = {ConsumerOf(OUT_DFB, "cb_out")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "out"}},
        .compile_time_args =
            {
                {"Mt", p.Mt},
                {"Nt", p.Nt},
                {"batch", p.B},
                {"per_core_M", p.per_core_M},
                {"per_core_N", p.per_core_N},
                {"out_subblock_h", p.out_subblock_h},
                {"out_subblock_w", p.out_subblock_w},
                {"num_block_cols", p.num_block_cols},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"block_start", "num_blocks"}},
        .hw_config =
            ttnn::create_writer_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // ---- Compute: the shared block matmul kernel, one spec per distinct blocks-per-core ----
    const bool fp32_dest_acc_en = get_fp32_dest_acc_en(operation_attributes.compute_kernel_config);
    std::map<std::string, std::string> mm_kernel_defines;
    if (p.packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    const ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level =
        ttnn::get_throttle_level(operation_attributes.compute_kernel_config);
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), p.cores.size(), mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), p.cores.size(), mm_kernel_defines, throttle_level);
    KernelSpec::CompilerOptions::Defines compute_defines(mm_kernel_defines);
    ComputeHardwareConfig compute_hw_config =
        ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config.value());
    if (fp32_dest_acc_en) {
        // With a 32-bit DST every consumed 32-bit DFB needs an explicit unpack mode. The partials are
        // reloaded with a data copy into DST, so unpack them straight to DST and keep fp32 precision;
        // fp32 operands feed the FPU and go through SrcA/SrcB.
        auto is_32bit = [](tt::DataFormat f) {
            return f == tt::DataFormat::Float32 || f == tt::DataFormat::Int32 || f == tt::DataFormat::UInt32;
        };
        ComputeUnpackModes& modes = unpack_modes(compute_hw_config);
        if (is_32bit(p.interm_format)) {
            modes.emplace(INTERM0_DFB, tt::tt_metal::UnpackMode::UnpackToDest);
        }
        if (is_32bit(p.in0_format)) {
            modes.emplace(IN0_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
        if (is_32bit(p.in1_format)) {
            modes.emplace(IN1_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
    }

    const uint32_t in0_num_subblocks = p.per_core_M / p.out_subblock_h;
    const uint32_t in1_num_subblocks = p.per_core_N / p.out_subblock_w;
    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t blocks_per_core) {
        return KernelSpec{
            .unique_id = unique_id,
            .source = std::filesystem::path(
                std::string(KERNEL_DIR) + "compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp"),
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings =
                {
                    ConsumerOf(IN0_DFB, "cb_in0"),
                    ConsumerOf(IN1_DFB, "cb_in1"),
                    ProducerOf(OUT_DFB, "cb_out"),
                    ProducerOf(INTERM0_DFB, "cb_intermed0"),
                    ConsumerOf(INTERM0_DFB, "cb_intermed0"),
                },
            .compile_time_args =
                {
                    {"in0_block_w", p.in0_block_w},
                    {"in0_num_subblocks", in0_num_subblocks},
                    {"in0_block_num_tiles", p.per_core_M * p.in0_block_w},
                    {"in0_subblock_num_tiles", p.out_subblock_h * p.in0_block_w},
                    {"in1_num_subblocks", in1_num_subblocks},
                    {"in1_block_num_tiles", p.in0_block_w * p.per_core_N},
                    {"in1_block_w", p.per_core_N},
                    {"num_blocks_inner_dim", p.num_blocks_inner_dim},
                    // The kernel walks a (h x w) grid of identical output blocks; this core's run of
                    // blocks is that grid with w = 1. Block positions live in the reader / writer.
                    {"num_blocks_w_dim", 1u},
                    {"num_blocks_h_dim", blocks_per_core},
                    {"out_subblock_h", p.out_subblock_h},
                    {"out_subblock_w", p.out_subblock_w},
                    {"out_subblock_num_tiles", p.out_subblock_h * p.out_subblock_w},
                    {"batch", p.B},
                    {"out_block_num_tiles", p.per_core_M * p.per_core_N},
                    {"untilize_out", 0u},
                    {"get_batch_from_reader", 0u},
                    {"bias_ntiles", 0u},
                },
            .hw_config = compute_hw_config,
        };
    };

    // Cores with q+1 blocks come first in assignment order, cores with q blocks after them.
    const uint32_t num_active = p.cores.size();
    uint32_t num_g1 = 0;
    while (num_g1 < num_active && p.blocks_per_core[num_g1] == p.max_blocks_per_core) {
        ++num_g1;
    }
    // CoreRangeSet's span-of-cores constructor merges the cores into as few rectangles as possible.
    auto cores_to_set = [&](uint32_t first, uint32_t count) {
        return CoreRangeSet(ttsl::Span<const CoreCoord>(p.cores.data() + first, count));
    };
    const CoreRangeSet group_1 = cores_to_set(0, num_g1);
    const bool group_2_present = num_g1 < num_active;

    Group<KernelSpec> kernels = {reader, writer, make_compute(COMPUTE_KERNEL_G1, p.max_blocks_per_core)};
    Group<WorkUnitSpec> work_units = {WorkUnitSpec{
        .name = "wu_g1",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL_G1},
        .target_nodes = group_1,
    }};
    if (group_2_present) {
        kernels.push_back(make_compute(COMPUTE_KERNEL_G2, p.blocks_per_core[num_g1]));
        work_units.push_back(WorkUnitSpec{
            .name = "wu_g2",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL_G2},
            .target_nodes = cores_to_set(num_g1, num_active - num_g1),
        });
    }

    // ---- Per-core runtime args ----
    ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = READER_KERNEL};
    ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = WRITER_KERNEL};
    for (uint32_t i = 0; i < num_active; ++i) {
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            p.cores[i],
            {{"block_start", p.block_start[i]}, {"num_blocks", p.blocks_per_core[i]}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            p.cores[i],
            {{"block_start", p.block_start[i]}, {"num_blocks", p.blocks_per_core[i]}});
    }

    ProgramSpec spec{
        .name = "matmul_unified",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };
    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)},
        .tensor_args = {{IN0_TENSOR, a}, {IN1_TENSOR, b}, {OUT_TENSOR, output}},
    };
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
