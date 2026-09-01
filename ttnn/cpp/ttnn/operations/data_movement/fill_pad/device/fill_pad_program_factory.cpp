// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_pad_program_factory.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <algorithm>
#include <bit>
#include <map>
#include <string>
#include <vector>
#include <fmt/format.h>
#include <cstdint>

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// The mask-generation / fill defines shared by both factories (same content the legacy
// factory emitted; passed to every kernel, unused defines are harmless). Kept file-local
// inside CMAKE_UNIQUE_NAMESPACE so the symbol stays collision-free under unity builds.
KernelSpec::CompilerOptions::Defines build_kernel_defines(
    const Tensor& input_tensor,
    tt::DataFormat cb_data_format,
    std::uint32_t input_element_size_bytes,
    bool is_float_type,
    bool is_fp32) {
    KernelSpec::CompilerOptions::Defines d;
    d.insert({"MASK_ELEM_UINT", (input_element_size_bytes == 2) ? "uint16_t" : "uint32_t"});
    d.insert({"MASK_VALUE", is_fp32 ? "0x3F800000u" : is_float_type ? "0x3F80u" : "1u"});
    d.insert({"FILL_PAD_DATA_FMT", detail::get_where_data_fmt(input_tensor.dtype())});
    if (!is_float_type) {
        d.insert({"FILL_PAD_FILL_DATA_FMT", fmt::format("DataFormat::{}", cb_data_format)});
    }
    d.insert({"FILL_PAD_FILL_FN", is_float_type ? "fill_tile_bitcast" : "fill_tile_int<FILL_PAD_FILL_DATA_FMT>"});
    d.insert({"FILL_PAD_FILL_ARG", "fill_bits"});
    return d;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts FillPadProgramFactory::create_program_artifacts(
    const FillPadParams& operation_attributes, const FillPadInputs& tensor_args, Tensor& /*tensor_return_value*/) {
    const Tensor& input_tensor = tensor_args.input;
    TT_FATAL(
        !input_tensor.is_sharded() || !input_tensor.memory_config().is_l1(),
        "FillPadProgramFactory called with L1-sharded tensor; use FillPadL1ShardedProgramFactory");
    TT_FATAL(
        detail::data_type_to_size.contains(input_tensor.dtype()),
        "FillPadProgramFactory: unsupported dtype {}",
        input_tensor.dtype());

    const ttnn::PadValue& fill_value = operation_attributes.fill_value;
    tt::tt_metal::IDevice* device = input_tensor.device();

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    TT_FATAL(input_tensor.buffer() != nullptr, "Input buffer should be allocated on device!");

    const std::uint32_t input_element_size_bytes = detail::data_type_to_size.at(input_tensor.dtype());
    const std::uint32_t tile_bytes = tt::tile_size(cb_data_format);

    const std::uint32_t height = input_tensor.logical_shape()[-2];
    const std::uint32_t width = input_tensor.logical_shape()[-1];
    const std::uint32_t N_slices = detail::num_slice_batches(input_tensor.logical_shape());

    const std::uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    const std::uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();

    const std::uint32_t H_tiles = tt::div_up(height, tile_height);
    const std::uint32_t W_tiles = tt::div_up(width, tile_width);
    const std::uint32_t H_mod32 = height % tile_height;
    const std::uint32_t W_mod32 = width % tile_width;
    const bool has_right_pad = W_mod32 != 0;
    const bool has_bottom_pad = H_mod32 != 0;

    const bool is_float_type =
        (input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::FLOAT32);
    const bool is_fp32 = (input_tensor.dtype() == DataType::FLOAT32);
    const bool is_uint32 = (input_tensor.dtype() == DataType::UINT32);
    const bool is_int32 = (input_tensor.dtype() == DataType::INT32);
    // 32-bit integer types need fp32_dest_acc_en so DST holds full 32-bit values
    // and where_tile<UInt32/Int32> can use INT32-mode SFPLOAD/SFPSTORE correctly.
    const bool need_fp32_dest_acc = is_fp32 || is_uint32 || is_int32;
    // Float types: raw bit pattern of fill_value for fill_tile_bitcast.
    // Integer types: packed native bit pattern for fill_tile_int.
    const std::uint32_t fill_bits = detail::pack_fill_value_for_dtype(input_tensor.dtype(), fill_value);

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const std::uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const std::uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Unified border-tile split across all slices.
    //   right_slice_stride  = rows per slice in the right block (H_tiles-1 if both pads, else H_tiles).
    //   bottom_slice_stride = cols per slice in the bottom block (W_tiles-1 if both pads, else W_tiles).
    // The global tile-index space is three contiguous blocks:
    //   [0, T_right)                 – right-column border tiles
    //   [T_right, T_right+T_bottom)  – bottom-row border tiles (incl. corner if !has_right_pad)
    //   [..., total_work)            – corner tiles (only when has_right_pad && has_bottom_pad)
    const std::uint32_t right_slice_stride = has_right_pad ? (has_bottom_pad ? (H_tiles - 1u) : H_tiles) : 0u;
    const std::uint32_t bottom_slice_stride = has_bottom_pad ? (has_right_pad ? (W_tiles - 1u) : W_tiles) : 0u;
    const std::uint32_t T_right = has_right_pad ? (N_slices * right_slice_stride) : 0u;
    const std::uint32_t T_bottom = has_bottom_pad ? (N_slices * bottom_slice_stride) : 0u;
    const std::uint32_t T_corner = (has_right_pad && has_bottom_pad) ? N_slices : 0u;
    const std::uint32_t total_work = T_right + T_bottom + T_corner;

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_work_per_core_group_1, num_work_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_work);
    const std::uint32_t g1_numcores = core_group_1.num_cores();

    // ---- Named resources ----
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName DATA_IN{"data_in"};
    const DFBSpecName RIGHT_MASK{"right_mask"};
    const DFBSpecName BOT_MASK{"bot_mask"};
    const DFBSpecName DATA_OUT{"data_out"};
    const TensorParamName INPUT{"input"};

    // ---- Dataflow buffers (placement derived from bindings) ----
    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = DATA_IN, .entry_size = tile_bytes, .num_entries = 2, .data_format_metadata = cb_data_format});
    if (has_right_pad) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RIGHT_MASK,
            .entry_size = tile_bytes,
            .num_entries = 1,
            .data_format_metadata = cb_data_format});
    }
    if (has_bottom_pad) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = BOT_MASK, .entry_size = tile_bytes, .num_entries = 1, .data_format_metadata = cb_data_format});
    }
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = DATA_OUT, .entry_size = tile_bytes, .num_entries = 2, .data_format_metadata = cb_data_format});

    // ---- Kernel source paths (this op's own directory) ----
    constexpr const char* kDramReaderSrc =
        "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_reader.cpp";
    constexpr const char* kDramWriterSrc =
        "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_writer.cpp";
    constexpr const char* kComputeSrc =
        "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/compute/fill_pad_compute.cpp";

    // ---- Defines ----
    const auto fill_defines = CMAKE_UNIQUE_NAMESPACE::build_kernel_defines(
        input_tensor, cb_data_format, input_element_size_bytes, is_float_type, is_fp32);
    // has_right_pad / has_bottom_pad are promoted to preprocessor defines on the writer and
    // compute kernels (they gate the conditionally-bound right / bottom mask DFBs).
    auto mask_defines = fill_defines;
    if (has_right_pad) {
        mask_defines.insert({"HAS_RIGHT_PAD", "1"});
    }
    if (has_bottom_pad) {
        mask_defines.insert({"HAS_BOTTOM_PAD", "1"});
    }

    // ---- Reader KernelSpec ----
    KernelSpec reader_spec{
        .unique_id = READER,
        .source = kDramReaderSrc,
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = DATA_IN, .accessor_name = "data_in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .compile_time_args =
            {{"W_tiles", W_tiles},
             {"H_tiles", H_tiles},
             {"has_right_pad", has_right_pad},
             {"has_bottom_pad", has_bottom_pad},
             {"elem_size", input_element_size_bytes}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"start_right", "num_right", "start_bottom", "num_bottom", "start_corner", "num_corner"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };
    reader_spec.compiler_options.defines = fill_defines;

    // ---- Writer KernelSpec ----
    Group<DFBBinding> writer_dfb_bindings = {
        DFBBinding{.dfb_spec_name = DATA_OUT, .accessor_name = "data_out", .endpoint_type = DFBEndpointType::CONSUMER}};
    KernelSpec::CompileTimeArgs writer_cta = {{"W_tiles", W_tiles}, {"H_tiles", H_tiles}};
    if (has_right_pad) {
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RIGHT_MASK, .accessor_name = "right_mask", .endpoint_type = DFBEndpointType::PRODUCER});
        writer_cta.insert({"W_mod32", W_mod32});
    }
    if (has_bottom_pad) {
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = BOT_MASK, .accessor_name = "bot_mask", .endpoint_type = DFBEndpointType::PRODUCER});
        writer_cta.insert({"H_mod32", H_mod32});
    }
    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = kDramWriterSrc,
        .dfb_bindings = std::move(writer_dfb_bindings),
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "dst"}},
        .compile_time_args = std::move(writer_cta),
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"start_right", "num_right", "start_bottom", "num_bottom", "start_corner", "num_corner"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };
    writer_spec.compiler_options.defines = mask_defines;

    // ---- Compute KernelSpec ----
    Group<DFBBinding> compute_dfb_bindings = {
        DFBBinding{.dfb_spec_name = DATA_IN, .accessor_name = "data_in", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = DATA_OUT, .accessor_name = "data_out", .endpoint_type = DFBEndpointType::PRODUCER}};
    if (has_right_pad) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RIGHT_MASK, .accessor_name = "right_mask", .endpoint_type = DFBEndpointType::CONSUMER});
    }
    if (has_bottom_pad) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = BOT_MASK, .accessor_name = "bot_mask", .endpoint_type = DFBEndpointType::CONSUMER});
    }

    ComputeGen1Config compute_hw{};
    compute_hw.enable_32_bit_dest = need_fp32_dest_acc;
    if (is_fp32) {
        // Match legacy UnpackToDestFp32 on the FP32 DFBs the compute kernel consumes.
        compute_hw.unpack_modes.insert({DATA_IN, UnpackMode::UnpackToDest});
        if (has_right_pad) {
            compute_hw.unpack_modes.insert({RIGHT_MASK, UnpackMode::UnpackToDest});
        }
        if (has_bottom_pad) {
            compute_hw.unpack_modes.insert({BOT_MASK, UnpackMode::UnpackToDest});
        }
    }
    KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = kComputeSrc,
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args =
            {{"W_tiles", W_tiles},
             {"H_tiles", H_tiles},
             {"elem_size", input_element_size_bytes},
             {"fill_bits", fill_bits}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_right", "num_bottom", "num_corner"}},
        .hw_config = ComputeHardwareConfig{std::move(compute_hw)},
    };
    compute_spec.compiler_options.defines = mask_defines;
    compute_spec.compiler_options.opt_level = KernelBuildOptLevel::O3;  // legacy ComputeConfig defaults to O3

    // ---- ProgramSpec ----
    ProgramSpec spec{
        .name = "fill_pad_dram",
        .kernels = {std::move(reader_spec), std::move(writer_spec), std::move(compute_spec)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = {TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()}},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}},
    };

    // ---- Per-core runtime args ----
    // Each core's global range [work_start, work_start + num_work) is intersected with the three
    // global blocks to produce per-phase (start, num) pairs. Phases with num==0 are skipped in the
    // kernels.
    const std::vector<CoreCoord> cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);
    KernelRunArgs reader_run{.kernel = READER};
    KernelRunArgs writer_run{.kernel = WRITER};
    KernelRunArgs compute_run{.kernel = COMPUTE};
    std::uint32_t work_start = 0;
    for (std::uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        const std::uint32_t num_work = (i < g1_numcores) ? num_work_per_core_group_1 : num_work_per_core_group_2;

        // Intersect this core's work range with each phase block and return the
        // block-relative (start, count) pair of tiles assigned to this core.
        const std::uint32_t work_end = work_start + num_work;
        auto clip_to_phase_block =
            [work_start, work_end](
                std::uint32_t block_start, std::uint32_t block_size, std::uint32_t& out_start, std::uint32_t& out_num) {
                if (block_size == 0u) {
                    out_start = 0u;
                    out_num = 0u;
                    return;
                }
                const std::uint32_t block_end = block_start + block_size;
                const std::uint32_t lo = std::max(work_start, block_start);
                const std::uint32_t hi = std::min(work_end, block_end);
                if (lo >= hi) {
                    out_start = 0u;
                    out_num = 0u;
                } else {
                    out_start = lo - block_start;
                    out_num = hi - lo;
                }
            };

        std::uint32_t start_right = 0, num_right = 0;
        std::uint32_t start_bottom = 0, num_bottom = 0;
        std::uint32_t start_corner = 0, num_corner = 0;
        clip_to_phase_block(0u, T_right, start_right, num_right);
        clip_to_phase_block(T_right, T_bottom, start_bottom, num_bottom);
        clip_to_phase_block(T_right + T_bottom, T_corner, start_corner, num_corner);

        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"start_right", start_right},
             {"num_right", num_right},
             {"start_bottom", start_bottom},
             {"num_bottom", num_bottom},
             {"start_corner", start_corner},
             {"num_corner", num_corner}});
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"start_right", start_right},
             {"num_right", num_right},
             {"start_bottom", start_bottom},
             {"num_bottom", num_bottom},
             {"start_corner", start_corner},
             {"num_corner", num_corner}});
        // Compute RT: per-phase counts; starts are not needed (DFBs are FIFO).
        AddRuntimeArgsForNode(
            compute_run.runtime_arg_values,
            core,
            {{"num_right", num_right}, {"num_bottom", num_bottom}, {"num_corner", num_corner}});

        work_start = work_end;
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run), std::move(compute_run)};
    run_args.tensor_args.insert({INPUT, TensorArgument{tensor_args.input.mesh_tensor()}});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts FillPadL1ShardedProgramFactory::create_program_artifacts(
    const FillPadParams& operation_attributes, const FillPadInputs& tensor_args, Tensor& /*tensor_return_value*/) {
    const Tensor& input_tensor = tensor_args.input;
    TT_FATAL(
        input_tensor.is_sharded() && input_tensor.memory_config().is_l1(),
        "FillPadL1ShardedProgramFactory requires an L1-sharded input tensor");
    TT_FATAL(
        detail::data_type_to_size.contains(input_tensor.dtype()),
        "FillPadL1ShardedProgramFactory: unsupported dtype {}",
        input_tensor.dtype());

    const ttnn::PadValue& fill_value = operation_attributes.fill_value;

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::Buffer* tens_buffer = input_tensor.buffer();
    TT_FATAL(tens_buffer != nullptr, "Input buffer should be allocated on device!");

    const std::uint32_t input_element_size_bytes = detail::data_type_to_size.at(input_tensor.dtype());
    const std::uint32_t tile_bytes = tt::tile_size(cb_data_format);

    const std::uint32_t height = input_tensor.logical_shape()[-2];
    const std::uint32_t width = input_tensor.logical_shape()[-1];
    const std::uint32_t N_slices = detail::num_slice_batches(input_tensor.logical_shape());

    TT_FATAL(N_slices == 1, "FillPadL1ShardedProgramFactory: N_slices > 1 not yet supported (got {})", N_slices);

    const std::uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    const std::uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();

    const std::uint32_t H_tiles = tt::div_up(height, tile_height);
    const std::uint32_t W_tiles_tensor = tt::div_up(width, tile_width);
    const std::uint32_t W_mod32 = width % tile_width;
    const std::uint32_t H_mod32 = height % tile_height;
    const bool has_right_pad = W_mod32 != 0;
    const bool has_bottom_pad = H_mod32 != 0;

    // ---- Shard geometry ----
    const auto layout = input_tensor.memory_config().memory_layout();
    const tt::tt_metal::ShardSpec& shard_spec = input_tensor.shard_spec().value();
    const bool rm_orientation = (shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    const tt::tt_metal::ShardSpecBuffer& buf_shard_spec = tens_buffer->shard_spec();
    const auto [pages_per_shard_y, pages_per_shard_x] = buf_shard_spec.shape_in_pages();
    const std::uint32_t W_tiles = pages_per_shard_x;  // shard width in tiles (CT arg for kernels)

    // Ordered shard cores — same ordering used by generate_buffer_page_mapping.
    const std::vector<CoreCoord> all_shard_cores = corerange_to_cores(shard_spec.grid, std::nullopt, rm_orientation);
    const CoreRange bb = shard_spec.grid.bounding_box();
    const std::uint32_t num_cols = bb.end_coord.x - bb.start_coord.x + 1;
    const std::uint32_t num_rows = bb.end_coord.y - bb.start_coord.y + 1;

    // ---- Per-core properties ----
    // Each core's (row, col) in the shard grid determines h_start and w_start, which in turn
    // determine whether it touches the right or bottom edge of the tensor.
    //   HEIGHT_SHARDED: row=i, col=0  (full-width shards; all touch the right edge)
    //   WIDTH_SHARDED:  row=0, col=i  (full-height shards; all touch the bottom edge)
    //   BLOCK_SHARDED:  row/col from 2-D grid layout
    struct ShardCoreInfo {
        CoreCoord coord;
        std::uint32_t shard_H_tiles;
        std::uint32_t has_right_pad;   // per-core right-edge flag (CT binary selector)
        std::uint32_t has_bottom_pad;  // per-core bottom-edge flag (CT binary selector for compute)
        std::uint32_t num_work;
        std::uint32_t local_valid_w;    // min(pages_per_shard_x, W_tiles_tensor - w_start)
        std::uint32_t local_right_col;  // local_valid_w - 1; right-border tile's column within this shard
    };

    std::vector<ShardCoreInfo> active;
    active.reserve(all_shard_cores.size());

    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(all_shard_cores.size()); ++i) {
        std::uint32_t row, col;
        if (layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            row = i;
            col = 0;
        } else if (layout == TensorMemoryLayout::WIDTH_SHARDED) {
            row = 0;
            col = i;
        } else {  // BLOCK_SHARDED
            if (rm_orientation) {
                row = i / num_cols;
                col = i % num_cols;
            } else {
                col = i / num_rows;
                row = i % num_rows;
            }
        }

        const std::uint32_t h_start = row * pages_per_shard_y;
        const std::uint32_t w_start = col * pages_per_shard_x;

        if (h_start >= H_tiles || w_start >= W_tiles_tensor) {
            continue;  // core's shard is outside the valid tile range
        }

        const std::uint32_t shard_h = std::min(pages_per_shard_y, H_tiles - h_start);
        const std::uint32_t core_has_right_pad =
            (has_right_pad && w_start + pages_per_shard_x >= W_tiles_tensor) ? 1u : 0u;
        const std::uint32_t core_has_bottom_pad = (has_bottom_pad && h_start + pages_per_shard_y >= H_tiles) ? 1u : 0u;
        const std::uint32_t nw = core_has_bottom_pad ? 1u : (core_has_right_pad ? shard_h : 0u);
        const std::uint32_t local_valid_w = std::min(pages_per_shard_x, W_tiles_tensor - w_start);
        const std::uint32_t local_right_col = local_valid_w - 1u;

        active.push_back(
            {all_shard_cores[i], shard_h, core_has_right_pad, core_has_bottom_pad, nw, local_valid_w, local_right_col});
    }

    TT_FATAL(!active.empty(), "FillPadL1ShardedProgramFactory: no active shard cores");

    // ---- Compute grouping key ----
    // Reader/writer/compute are all grouped by this key (one WorkUnitSpec per key), so each work
    // unit carries a matched {reader, writer, compute} triple and every DFB's producer + consumer
    // land on the same nodes (per-node 1P+1C). The reader/writer binaries depend only on
    // has_right_pad; splitting them per-key produces identical binaries over disjoint nodes.
    //   For has_bottom_pad=0 (Mode A), H is unused — such cores share a key with H=pages_per_shard_y.
    //   For has_bottom_pad=1 (Mode B), H drives right_rows = H-1; use actual shard height.
    //   effective_W = local_valid_w; equals pages_per_shard_x for fully-packed shards, less for
    //   partially-filled rightmost shards (W_tiles_tensor % pages_per_shard_x != 0).
    struct ComputeKey {
        std::uint32_t has_right_pad, has_bottom_pad, H, effective_W;
        bool operator<(const ComputeKey& o) const {
            if (has_right_pad != o.has_right_pad) {
                return has_right_pad < o.has_right_pad;
            }
            if (has_bottom_pad != o.has_bottom_pad) {
                return has_bottom_pad < o.has_bottom_pad;
            }
            if (H != o.H) {
                return H < o.H;
            }
            return effective_W < o.effective_W;
        }
    };
    auto key_of = [&](const ShardCoreInfo& ci) {
        const std::uint32_t key_H = ci.has_bottom_pad ? ci.shard_H_tiles : pages_per_shard_y;
        return ComputeKey{ci.has_right_pad, ci.has_bottom_pad, key_H, ci.local_valid_w};
    };

    std::map<ComputeKey, std::vector<CoreRange>> group_ranges;
    for (const auto& ci : active) {
        group_ranges[key_of(ci)].emplace_back(ci.coord, ci.coord);
    }

    const bool is_float_type =
        (input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::FLOAT32);
    const bool is_fp32 = (input_tensor.dtype() == DataType::FLOAT32);
    const bool is_uint32 = (input_tensor.dtype() == DataType::UINT32);
    const bool is_int32 = (input_tensor.dtype() == DataType::INT32);
    const bool need_fp32_dest_acc = is_fp32 || is_uint32 || is_int32;
    const std::uint32_t fill_bits = detail::pack_fill_value_for_dtype(input_tensor.dtype(), fill_value);

    // ---- Named resources ----
    const DFBSpecName DATA_IN{"data_in"};
    const DFBSpecName RIGHT_MASK{"right_mask"};
    const DFBSpecName BOT_MASK{"bot_mask"};
    const DFBSpecName DATA_OUT{"data_out"};
    const TensorParamName INPUT{"input"};

    // ---- Dataflow buffers ----
    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = DATA_IN, .entry_size = tile_bytes, .num_entries = 2, .data_format_metadata = cb_data_format});
    if (has_right_pad) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RIGHT_MASK,
            .entry_size = tile_bytes,
            .num_entries = 1,
            .data_format_metadata = cb_data_format});
    }
    if (has_bottom_pad) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = BOT_MASK, .entry_size = tile_bytes, .num_entries = 1, .data_format_metadata = cb_data_format});
    }
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = DATA_OUT, .entry_size = tile_bytes, .num_entries = 2, .data_format_metadata = cb_data_format});

    // ---- Kernel source paths (this op's own directory) ----
    constexpr const char* kShardedReaderSrc =
        "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_sharded_reader.cpp";
    constexpr const char* kShardedWriterSrc =
        "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_sharded_writer.cpp";
    constexpr const char* kComputeSrc =
        "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/compute/fill_pad_compute.cpp";

    const auto fill_defines = CMAKE_UNIQUE_NAMESPACE::build_kernel_defines(
        input_tensor, cb_data_format, input_element_size_bytes, is_float_type, is_fp32);

    // ---- Build kernels + work units per group ----
    struct GroupInfo {
        KernelSpecName reader_id, writer_id, compute_id;
        KernelRunArgs reader_run, writer_run, compute_run;
    };
    std::map<ComputeKey, GroupInfo> groups;
    Group<KernelSpec> kernels;
    Group<WorkUnitSpec> work_units;

    std::uint32_t g = 0;
    for (const auto& [key, ranges] : group_ranges) {
        const std::string gs = std::to_string(g);
        GroupInfo gi{
            .reader_id = KernelSpecName{"reader_" + gs},
            .writer_id = KernelSpecName{"writer_" + gs},
            .compute_id = KernelSpecName{"compute_" + gs},
            .reader_run = KernelRunArgs{.kernel = KernelSpecName{"reader_" + gs}},
            .writer_run = KernelRunArgs{.kernel = KernelSpecName{"writer_" + gs}},
            .compute_run = KernelRunArgs{.kernel = KernelSpecName{"compute_" + gs}}};

        // Bind/define the mask DFBs uniformly on ALL sharded cores (matching the interleaved
        // factory and main's all-cores CB allocation). Per-core right/bottom behavior is driven
        // by the runtime tile counts and has_right_pad_core / has_bottom_pad_core, not by the
        // binding. Non-uniform mask placement (only on rp/bp cores) left the data_out DFB
        // producer/consumer rendezvous unsatisfiable on bottom-only cores, deadlocking the
        // simulator (#50804); uniform placement keeps every core's DFB layout identical.
        const bool k_right = has_right_pad;
        const bool k_bottom = has_bottom_pad;

        // Mask defines for this group's writer + compute.
        auto mask_defines = fill_defines;
        if (k_right) {
            mask_defines.insert({"HAS_RIGHT_PAD", "1"});
        }
        if (k_bottom) {
            mask_defines.insert({"HAS_BOTTOM_PAD", "1"});
        }

        // Reader.
        KernelSpec reader_spec{
            .unique_id = gi.reader_id,
            .source = kShardedReaderSrc,
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = DATA_IN, .accessor_name = "data_in", .endpoint_type = DFBEndpointType::PRODUCER}},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
            .compile_time_args =
                {{"W_tiles", W_tiles}, {"has_right_pad", key.has_right_pad}, {"elem_size", input_element_size_bytes}},
            .runtime_arg_schema =
                {.runtime_arg_names = {"shard_H_tiles", "has_bottom_pad_core", "num_work", "local_right_col"}},
            .hw_config = ttnn::create_reader_datamovement_config(input_tensor.device()->arch()),
        };

        // Writer.
        Group<DFBBinding> writer_dfb_bindings = {DFBBinding{
            .dfb_spec_name = DATA_OUT, .accessor_name = "data_out", .endpoint_type = DFBEndpointType::CONSUMER}};
        KernelSpec::CompileTimeArgs writer_cta = {{"W_tiles", W_tiles}};
        if (k_right) {
            writer_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = RIGHT_MASK,
                .accessor_name = "right_mask",
                .endpoint_type = DFBEndpointType::PRODUCER});
            writer_cta.insert({"W_mod32", W_mod32});
        }
        if (k_bottom) {
            writer_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = BOT_MASK, .accessor_name = "bot_mask", .endpoint_type = DFBEndpointType::PRODUCER});
            writer_cta.insert({"H_mod32", H_mod32});
        }
        KernelSpec writer_spec{
            .unique_id = gi.writer_id,
            .source = kShardedWriterSrc,
            .dfb_bindings = std::move(writer_dfb_bindings),
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "dst"}},
            .compile_time_args = std::move(writer_cta),
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"shard_H_tiles", "has_bottom_pad_core", "num_work", "local_right_col", "has_right_pad_core"}},
            .hw_config = ttnn::create_writer_datamovement_config(input_tensor.device()->arch()),
        };
        writer_spec.compiler_options.defines = mask_defines;

        // Compute.
        Group<DFBBinding> compute_dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = DATA_IN, .accessor_name = "data_in", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = DATA_OUT, .accessor_name = "data_out", .endpoint_type = DFBEndpointType::PRODUCER}};
        if (k_right) {
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = RIGHT_MASK,
                .accessor_name = "right_mask",
                .endpoint_type = DFBEndpointType::CONSUMER});
        }
        if (k_bottom) {
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = BOT_MASK, .accessor_name = "bot_mask", .endpoint_type = DFBEndpointType::CONSUMER});
        }
        ComputeGen1Config compute_hw{};
        compute_hw.enable_32_bit_dest = need_fp32_dest_acc;
        if (is_fp32) {
            compute_hw.unpack_modes.insert({DATA_IN, UnpackMode::UnpackToDest});
            if (k_right) {
                compute_hw.unpack_modes.insert({RIGHT_MASK, UnpackMode::UnpackToDest});
            }
            if (k_bottom) {
                compute_hw.unpack_modes.insert({BOT_MASK, UnpackMode::UnpackToDest});
            }
        }
        KernelSpec compute_spec{
            .unique_id = gi.compute_id,
            .source = kComputeSrc,
            .dfb_bindings = std::move(compute_dfb_bindings),
            .compile_time_args =
                {{"W_tiles", key.effective_W},
                 {"H_tiles", key.H},
                 {"elem_size", input_element_size_bytes},
                 {"fill_bits", fill_bits}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_right", "num_bottom", "num_corner"}},
            .hw_config = ComputeHardwareConfig{std::move(compute_hw)},
        };
        compute_spec.compiler_options.defines = mask_defines;
        compute_spec.compiler_options.opt_level = KernelBuildOptLevel::O3;

        kernels.push_back(std::move(reader_spec));
        kernels.push_back(std::move(writer_spec));
        kernels.push_back(std::move(compute_spec));
        work_units.push_back(WorkUnitSpec{
            .name = "wu_" + gs,
            .kernels = {gi.reader_id, gi.writer_id, gi.compute_id},
            .target_nodes = CoreRangeSet(ranges)});

        groups.emplace(key, std::move(gi));
        ++g;
    }

    // ---- Per-core runtime args ----
    // The sharded reader/writer push tiles in the order (right, bottom, corner); the per-phase
    // counts below let the shared compute kernel process them in lock-step (the ordering discipline
    // whose earlier violation caused #50904's revert — kept intact here).
    for (const auto& ci : active) {
        GroupInfo& gi = groups.at(key_of(ci));
        AddRuntimeArgsForNode(
            gi.reader_run.runtime_arg_values,
            ci.coord,
            {{"shard_H_tiles", ci.shard_H_tiles},
             {"has_bottom_pad_core", ci.has_bottom_pad},
             {"num_work", ci.num_work},
             {"local_right_col", ci.local_right_col}});
        AddRuntimeArgsForNode(
            gi.writer_run.runtime_arg_values,
            ci.coord,
            {{"shard_H_tiles", ci.shard_H_tiles},
             {"has_bottom_pad_core", ci.has_bottom_pad},
             {"num_work", ci.num_work},
             {"local_right_col", ci.local_right_col},
             {"has_right_pad_core", ci.has_right_pad}});

        std::uint32_t num_right = 0, num_bottom = 0, num_corner = 0;
        if (ci.has_bottom_pad == 0u) {
            // Mode A: right-column tiles only (only cores with has_right_pad=1 have work).
            num_right = ci.has_right_pad ? ci.shard_H_tiles : 0u;
        } else if (ci.has_right_pad) {
            // Mode B with right pad: right strip (H-1) + bottom non-corner (local_valid_w-1) + corner.
            num_right = ci.shard_H_tiles - 1u;
            num_bottom = ci.local_valid_w - 1u;  // = local_right_col
            num_corner = 1u;
        } else {
            // Mode B, bottom pad only: full bottom row of this shard.
            num_bottom = ci.local_valid_w;
        }
        AddRuntimeArgsForNode(
            gi.compute_run.runtime_arg_values,
            ci.coord,
            {{"num_right", num_right}, {"num_bottom", num_bottom}, {"num_corner", num_corner}});
    }

    // ---- Assemble ----
    ProgramSpec spec{
        .name = "fill_pad_l1_sharded",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = {TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()}},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    for (auto& [key, gi] : groups) {
        run_args.kernel_run_args.push_back(std::move(gi.reader_run));
        run_args.kernel_run_args.push_back(std::move(gi.writer_run));
        run_args.kernel_run_args.push_back(std::move(gi.compute_run));
    }
    run_args.tensor_args.insert({INPUT, TensorArgument{tensor_args.input.mesh_tensor()}});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
