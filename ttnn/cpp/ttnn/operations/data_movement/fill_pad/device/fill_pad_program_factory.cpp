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
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include <fmt/format.h>

namespace ttnn::prim {

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace {

// Kernel source paths (relative to TT_METAL_HOME).
constexpr const char* READER_SRC =
    "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_reader.cpp";
constexpr const char* WRITER_SRC =
    "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_writer.cpp";
constexpr const char* SHARDED_READER_SRC =
    "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_sharded_reader.cpp";
constexpr const char* SHARDED_WRITER_SRC =
    "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_sharded_writer.cpp";
constexpr const char* COMPUTE_SRC =
    "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/compute/fill_pad_compute.cpp";

// Builds the shared kernel defines (mask element type, mask 1-value, and the dtype-directed
// fill-function / fill-format selectors). Identical to the legacy KernelDescriptor::Defines,
// now as a Metal 2.0 defines Table. HAS_RIGHT_PAD / HAS_BOTTOM_PAD are layered on per kernel.
m2::KernelSpec::CompilerOptions::Defines make_fill_defines(
    const Tensor& input_tensor, uint32_t input_element_size_bytes, tt::DataFormat cb_data_format) {
    const bool is_float_type =
        (input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::FLOAT32);
    const bool is_fp32 = (input_tensor.dtype() == DataType::FLOAT32);

    m2::KernelSpec::CompilerOptions::Defines defines;
    defines.insert({"MASK_ELEM_UINT", (input_element_size_bytes == 2) ? "uint16_t" : "uint32_t"});
    defines.insert({"MASK_VALUE", is_fp32 ? "0x3F800000u" : is_float_type ? "0x3F80u" : "1u"});
    defines.insert({"FILL_PAD_DATA_FMT", detail::get_where_data_fmt(input_tensor.dtype())});
    if (!is_float_type) {
        defines.insert({"FILL_PAD_FILL_DATA_FMT", fmt::format("DataFormat::{}", cb_data_format)});
    }
    defines.insert({"FILL_PAD_FILL_FN", is_float_type ? "fill_tile_bitcast" : "fill_tile_int<FILL_PAD_FILL_DATA_FMT>"});
    defines.insert({"FILL_PAD_FILL_ARG", "fill_bits"});
    return defines;
}

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

    // Metal 2.0 named resource handles for this factory's ProgramSpec.
    const m2::DFBSpecName DATA_IN{"data_in"};
    const m2::DFBSpecName RIGHT_MASK{"right_mask"};
    const m2::DFBSpecName BOT_MASK{"bot_mask"};
    const m2::DFBSpecName DATA_OUT{"data_out"};
    const m2::TensorParamName INPUT{"input"};
    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};

    const tt::tt_metal::PadValue& fill_value = operation_attributes.fill_value;
    tt::tt_metal::IDevice* device = input_tensor.device();

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    const uint32_t input_element_size_bytes = detail::data_type_to_size.at(input_tensor.dtype());
    const uint32_t tile_bytes = tt::tile_size(cb_data_format);

    const uint32_t height = input_tensor.logical_shape()[-2];
    const uint32_t width = input_tensor.logical_shape()[-1];
    const uint32_t N_slices = input_tensor.logical_shape().rank() > 2 ? input_tensor.logical_shape()[-3] : 1u;

    const uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    const uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();

    const uint32_t H_tiles = tt::div_up(height, tile_height);
    const uint32_t W_tiles = tt::div_up(width, tile_width);
    const uint32_t H_mod32 = height % tile_height;
    const uint32_t W_mod32 = width % tile_width;
    const bool has_right_pad = W_mod32 != 0;
    const bool has_bottom_pad = H_mod32 != 0;

    const bool is_fp32 = (input_tensor.dtype() == DataType::FLOAT32);
    const bool is_uint32 = (input_tensor.dtype() == DataType::UINT32);
    const bool is_int32 = (input_tensor.dtype() == DataType::INT32);
    // 32-bit integer types need enable_32_bit_dest so DST holds full 32-bit values
    // and where_tile<UInt32/Int32> can use INT32-mode SFPLOAD/SFPSTORE correctly.
    const bool need_fp32_dest_acc = is_fp32 || is_uint32 || is_int32;
    // Float types: raw bit pattern of fill_value for fill_tile_bitcast.
    // Integer types: packed native bit pattern for fill_tile_int.
    const uint32_t fill_bits = detail::pack_fill_value_for_dtype(input_tensor.dtype(), fill_value);

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Unified border-tile split across all slices.
    //   right_slice_stride  = rows per slice in the right block (H_tiles-1 if both pads, else H_tiles).
    //   bottom_slice_stride = cols per slice in the bottom block (W_tiles-1 if both pads, else W_tiles).
    // The global tile-index space is three contiguous blocks:
    //   [0, T_right)                 – right-column border tiles
    //   [T_right, T_right+T_bottom)  – bottom-row border tiles (incl. corner if !has_right_pad)
    //   [..., total_work)            – corner tiles (only when has_right_pad && has_bottom_pad)
    const uint32_t right_slice_stride = has_right_pad ? (has_bottom_pad ? (H_tiles - 1u) : H_tiles) : 0u;
    const uint32_t bottom_slice_stride = has_bottom_pad ? (has_right_pad ? (W_tiles - 1u) : W_tiles) : 0u;
    const uint32_t T_right = has_right_pad ? (N_slices * right_slice_stride) : 0u;
    const uint32_t T_bottom = has_bottom_pad ? (N_slices * bottom_slice_stride) : 0u;
    const uint32_t T_corner = (has_right_pad && has_bottom_pad) ? N_slices : 0u;
    const uint32_t total_work = T_right + T_bottom + T_corner;

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_work_per_core_group_1, num_work_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_work);
    const uint32_t g1_numcores = core_group_1.num_cores();

    // ---- Dataflow buffers (one per legacy CB index). Placement derived from kernel bindings.
    // Mask DFBs are created only under their respective pad config (matching legacy CB creation).
    m2::DataflowBufferSpec data_in_dfb{
        .unique_id = DATA_IN, .entry_size = tile_bytes, .num_entries = 2, .data_format_metadata = cb_data_format};
    m2::DataflowBufferSpec right_mask_dfb{
        .unique_id = RIGHT_MASK, .entry_size = tile_bytes, .num_entries = 1, .data_format_metadata = cb_data_format};
    m2::DataflowBufferSpec bot_mask_dfb{
        .unique_id = BOT_MASK, .entry_size = tile_bytes, .num_entries = 1, .data_format_metadata = cb_data_format};
    m2::DataflowBufferSpec data_out_dfb{
        .unique_id = DATA_OUT, .entry_size = tile_bytes, .num_entries = 2, .data_format_metadata = cb_data_format};

    m2::Group<m2::DataflowBufferSpec> dataflow_buffers = {data_in_dfb, data_out_dfb};
    if (has_right_pad) {
        dataflow_buffers.push_back(right_mask_dfb);
    }
    if (has_bottom_pad) {
        dataflow_buffers.push_back(bot_mask_dfb);
    }

    // ---- Tensor parameter (in-place op: one io tensor, read by reader and written by writer).
    m2::TensorParameter input_param{.unique_id = INPUT, .spec = input_tensor.tensor_spec()};

    // ---- Kernel defines. HAS_RIGHT_PAD / HAS_BOTTOM_PAD gate the conditionally-bound mask DFBs
    // in the compute and writer kernels (promoting the legacy `if constexpr(has_*_pad)` CTA gates).
    const m2::KernelSpec::CompilerOptions::Defines fill_defines =
        make_fill_defines(input_tensor, input_element_size_bytes, cb_data_format);
    m2::KernelSpec::CompilerOptions::Defines writer_defines = fill_defines;
    m2::KernelSpec::CompilerOptions::Defines compute_defines = fill_defines;
    if (has_right_pad) {
        writer_defines.insert({"HAS_RIGHT_PAD", "1"});
        compute_defines.insert({"HAS_RIGHT_PAD", "1"});
    }
    if (has_bottom_pad) {
        writer_defines.insert({"HAS_BOTTOM_PAD", "1"});
        compute_defines.insert({"HAS_BOTTOM_PAD", "1"});
    }

    // ---- Reader KernelSpec ----
    // has_right_pad / has_bottom_pad stay named CTAs (reader binds no conditional DFB — its
    // if constexpr phase gates reference only cb_data_in). N_slices / elem_size / fill_bits are
    // dead here (preserved verbatim per the audit's Misc-anomalies note).
    m2::KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path{READER_SRC},
        .compiler_options = {.defines = fill_defines},
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = DATA_IN, .accessor_name = "data_in", .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .compile_time_args =
            {{"W_tiles", W_tiles},
             {"H_tiles", H_tiles},
             {"N_slices", N_slices},
             {"has_right_pad", static_cast<uint32_t>(has_right_pad)},
             {"has_bottom_pad", static_cast<uint32_t>(has_bottom_pad)},
             {"W_mod32", W_mod32},
             {"H_mod32", H_mod32},
             {"elem_size", input_element_size_bytes},
             {"fill_bits", fill_bits}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"start_right", "num_right", "start_bottom", "num_bottom", "start_corner", "num_corner"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // ---- Writer KernelSpec ----
    m2::Group<m2::DFBBinding> writer_dfb_bindings = {m2::DFBBinding{
        .dfb_spec_name = DATA_OUT, .accessor_name = "data_out", .endpoint_type = m2::DFBEndpointType::CONSUMER}};
    if (has_right_pad) {
        writer_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = RIGHT_MASK,
            .accessor_name = "right_mask",
            .endpoint_type = m2::DFBEndpointType::PRODUCER});
    }
    if (has_bottom_pad) {
        writer_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = BOT_MASK, .accessor_name = "bot_mask", .endpoint_type = m2::DFBEndpointType::PRODUCER});
    }
    m2::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path{WRITER_SRC},
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings = writer_dfb_bindings,
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "dst"}},
        .compile_time_args =
            {{"W_tiles", W_tiles},
             {"H_tiles", H_tiles},
             {"N_slices", N_slices},
             {"W_mod32", W_mod32},
             {"H_mod32", H_mod32}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"start_right", "num_right", "start_bottom", "num_bottom", "start_corner", "num_corner"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // ---- Compute KernelSpec ----
    m2::Group<m2::DFBBinding> compute_dfb_bindings = {
        m2::DFBBinding{
            .dfb_spec_name = DATA_IN, .accessor_name = "data_in", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = DATA_OUT, .accessor_name = "data_out", .endpoint_type = m2::DFBEndpointType::PRODUCER}};
    if (has_right_pad) {
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = RIGHT_MASK,
            .accessor_name = "right_mask",
            .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }
    if (has_bottom_pad) {
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = BOT_MASK, .accessor_name = "bot_mask", .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }
    // Style B compute config: reproduce the legacy ComputeConfigDescriptor field-for-field.
    // Only fp32_dest_acc_en (→ enable_32_bit_dest) and unpack_to_dest_mode (→ unpack_modes) were set;
    // all other fields stay at their ComputeGen1Config defaults (which match the legacy defaults).
    // unpack_modes: only Float32 DFBs the kernel consumes need an entry (Int32/UInt32 deferred, #49936);
    // legacy set UnpackToDestFp32 (→ UnpackMode::UnpackToDest) on cb_data_in / mask CBs only under fp32.
    m2::ComputeGen1Config compute_hw{};
    compute_hw.enable_32_bit_dest = need_fp32_dest_acc;
    if (is_fp32) {
        compute_hw.unpack_modes.insert({DATA_IN, tt::tt_metal::UnpackMode::UnpackToDest});
        if (has_right_pad) {
            compute_hw.unpack_modes.insert({RIGHT_MASK, tt::tt_metal::UnpackMode::UnpackToDest});
        }
        if (has_bottom_pad) {
            compute_hw.unpack_modes.insert({BOT_MASK, tt::tt_metal::UnpackMode::UnpackToDest});
        }
    }
    m2::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = std::filesystem::path{COMPUTE_SRC},
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings = compute_dfb_bindings,
        // W_tiles / H_tiles / elem_size are dead in the compute body (preserved verbatim).
        .compile_time_args =
            {{"W_tiles", W_tiles},
             {"H_tiles", H_tiles},
             {"elem_size", input_element_size_bytes},
             {"fill_bits", fill_bits}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_right", "num_bottom", "num_corner"}},
        .hw_config = compute_hw,
    };

    // ---- Per-core runtime args ----
    // Each core's global range [work_start, work_start + num_work) is intersected with the three
    // global blocks to produce per-phase (start, num) pairs. Phases with num==0 are skipped in the
    // kernels.
    const std::vector<CoreCoord> cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);
    m2::KernelRunArgs reader_run{.kernel = READER};
    m2::KernelRunArgs writer_run{.kernel = WRITER};
    m2::KernelRunArgs compute_run{.kernel = COMPUTE};
    uint32_t work_start = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t num_work = (i < g1_numcores) ? num_work_per_core_group_1 : num_work_per_core_group_2;

        // Intersect this core's work range with each phase block and return the
        // block-relative (start, count) pair of tiles assigned to this core.
        const uint32_t work_end = work_start + num_work;
        auto clip_to_phase_block =
            [work_start, work_end](uint32_t block_start, uint32_t block_size, uint32_t& out_start, uint32_t& out_num) {
                if (block_size == 0u) {
                    out_start = 0u;
                    out_num = 0u;
                    return;
                }
                const uint32_t block_end = block_start + block_size;
                const uint32_t lo = std::max(work_start, block_start);
                const uint32_t hi = std::min(work_end, block_end);
                if (lo >= hi) {
                    out_start = 0u;
                    out_num = 0u;
                } else {
                    out_start = lo - block_start;
                    out_num = hi - lo;
                }
            };

        uint32_t start_right = 0, num_right = 0;
        uint32_t start_bottom = 0, num_bottom = 0;
        uint32_t start_corner = 0, num_corner = 0;
        clip_to_phase_block(0u, T_right, start_right, num_right);
        clip_to_phase_block(T_right, T_bottom, start_bottom, num_bottom);
        clip_to_phase_block(T_right + T_bottom, T_corner, start_corner, num_corner);

        m2::AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"start_right", start_right},
             {"num_right", num_right},
             {"start_bottom", start_bottom},
             {"num_bottom", num_bottom},
             {"start_corner", start_corner},
             {"num_corner", num_corner}});
        m2::AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"start_right", start_right},
             {"num_right", num_right},
             {"start_bottom", start_bottom},
             {"num_bottom", num_bottom},
             {"start_corner", start_corner},
             {"num_corner", num_corner}});

        // Compute RT: per-phase counts; starts are not needed (CBs are FIFO).
        m2::AddRuntimeArgsForNode(
            compute_run.runtime_arg_values,
            core,
            {{"num_right", num_right}, {"num_bottom", num_bottom}, {"num_corner", num_corner}});

        work_start = work_end;
    }

    // ---- Assemble ----
    m2::ProgramSpec spec{
        .name = "fill_pad",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = dataflow_buffers,
        .tensor_parameters = {input_param},
        .work_units = {m2::WorkUnitSpec{
            .name = "fill_pad", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}},
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run, compute_run};
    run_args.tensor_args.emplace(INPUT, m2::TensorArgument{input_tensor.mesh_tensor()});

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

    // Metal 2.0 named resource handles.
    const m2::DFBSpecName DATA_IN{"data_in"};
    const m2::DFBSpecName RIGHT_MASK{"right_mask"};
    const m2::DFBSpecName BOT_MASK{"bot_mask"};
    const m2::DFBSpecName DATA_OUT{"data_out"};
    const m2::TensorParamName INPUT{"input"};

    const tt::tt_metal::PadValue& fill_value = operation_attributes.fill_value;

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    const uint32_t input_element_size_bytes = detail::data_type_to_size.at(input_tensor.dtype());
    const uint32_t tile_bytes = tt::tile_size(cb_data_format);

    const uint32_t height = input_tensor.logical_shape()[-2];
    const uint32_t width = input_tensor.logical_shape()[-1];
    const uint32_t N_slices = input_tensor.logical_shape().rank() > 2 ? input_tensor.logical_shape()[-3] : 1u;

    TT_FATAL(N_slices == 1, "FillPadL1ShardedProgramFactory: N_slices > 1 not yet supported (got {})", N_slices);

    const uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    const uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();

    const uint32_t H_tiles = tt::div_up(height, tile_height);
    const uint32_t W_tiles_tensor = tt::div_up(width, tile_width);
    const uint32_t W_mod32 = width % tile_width;
    const uint32_t H_mod32 = height % tile_height;
    const bool has_right_pad = W_mod32 != 0;
    const bool has_bottom_pad = H_mod32 != 0;

    // ---- Shard geometry ----
    const auto layout = input_tensor.memory_config().memory_layout();
    const tt::tt_metal::ShardSpec& shard_spec = input_tensor.shard_spec().value();
    const bool rm_orientation = (shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    const tt::tt_metal::ShardSpecBuffer& buf_shard_spec = input_tensor.buffer()->shard_spec();
    const auto [pages_per_shard_y, pages_per_shard_x] = buf_shard_spec.shape_in_pages();
    const uint32_t W_tiles = pages_per_shard_x;  // shard width in tiles (CT arg for kernels)

    // Ordered shard cores — same ordering used by generate_buffer_page_mapping.
    const std::vector<CoreCoord> all_shard_cores = corerange_to_cores(shard_spec.grid, std::nullopt, rm_orientation);
    const CoreRange bb = shard_spec.grid.bounding_box();
    const uint32_t num_cols = bb.end_coord.x - bb.start_coord.x + 1;
    const uint32_t num_rows = bb.end_coord.y - bb.start_coord.y + 1;

    // ---- Per-core properties ----
    // Each core's (row, col) in the shard grid determines h_start and w_start, which in turn
    // determine whether it touches the right or bottom edge of the tensor.
    //   HEIGHT_SHARDED: row=i, col=0  (full-width shards; all touch the right edge)
    //   WIDTH_SHARDED:  row=0, col=i  (full-height shards; all touch the bottom edge)
    //   BLOCK_SHARDED:  row/col from 2-D grid layout
    struct ShardCoreInfo {
        CoreCoord coord;
        uint32_t shard_H_tiles;
        uint32_t has_right_pad;   // per-core right-edge flag (CT binary selector)
        uint32_t has_bottom_pad;  // per-core bottom-edge flag (CT binary selector for compute)
        uint32_t num_work;
        uint32_t local_valid_w;    // min(pages_per_shard_x, W_tiles_tensor - w_start)
        uint32_t local_right_col;  // local_valid_w - 1; right-border tile's column within this shard
    };

    std::vector<ShardCoreInfo> active;

    for (uint32_t i = 0; i < static_cast<uint32_t>(all_shard_cores.size()); ++i) {
        uint32_t row, col;
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

        const uint32_t h_start = row * pages_per_shard_y;
        const uint32_t w_start = col * pages_per_shard_x;

        if (h_start >= H_tiles || w_start >= W_tiles_tensor) {
            continue;  // core's shard is outside the valid tile range
        }

        const uint32_t shard_h = std::min(pages_per_shard_y, H_tiles - h_start);
        const uint32_t core_has_right_pad = (has_right_pad && w_start + pages_per_shard_x >= W_tiles_tensor) ? 1u : 0u;
        const uint32_t core_has_bottom_pad = (has_bottom_pad && h_start + pages_per_shard_y >= H_tiles) ? 1u : 0u;
        const uint32_t nw = core_has_bottom_pad ? 1u : (core_has_right_pad ? shard_h : 0u);
        const uint32_t local_valid_w = std::min(pages_per_shard_x, W_tiles_tensor - w_start);
        const uint32_t local_right_col = local_valid_w - 1u;

        active.push_back(
            {all_shard_cores[i], shard_h, core_has_right_pad, core_has_bottom_pad, nw, local_valid_w, local_right_col});
    }

    TT_FATAL(!active.empty(), "FillPadL1ShardedProgramFactory: no active shard cores");

    const bool is_fp32 = (input_tensor.dtype() == DataType::FLOAT32);
    const bool is_uint32 = (input_tensor.dtype() == DataType::UINT32);
    const bool is_int32 = (input_tensor.dtype() == DataType::INT32);
    const bool need_fp32_dest_acc = is_fp32 || is_uint32 || is_int32;
    const uint32_t fill_bits = detail::pack_fill_value_for_dtype(input_tensor.dtype(), fill_value);

    // ---- Compute grouping (binary key = has_right_pad, has_bottom_pad, H, effective_W) ----
    //   For has_bottom_pad=0 (Mode A), H_tiles is unused — all such cores share a key with
    //   H=pages_per_shard_y. For has_bottom_pad=1 (Mode B), H drives right_rows = H-1; use actual
    //   shard height. effective_W = local_valid_w (< pages_per_shard_x for partially-filled shards).
    struct ComputeKey {
        uint32_t has_right_pad, has_bottom_pad, H, effective_W;
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

    // Which specialization combos are present, so we create exactly the specs (and mask DFBs) needed.
    std::array<bool, 2> reader_present{false, false};                                     // [rp]
    std::array<std::array<bool, 2>, 2> writer_present{{{false, false}, {false, false}}};  // [rp][hbp]
    std::map<ComputeKey, std::vector<CoreCoord>> compute_cores;
    bool any_rp = false, any_bp = false;
    for (const auto& ci : active) {
        // Interior shard cores (neither right- nor bottom-edge) have no border tiles to fill.
        // Legacy created no-op kernels on them; Metal 2.0 derives placement from bindings, so we
        // simply do not place any kernel there (identical output — those cores are never touched).
        if (ci.has_right_pad == 0u && ci.has_bottom_pad == 0u) {
            continue;
        }
        reader_present[ci.has_right_pad] = true;
        writer_present[ci.has_right_pad][ci.has_bottom_pad] = true;
        const uint32_t key_H = ci.has_bottom_pad ? ci.shard_H_tiles : pages_per_shard_y;
        compute_cores[{ci.has_right_pad, ci.has_bottom_pad, key_H, ci.local_valid_w}].push_back(ci.coord);
        any_rp |= (ci.has_right_pad != 0u);
        any_bp |= (ci.has_bottom_pad != 0u);
    }

    // ---- Dataflow buffers. Mask DFBs created only if some active core binds them.
    m2::Group<m2::DataflowBufferSpec> dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = DATA_IN, .entry_size = tile_bytes, .num_entries = 2, .data_format_metadata = cb_data_format},
        m2::DataflowBufferSpec{
            .unique_id = DATA_OUT, .entry_size = tile_bytes, .num_entries = 2, .data_format_metadata = cb_data_format}};
    if (any_rp) {
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = RIGHT_MASK,
            .entry_size = tile_bytes,
            .num_entries = 1,
            .data_format_metadata = cb_data_format});
    }
    if (any_bp) {
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = BOT_MASK, .entry_size = tile_bytes, .num_entries = 1, .data_format_metadata = cb_data_format});
    }

    m2::TensorParameter input_param{.unique_id = INPUT, .spec = input_tensor.tensor_spec()};

    const m2::KernelSpec::CompilerOptions::Defines fill_defines =
        make_fill_defines(input_tensor, input_element_size_bytes, cb_data_format);
    const auto arch = input_tensor.device()->arch();

    // ---- Reader KernelSpecs (one per rp_idx). has_right_pad is a named CTA; has_bottom_pad_core
    // stays a runtime RTA (reader binds no conditional DFB).
    std::vector<m2::KernelSpec> kernel_specs;
    std::vector<m2::KernelRunArgs> kernel_runs;
    std::array<int, 2> reader_idx{-1, -1};
    for (uint32_t rp = 0; rp <= 1; ++rp) {
        if (!reader_present[rp]) {
            continue;
        }
        m2::KernelSpecName name{"sh_reader_" + std::to_string(rp)};
        kernel_specs.push_back(m2::KernelSpec{
            .unique_id = name,
            .source = std::filesystem::path{SHARDED_READER_SRC},
            .compiler_options = {.defines = fill_defines},
            .dfb_bindings = {m2::DFBBinding{
                .dfb_spec_name = DATA_IN, .accessor_name = "data_in", .endpoint_type = m2::DFBEndpointType::PRODUCER}},
            .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
            .compile_time_args = {{"W_tiles", W_tiles}, {"has_right_pad", rp}, {"elem_size", input_element_size_bytes}},
            .runtime_arg_schema =
                {.runtime_arg_names = {"shard_H_tiles", "has_bottom_pad_core", "num_work", "local_right_col"}},
            .hw_config = ttnn::create_reader_datamovement_config(arch),
        });
        reader_idx[rp] = static_cast<int>(kernel_specs.size()) - 1;
        kernel_runs.push_back(m2::KernelRunArgs{.kernel = name});
    }

    // ---- Writer KernelSpecs (one per (rp_idx, has_bottom_pad_core)). Split by has_bottom_pad_core
    // so the conditional BOT_MASK producer binding is per-node consistent with the compute consumer
    // (see METAL2_PORT_PLAN.md — sharded writer split). has_right_pad / has_bottom_pad become #defines.
    std::array<std::array<int, 2>, 2> writer_idx{{{-1, -1}, {-1, -1}}};
    for (uint32_t rp = 0; rp <= 1; ++rp) {
        for (uint32_t hbp = 0; hbp <= 1; ++hbp) {
            if (!writer_present[rp][hbp]) {
                continue;
            }
            m2::KernelSpecName name{"sh_writer_" + std::to_string(rp) + "_" + std::to_string(hbp)};
            m2::Group<m2::DFBBinding> wb = {m2::DFBBinding{
                .dfb_spec_name = DATA_OUT,
                .accessor_name = "data_out",
                .endpoint_type = m2::DFBEndpointType::CONSUMER}};
            m2::KernelSpec::CompilerOptions::Defines wd = fill_defines;
            if (rp == 1u) {
                wb.push_back(m2::DFBBinding{
                    .dfb_spec_name = RIGHT_MASK,
                    .accessor_name = "right_mask",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER});
                wd.insert({"HAS_RIGHT_PAD", "1"});
            }
            if (hbp == 1u) {
                wb.push_back(m2::DFBBinding{
                    .dfb_spec_name = BOT_MASK,
                    .accessor_name = "bot_mask",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER});
                wd.insert({"HAS_BOTTOM_PAD", "1"});
            }
            kernel_specs.push_back(m2::KernelSpec{
                .unique_id = name,
                .source = std::filesystem::path{SHARDED_WRITER_SRC},
                .compiler_options = {.defines = wd},
                .dfb_bindings = wb,
                .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "dst"}},
                .compile_time_args = {{"W_tiles", W_tiles}, {"W_mod32", W_mod32}, {"H_mod32", H_mod32}},
                .runtime_arg_schema = {.runtime_arg_names = {"shard_H_tiles", "num_work", "local_right_col"}},
                .hw_config = ttnn::create_writer_datamovement_config(arch),
            });
            writer_idx[rp][hbp] = static_cast<int>(kernel_specs.size()) - 1;
            kernel_runs.push_back(m2::KernelRunArgs{.kernel = name});
        }
    }

    // ---- Compute KernelSpecs (one per ComputeKey) ----
    std::map<ComputeKey, int> compute_idx;
    {
        int ck = 0;
        for (const auto& [key, group_cores] : compute_cores) {
            m2::KernelSpecName name{"sh_compute_" + std::to_string(ck++)};
            m2::Group<m2::DFBBinding> cb = {
                m2::DFBBinding{
                    .dfb_spec_name = DATA_IN,
                    .accessor_name = "data_in",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{
                    .dfb_spec_name = DATA_OUT,
                    .accessor_name = "data_out",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER}};
            m2::KernelSpec::CompilerOptions::Defines cd = fill_defines;
            m2::ComputeGen1Config compute_hw{};
            compute_hw.enable_32_bit_dest = need_fp32_dest_acc;
            if (is_fp32) {
                compute_hw.unpack_modes.insert({DATA_IN, tt::tt_metal::UnpackMode::UnpackToDest});
            }
            if (key.has_right_pad) {
                cb.push_back(m2::DFBBinding{
                    .dfb_spec_name = RIGHT_MASK,
                    .accessor_name = "right_mask",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER});
                cd.insert({"HAS_RIGHT_PAD", "1"});
                if (is_fp32) {
                    compute_hw.unpack_modes.insert({RIGHT_MASK, tt::tt_metal::UnpackMode::UnpackToDest});
                }
            }
            if (key.has_bottom_pad) {
                cb.push_back(m2::DFBBinding{
                    .dfb_spec_name = BOT_MASK,
                    .accessor_name = "bot_mask",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER});
                cd.insert({"HAS_BOTTOM_PAD", "1"});
                if (is_fp32) {
                    compute_hw.unpack_modes.insert({BOT_MASK, tt::tt_metal::UnpackMode::UnpackToDest});
                }
            }
            kernel_specs.push_back(m2::KernelSpec{
                .unique_id = name,
                .source = std::filesystem::path{COMPUTE_SRC},
                .compiler_options = {.defines = cd},
                .dfb_bindings = cb,
                // W_tiles(=effective_W) / H_tiles(=key.H) / elem_size are dead in the compute body
                // (preserved verbatim per the audit's Misc-anomalies note).
                .compile_time_args =
                    {{"W_tiles", key.effective_W},
                     {"H_tiles", key.H},
                     {"elem_size", input_element_size_bytes},
                     {"fill_bits", fill_bits}},
                .runtime_arg_schema = {.runtime_arg_names = {"num_right", "num_bottom", "num_corner"}},
                .hw_config = compute_hw,
            });
            compute_idx[key] = static_cast<int>(kernel_specs.size()) - 1;
            // Keep kernel_runs in lockstep with kernel_specs so compute_idx indexes both.
            kernel_runs.push_back(m2::KernelRunArgs{.kernel = name});
        }
    }

    // ---- Per-core runtime args ----
    // RT layout mirrors the legacy sharded reader/writer/compute args, minus the buffer-address arg
    // (Case 2 base pulled via TensorAccessor::get_bank_base_address) and, for the writer, the
    // has_bottom_pad_core arg (now the HAS_BOTTOM_PAD compile define).
    for (const auto& ci : active) {
        if (ci.has_right_pad == 0u && ci.has_bottom_pad == 0u) {
            continue;  // interior core — no kernels placed (see pass above)
        }
        const uint32_t key_H = ci.has_bottom_pad ? ci.shard_H_tiles : pages_per_shard_y;
        const ComputeKey key{ci.has_right_pad, ci.has_bottom_pad, key_H, ci.local_valid_w};

        m2::AddRuntimeArgsForNode(
            kernel_runs[reader_idx[ci.has_right_pad]].runtime_arg_values,
            ci.coord,
            {{"shard_H_tiles", ci.shard_H_tiles},
             {"has_bottom_pad_core", ci.has_bottom_pad},
             {"num_work", ci.num_work},
             {"local_right_col", ci.local_right_col}});

        m2::AddRuntimeArgsForNode(
            kernel_runs[writer_idx[ci.has_right_pad][ci.has_bottom_pad]].runtime_arg_values,
            ci.coord,
            {{"shard_H_tiles", ci.shard_H_tiles}, {"num_work", ci.num_work}, {"local_right_col", ci.local_right_col}});

        // Compute RT: (num_right, num_bottom, num_corner) per the unified phase layout.
        // The sharded reader/writer push tiles in the same order (right, bottom, corner),
        // so these counts let the shared compute kernel process them in lock-step.
        uint32_t num_right = 0, num_bottom = 0, num_corner = 0;
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
        m2::AddRuntimeArgsForNode(
            kernel_runs[compute_idx.at(key)].runtime_arg_values,
            ci.coord,
            {{"num_right", num_right}, {"num_bottom", num_bottom}, {"num_corner", num_corner}});
    }

    // ---- Work units: one per active ComputeKey group, wiring the group's reader / writer / compute.
    m2::Group<m2::WorkUnitSpec> work_units;
    for (const auto& [key, group_cores] : compute_cores) {
        std::vector<CoreRange> ranges;
        ranges.reserve(group_cores.size());
        for (const auto& c : group_cores) {
            ranges.emplace_back(c, c);
        }
        work_units.push_back(m2::WorkUnitSpec{
            .name = "fill_pad_sharded",
            .kernels =
                {kernel_specs[reader_idx[key.has_right_pad]].unique_id,
                 kernel_specs[writer_idx[key.has_right_pad][key.has_bottom_pad]].unique_id,
                 kernel_specs[compute_idx.at(key)].unique_id},
            .target_nodes = CoreRangeSet(ranges)});
    }

    m2::ProgramSpec spec{
        .name = "fill_pad_sharded",
        .kernels = kernel_specs,
        .dataflow_buffers = dataflow_buffers,
        .tensor_parameters = {input_param},
        .work_units = work_units,
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = kernel_runs;
    run_args.tensor_args.emplace(INPUT, m2::TensorArgument{input_tensor.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
