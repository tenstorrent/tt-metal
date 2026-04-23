// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_pad_program_factory.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <algorithm>
#include <bit>
#include <fmt/format.h>

namespace ttnn::prim {

FillPadProgramFactory::cached_program_t FillPadProgramFactory::create(
    const FillPadParams& operation_attributes, const FillPadInputs& tensor_args, Tensor& /*tensor_return_value*/) {
    const Tensor& input_tensor = tensor_args.input;
    TT_FATAL(
        !input_tensor.is_sharded() || !input_tensor.memory_config().is_l1(),
        "FillPadProgramFactory called with L1-sharded tensor; use FillPadL1ShardedProgramFactory");
    TT_FATAL(
        detail::data_type_to_size.contains(input_tensor.dtype()),
        "FillPadProgramFactory: unsupported dtype {}",
        input_tensor.dtype());

    const float fill_value = operation_attributes.fill_value;
    tt::tt_metal::IDevice* device = input_tensor.device();
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::Buffer* tens_buffer = input_tensor.buffer();
    TT_FATAL(tens_buffer != nullptr, "Input buffer should be allocated on device!");

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

    // ---- Circular buffers ----
    constexpr uint32_t cb_data_in_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_right_mask_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_bot_mask_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_data_out_idx = tt::CBIndex::c_16;

    // CB[0]: data in, double-buffered (reader → compute)
    {
        const tt::tt_metal::CircularBufferConfig cb_config =
            tt::tt_metal::CircularBufferConfig(tile_bytes * 2, {{cb_data_in_idx, cb_data_format}})
                .set_page_size(cb_data_in_idx, tile_bytes);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);
    }

    // CB[1]: right mask, capacity=1 (writer → compute, persistent; only when has_right_pad)
    if (has_right_pad) {
        const tt::tt_metal::CircularBufferConfig cb_config =
            tt::tt_metal::CircularBufferConfig(tile_bytes, {{cb_right_mask_idx, cb_data_format}})
                .set_page_size(cb_right_mask_idx, tile_bytes);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);
    }

    // CB[2]: bottom mask, capacity=1 (writer → compute, persistent; only when has_bottom_pad)
    if (has_bottom_pad) {
        const tt::tt_metal::CircularBufferConfig cb_config =
            tt::tt_metal::CircularBufferConfig(tile_bytes, {{cb_bot_mask_idx, cb_data_format}})
                .set_page_size(cb_bot_mask_idx, tile_bytes);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);
    }

    // CB[16]: data out, double-buffered (compute → writer)
    {
        const tt::tt_metal::CircularBufferConfig cb_config =
            tt::tt_metal::CircularBufferConfig(tile_bytes * 2, {{cb_data_out_idx, cb_data_format}})
                .set_page_size(cb_data_out_idx, tile_bytes);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);
    }

    // ---- Kernel defines ----
    std::map<std::string, std::string> kernel_defines;
    kernel_defines["MASK_ELEM_UINT"] = (input_element_size_bytes == 2) ? "uint16_t" : "uint32_t";
    kernel_defines["MASK_VALUE"] = is_fp32 ? "0x3F800000u" : is_float_type ? "0x3F80u" : "1u";
    kernel_defines["FILL_PAD_DATA_FMT"] = detail::get_where_data_fmt(input_tensor.dtype());
    if (!is_float_type) {
        kernel_defines["FILL_PAD_FILL_DATA_FMT"] = fmt::format("DataFormat::{}", cb_data_format);
    }
    kernel_defines["FILL_PAD_FILL_FN"] = is_float_type ? "fill_tile_bitcast" : "fill_tile_int<FILL_PAD_FILL_DATA_FMT>";
    kernel_defines["FILL_PAD_FILL_ARG"] = "fill_bits";

    // ---- Reader CT args ----
    // [0] W_tiles, [1] H_tiles, [2] N_slices, [3] has_right_pad, [4] has_bottom_pad,
    // [5] W_mod32, [6] H_mod32, [7] elem_size, [8] fill_bits (unused by reader), [9] cb_data_in_idx=0,
    // [10+] accessor args
    std::vector<uint32_t> reader_ct = {
        W_tiles,
        H_tiles,
        N_slices,
        has_right_pad,
        has_bottom_pad,
        W_mod32,
        H_mod32,
        input_element_size_bytes,
        fill_bits,
        static_cast<uint32_t>(cb_data_in_idx),
    };
    tt::tt_metal::TensorAccessorArgs(*tens_buffer).append_to(reader_ct);

    // ---- Writer CT args ----
    // [0] W_tiles, [1] H_tiles, [2] N_slices, [3] has_right_pad, [4] has_bottom_pad,
    // [5] W_mod32, [6] H_mod32,
    // [7] cb_right_mask=1, [8] cb_bot_mask=2, [9] cb_data_out=16,
    // [10+] accessor args
    std::vector<uint32_t> writer_ct = {
        W_tiles,
        H_tiles,
        N_slices,
        has_right_pad,
        has_bottom_pad,
        W_mod32,
        H_mod32,
        static_cast<uint32_t>(cb_right_mask_idx),
        static_cast<uint32_t>(cb_bot_mask_idx),
        static_cast<uint32_t>(cb_data_out_idx),
    };
    tt::tt_metal::TensorAccessorArgs(*tens_buffer).append_to(writer_ct);

    // ---- Compute CT args (no accessor args) ----
    // [0] W_tiles, [1] H_tiles, [2] has_right_pad, [3] has_bottom_pad,
    // [4] elem_size, [5] fill_bits, [6] cb_data_in=0, [7] cb_right_mask=1,
    // [8] cb_bot_mask=2, [9] cb_data_out=16
    const std::vector<uint32_t> compute_ct = {
        W_tiles,
        H_tiles,
        has_right_pad,
        has_bottom_pad,
        input_element_size_bytes,
        fill_bits,
        static_cast<uint32_t>(cb_data_in_idx),
        static_cast<uint32_t>(cb_right_mask_idx),
        static_cast<uint32_t>(cb_bot_mask_idx),
        static_cast<uint32_t>(cb_data_out_idx),
    };

    // ---- Kernel creation ----
    const tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_reader.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct, kernel_defines));

    const tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_writer.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct, kernel_defines));

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (is_fp32) {
        unpack_to_dest_mode[cb_data_in_idx] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[cb_right_mask_idx] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[cb_bot_mask_idx] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    const tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/compute/fill_pad_compute.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = need_fp32_dest_acc,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_ct,
            .defines = kernel_defines,
        });

    // ---- Per-core runtime args ----
    // Each core's global range [work_start, work_start + num_work) is intersected with the three
    // global blocks to produce per-phase (start, num) pairs. Phases with num==0 are skipped in the
    // kernels.
    const std::vector<CoreCoord> cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);
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

        const std::vector<uint32_t> rt_args = {
            static_cast<uint32_t>(tens_buffer->address()),
            start_right,
            num_right,
            start_bottom,
            num_bottom,
            start_corner,
            num_corner,
        };
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, rt_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, rt_args);

        // Compute RT: per-phase counts; starts are not needed (CBs are FIFO).
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {num_right, num_bottom, num_corner});

        work_start = work_end;
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .reader_kernel_id = reader_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .compute_kernel_id = compute_kernel_id,
            .cores = cores,
        }};
}

void FillPadProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const FillPadParams& /*operation_attributes*/,
    const FillPadInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    const Tensor& input_tensor = tensor_args.input;
    tt::tt_metal::Buffer* tens_buffer = input_tensor.buffer();

    tt::tt_metal::Program& program = cached_program.program;
    const tt::tt_metal::KernelHandle reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const tt::tt_metal::KernelHandle writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id);

    for (const auto& core : cores) {
        reader_runtime_args[core.x][core.y][0] = tens_buffer->address();
        writer_runtime_args[core.x][core.y][0] = tens_buffer->address();
    }
}

FillPadL1ShardedProgramFactory::cached_program_t FillPadL1ShardedProgramFactory::create(
    const FillPadParams& operation_attributes, const FillPadInputs& tensor_args, Tensor& /*tensor_return_value*/) {
    const Tensor& input_tensor = tensor_args.input;
    TT_FATAL(
        input_tensor.is_sharded() && input_tensor.memory_config().is_l1(),
        "FillPadL1ShardedProgramFactory requires an L1-sharded input tensor");
    TT_FATAL(
        detail::data_type_to_size.contains(input_tensor.dtype()),
        "FillPadL1ShardedProgramFactory: unsupported dtype {}",
        input_tensor.dtype());

    const float fill_value = operation_attributes.fill_value;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::Buffer* tens_buffer = input_tensor.buffer();
    TT_FATAL(tens_buffer != nullptr, "Input buffer should be allocated on device!");

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
    const tt::tt_metal::ShardSpecBuffer& buf_shard_spec = tens_buffer->shard_spec();
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

    // ---- Build kernel groups ----
    // Reader/writer: binary key = has_right_pad; has_bottom_pad_core is RT.
    // Compute: binary key = (has_right_pad, has_bottom_pad, H_tiles, effective_W).
    //   For has_bottom_pad=0 (Mode A), H_tiles is unused — all such cores share a binary
    //   with H=pages_per_shard_y.
    //   For has_bottom_pad=1 (Mode B), H_tiles drives right_rows = H_tiles - 1; use actual shard height.
    //   effective_W = local_valid_w; equals pages_per_shard_x for fully-packed shards, less for
    //   partially-filled rightmost shards (W_tiles_tensor % pages_per_shard_x != 0).
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

    std::array<std::vector<CoreRange>, 2> rw_ranges;
    std::map<ComputeKey, std::vector<CoreRange>> compute_ranges;

    for (const auto& ci : active) {
        rw_ranges[ci.has_right_pad].emplace_back(ci.coord, ci.coord);
        const uint32_t key_H = ci.has_bottom_pad ? ci.shard_H_tiles : pages_per_shard_y;
        compute_ranges[{ci.has_right_pad, ci.has_bottom_pad, key_H, ci.local_valid_w}].emplace_back(ci.coord, ci.coord);
    }

    // All-active CoreRangeSet for CB creation.
    std::vector<CoreRange> all_active_vec;
    for (const auto& ci : active) {
        all_active_vec.emplace_back(ci.coord, ci.coord);
    }
    const CoreRangeSet all_active_set(all_active_vec);

    const bool is_float_type =
        (input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::FLOAT32);
    const bool is_fp32 = (input_tensor.dtype() == DataType::FLOAT32);
    const bool is_uint32 = (input_tensor.dtype() == DataType::UINT32);
    const bool is_int32 = (input_tensor.dtype() == DataType::INT32);
    const bool need_fp32_dest_acc = is_fp32 || is_uint32 || is_int32;
    const uint32_t fill_bits = detail::pack_fill_value_for_dtype(input_tensor.dtype(), fill_value);

    // ---- CB indices ----
    constexpr uint32_t cb_data_in_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_right_mask_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_bot_mask_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_data_out_idx = tt::CBIndex::c_16;

    // ---- Circular buffers (all active cores) ----
    {
        const tt::tt_metal::CircularBufferConfig cb_config =
            tt::tt_metal::CircularBufferConfig(tile_bytes * 2, {{cb_data_in_idx, cb_data_format}})
                .set_page_size(cb_data_in_idx, tile_bytes);
        tt::tt_metal::CreateCircularBuffer(program, all_active_set, cb_config);
    }
    if (has_right_pad) {
        const tt::tt_metal::CircularBufferConfig cb_config =
            tt::tt_metal::CircularBufferConfig(tile_bytes, {{cb_right_mask_idx, cb_data_format}})
                .set_page_size(cb_right_mask_idx, tile_bytes);
        tt::tt_metal::CreateCircularBuffer(program, all_active_set, cb_config);
    }
    if (has_bottom_pad) {
        const tt::tt_metal::CircularBufferConfig cb_config =
            tt::tt_metal::CircularBufferConfig(tile_bytes, {{cb_bot_mask_idx, cb_data_format}})
                .set_page_size(cb_bot_mask_idx, tile_bytes);
        tt::tt_metal::CreateCircularBuffer(program, all_active_set, cb_config);
    }
    {
        const tt::tt_metal::CircularBufferConfig cb_config =
            tt::tt_metal::CircularBufferConfig(tile_bytes * 2, {{cb_data_out_idx, cb_data_format}})
                .set_page_size(cb_data_out_idx, tile_bytes);
        tt::tt_metal::CreateCircularBuffer(program, all_active_set, cb_config);
    }

    // ---- Kernel defines ----
    std::map<std::string, std::string> kernel_defines;
    kernel_defines["MASK_ELEM_UINT"] = (input_element_size_bytes == 2) ? "uint16_t" : "uint32_t";
    kernel_defines["MASK_VALUE"] = is_fp32 ? "0x3F800000u" : is_float_type ? "0x3F80u" : "1u";
    kernel_defines["FILL_PAD_DATA_FMT"] = detail::get_where_data_fmt(input_tensor.dtype());
    if (!is_float_type) {
        kernel_defines["FILL_PAD_FILL_DATA_FMT"] = fmt::format("DataFormat::{}", cb_data_format);
    }
    kernel_defines["FILL_PAD_FILL_FN"] = is_float_type ? "fill_tile_bitcast" : "fill_tile_int<FILL_PAD_FILL_DATA_FMT>";
    kernel_defines["FILL_PAD_FILL_ARG"] = "fill_bits";

    // ---- Reader kernels (one per has_right_pad value) ----
    // CT: [0] W_tiles, [1] has_right_pad, [2] elem_size, [3] cb_data_in
    std::array<tt::tt_metal::KernelHandle, 2> reader_kernel_ids = {0, 0};
    for (uint32_t rp_idx = 0; rp_idx <= 1; ++rp_idx) {
        if (rw_ranges[rp_idx].empty()) {
            continue;
        }
        const std::vector<uint32_t> reader_ct = {
            W_tiles, rp_idx, input_element_size_bytes, static_cast<uint32_t>(cb_data_in_idx)};
        reader_kernel_ids[rp_idx] = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_sharded_reader.cpp",
            CoreRangeSet(rw_ranges[rp_idx]),
            tt::tt_metal::ReaderDataMovementConfig(reader_ct, kernel_defines));
    }

    // ---- Writer kernels (one per has_right_pad value) ----
    // CT: [0] W_tiles, [1] has_right_pad, [2] W_mod32, [3] H_mod32,
    //     [4] cb_right_mask, [5] cb_bot_mask, [6] cb_data_out
    std::array<tt::tt_metal::KernelHandle, 2> writer_kernel_ids = {0, 0};
    for (uint32_t rp_idx = 0; rp_idx <= 1; ++rp_idx) {
        if (rw_ranges[rp_idx].empty()) {
            continue;
        }
        const std::vector<uint32_t> writer_ct = {
            W_tiles,
            rp_idx,
            W_mod32,
            H_mod32,
            static_cast<uint32_t>(cb_right_mask_idx),
            static_cast<uint32_t>(cb_bot_mask_idx),
            static_cast<uint32_t>(cb_data_out_idx)};
        writer_kernel_ids[rp_idx] = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_sharded_writer.cpp",
            CoreRangeSet(rw_ranges[rp_idx]),
            tt::tt_metal::WriterDataMovementConfig(writer_ct, kernel_defines));
    }

    // ---- Compute kernel unpack-to-dest mode ----
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (is_fp32) {
        unpack_to_dest_mode[cb_data_in_idx] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[cb_right_mask_idx] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[cb_bot_mask_idx] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    // ---- Compute kernels (one per (has_right_pad, has_bottom_pad, H_tiles, effective_W) group) ----
    // CT: [0] W_tiles (= effective_W for this group), [1] H_tiles, [2] has_right_pad, [3] has_bottom_pad,
    //     [4] elem_size, [5] fill_bits, [6] cb_data_in, [7] cb_right_mask,
    //     [8] cb_bot_mask, [9] cb_data_out
    std::map<ComputeKey, tt::tt_metal::KernelHandle> compute_kernel_map;
    std::vector<tt::tt_metal::KernelHandle> compute_kernel_ids_vec;
    for (auto& [key, ranges] : compute_ranges) {
        const std::vector<uint32_t> compute_ct = {
            key.effective_W,
            key.H,
            key.has_right_pad,
            key.has_bottom_pad,
            input_element_size_bytes,
            fill_bits,
            static_cast<uint32_t>(cb_data_in_idx),
            static_cast<uint32_t>(cb_right_mask_idx),
            static_cast<uint32_t>(cb_bot_mask_idx),
            static_cast<uint32_t>(cb_data_out_idx)};
        const tt::tt_metal::KernelHandle h = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/compute/fill_pad_compute.cpp",
            CoreRangeSet(ranges),
            tt::tt_metal::ComputeConfig{
                .fp32_dest_acc_en = need_fp32_dest_acc,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .compile_args = compute_ct,
                .defines = kernel_defines});
        compute_kernel_map[key] = h;
        compute_kernel_ids_vec.push_back(h);
    }

    // ---- Per-core runtime args ----
    // RT layout: [0] buf_addr, [1] shard_H_tiles, [2] has_bottom_pad_core, [3] num_work,
    //            [4] local_right_col
    const uint32_t buf_addr = static_cast<uint32_t>(tens_buffer->address());
    std::vector<CoreCoord> active_coords;
    std::vector<uint32_t> active_has_right_pad;

    for (const auto& ci : active) {
        const std::vector<uint32_t> rt = {
            buf_addr, ci.shard_H_tiles, ci.has_bottom_pad, ci.num_work, ci.local_right_col};
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_ids[ci.has_right_pad], ci.coord, rt);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_ids[ci.has_right_pad], ci.coord, rt);

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
        const uint32_t key_H = ci.has_bottom_pad ? ci.shard_H_tiles : pages_per_shard_y;
        tt::tt_metal::SetRuntimeArgs(
            program,
            compute_kernel_map.at({ci.has_right_pad, ci.has_bottom_pad, key_H, ci.local_valid_w}),
            ci.coord,
            {num_right, num_bottom, num_corner});
        active_coords.push_back(ci.coord);
        active_has_right_pad.push_back(ci.has_right_pad);
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .reader_kernel_ids = reader_kernel_ids,
            .writer_kernel_ids = writer_kernel_ids,
            .compute_kernel_ids = std::move(compute_kernel_ids_vec),
            .active_cores = std::move(active_coords),
            .active_core_has_right_pad = std::move(active_has_right_pad),
        }};
}

void FillPadL1ShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const FillPadParams& /*operation_attributes*/,
    const FillPadInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    const uint32_t buf_addr = static_cast<uint32_t>(tensor_args.input.buffer()->address());

    tt::tt_metal::Program& program = cached_program.program;
    const auto& sv = cached_program.shared_variables;

    for (uint32_t rp_idx = 0; rp_idx <= 1; ++rp_idx) {
        if (sv.reader_kernel_ids[rp_idx] == 0) {
            continue;
        }
        auto& rrt = GetRuntimeArgs(program, sv.reader_kernel_ids[rp_idx]);
        auto& wrt = GetRuntimeArgs(program, sv.writer_kernel_ids[rp_idx]);
        for (uint32_t i = 0; i < sv.active_cores.size(); ++i) {
            if (sv.active_core_has_right_pad[i] != rp_idx) {
                continue;
            }
            const CoreCoord& core = sv.active_cores[i];
            rrt[core.x][core.y][0] = buf_addr;
            wrt[core.x][core.y][0] = buf_addr;
        }
    }
}

}  // namespace ttnn::prim
