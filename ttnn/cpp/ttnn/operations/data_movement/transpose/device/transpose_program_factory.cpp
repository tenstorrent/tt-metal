// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/host_buffer/functions.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/math.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include "ttnn/operation.hpp"

#include <cstdint>
#include <math.h>
#include <vector>

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

using namespace tt::constants;

template <bool IS_CREATING>
void override_runtime_args_mc_cn(
    const Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_tiles_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_tiles_per_core_group_2) {
    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_padded_shape();
    auto output_shape = output_tensor.get_padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;

    uint32_t num_tensor_tiles = N * C * H * W / TILE_HW;
    uint32_t HtWt = Ht * Wt;
    uint32_t CHtWt = C * HtWt;
    uint32_t NCHtWt = num_tensor_tiles;
    uint32_t batch_step = CHtWt - HtWt;
    uint32_t channel_step = NCHtWt - HtWt;

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core;

        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op
            num_tiles_per_core = 0;
        }

        uint32_t hw = num_tiles_read % HtWt;
        uint32_t curr_c = num_tiles_read / HtWt;
        uint32_t n = curr_c % N;
        uint32_t start_tile = num_tiles_read + curr_c * batch_step - curr_c / N * channel_step;

        if constexpr (IS_CREATING) {
            tt::tt_metal::SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {input_buffer->address(), N, C, HtWt, batch_step, channel_step, num_tiles_per_core, start_tile, hw, n});

            tt::tt_metal::SetRuntimeArgs(
                program, writer_kernel_id, core, {output_buffer->address(), num_tiles_per_core, num_tiles_read});
        } else {
            auto& reader_args = cached_reader_args.at(core.x).at(core.y);
            auto& writer_args = cached_writer_args.at(core.x).at(core.y);

            reader_args[0] = input_buffer->address();
            reader_args[1] = N;
            reader_args[2] = C;
            reader_args[3] = HtWt;
            reader_args[4] = batch_step;
            reader_args[5] = channel_step;
            reader_args[6] = num_tiles_per_core;
            reader_args[7] = start_tile;
            reader_args[8] = hw;
            reader_args[9] = n;

            writer_args[0] = output_buffer->address();
            writer_args[1] = num_tiles_per_core;
            writer_args[2] = num_tiles_read;
        }

        num_tiles_read += num_tiles_per_core;
    }
}

operation::ProgramWithCallbacks transpose_cn_multi_core(const Tensor& a, Tensor& output) {
    TT_ASSERT(a.storage_type() == StorageType::DEVICE, "Operand to transpose_cn needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to transpose_cn needs to be allocated in a buffer on device!");

    tt::tt_metal::Program program = tt::tt_metal::Program();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::IDevice* device = a.device();

    uint32_t num_tensor_tiles = a.volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_cb_index, (std::uint32_t)src0_is_dram};
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)src0_cb_index, (std::uint32_t)dst_is_dram};

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_cn_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    override_runtime_args_mc_cn<true>(
        program,
        reader_kernel_id,
        writer_kernel_id,
        a,
        output,
        num_cores_total,
        num_cores,
        num_cores_y,
        core_group_1,
        num_tiles_per_core_group_1,
        core_group_2,
        num_tiles_per_core_group_2);

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, compute_with_storage_grid_size](
                                              const void* operation,
                                              const Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_tensor = input_tensors.at(0);
        auto dst_tensor = output_tensors.at(0);

        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;

        uint32_t num_cores_total = num_cores_x * num_cores_y;
        uint32_t num_tensor_tiles = src_tensor.volume() / TILE_HW;

        auto
            [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
                tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);

        override_runtime_args_mc_cn<false>(
            program,
            reader_kernel_id,
            writer_kernel_id,
            src_tensor,
            dst_tensor,
            num_cores_total,
            num_cores,
            num_cores_y,
            core_group_1,
            num_tiles_per_core_group_1,
            core_group_2,
            num_tiles_per_core_group_2);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

template <bool IS_CREATING>
void override_runtime_args_mc_hc(
    const Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_tiles_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_tiles_per_core_group_2) {
    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_padded_shape();
    auto output_shape = output_tensor.get_padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t HW = H * W;
    uint32_t HW_bytes = HW * input_tensor.element_size();
    uint32_t CHW = C * H * W;
    uint32_t CHW_bytes = CHW * input_tensor.element_size();

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t Ct = C / TILE_HEIGHT;
    uint32_t CtHWt = Ct * H * Wt;
    uint32_t CtWt = Ct * Wt;

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core;

        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op
            num_tiles_per_core = 0;
        }

        uint32_t h = num_tiles_read / CtWt % H;  // Current h index output of current batch
        uint32_t ct = num_tiles_read / Wt % Ct;  // Current Ct index output tile of current batch

        if constexpr (IS_CREATING) {
            tt::tt_metal::SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {input_buffer->address(),
                 Wt,
                 H,
                 Ct,
                 HW_bytes,
                 CHW_bytes,
                 num_tiles_read,
                 num_tiles_per_core,
                 num_tiles_read / CtHWt * CHW_bytes,
                 h,
                 h / TILE_HEIGHT * Wt,
                 ct,
                 ct * TILE_HEIGHT * HW_bytes,
                 num_tiles_read % Wt});

            tt::tt_metal::SetRuntimeArgs(
                program, writer_kernel_id, core, {output_buffer->address(), num_tiles_per_core, num_tiles_read});
        } else {
            auto& reader_args = cached_reader_args.at(core.x).at(core.y);
            auto& writer_args = cached_writer_args.at(core.x).at(core.y);

            reader_args[0] = input_buffer->address();
            reader_args[1] = Wt;
            reader_args[2] = H;
            reader_args[3] = Ct;
            reader_args[4] = HW_bytes;
            reader_args[5] = CHW_bytes;
            reader_args[6] = num_tiles_read;
            reader_args[7] = num_tiles_per_core;
            reader_args[8] = num_tiles_read / CtHWt * CHW_bytes;
            reader_args[9] = h;
            reader_args[10] = h / TILE_HEIGHT * Wt;
            reader_args[11] = ct;
            reader_args[12] = ct * TILE_HEIGHT * HW_bytes;
            reader_args[13] = num_tiles_read % Wt;

            writer_args[0] = output_buffer->address();
            writer_args[1] = num_tiles_per_core;
            writer_args[2] = num_tiles_read;
        }

        num_tiles_read += num_tiles_per_core;
    }
}

template <bool IS_CREATING>
void override_runtime_args_mc_hc_rm(
    const Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_w_sticks_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_w_sticks_per_core_group_2) {
    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_padded_shape();
    auto output_shape = output_tensor.get_padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t W_bytes = W * input_tensor.element_size();

    uint32_t max_read_size = 2048;  // TILE size
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    for (uint32_t i = 0, curr_sticks_read = 0, curr_sticks_write = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_sticks_per_core;

        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_w_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_w_sticks_per_core_group_2;
        } else {
            // no-op
            num_sticks_per_core = 0;
        }

        // issue more reads before calling barrier
        uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
        if (num_sticks_per_core != 0) {
            num_sticks_per_core_read =
                tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core, W_bytes, max_read_size);
            num_read_per_barrier = num_sticks_per_core / num_sticks_per_core_read;
        }

        if constexpr (IS_CREATING) {
            tt::tt_metal::SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {input_buffer->address(),
                 num_sticks_per_core_read,
                 num_read_per_barrier,
                 curr_sticks_read,
                 curr_c,
                 curr_h,
                 curr_n});

            tt::tt_metal::SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {output_buffer->address(), num_sticks_per_core_read, num_read_per_barrier, curr_sticks_write});
        } else {
            auto& reader_args = cached_reader_args.at(core.x).at(core.y);
            auto& writer_args = cached_writer_args.at(core.x).at(core.y);

            reader_args[0] = input_buffer->address();
            reader_args[1] = num_sticks_per_core_read;
            reader_args[2] = num_read_per_barrier;
            reader_args[3] = curr_sticks_read;
            reader_args[4] = curr_c;
            reader_args[5] = curr_h;
            reader_args[6] = curr_n;

            writer_args[0] = output_buffer->address();
            writer_args[1] = num_sticks_per_core_read;
            writer_args[2] = num_read_per_barrier;
            writer_args[3] = curr_sticks_write;
        }

        curr_sticks_write += num_sticks_per_core;

        for (uint32_t i = 0; i < num_sticks_per_core; ++i) {
            curr_c++;
            curr_sticks_read += H;
            if (curr_c == C) {  // end of channel dim
                curr_h++;
                curr_c = 0;
                if (curr_h == H) {  // end of H dim
                    curr_n++;
                    curr_c = 0;
                    curr_h = 0;
                    curr_sticks_read = curr_sticks_read - H + 1;
                } else {
                    curr_sticks_read = curr_sticks_read - C * H + 1;
                }
            }
        }
    }
}

template <bool IS_CREATING>
void override_runtime_args_mc_hc_tiled_interleaved(
    const Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const Tensor& input_tensor,
    Tensor& output_tensor) {
    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();

    auto tile_shape = input_tensor.get_tensor_spec().tile().get_tile_shape();
    auto tile_hw = (tile_shape[0] * tile_shape[1]);
    uint32_t num_tensor_tiles = input_tensor.volume() / tile_hw;
    uint32_t num_output_tiles = output_tensor.volume() / tile_hw;
    uint32_t W = input_tensor.get_logical_shape()[3], H = input_tensor.get_logical_shape()[2],
             C = input_tensor.get_logical_shape()[1], N = input_tensor.get_logical_shape()[0];
    bool needs_padding = C % tile_shape[0] != 0;
    uint32_t padded_num_tensor_tiles = num_output_tiles / (output_tensor.get_padded_shape()[2] /
                                                           tile_shape[0]);  // only last row of Ct should have padding

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);
    auto
        [padded_num_cores,
         padded_all_cores,
         padded_core_group_1,
         padded_core_group_2,
         padded_num_tiles_per_core_group_1,
         padded_num_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, padded_num_tensor_tiles);

    all_cores = num_cores > padded_num_cores ? all_cores : padded_all_cores;
    auto cores = corerange_to_cores(all_cores, std::nullopt);

    uint32_t start_idx = 0;
    uint32_t padded_start_idx = 0;
    for (const auto& core : cores) {
        uint32_t num_tiles_per_core;
        uint32_t padded_tiles_per_core;

        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op
            num_tiles_per_core = 0;
        }

        if (padded_core_group_1.contains(core)) {
            padded_tiles_per_core = padded_num_tiles_per_core_group_1;
        } else if (padded_core_group_2.contains(core)) {
            padded_tiles_per_core = padded_num_tiles_per_core_group_2;
        } else {
            // no-op
            padded_tiles_per_core = 0;
        }

        uint32_t end_idx = start_idx + num_tiles_per_core;
        uint32_t padded_end_idx = padded_start_idx + padded_tiles_per_core;
        if constexpr (IS_CREATING) {
            tt::tt_metal::SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {
                    input_buffer->address(),
                    num_tiles_per_core,
                    start_idx,
                });

            tt::tt_metal::SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {output_buffer->address(), start_idx, end_idx, padded_start_idx, padded_end_idx});
        } else {
            auto& reader_args = cached_reader_args.at(core.x).at(core.y);
            auto& writer_args = cached_writer_args.at(core.x).at(core.y);

            reader_args[0] = input_buffer->address();
            writer_args[0] = output_buffer->address();
        }
        start_idx = end_idx;
        padded_start_idx = padded_end_idx;
    }
}

operation::ProgramWithCallbacks transpose_hc_multi_core_tiled_interleaved(
    const Tensor& a, Tensor& output, const std::optional<float>& pad_value) {
    TT_ASSERT(a.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    tt::tt_metal::Program program = tt::tt_metal::Program();
    auto tile = a.get_tensor_spec().tile();
    auto tile_shape = tile.get_tile_shape();
    auto face_shape = tile.get_face_shape();
    uint32_t num_tensor_tiles = a.volume() / (tile_shape[0] * tile_shape[1]);
    uint32_t num_output_tiles = output.volume() / (tile_shape[0] * tile_shape[1]);
    uint32_t W = a.get_logical_shape()[3], H = a.get_logical_shape()[2], C = a.get_logical_shape()[1],
             N = a.get_logical_shape()[0];
    bool needs_padding = (C % tile_shape[1] != 0) && pad_value.has_value();
    uint32_t padded_num_tensor_tiles =
        num_output_tiles / (output.get_padded_shape()[2] / tile_shape[0]);  // only last row of Ct should have padding

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);

    auto compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t padding_cb_index = tt::CBIndex::c_1;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(2 * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    if (needs_padding) {
        tt::tt_metal::CircularBufferConfig cb_src1_config =
            tt::tt_metal::CircularBufferConfig(face_shape[1] * a.element_size(), {{padding_cb_index, cb_data_format}})
                .set_page_size(padding_cb_index, face_shape[1] * a.element_size());
        auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src1_config);
    }

    // create reader kernel with compile time and runtime args
    tt::tt_metal::Buffer* src_buffer = a.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    uint32_t element_size = a.element_size();
    uint32_t padding_val_packed = 0;
    uint32_t num_writes = 0;
    if (pad_value.has_value()) {
        if (C % tile_shape[1] != 0) {
            uint32_t num_packed_values = sizeof(uint32_t) / element_size;
            num_writes = face_shape[1] / num_packed_values;
            if (a.get_dtype() == DataType::BFLOAT16) {
                padding_val_packed =
                    pack_two_bfloat16_into_uint32({bfloat16(pad_value.value()), bfloat16(pad_value.value())});
            } else if (num_packed_values == 2) {
                padding_val_packed =
                    static_cast<uint32_t>(pad_value.value()) | (static_cast<uint32_t>(pad_value.value()) << 16);
            } else {
                padding_val_packed = std::bit_cast<uint32_t>(pad_value.value());
            }
        }
    }
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)src_is_dram, num_writes, padding_val_packed, (uint32_t)needs_padding, (uint32_t)0, 1, 1, 1, 1, 1};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // create writer kernel with compile time and runtime args

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)dst_is_dram,
        a.element_size(),
        tt::CBIndex::c_0,
        C,
        H,
        W,
        tile_shape[0],
        tile_shape[1],
        face_shape[0],
        face_shape[1],
        (uint32_t)needs_padding};

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    override_runtime_args_mc_hc_tiled_interleaved<true>(
        program, unary_reader_kernel_id, unary_writer_kernel_id, a, output);

    auto override_runtime_args_callback =
        [unary_reader_kernel_id, unary_writer_kernel_id, compute_with_storage_grid_size](
            const void* operation,
            const Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>&,
            const std::vector<Tensor>& output_tensors) {
            auto src_tensor = input_tensors.at(0);
            auto dst_tensor = output_tensors.at(0);

            override_runtime_args_mc_hc_tiled_interleaved<false>(
                program, unary_reader_kernel_id, unary_writer_kernel_id, src_tensor, dst_tensor);
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

operation::ProgramWithCallbacks transpose_hc_multi_core(
    const Tensor& a, Tensor& output, const std::optional<float>& pad_value) {
    const auto shape = a.get_padded_shape();
    if (a.get_layout() == Layout::TILE && !a.is_sharded()) {
        return transpose_hc_multi_core_tiled_interleaved(a, output, pad_value);
    }
    uint32_t sub_tile_line_bytes = 16 * a.element_size();

    uint32_t num_tensor_tiles = a.volume() / TILE_HW;
    const auto& a_shape = a.get_logical_shape();
    uint32_t W = a_shape[3], H = a_shape[2], C = a_shape[1], N = a_shape[0];
    uint32_t NCH = N * C * H;
    bool row_major = a.get_layout() == Layout::ROW_MAJOR;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);

    tt::tt_metal::Buffer* src0_dram_buffer = a.buffer();

    tt::log_debug("transpose_hc_multi_core");
    tt::log_debug("sub_tile_line_bytes: {}", sub_tile_line_bytes);
    tt::log_debug("cb_data_format: {}", cb_data_format);
    tt::log_debug("single_tile_size: {}", single_tile_size);

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::IDevice* device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, row_major ? NCH : num_tensor_tiles);

    auto output_shape = output.get_padded_shape();

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    // check if we need to allocate a scratch buffer
    // The kernel reads several 16 element face lines (32B for BFLOAT16) from different input tiles to form a single
    // output tile, one output tile at a time Each face line is 32 bytes, so if our minimum read alignment is greater
    // than that (64B for Blackhole) then we will have reads from unaligned face-lines into differently aligned
    // destination face-lines
    // TODO: noc_async_write only require 16B alignment for both DRAM and L1 for Blackhole, so instead of reading in
    // face-lines from C tiles to form a single tile, we can load a single tile and then write out its face-lines to C
    // tiles
    uint32_t alignment = dst_buffer->alignment();
    bool misaligned = alignment > sub_tile_line_bytes;
    if (row_major) {
        auto num_sticks = num_tiles_per_core_group_1 > num_tiles_per_core_group_2 ? num_tiles_per_core_group_1
                                                                                  : num_tiles_per_core_group_2;
        auto stick_size = W * a.element_size();
        tt::tt_metal::CircularBufferConfig cb_src0_config =
            tt::tt_metal::CircularBufferConfig(num_sticks * stick_size, {{src0_cb_index, cb_data_format}})
                .set_page_size(src0_cb_index, stick_size);
        auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);
    } else {
        uint32_t num_input_tiles = 2;
        tt::tt_metal::CircularBufferConfig cb_src0_config =
            tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
                .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

        // need some scratch memory here - if we need data from a misaligned address then we need to read from the
        // nearest aligned address and then copy the data to the correct location
        if (misaligned) {
            uint32_t src1_cb_index = 1;
            tt::tt_metal::CircularBufferConfig cb_src1_config =
                tt::tt_metal::CircularBufferConfig(alignment, {{src1_cb_index, cb_data_format}})
                    .set_page_size(src1_cb_index, alignment);
            auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src1_config);
        }
    }

    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_is_dram};
    if (row_major) {
        reader_compile_time_args.push_back((std::uint32_t)N);
        reader_compile_time_args.push_back((std::uint32_t)H);
        reader_compile_time_args.push_back((std::uint32_t)C);
        reader_compile_time_args.push_back((std::uint32_t)W * a.element_size());

        auto stick_size = W * a.element_size();
        bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
        reader_compile_time_args.push_back((std::uint32_t)stick_size_is_power_of_two);
        if (stick_size_is_power_of_two) {
            uint32_t log2_stick_size = (std::uint32_t)std::log2(stick_size);
            reader_compile_time_args.push_back((std::uint32_t)log2_stick_size);
        } else {
            reader_compile_time_args.push_back(stick_size);
        }
    } else {
        reader_compile_time_args.push_back((std::uint32_t)sub_tile_line_bytes);
        reader_compile_time_args.push_back((std::uint32_t)(cb_data_format == tt::DataFormat::Float32));
        reader_compile_time_args.push_back((std::uint32_t)alignment);
    }
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)src0_cb_index, (std::uint32_t)dst_is_dram};
    if (row_major) {
        writer_compile_time_args.push_back((std::uint32_t)W * a.element_size());

        auto stick_size = W * a.element_size();
        bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
        writer_compile_time_args.push_back((std::uint32_t)stick_size_is_power_of_two);
        if (stick_size_is_power_of_two) {
            uint32_t log2_stick_size = (std::uint32_t)std::log2(stick_size);
            writer_compile_time_args.push_back((std::uint32_t)log2_stick_size);
        } else {
            writer_compile_time_args.push_back(stick_size);
        }
    }

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        row_major ? "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                    "reader_unary_transpose_hc_interleaved_partitioned_rm.cpp"
                  : "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                    "reader_unary_transpose_hc_interleaved_partitioned.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        row_major
            ? "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
              "writer_unary_transpose_hc_interleaved_start_id_rm.cpp"
            : "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    if (row_major) {
        override_runtime_args_mc_hc_rm<true>(
            program,
            reader_kernel_id,
            writer_kernel_id,
            a,
            output,
            num_cores_total,
            num_cores,
            num_cores_y,
            core_group_1,
            num_tiles_per_core_group_1,
            core_group_2,
            num_tiles_per_core_group_2);
    } else {
        override_runtime_args_mc_hc<true>(
            program,
            reader_kernel_id,
            writer_kernel_id,
            a,
            output,
            num_cores_total,
            num_cores,
            num_cores_y,
            core_group_1,
            num_tiles_per_core_group_1,
            core_group_2,
            num_tiles_per_core_group_2);
    }

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, compute_with_storage_grid_size](
                                              const void* operation,
                                              const Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_tensor = input_tensors.at(0);
        auto dst_tensor = output_tensors.at(0);

        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;

        uint32_t num_cores_total = num_cores_x * num_cores_y;

        uint32_t num_tensor_tiles = src_tensor.volume() / TILE_HW;

        uint32_t H = src_tensor.get_logical_shape()[2], C = src_tensor.get_logical_shape()[1],
                 N = src_tensor.get_logical_shape()[0];
        uint32_t NCH = N * C * H;
        bool row_major = src_tensor.get_layout() == Layout::ROW_MAJOR;

        auto
            [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
                tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, row_major ? NCH : num_tensor_tiles);

        if (row_major) {
            override_runtime_args_mc_hc_rm<false>(
                program,
                reader_kernel_id,
                writer_kernel_id,
                src_tensor,
                dst_tensor,
                num_cores_total,
                num_cores,
                num_cores_y,
                core_group_1,
                num_tiles_per_core_group_1,
                core_group_2,
                num_tiles_per_core_group_2);
        } else {
            override_runtime_args_mc_hc<false>(
                program,
                reader_kernel_id,
                writer_kernel_id,
                src_tensor,
                dst_tensor,
                num_cores_total,
                num_cores,
                num_cores_y,
                core_group_1,
                num_tiles_per_core_group_1,
                core_group_2,
                num_tiles_per_core_group_2);
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_runtime_args_mc_hc_rm_sharded(
    const Tensor& input_tensor, Tensor& output_tensor, uint32_t num_cores, uint32_t num_cores_x, uint32_t num_cores_y) {
    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_padded_shape();
    auto output_shape = output_tensor.get_padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t W_bytes = W * input_tensor.element_size();

    auto shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    uint32_t shard_width = shard_spec.shape[1];
    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    tt::tt_metal::IDevice* device = input_tensor.device();

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores);

    std::vector<uint32_t> shard_grid_x_map;
    for (uint32_t i = 0; i < num_cores_x; ++i) {
        auto physical_core = device->worker_core_from_logical_core(CoreCoord(i, 0));
        shard_grid_x_map.push_back(physical_core.x);
    }
    std::vector<uint32_t> shard_grid_y_map;
    for (uint32_t i = 0; i < num_cores_y; ++i) {
        auto physical_core = device->worker_core_from_logical_core(CoreCoord(0, i));
        shard_grid_y_map.push_back(physical_core.y);
    }

    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    for (uint32_t i = 0, curr_sticks_read = 0; i < num_cores; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x, i / num_cores_x};
        } else {
            core = {i / num_cores_y, i % num_cores_y};
        }
        uint32_t num_sticks_per_core = shard_height;

        // reader
        std::vector<uint32_t> reader_runtime_args = {num_sticks_per_core, curr_sticks_read, curr_c, curr_h, curr_n};
        reader_runtime_args.insert(reader_runtime_args.end(), shard_grid_x_map.begin(), shard_grid_x_map.end());
        reader_runtime_args.insert(reader_runtime_args.end(), shard_grid_y_map.begin(), shard_grid_y_map.end());

        // writer
        std::vector<uint32_t> writer_runtime_args;

        ret_val[i] = {reader_runtime_args, writer_runtime_args};

        for (uint32_t i = 0; i < num_sticks_per_core; ++i) {
            curr_c++;
            curr_sticks_read += H;
            if (curr_c == C) {  // end of channel dim
                curr_h++;
                curr_c = 0;
                if (curr_h == H) {  // end of H dim
                    curr_n++;
                    curr_c = 0;
                    curr_h = 0;
                    curr_sticks_read = curr_sticks_read - H + 1;
                } else {
                    curr_sticks_read = curr_sticks_read - C * H + 1;
                }
            }
        }
    }

    return ret_val;
}

std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_runtime_args_mc_hc_rm_sharded_special_case(
    const Tensor& input_tensor, Tensor& output_tensor, uint32_t num_cores, uint32_t num_cores_x, uint32_t num_cores_y) {
    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_padded_shape();
    auto output_shape = output_tensor.get_padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t W_bytes = W * input_tensor.element_size();
    uint32_t total_height = N * C * H;
    uint32_t stick_size_bytes = W * input_tensor.element_size();

    auto shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    uint32_t shard_width = shard_spec.shape[1];
    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    tt::tt_metal::IDevice* device = input_tensor.device();

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores);

    uint32_t height = 0;
    std::vector<CoreCoord> cores;
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x, i / num_cores_x};
        } else {
            core = {i / num_cores_y, i % num_cores_y};
        }

        height += shard_height;

        if (height <= total_height) {
            cores.push_back(core);
        }

        tt::log_debug("core: {}", core);
    }

    uint32_t CH = C * H;
    uint32_t NCH = N * C * H;

    uint32_t num_H_per_core = shard_height / H > 0 ? shard_height / H : 1;    // the number of H blocks in a shard
    uint32_t num_N_per_core = shard_height / CH > 0 ? shard_height / CH : 1;  // the number of N blocks in a shard

    uint32_t shard_C_per_core = shard_height > C ? C : shard_height;  // the number of shards of (dst) C blocks per core
    uint32_t shard_H_per_core = shard_height > H ? H : shard_height;  // the number of shards of H blocks per core

    uint32_t num_core_per_C = C / shard_height > 0 ? C / shard_height : 1;  // the number of cores for (dst) C block
    uint32_t num_core_per_H = H / shard_height > 0 ? H / shard_height : 1;  // the number of cores for H block

    uint32_t num_C_blocks_per_core = shard_height > C ? shard_height / C : 1;

    uint32_t curr_core_offset = 0;
    uint32_t curr_height = 0;
    uint32_t curr_core = 0;
    uint32_t curr_N = 0;
    uint32_t curr_C = 0;
    uint32_t curr_H = 0;
    uint32_t curr_C_shard = 0;
    uint32_t curr_H_shard = 0;

    uint32_t curr_c = 0, curr_h = 0;
    for (uint32_t i = 0, curr_sticks_read = 0; i < num_cores; i++) {
        auto core = cores[i];
        uint32_t pre_core = curr_core;
        uint32_t pre_N = curr_N;
        std::vector<uint32_t> read_cores_indices;
        std::vector<uint32_t> read_cores_noc_x;
        std::vector<uint32_t> read_cores_noc_y;
        std::vector<uint32_t> read_stick_offset;

        uint32_t num_sticks_per_core = shard_height;

        std::vector<uint32_t> stick_ids_per_core;
        for (uint32_t j = 0; j < num_sticks_per_core; ++j) {
            stick_ids_per_core.push_back(curr_sticks_read);
            curr_c++;
            curr_sticks_read += H;
            if (curr_c == C) {  // end of channel dim
                curr_h++;
                curr_c = 0;
                if (curr_h == H) {  // end of H dim
                    curr_c = 0;
                    curr_h = 0;
                    curr_sticks_read = curr_sticks_read - H + 1;
                } else {
                    curr_sticks_read = curr_sticks_read - C * H + 1;
                }
            }
        }

        // figure out the stick id in a shard, and the core id for the stick.
        std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> core_stick_map;
        for (uint32_t j = 0; j < num_sticks_per_core; ++j) {
            uint32_t stick_id = stick_ids_per_core[j];
            uint32_t shard_id = stick_id / num_sticks_per_core;
            uint32_t stick_id_in_shard = stick_id - (shard_id * num_sticks_per_core);

            uint32_t shard_grid_inner_dim = row_major ? num_cores_x : num_cores_y;
            uint32_t shard_grid_outer_dim_id = shard_id / shard_grid_inner_dim;
            uint32_t shard_grid_inner_dim_id = shard_id - (shard_grid_outer_dim_id * shard_grid_inner_dim);

            uint32_t worker_y_logical = row_major ? shard_grid_outer_dim_id : shard_grid_inner_dim_id;
            uint32_t worker_x_logical = row_major ? shard_grid_inner_dim_id : shard_grid_outer_dim_id;

            if (worker_x_logical < num_cores_x and worker_y_logical < num_cores_y) {
                auto core_physical =
                    device->worker_core_from_logical_core(CoreCoord{worker_x_logical, worker_y_logical});

                read_cores_indices.push_back(shard_id);
                read_stick_offset.push_back(stick_id_in_shard * stick_size_bytes);
                read_cores_noc_x.push_back(core_physical.x);
                read_cores_noc_y.push_back(core_physical.y);
            }
        }

        // reader rt args
        std::vector<uint32_t> non_repeat_stick_offset_values;
        std::vector<uint32_t> non_repeat_noc_x_values;
        std::vector<uint32_t> non_repeat_noc_y_values;

        uint32_t num_sticks_per_shard_core = 0;
        uint32_t num_sticks_per_shard_core_reader = 0, num_sticks_per_shard_core_writer = 0,
                 writer_read_stick_offset = 0, writer_write_stick_offset = 0;
        uint32_t num_C_blocks_per_core_reader = num_C_blocks_per_core, num_C_blocks_per_core_writer = 0;

        uint32_t num_non_repeat_cores = read_cores_indices.size();
        uint32_t read_stick_stride = read_stick_offset.size() > 1 ? read_stick_offset[1] - read_stick_offset[0] : 0;

        if (num_H_per_core == 1) {  // each core only has one H block or part of H block
            for (uint32_t i = 1; i < read_cores_indices.size(); ++i) {
                if (read_cores_indices[i] == read_cores_indices[0]) {
                    num_non_repeat_cores = i;
                    read_stick_stride = read_stick_offset[i] - read_stick_offset[0];
                    break;
                }
            }

            num_sticks_per_shard_core = shard_height / num_non_repeat_cores;
            num_sticks_per_shard_core_reader = num_sticks_per_shard_core;
            bool split_reader = num_sticks_per_shard_core > 2;
            if (split_reader) {
                num_sticks_per_shard_core_reader = num_sticks_per_shard_core / 2;
                num_sticks_per_shard_core_writer = num_sticks_per_shard_core - num_sticks_per_shard_core_reader;
                writer_read_stick_offset = num_sticks_per_shard_core_reader * read_stick_stride;
                writer_write_stick_offset = writer_read_stick_offset * num_non_repeat_cores;
            }

            for (uint32_t i = 0; i < num_non_repeat_cores; ++i) {
                non_repeat_stick_offset_values.push_back(read_stick_offset[i]);
                non_repeat_noc_x_values.push_back(read_cores_noc_x[i]);
                non_repeat_noc_y_values.push_back(read_cores_noc_y[i]);
            }
        } else {  // contains multiple H blocks
            std::set<uint32_t> unique_values(read_cores_indices.begin(), read_cores_indices.end());
            num_non_repeat_cores = unique_values.size();
            read_stick_stride = read_stick_offset[1] - read_stick_offset[0];

            // TODO: add the second batch args (num_non_repeat_cores, read_stick_offset, non_repeat_noc_x_values,
            // non_repeat_noc_y_values) to support multiple batch in a shard
            for (uint32_t j = 1; j < num_sticks_per_core; ++j) {
                if ((read_cores_indices[j - 1] == read_cores_indices[j]) and
                    (read_stick_offset[j] == read_stick_offset[j - 1] + stick_size_bytes)) {
                    break;
                }
            }

            num_sticks_per_shard_core = shard_height / num_non_repeat_cores / num_C_blocks_per_core;
            num_sticks_per_shard_core_reader = num_sticks_per_shard_core;
            num_sticks_per_shard_core_writer = num_sticks_per_shard_core;
            bool split_reader = num_C_blocks_per_core > 2;
            if (split_reader) {
                num_C_blocks_per_core_reader = num_C_blocks_per_core / 2;
                num_C_blocks_per_core_writer = num_C_blocks_per_core - num_C_blocks_per_core_reader;
                writer_read_stick_offset = num_C_blocks_per_core_reader * stick_size_bytes;
                writer_write_stick_offset =
                    num_C_blocks_per_core_reader * num_non_repeat_cores * num_sticks_per_shard_core * stick_size_bytes;
            }

            for (uint32_t i = 0; i < num_non_repeat_cores; ++i) {
                non_repeat_stick_offset_values.push_back(read_stick_offset[i * num_sticks_per_shard_core]);
                non_repeat_noc_x_values.push_back(read_cores_noc_x[i * num_sticks_per_shard_core]);
                non_repeat_noc_y_values.push_back(read_cores_noc_y[i * num_sticks_per_shard_core]);
            }
        }

        bool read_single_h_block_per_core = num_H_per_core == 1;

        std::vector<uint32_t> reader_runtime_args = {
            (std::uint32_t)read_single_h_block_per_core,
            (std::uint32_t)num_C_blocks_per_core_reader,
            (std::uint32_t)num_sticks_per_shard_core_reader,
            (std::uint32_t)num_non_repeat_cores,
            (std::uint32_t)read_stick_stride,
        };

        reader_runtime_args.insert(
            reader_runtime_args.end(), non_repeat_stick_offset_values.begin(), non_repeat_stick_offset_values.end());
        reader_runtime_args.insert(
            reader_runtime_args.end(), non_repeat_noc_x_values.begin(), non_repeat_noc_x_values.end());
        reader_runtime_args.insert(
            reader_runtime_args.end(), non_repeat_noc_y_values.begin(), non_repeat_noc_y_values.end());

        // writer rt args
        std::vector<uint32_t> writer_runtime_args = {
            (std::uint32_t)read_single_h_block_per_core,
            (std::uint32_t)num_C_blocks_per_core_writer,
            (std::uint32_t)num_sticks_per_shard_core_writer,
            (std::uint32_t)num_non_repeat_cores,
            (std::uint32_t)read_stick_stride,
            (std::uint32_t)writer_read_stick_offset,
            (std::uint32_t)writer_write_stick_offset,

        };

        writer_runtime_args.insert(
            writer_runtime_args.end(), non_repeat_stick_offset_values.begin(), non_repeat_stick_offset_values.end());
        writer_runtime_args.insert(
            writer_runtime_args.end(), non_repeat_noc_x_values.begin(), non_repeat_noc_x_values.end());
        writer_runtime_args.insert(
            writer_runtime_args.end(), non_repeat_noc_y_values.begin(), non_repeat_noc_y_values.end());

        ret_val[i] = {reader_runtime_args, writer_runtime_args};
    }

    return ret_val;
}

operation::ProgramWithCallbacks transpose_hc_multi_core_sharded(const Tensor& a, Tensor& output) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt::tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt::tt_metal::detail::TileSize(dst_cb_data_format);

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    const auto shape = a.get_padded_shape();
    uint32_t W = a.get_logical_shape()[3], H = a.get_logical_shape()[2], C = a.get_logical_shape()[1],
             N = a.get_logical_shape()[0];
    uint32_t total_height = N * C * H;
    uint32_t stick_size_bytes = W * a.element_size();

    tt::tt_metal::IDevice* device = a.device();

    auto shard_spec = a.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    uint32_t shard_width = shard_spec.shape[1];
    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    bool is_special_case = false;
    if ((shard_spec.shape[0] % H == 0 or H % shard_spec.shape[0] == 0) &&
        (shard_spec.shape[0] % C == 0 or C % shard_spec.shape[0] == 0) && (C % H == 0 or H % C == 0) &&
        (shard_height <= C * H)) {
        is_special_case = true;
    }

    auto& all_cores = shard_spec.grid;
    uint32_t num_cores = shard_spec.num_cores();
    auto bbox = shard_spec.grid.bounding_box();
    CoreCoord grid_size = {bbox.end_coord.x + 1, bbox.end_coord.y + 1};
    uint32_t num_cores_x = grid_size.x;
    uint32_t num_cores_y = grid_size.y;

    tt::log_debug("all_cores: {}", all_cores);
    tt::log_debug("num_cores: {}", num_cores);

    auto output_shape = output.get_padded_shape();

    tt::tt_metal::Buffer* dst_buffer = output.buffer();

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(shard_height * stick_size_bytes, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, stick_size_bytes)
            .set_globally_allocated_address(*a.buffer());
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(shard_height * stick_size_bytes, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, stick_size_bytes)
            .set_globally_allocated_address(*output.buffer());
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    std::vector<uint32_t> reader_compile_time_args;
    if (is_special_case) {
        reader_compile_time_args = {
            (std::uint32_t)src0_cb_index, (std::uint32_t)output_cb_index, (std::uint32_t)stick_size_bytes};
    } else {
        reader_compile_time_args = {
            (std::uint32_t)src0_cb_index,
            (std::uint32_t)output_cb_index,
            (std::uint32_t)N,
            (std::uint32_t)H,
            (std::uint32_t)C,
            (std::uint32_t)stick_size_bytes,
            (std::uint32_t)row_major,
            (std::uint32_t)num_cores_x,
            (std::uint32_t)num_cores_y};
    }

    // defines
    std::map<string, string> reader_defines;
    if (is_special_case) {
        reader_defines["USE_SPECIAL_CASE"] = "1";
    }

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_sharded_rm.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    tt::tt_metal::KernelHandle writer_kernel_id;
    if (is_special_case) {
        std::vector<uint32_t> writer_compile_time_args = {
            (std::uint32_t)src0_cb_index, (std::uint32_t)output_cb_index, (std::uint32_t)stick_size_bytes};

        writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
            "writer_unary_transpose_hc_sharded_rm.cpp",
            all_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    }

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> all_runtime_args;
    if (is_special_case) {
        all_runtime_args =
            get_runtime_args_mc_hc_rm_sharded_special_case(a, output, num_cores, num_cores_x, num_cores_y);
    } else {
        all_runtime_args = get_runtime_args_mc_hc_rm_sharded(a, output, num_cores, num_cores_x, num_cores_y);
    }

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x, i / num_cores_x};
        } else {
            core = {i / num_cores_y, i % num_cores_y};
        }

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args[i].first);

        tt::tt_metal::SetRuntimeArgs(
            program, writer_kernel_id, core, all_runtime_args[i].second

        );
    }

    auto override_runtime_args_callback =
        [cb_src0, cb_output, src0_single_tile_size, dst_single_tile_size, num_cores_x, num_cores_y](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>&,
            const std::vector<Tensor>& output_tensors) {
            const auto& src_tensor = input_tensors.at(0);
            const auto& dst_tensor = output_tensors.at(0);

            const auto src_buffer = src_tensor.buffer();
            const auto dst_buffer = dst_tensor.buffer();

            UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
            UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

template <bool IS_CREATING>
void override_runtime_args_wh(
    const Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle compute_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_tiles_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_tiles_per_core_group_2) {
    auto input_shape = input_tensor.get_padded_shape();
    auto output_shape = output_tensor.get_padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], NC = input_shape[1] * input_shape[0];
    uint32_t HW = H * W;

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;

    uint32_t num_tensor_tiles = input_tensor.volume() / TILE_HW;
    auto HtWt = Ht * Wt;

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_compute_args = GetRuntimeArgs(program, compute_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core;

        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // noop
            num_tiles_per_core = 0;

            if constexpr (!IS_CREATING) {
                auto& reader_args = cached_reader_args.at(core.x).at(core.y);
                auto& compute_args = cached_compute_args.at(core.x).at(core.y);
                auto& writer_args = cached_writer_args.at(core.x).at(core.y);

                reader_args[1] = 0;
                compute_args[0] = 0;
                writer_args[1] = 0;
                continue;
            }
        }

        uint32_t h = num_tiles_read % Ht;
        uint32_t w = num_tiles_read / Ht % Wt;

        if constexpr (IS_CREATING) {
            tt::tt_metal::SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {input_tensor.buffer()->address(),
                 num_tiles_per_core,
                 tt::round_down(num_tiles_read, HtWt) + h * Wt + w,
                 h,
                 w,
                 Ht,
                 Wt,
                 HtWt});

            tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {num_tiles_per_core});

            tt::tt_metal::SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {output_tensor.buffer()->address(), num_tiles_per_core, num_tiles_read});
        } else {
            auto& reader_args = cached_reader_args.at(core.x).at(core.y);
            auto& compute_args = cached_compute_args.at(core.x).at(core.y);
            auto& writer_args = cached_writer_args.at(core.x).at(core.y);

            reader_args[0] = input_tensor.buffer()->address();
            reader_args[1] = num_tiles_per_core;
            reader_args[2] = tt::round_down(num_tiles_read, HtWt) + h * Wt + w;
            reader_args[3] = h;
            reader_args[4] = w;
            reader_args[5] = Ht;
            reader_args[6] = Wt;
            reader_args[7] = HtWt;

            compute_args[0] = num_tiles_per_core;

            writer_args[0] = output_tensor.buffer()->address();
            writer_args[1] = num_tiles_per_core;
            writer_args[2] = num_tiles_read;
        }

        // std::vector<uint32_t> compute_runtime_args = {num_tiles_per_core};

        // std::vector<uint32_t> reader_runtime_args = {
        //         input_tensor.buffer()->address(),
        //         num_tiles_per_core,
        //         tt::round_down(num_tiles_read, HtWt) + h * Wt + w,
        //         h,
        //         w,
        //         Ht,
        //         Wt,
        //         HtWt
        // };

        // std::vector<uint32_t> writer_runtime_args = {
        //         output_tensor.buffer()->address(),
        //         num_tiles_per_core,
        //         num_tiles_read
        // };

        num_tiles_read += num_tiles_per_core;
    }
}

template <bool IS_CREATING>
void override_runtime_args_wh_rm(
    const Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle compute_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_hw_blocks_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_hw_blocks_per_core_group_2) {
    auto input_shape = input_tensor.get_logical_shape();
    auto output_shape = output_tensor.get_logical_shape();

    uint32_t W = input_shape[3], H = input_shape[2], NC = input_shape[1] * input_shape[0];
    uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_compute_args = GetRuntimeArgs(program, compute_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    for (uint32_t i = 0, num_sticks_read = 0, num_sticks_write = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_hw_blocks_per_core;

        if (core_group_1.contains(core)) {
            num_hw_blocks_per_core = num_hw_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_hw_blocks_per_core = num_hw_blocks_per_core_group_2;
        } else {
            // noop
            num_hw_blocks_per_core = 0;
        }

        if constexpr (IS_CREATING) {
            tt::tt_metal::SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {
                    input_tensor.buffer()->address(),
                    num_sticks_read,
                    num_hw_blocks_per_core,
                });

            tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {num_hw_blocks_per_core});

            tt::tt_metal::SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {
                    output_tensor.buffer()->address(),
                    num_sticks_write,
                    num_hw_blocks_per_core,
                });
        } else {
            auto& reader_args = cached_reader_args.at(core.x).at(core.y);
            auto& compute_args = cached_compute_args.at(core.x).at(core.y);
            auto& writer_args = cached_writer_args.at(core.x).at(core.y);

            reader_args[0] = input_tensor.buffer()->address();
            reader_args[1] = num_sticks_read;
            reader_args[2] = num_hw_blocks_per_core;

            compute_args[0] = num_hw_blocks_per_core;

            writer_args[0] = output_tensor.buffer()->address();
            writer_args[1] = num_sticks_write;
            writer_args[2] = num_hw_blocks_per_core;
        }

        // // compute
        // std::vector<uint32_t> compute_runtime_args = {num_hw_blocks_per_core};

        // // reader
        // std::vector<uint32_t> reader_runtime_args = {
        //         input_tensor.buffer()->address(),
        //         num_sticks_read,
        //         num_hw_blocks_per_core,
        // };

        // // writer
        // std::vector<uint32_t> writer_runtime_args = {
        //         output_tensor.buffer()->address(),
        //         num_sticks_write,
        //         num_hw_blocks_per_core,
        // };

        num_sticks_read += num_hw_blocks_per_core * H;
        num_sticks_write += num_hw_blocks_per_core * W;
    }
}

operation::ProgramWithCallbacks transpose_wh_multi_core(const Tensor& a, Tensor& output) {
    uint32_t num_tensor_tiles = a.volume() / TILE_HW;
    uint32_t W = a.get_logical_shape()[3], H = a.get_logical_shape()[2], C = a.get_logical_shape()[1],
             N = a.get_logical_shape()[0], NC = a.get_logical_shape()[1] * a.get_logical_shape()[0];
    bool row_major = a.get_layout() == Layout::ROW_MAJOR;

    uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt::tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt::tt_metal::detail::TileSize(dst_cb_data_format);

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::IDevice* device = a.device();

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, row_major ? NC : num_tensor_tiles);

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = row_major ? wt * 2 : 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = row_major ? ht * 2 : 2;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size);
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_output_config);

    if (row_major) {
        // tilize cb
        uint32_t im_cb_index = 24;
        uint32_t num_im_tiles = ht * wt;
        tt::tt_metal::CircularBufferConfig cb_im_config =
            tt::tt_metal::CircularBufferConfig(
                num_im_tiles * src0_single_tile_size, {{im_cb_index, src0_cb_data_format}})
                .set_page_size(im_cb_index, src0_single_tile_size);
        auto cb_im = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_im_config);

        // untilize cb
        uint32_t im2_cb_index = 25;
        uint32_t num_im2_tiles = ht;
        tt::tt_metal::CircularBufferConfig cb_im2_config =
            tt::tt_metal::CircularBufferConfig(
                num_im2_tiles * dst_single_tile_size, {{im2_cb_index, dst_cb_data_format}})
                .set_page_size(im2_cb_index, dst_single_tile_size);
        auto cb_im2 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_im2_config);
    }

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_is_dram,
    };
    if (row_major) {
        reader_compile_time_args.push_back(ht);
        reader_compile_time_args.push_back(H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT);
        reader_compile_time_args.push_back(H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT);
        reader_compile_time_args.push_back(wt);
        reader_compile_time_args.push_back(W);
        reader_compile_time_args.push_back(ht * wt);
        reader_compile_time_args.push_back(W * a.element_size());
        reader_compile_time_args.push_back(wt * a.element_size() * TILE_WIDTH);

        auto stick_size = W * a.element_size();
        bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
        reader_compile_time_args.push_back((std::uint32_t)stick_size_is_power_of_two);
        if (stick_size_is_power_of_two) {
            uint32_t log2_stick_size = (std::uint32_t)std::log2(stick_size);
            reader_compile_time_args.push_back((std::uint32_t)log2_stick_size);
        } else {
            reader_compile_time_args.push_back(stick_size);
        }
    }

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};
    if (row_major) {
        writer_compile_time_args.push_back(ht);
        writer_compile_time_args.push_back(H);
        writer_compile_time_args.push_back(wt);
        writer_compile_time_args.push_back(W > TILE_WIDTH ? TILE_WIDTH : W % TILE_WIDTH);
        writer_compile_time_args.push_back(W % TILE_WIDTH == 0 ? TILE_WIDTH : W % TILE_WIDTH);
        writer_compile_time_args.push_back(ht * wt);
        writer_compile_time_args.push_back(H * output.element_size());
        writer_compile_time_args.push_back(ht * output.element_size() * TILE_HEIGHT);

        auto stick_size = H * output.element_size();
        bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
        writer_compile_time_args.push_back((std::uint32_t)stick_size_is_power_of_two);
        if (stick_size_is_power_of_two) {
            uint32_t log2_stick_size = (std::uint32_t)std::log2(stick_size);
            writer_compile_time_args.push_back((std::uint32_t)log2_stick_size);
        } else {
            writer_compile_time_args.push_back(stick_size);
        }
    }

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        row_major ? "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                    "reader_unary_transpose_wh_interleaved_start_id_rm.cpp"
                  : "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                    "reader_unary_transpose_wh_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        row_major
            ? "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
              "writer_unary_transpose_wh_interleaved_start_id_rm.cpp"
            : "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_kernel_args = {};
    if (row_major) {
        compute_kernel_args.push_back(ht);
        compute_kernel_args.push_back(wt);
        compute_kernel_args.push_back(ht * wt);
    }
    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        row_major ? "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_rm.cpp"
                  : "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh.cpp",
        total_cores,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_kernel_args,
        });

    if (row_major) {
        override_runtime_args_wh_rm<true>(
            program,
            reader_kernel_id,
            compute_kernel_id,
            writer_kernel_id,
            a,
            output,
            num_cores_total,
            num_cores,
            num_cores_y,
            core_group_1,
            num_tiles_per_core_group_1,
            core_group_2,
            num_tiles_per_core_group_2);
    } else {
        override_runtime_args_wh<true>(
            program,
            reader_kernel_id,
            compute_kernel_id,
            writer_kernel_id,
            a,
            output,
            num_cores_total,
            num_cores,
            num_cores_y,
            core_group_1,
            num_tiles_per_core_group_1,
            core_group_2,
            num_tiles_per_core_group_2);
    }

    auto override_runtime_args_callback = [reader_kernel_id,
                                           compute_kernel_id,
                                           writer_kernel_id,
                                           compute_with_storage_grid_size](
                                              const void* operation,
                                              const Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_tensor = input_tensors.at(0);
        auto dst_tensor = output_tensors.at(0);

        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        uint32_t num_cores_total = num_cores_x * num_cores_y;
        uint32_t num_tensor_tiles = src_tensor.volume() / TILE_HW;
        uint32_t NC = src_tensor.get_logical_shape()[1] * src_tensor.get_logical_shape()[0];
        bool row_major = src_tensor.get_layout() == Layout::ROW_MAJOR;

        auto
            [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
                tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, row_major ? NC : num_tensor_tiles);

        if (row_major) {
            override_runtime_args_wh_rm<false>(
                program,
                reader_kernel_id,
                compute_kernel_id,
                writer_kernel_id,
                src_tensor,
                dst_tensor,
                num_cores_total,
                num_cores,
                num_cores_y,
                core_group_1,
                num_tiles_per_core_group_1,
                core_group_2,
                num_tiles_per_core_group_2);
        } else {
            override_runtime_args_wh<false>(
                program,
                reader_kernel_id,
                compute_kernel_id,
                writer_kernel_id,
                src_tensor,
                dst_tensor,
                num_cores_total,
                num_cores,
                num_cores_y,
                core_group_1,
                num_tiles_per_core_group_1,
                core_group_2,
                num_tiles_per_core_group_2);
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

operation::ProgramWithCallbacks transpose_wh_multi_core_sharded(const Tensor& a, Tensor& output) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt::tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt::tt_metal::detail::TileSize(dst_cb_data_format);

    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    const auto tile = a.get_tensor_spec().tile();
    const uint32_t tile_hw = tile.get_tile_hw();
    int32_t num_tiles = a.volume() / tile_hw;

    tt::tt_metal::IDevice* device = a.device();

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto shard_spec = a.shard_spec().value();

    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    auto& all_cores = shard_spec.grid;
    uint32_t num_cores = all_cores.num_cores();
    uint32_t num_tiles_per_shard = shard_spec.numel() / tile_hw;

    auto output_shape = output.get_padded_shape();

    tt::tt_metal::Buffer* dst_buffer = output.buffer();

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = num_tiles_per_shard;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size)
            .set_globally_allocated_address(*a.buffer());
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = num_tiles_per_shard;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_output_config);

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_cb_index,
    };

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_cb_index,
    };

    std::vector<uint32_t> compute_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)output_cb_index,
    };

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_sharded.cpp",
        total_cores,
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_compile_time_args});

    auto padded_shape = a.get_padded_shape();
    auto shard_shape = shard_spec.shape;

    uint32_t H = padded_shape[2], W = padded_shape[3];
    uint32_t Hs = shard_shape[0], Ws = shard_shape[1];

    uint32_t Hts = Hs / tile.tile_shape[0];
    uint32_t Wts = Ws / tile.tile_shape[1];

    uint32_t Ht = H / tile.tile_shape[0];
    uint32_t Ht_per_shard = std::min(Ht, Hts);

    uint32_t num_hw_blocks_per_shard = Hts > Ht ? Hts / Ht : 1;

    uint32_t HtWt_tile_size = Ht_per_shard * Wts;
    uint32_t num_blocks = num_hw_blocks_per_shard * HtWt_tile_size;

    auto bbox = all_cores.bounding_box();
    std::vector<CoreCoord> cores =
        grid_to_cores_with_noop(bbox.end_coord.x, bbox.end_coord.y, num_cores_x, num_cores_y, row_major);

    std::vector<std::vector<uint32_t>> unary_reader_args = {cores.size(), std::vector<uint32_t>(1)};
    std::vector<std::vector<uint32_t>> unary_compute_args = {cores.size(), std::vector<uint32_t>(5)};
    std::vector<std::vector<uint32_t>> unary_writer_args = {cores.size(), std::vector<uint32_t>(1)};
    std::fill(
        unary_reader_args.begin(),
        unary_reader_args.begin() + all_cores.num_cores(),
        std::vector<uint32_t>{num_blocks});
    std::fill(
        unary_compute_args.begin(),
        unary_compute_args.begin() + all_cores.num_cores(),
        std::vector<uint32_t>{num_blocks, HtWt_tile_size, num_hw_blocks_per_shard, Ht_per_shard, Wts});
    std::fill(
        unary_writer_args.begin(),
        unary_writer_args.begin() + all_cores.num_cores(),
        std::vector<uint32_t>{num_blocks});

    tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, cores, unary_reader_args);
    tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, cores, unary_compute_args);
    tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, cores, unary_writer_args);

    auto override_runtime_args_callback = [reader_kernel_id,
                                           compute_kernel_id,
                                           writer_kernel_id,
                                           cb_src0,
                                           cb_output,
                                           src0_single_tile_size,
                                           dst_single_tile_size,
                                           num_cores_x,
                                           num_cores_y](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        const auto& src_tensor = input_tensors.at(0);
        const auto& dst_tensor = output_tensors.at(0);

        const auto src_buffer = src_tensor.buffer();
        const auto dst_buffer = dst_tensor.buffer();

        bool src0_sharded = src_tensor.is_sharded();
        bool out_sharded = dst_tensor.is_sharded();

        auto shard_spec = src_tensor.shard_spec().value();

        const auto tile = src_tensor.get_tensor_spec().tile();
        const uint32_t tile_hw = tile.get_tile_hw();
        int32_t num_tiles = src_tensor.volume() / tile_hw;

        uint32_t num_tiles_per_shard = shard_spec.numel() / tile_hw;

        if (src0_sharded) {
            UpdateDynamicCircularBufferAddressAndTotalSize(
                program, cb_src0, *src_buffer, num_tiles_per_shard * src0_single_tile_size);
        }

        if (out_sharded) {
            UpdateDynamicCircularBufferAddressAndTotalSize(
                program, cb_output, *dst_buffer, num_tiles_per_shard * dst_single_tile_size);
        }

        auto padded_shape = src_tensor.get_padded_shape();
        auto shard_shape = shard_spec.shape;

        uint32_t H = padded_shape[2], W = padded_shape[3];
        uint32_t Hs = shard_shape[0], Ws = shard_shape[1];

        uint32_t Hts = Hs / tile.tile_shape[0];
        uint32_t Wts = Ws / tile.tile_shape[1];

        uint32_t Ht = H / tile.tile_shape[0];
        uint32_t Ht_per_shard = std::min(Ht, Hts);

        uint32_t num_hw_blocks_per_shard = Hts > Ht ? Hts / Ht : 1;

        uint32_t HtWt_tile_size = Ht_per_shard * Wts;
        uint32_t num_blocks = num_hw_blocks_per_shard * HtWt_tile_size;

        const auto& all_cores = shard_spec.grid;
        bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

        auto bbox = all_cores.bounding_box();
        std::vector<CoreCoord> cores =
            grid_to_cores_with_noop(bbox.end_coord.x, bbox.end_coord.y, num_cores_x, num_cores_y, row_major);
        std::vector<std::vector<uint32_t>> unary_reader_args = {cores.size(), std::vector<uint32_t>(1)};
        std::vector<std::vector<uint32_t>> unary_compute_args = {cores.size(), std::vector<uint32_t>(5)};
        std::vector<std::vector<uint32_t>> unary_writer_args = {cores.size(), std::vector<uint32_t>(1)};
        std::fill(
            unary_reader_args.begin(),
            unary_reader_args.begin() + all_cores.num_cores(),
            std::vector<uint32_t>{num_blocks});
        std::fill(
            unary_compute_args.begin(),
            unary_compute_args.begin() + all_cores.num_cores(),
            std::vector<uint32_t>{num_blocks, HtWt_tile_size, num_hw_blocks_per_shard, Ht_per_shard, Wts});
        std::fill(
            unary_writer_args.begin(),
            unary_writer_args.begin() + all_cores.num_cores(),
            std::vector<uint32_t>{num_blocks});

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, cores, unary_reader_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, cores, unary_compute_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, cores, unary_writer_args);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

operation::ProgramWithCallbacks transpose_wh_multi_core_sharded_rm(const Tensor& a, Tensor& output) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt::tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt::tt_metal::detail::TileSize(dst_cb_data_format);

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    const auto shape = a.get_padded_shape();
    uint32_t W = a.get_logical_shape()[3], H = a.get_logical_shape()[2], C = a.get_logical_shape()[1],
             N = a.get_logical_shape()[0];
    uint32_t total_height = N * C * H;
    uint32_t stick_size_bytes = W * a.element_size();
    uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    uint32_t output_page_size, pack_num_pages, pack_num_pages_last_col, pack_num_pages_last_row,
        pack_num_pages_last_row_col;
    if ((W % TILE_WIDTH) != 0 and (H % TILE_HEIGHT) != 0) {
        output_page_size = (W % TILE_WIDTH) * (H % TILE_HEIGHT) * output.element_size();
        pack_num_pages = dst_single_tile_size / output_page_size;
        auto output_page_size_last_col = TILE_WIDTH * (H % TILE_HEIGHT) * output.element_size();
        pack_num_pages_last_col = dst_single_tile_size / output_page_size_last_col;
        auto output_page_size_last_row = TILE_HEIGHT * (W % TILE_WIDTH) * output.element_size();
        pack_num_pages_last_row = dst_single_tile_size / output_page_size_last_row;
        pack_num_pages_last_row_col = 1;
    } else if ((W % TILE_WIDTH) != 0 and (H % TILE_HEIGHT) == 0) {
        output_page_size = (W % TILE_WIDTH) * (TILE_HEIGHT)*output.element_size();
        pack_num_pages = dst_single_tile_size / output_page_size;
        pack_num_pages_last_col = pack_num_pages;
        pack_num_pages_last_row = 1;
        pack_num_pages_last_row_col = 1;
    } else if ((W % TILE_WIDTH) == 0 and (H % TILE_HEIGHT) != 0) {
        output_page_size = (TILE_WIDTH) * (H % TILE_HEIGHT) * output.element_size();
        pack_num_pages = dst_single_tile_size / output_page_size;
        pack_num_pages_last_col = 1;
        pack_num_pages_last_row = pack_num_pages;
        pack_num_pages_last_row_col = 1;
    } else {
        output_page_size = dst_single_tile_size;
        pack_num_pages = 1;
        pack_num_pages_last_col = 1;
        pack_num_pages_last_row = 1;
        pack_num_pages_last_row_col = 1;
    }

    tt::log_debug("output_page_size: {}", output_page_size);
    tt::log_debug("pack_num_pages: {}", pack_num_pages);
    tt::log_debug("pack_num_pages_last_col: {}", pack_num_pages_last_col);
    tt::log_debug("pack_num_pages_last_row: {}", pack_num_pages_last_row);
    tt::log_debug("pack_num_pages_last_row_col: {}", pack_num_pages_last_row_col);

    auto shard_spec = a.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    uint32_t shard_width = shard_spec.shape[1];
    uint32_t num_hw_blocks_per_core = shard_height / H;

    tt::log_debug("shard_height: {}", shard_height);
    tt::log_debug("dst_single_tile_size: {}", dst_single_tile_size);

    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    tt::tt_metal::IDevice* device = a.device();

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;

    auto& all_cores = shard_spec.grid;
    uint32_t num_cores = shard_spec.num_cores();
    auto bbox = shard_spec.grid.bounding_box();
    CoreCoord grid_size = {bbox.end_coord.x + 1, bbox.end_coord.y + 1};
    uint32_t num_cores_x = grid_size.x;
    uint32_t num_cores_y = grid_size.y;

    tt::log_debug("all_cores: {}", all_cores);
    tt::log_debug("num_cores: {}", num_cores);

    auto output_shape = output.get_padded_shape();

    // sharded cb
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(shard_height * stick_size_bytes, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, stick_size_bytes)
            .set_globally_allocated_address(*a.buffer());
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    // sharded cb
    uint32_t output_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(stick_size_bytes * shard_height, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, output_page_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    // cb_in
    uint32_t in_cb_index = tt::CBIndex::c_24;
    uint32_t num_in_tiles = wt * 2;  // double buffer
    tt::tt_metal::CircularBufferConfig cb_in_config =
        tt::tt_metal::CircularBufferConfig(num_in_tiles * src0_single_tile_size, {{in_cb_index, src0_cb_data_format}})
            .set_page_size(in_cb_index, src0_single_tile_size);
    auto cb_in = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_in_config);

    // tilize cb
    uint32_t im_cb_index = tt::CBIndex::c_25;
    uint32_t num_im_tiles = ht * wt;
    tt::tt_metal::CircularBufferConfig cb_im_config =
        tt::tt_metal::CircularBufferConfig(num_im_tiles * src0_single_tile_size, {{im_cb_index, src0_cb_data_format}})
            .set_page_size(im_cb_index, src0_single_tile_size);
    auto cb_im = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im_config);

    // untilize cb
    if (ht > 8) {
        uint32_t im2_cb_index = tt::CBIndex::c_26;
        uint32_t num_im2_tiles = ht;
        tt::tt_metal::CircularBufferConfig cb_im2_config =
            tt::tt_metal::CircularBufferConfig(
                num_im2_tiles * dst_single_tile_size, {{im2_cb_index, dst_cb_data_format}})
                .set_page_size(im2_cb_index, dst_single_tile_size);
        auto cb_im2 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im2_config);

        // compute_output_cb
        uint32_t out_cb_index = tt::CBIndex::c_27;
        uint32_t num_out_tiles = ht * 2;  // double buffer
        tt::tt_metal::CircularBufferConfig cb_out_config =
            tt::tt_metal::CircularBufferConfig(
                num_out_tiles * dst_single_tile_size, {{out_cb_index, dst_cb_data_format}})
                .set_page_size(out_cb_index, dst_single_tile_size);
        auto cb_out = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);
    }

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)num_hw_blocks_per_core,
        (std::uint32_t)ht,
        (std::uint32_t)H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT,
        (std::uint32_t)H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT,
        (std::uint32_t)wt,
        (std::uint32_t)stick_size_bytes,
        (std::uint32_t)wt * a.element_size() * TILE_WIDTH,
    };
    reader_compile_time_args.push_back(H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT);
    reader_compile_time_args.push_back(H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT);

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_wh_sharded_rm.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)num_hw_blocks_per_core,
        (std::uint32_t)ht,
        (std::uint32_t)wt,
        (std::uint32_t)W > TILE_WIDTH ? TILE_WIDTH : W % TILE_WIDTH,
        (std::uint32_t)W % TILE_WIDTH == 0 ? TILE_WIDTH : W % TILE_WIDTH,
        (std::uint32_t)H * output.element_size(),
        (std::uint32_t)ht * output.element_size() * TILE_HEIGHT,
    };

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "writer_unary_transpose_wh_sharded_rm.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_compile_time_args = {
        (std::uint32_t)ht,
        (std::uint32_t)wt,
        (std::uint32_t)ht * wt,
        (std::uint32_t)num_hw_blocks_per_core,
        (std::uint32_t)H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT,  // last_output_row_num_datums
        (std::uint32_t)pack_num_pages,
        (std::uint32_t)pack_num_pages_last_col,
        (std::uint32_t)pack_num_pages_last_row,
        (std::uint32_t)pack_num_pages_last_row_col,
    };

    std::map<string, string> compute_defines;
    compute_defines["SHARDED"] = "1";

    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_rm.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_compile_time_args,
            .defines = compute_defines});

    auto override_runtime_args_callback =
        [reader_kernel_id, cb_src0, cb_output, src0_single_tile_size, dst_single_tile_size, num_cores_x, num_cores_y](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>&,
            const std::vector<Tensor>& output_tensors) {
            const auto& src_tensor = input_tensors.at(0);
            const auto& dst_tensor = output_tensors.at(0);

            const auto src_buffer = src_tensor.buffer();
            const auto dst_buffer = dst_tensor.buffer();

            UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
            UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement::detail
