// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_log.h"

namespace ttnn::operations::data_movement::detail {

using namespace tt::constants;

template <bool IS_CREATING>
void override_runtime_args_mc_cn(
    const Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const Tensor &input_tensor,
    Tensor &output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores,
    uint32_t num_cores_y,
    CoreRangeSet core_group_1,
    uint32_t num_tiles_per_core_group_1,
    CoreRangeSet core_group_2,
    uint32_t num_tiles_per_core_group_2
){
    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = N*C*H*W / TILE_HW;
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

        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            //no-op
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
                {
                    input_buffer->address(),
                    N,
                    C,
                    HtWt,
                    batch_step,
                    channel_step,
                    num_tiles_per_core,
                    start_tile,
                    hw,
                    n
                }
            );

            tt::tt_metal::SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {
                    output_buffer->address(),
                    num_tiles_per_core,
                    num_tiles_read
                }
            );
        }
        else {
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


operation::ProgramWithCallbacks transpose_cn_multi_core(const Tensor &a, Tensor &output) {

    TT_ASSERT(a.storage_type() == StorageType::DEVICE, "Operand to transpose_cn needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to transpose_cn needs to be allocated in a buffer on device!");

    tt::tt_metal::Program program = tt::tt_metal::Program();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);

    tt::tt_metal::Buffer *src0_buffer = a.buffer();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::Device *device = a.device();

    uint32_t num_tensor_tiles = a.volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);

    tt::tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config = tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) src0_is_dram
    };
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) dst_is_dram
    };

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/reader_unary_transpose_cn_interleaved_start_id.cpp",
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
        a, output, num_cores_total, num_cores, num_cores_y, core_group_1, num_tiles_per_core_group_1, core_group_2, num_tiles_per_core_group_2);

    auto override_runtime_args_callback = [
            reader_kernel_id,
            writer_kernel_id,
            compute_with_storage_grid_size
        ]
    (
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_tensor = input_tensors.at(0);
        auto dst_tensor = output_tensors.at(0);

        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;

        uint32_t num_cores_total = num_cores_x * num_cores_y;
        uint32_t num_tensor_tiles = src_tensor.volume() / TILE_HW;

        auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);

        override_runtime_args_mc_cn<false>(
            program,
            reader_kernel_id,
            writer_kernel_id, src_tensor, dst_tensor, num_cores_total, num_cores, num_cores_y, core_group_1, num_tiles_per_core_group_1, core_group_2, num_tiles_per_core_group_2);
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

template <bool IS_CREATING>
void override_runtime_args_mc_hc(
    const Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const Tensor &input_tensor,
    Tensor &output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores,
    uint32_t num_cores_y,
    CoreRangeSet core_group_1,
    uint32_t num_tiles_per_core_group_1,
    CoreRangeSet core_group_2,
    uint32_t num_tiles_per_core_group_2
){
    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t HW = H*W;
    uint32_t HW_bytes = HW * input_tensor.element_size();
    uint32_t CHW = C*H*W;
    uint32_t CHW_bytes = CHW * input_tensor.element_size();

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;
    uint32_t Ct = C/TILE_HEIGHT;
    uint32_t CtHWt = Ct*H*Wt;
    uint32_t CtWt = Ct * Wt;

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    for(uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core;

        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            //no-op
            num_tiles_per_core = 0;
        }

        uint32_t h = num_tiles_read / CtWt % H; // Current h index output of current batch
        uint32_t ct = num_tiles_read / Wt % Ct; // Current Ct index output tile of current batch

        if constexpr (IS_CREATING) {
            tt::tt_metal::SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {
                    input_buffer->address(),
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
                    num_tiles_read % Wt
                }
            );

            tt::tt_metal::SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {
                    output_buffer->address(),
                    num_tiles_per_core,
                    num_tiles_read
                }
            );
        }
        else {
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
    const Tensor &input_tensor,
    Tensor &output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores,
    uint32_t num_cores_y,
    CoreRangeSet core_group_1,
    uint32_t num_w_sticks_per_core_group_1,
    CoreRangeSet core_group_2,
    uint32_t num_w_sticks_per_core_group_2
){
    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t W_bytes = W * input_tensor.element_size();

    uint32_t max_read_size = 2048; // TILE size
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    for (uint32_t i = 0, curr_sticks_read = 0, curr_sticks_write = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_sticks_per_core;

        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_sticks_per_core = num_w_sticks_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_sticks_per_core = num_w_sticks_per_core_group_2;
        } else {
            //no-op
            num_sticks_per_core = 0;
        }

        // issue more reads before calling barrier
        uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
        if (num_sticks_per_core != 0) {
            num_sticks_per_core_read = merge_num_sticks_to_read(num_sticks_per_core, W_bytes, max_read_size);
            num_read_per_barrier = num_sticks_per_core / num_sticks_per_core_read;
        }

        if constexpr (IS_CREATING) {
            tt::tt_metal::SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {
                    input_buffer->address(),
                    num_sticks_per_core_read,
                    num_read_per_barrier,
                    curr_sticks_read,
                    curr_c,
                    curr_h,
                    curr_n
                }
            );

            tt::tt_metal::SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {
                    output_buffer->address(),
                    num_sticks_per_core_read,
                    num_read_per_barrier,
                    curr_sticks_write
                }
            );
        }
        else {
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
            if (curr_c == C) { // end of channel dim
                curr_h++;
                curr_c = 0;
                if (curr_h == H) { // end of H dim
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

operation::ProgramWithCallbacks transpose_hc_multi_core(const Tensor &a, Tensor &output) {

    const auto shape = a.get_legacy_shape();
    uint32_t sub_tile_line_bytes = 16 * a.element_size();

    uint32_t num_tensor_tiles = a.volume() / TILE_HW;
    uint32_t W = a.shape()[3], H = a.shape()[2], C = a.shape()[1], N = a.shape()[0];
    uint32_t NCH = N * C * H;
    bool row_major = a.get_layout() == Layout::ROW_MAJOR;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);

    tt::tt_metal::Buffer *src0_dram_buffer = a.buffer();

    tt::log_debug("transpose_hc_multi_core");
    tt::log_debug("sub_tile_line_bytes: {}", sub_tile_line_bytes);
    tt::log_debug("cb_data_format: {}", cb_data_format);
    tt::log_debug("single_tile_size: {}", single_tile_size);

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::Device *device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x-1, num_cores_y-1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, row_major ? NCH : num_tensor_tiles);

    tt::tt_metal::Shape output_shape = output.get_legacy_shape();

    tt::tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    if (row_major) {
        auto num_sticks = num_tiles_per_core_group_1 > num_tiles_per_core_group_2 ? num_tiles_per_core_group_1 : num_tiles_per_core_group_2;
        auto stick_size = W * a.element_size();
        tt::tt_metal::CircularBufferConfig cb_src0_config = tt::tt_metal::CircularBufferConfig(num_sticks * stick_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, stick_size);
        auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);
    } else {
        uint32_t num_input_tiles = 2;
        tt::tt_metal::CircularBufferConfig cb_src0_config = tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);
    }

    tt::tt_metal::Buffer *src0_buffer = a.buffer();
    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram
    };
    if (row_major) {
        reader_compile_time_args.push_back((std::uint32_t) N);
        reader_compile_time_args.push_back((std::uint32_t) H);
        reader_compile_time_args.push_back((std::uint32_t) C);
        reader_compile_time_args.push_back((std::uint32_t) W * a.element_size());

        auto stick_size = W * a.element_size();
        bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
        reader_compile_time_args.push_back((std::uint32_t) stick_size_is_power_of_two);
        if (stick_size_is_power_of_two) {
            uint32_t log2_stick_size = (std::uint32_t)log2(stick_size);
            reader_compile_time_args.push_back((std::uint32_t) log2_stick_size);
        } else {
            reader_compile_time_args.push_back(stick_size);
        }
    } else {
        reader_compile_time_args.push_back((std::uint32_t) sub_tile_line_bytes);
        reader_compile_time_args.push_back((std::uint32_t) (cb_data_format == tt::DataFormat::Float32));
    }
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) dst_is_dram
    };
    if (row_major) {
        writer_compile_time_args.push_back((std::uint32_t) W * a.element_size());

        auto stick_size = W * a.element_size();
        bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
        writer_compile_time_args.push_back((std::uint32_t) stick_size_is_power_of_two);
        if (stick_size_is_power_of_two) {
            uint32_t log2_stick_size = (std::uint32_t)log2(stick_size);
            writer_compile_time_args.push_back((std::uint32_t) log2_stick_size);
        } else {
            writer_compile_time_args.push_back(stick_size);
        }
    }

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        row_major ?
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/reader_unary_transpose_hc_interleaved_partitioned_rm.cpp" :
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/reader_unary_transpose_hc_interleaved_partitioned.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        row_major ?
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/writer_unary_transpose_hc_interleaved_start_id_rm.cpp" :
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    if (row_major) {
        override_runtime_args_mc_hc_rm<true>(
            program,
            reader_kernel_id,
            writer_kernel_id,
            a, output, num_cores_total, num_cores, num_cores_y, core_group_1, num_tiles_per_core_group_1, core_group_2, num_tiles_per_core_group_2);
    }
    else {
        override_runtime_args_mc_hc<true>(
            program,
            reader_kernel_id,
            writer_kernel_id,
            a, output, num_cores_total, num_cores, num_cores_y, core_group_1, num_tiles_per_core_group_1, core_group_2, num_tiles_per_core_group_2);
    }

    auto override_runtime_args_callback = [
            reader_kernel_id,
            writer_kernel_id,
            compute_with_storage_grid_size
        ]
    (
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_tensor = input_tensors.at(0);
        auto dst_tensor = output_tensors.at(0);

        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;

        uint32_t num_cores_total = num_cores_x * num_cores_y;

        uint32_t num_tensor_tiles = src_tensor.volume() / TILE_HW;

        uint32_t H = src_tensor.shape()[2], C = src_tensor.shape()[1], N = src_tensor.shape()[0];
        uint32_t NCH = N * C * H;
        bool row_major = src_tensor.get_layout() == Layout::ROW_MAJOR;

        auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, row_major ? NCH : num_tensor_tiles);

        if (row_major) {
            override_runtime_args_mc_hc_rm<false>(
                program,
                reader_kernel_id,
                writer_kernel_id,
                src_tensor, dst_tensor, num_cores_total, num_cores, num_cores_y, core_group_1, num_tiles_per_core_group_1, core_group_2, num_tiles_per_core_group_2);
        }
        else {
            override_runtime_args_mc_hc<false>(
                program,
                reader_kernel_id,
                writer_kernel_id,
                src_tensor, dst_tensor, num_cores_total, num_cores, num_cores_y, core_group_1, num_tiles_per_core_group_1, core_group_2, num_tiles_per_core_group_2);
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t> > > get_runtime_args_mc_hc_rm_sharded(
                                                                                        const Tensor &input_tensor,
                                                                                        Tensor &output_tensor,
                                                                                        uint32_t num_cores,
                                                                                        uint32_t num_cores_x,
                                                                                        uint32_t num_cores_y
                                                                                        ){
    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t W_bytes = W * input_tensor.element_size();

    auto shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    uint32_t shard_width = shard_spec.shape[1];
    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    tt::tt_metal::Device *device = input_tensor.device();

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t> > > ret_val(num_cores);

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
    for(uint32_t i = 0, curr_sticks_read = 0, curr_sticks_write = 0; i < num_cores; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x, i / num_cores_x};
        } else {
            core = {i / num_cores_y, i % num_cores_y};
        }
        uint32_t num_sticks_per_core = shard_height;

        // reader
        std::vector<uint32_t> reader_runtime_args = {
            num_sticks_per_core,
            curr_sticks_read,
            curr_c,
            curr_h,
            curr_n
        };
        reader_runtime_args.insert(reader_runtime_args.end(), shard_grid_x_map.begin(), shard_grid_x_map.end());
        reader_runtime_args.insert(reader_runtime_args.end(), shard_grid_y_map.begin(), shard_grid_y_map.end());

        // writer
        std::vector<uint32_t> writer_runtime_args;

        ret_val[i] = {reader_runtime_args, writer_runtime_args};

        curr_sticks_write += num_sticks_per_core;

        for (uint32_t i = 0; i < num_sticks_per_core; ++i) {
            curr_c++;
            curr_sticks_read += H;
            if (curr_c == C) { // end of channel dim
                curr_h++;
                curr_c = 0;
                if (curr_h == H) { // end of H dim
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

std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t> > > get_runtime_args_mc_hc_rm_sharded_special_case(
                                                                                        const Tensor &input_tensor,
                                                                                        Tensor &output_tensor,
                                                                                        uint32_t num_cores,
                                                                                        uint32_t num_cores_x,
                                                                                        uint32_t num_cores_y
                                                                                        ){

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t W_bytes = W * input_tensor.element_size();
    uint32_t total_height = N * C * H;
    uint32_t stick_size_bytes = W * input_tensor.element_size();

    auto shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    uint32_t shard_width = shard_spec.shape[1];
    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    tt::tt_metal::Device *device = input_tensor.device();

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t> > > ret_val(num_cores);

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

    uint32_t num_H_per_core = shard_height / H > 0 ? shard_height / H : 1; // the number of H blocks in a shard
    uint32_t num_N_per_core = shard_height / CH > 0 ? shard_height / CH : 1; // the number of N blocks in a shard

    uint32_t shard_C_per_core = shard_height > C ? C : shard_height; // the number of shards of (dst) C blocks per core
    uint32_t shard_H_per_core = shard_height > H ? H : shard_height; // the number of shards of H blocks per core

    uint32_t num_core_per_C = C / shard_height > 0 ? C / shard_height : 1; // the number of cores for (dst) C block
    uint32_t num_core_per_H = H / shard_height > 0 ? H / shard_height : 1; // the number of cores for H block

    uint32_t num_C_blocks_per_core = shard_height > C ? shard_height / C : 1;

    uint32_t curr_core_offset = 0;
    uint32_t curr_height = 0;
    uint32_t curr_core = 0;
    uint32_t curr_N = 0;
    uint32_t curr_C = 0;
    uint32_t curr_H = 0;
    uint32_t curr_C_shard = 0;
    uint32_t curr_H_shard = 0;

    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
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
            if (curr_c == C) { // end of channel dim
                curr_h++;
                curr_c = 0;
                if (curr_h == H) { // end of H dim
                    curr_n++;
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
                auto core_physical = device->worker_core_from_logical_core(CoreCoord{worker_x_logical, worker_y_logical});

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
        uint32_t num_sticks_per_shard_core_reader = 0, num_sticks_per_shard_core_writer = 0, writer_read_stick_offset = 0, writer_write_stick_offset = 0;
        uint32_t num_C_blocks_per_core_reader = num_C_blocks_per_core, num_C_blocks_per_core_writer = 0;

        uint32_t num_non_repeat_cores = read_cores_indices.size();
        uint32_t read_stick_stride = read_stick_offset.size() > 1 ? read_stick_offset[1] - read_stick_offset[0] : 0;

        bool has_second_batch = false;
        uint32_t num_sticks_before_second_batch = 0;

        if (num_H_per_core == 1) { // each core only has one H block or part of H block
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
        } else { // contains multiple H blocks
            std::set<uint32_t> unique_values(read_cores_indices.begin(), read_cores_indices.end());
            num_non_repeat_cores = unique_values.size();
            read_stick_stride = read_stick_offset[1] - read_stick_offset[0];

            // TODO: add the second batch args (num_non_repeat_cores, read_stick_offset, non_repeat_noc_x_values, non_repeat_noc_y_values) to support multiple batch in a shard
            for (uint32_t j = 1; j < num_sticks_per_core; ++j) {
                num_sticks_before_second_batch ++;
                if ((read_cores_indices[j-1] == read_cores_indices[j]) and (read_stick_offset[j] == read_stick_offset[j-1] + stick_size_bytes)) {
                    has_second_batch = true;
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
                writer_write_stick_offset = num_C_blocks_per_core_reader * num_non_repeat_cores * num_sticks_per_shard_core * stick_size_bytes;
            }

            for (uint32_t i = 0; i < num_non_repeat_cores; ++i) {
                non_repeat_stick_offset_values.push_back(read_stick_offset[i * num_sticks_per_shard_core]);
                non_repeat_noc_x_values.push_back(read_cores_noc_x[i * num_sticks_per_shard_core]);
                non_repeat_noc_y_values.push_back(read_cores_noc_y[i * num_sticks_per_shard_core]);
            }
        }

        bool read_single_h_block_per_core = num_H_per_core == 1;

        std::vector<uint32_t> reader_runtime_args = {
            (std::uint32_t) read_single_h_block_per_core,
            (std::uint32_t) num_C_blocks_per_core_reader,
            (std::uint32_t) num_sticks_per_shard_core_reader,
            (std::uint32_t) num_non_repeat_cores,
            (std::uint32_t) read_stick_stride,
        };

        reader_runtime_args.insert(reader_runtime_args.end(), non_repeat_stick_offset_values.begin(), non_repeat_stick_offset_values.end());
        reader_runtime_args.insert(reader_runtime_args.end(), non_repeat_noc_x_values.begin(), non_repeat_noc_x_values.end());
        reader_runtime_args.insert(reader_runtime_args.end(), non_repeat_noc_y_values.begin(), non_repeat_noc_y_values.end());

        // writer rt args
        std::vector<uint32_t> writer_runtime_args = {
            (std::uint32_t) read_single_h_block_per_core,
            (std::uint32_t) num_C_blocks_per_core_writer,
            (std::uint32_t) num_sticks_per_shard_core_writer,
            (std::uint32_t) num_non_repeat_cores,
            (std::uint32_t) read_stick_stride,
            (std::uint32_t) writer_read_stick_offset,
            (std::uint32_t) writer_write_stick_offset,

        };

        writer_runtime_args.insert(writer_runtime_args.end(), non_repeat_stick_offset_values.begin(), non_repeat_stick_offset_values.end());
        writer_runtime_args.insert(writer_runtime_args.end(), non_repeat_noc_x_values.begin(), non_repeat_noc_x_values.end());
        writer_runtime_args.insert(writer_runtime_args.end(), non_repeat_noc_y_values.begin(), non_repeat_noc_y_values.end());

        ret_val[i] = {reader_runtime_args, writer_runtime_args};
    }

    return ret_val;
}

operation::ProgramWithCallbacks transpose_hc_multi_core_sharded(const Tensor &a, Tensor &output) {


    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt::tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt::tt_metal::detail::TileSize(dst_cb_data_format);

    tt::tt_metal::Buffer *src0_buffer = a.buffer();

    const auto shape = a.get_legacy_shape();
    uint32_t W = a.shape()[3], H = a.shape()[2], C = a.shape()[1], N = a.shape()[0];
    uint32_t total_height = N * C * H;
    uint32_t stick_size_bytes = W * a.element_size();

    tt::tt_metal::Device *device = a.device();

    auto shard_spec = a.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    uint32_t shard_width = shard_spec.shape[1];
    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    bool is_special_case = false;
    if ((shard_spec.shape[0] % H == 0 or H % shard_spec.shape[0] == 0) &&
        (shard_spec.shape[0] % C == 0 or C % shard_spec.shape[0] == 0) &&
        (C % H == 0 or H % C == 0) &&
        (shard_height <= C * H)) {
        is_special_case = true;
    }

    auto& all_cores = shard_spec.grid;
    uint32_t num_cores = shard_spec.num_cores();
    auto bbox = shard_spec.grid.bounding_box();
    CoreCoord grid_size = {bbox.end_coord.x + 1, bbox.end_coord.y+1};
    uint32_t num_cores_x = grid_size.x;
    uint32_t num_cores_y = grid_size.y;

    tt::log_debug("all_cores: {}", all_cores);
    tt::log_debug("num_cores: {}", num_cores);

    tt::tt_metal::Shape output_shape = output.get_legacy_shape();

    tt::tt_metal::Buffer *dst_buffer = output.buffer();

    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_src0_config = tt::tt_metal::CircularBufferConfig(shard_height * stick_size_bytes, {{src0_cb_index, src0_cb_data_format}})
        .set_page_size(src0_cb_index, stick_size_bytes).set_globally_allocated_address(*a.buffer());
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CB::c_out0; // output operands start at index 16
    tt::tt_metal::CircularBufferConfig cb_output_config = tt::tt_metal::CircularBufferConfig(shard_height * stick_size_bytes, {{output_cb_index, dst_cb_data_format}})
        .set_page_size(output_cb_index, stick_size_bytes).set_globally_allocated_address(*output.buffer());
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    std::vector<uint32_t> reader_compile_time_args;
    if (is_special_case) {
        reader_compile_time_args = {
            (std::uint32_t) src0_cb_index,
            (std::uint32_t) output_cb_index,
            (std::uint32_t) stick_size_bytes
        };
    } else {
        reader_compile_time_args = {
            (std::uint32_t) src0_cb_index,
            (std::uint32_t) output_cb_index,
            (std::uint32_t) N,
            (std::uint32_t) H,
            (std::uint32_t) C,
            (std::uint32_t) stick_size_bytes,
            (std::uint32_t) row_major,
            (std::uint32_t) num_cores_x,
            (std::uint32_t) num_cores_y
        };
    }

    // defines
    std::map<string, string> reader_defines;
    if (is_special_case) {
        reader_defines["USE_SPECIAL_CASE"] = "1";
    }

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/reader_unary_transpose_hc_sharded_rm.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    tt::tt_metal::KernelHandle writer_kernel_id;
    if (is_special_case) {
        std::vector<uint32_t> writer_compile_time_args = {
            (std::uint32_t) src0_cb_index,
            (std::uint32_t) output_cb_index,
            (std::uint32_t) stick_size_bytes
        };

        writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/writer_unary_transpose_hc_sharded_rm.cpp",
            all_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    }

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t> > > all_runtime_args;
    if (is_special_case) {
        all_runtime_args = get_runtime_args_mc_hc_rm_sharded_special_case(a, output, num_cores, num_cores_x, num_cores_y);
    } else {
        all_runtime_args = get_runtime_args_mc_hc_rm_sharded(a, output, num_cores, num_cores_x, num_cores_y);
    }

    for(uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x, i / num_cores_x};
        } else {
            core = {i / num_cores_y, i % num_cores_y};
        }

        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            all_runtime_args[i].first
        );

        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            all_runtime_args[i].second

        );
    }

    auto override_runtime_args_callback = [
            cb_src0,
            cb_output,
            src0_single_tile_size,
            dst_single_tile_size,
            num_cores_x,
            num_cores_y
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        const auto& src_tensor = input_tensors.at(0);
        const auto& dst_tensor = output_tensors.at(0);

        const auto src_buffer = src_tensor.buffer();
        const auto dst_buffer = dst_tensor.buffer();

        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
    };

   return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

template <bool IS_CREATING>
void override_runtime_args_wh(
    const Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle compute_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const Tensor &input_tensor,
    Tensor &output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores,
    uint32_t num_cores_y,
    CoreRangeSet core_group_1,
    uint32_t num_tiles_per_core_group_1,
    CoreRangeSet core_group_2,
    uint32_t num_tiles_per_core_group_2
){
    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();

    uint32_t W = input_shape[3], H = input_shape[2], NC = input_shape[1]*input_shape[0];
    uint32_t HW = H*W;

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = input_tensor.volume() / TILE_HW;
    auto HtWt = Ht * Wt;

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_compute_args = GetRuntimeArgs(program, compute_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    for(uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core;

        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            //noop
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
                {
                    input_tensor.buffer()->address(),
                    num_tiles_per_core,
                    tt::round_down(num_tiles_read, HtWt) + h * Wt + w,
                    h,
                    w,
                    Ht,
                    Wt,
                    HtWt
                }
            );

            tt::tt_metal::SetRuntimeArgs(
                program,
                compute_kernel_id,
                core,
                { num_tiles_per_core }
            );

            tt::tt_metal::SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {
                    output_tensor.buffer()->address(),
                    num_tiles_per_core,
                    num_tiles_read
                }
            );
        }
        else {
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


template<bool IS_CREATING>
void override_runtime_args_wh_rm(
    const Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle compute_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const Tensor &input_tensor,
    Tensor &output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores,
    uint32_t num_cores_y,
    CoreRangeSet core_group_1,
    uint32_t num_hw_blocks_per_core_group_1,
    CoreRangeSet core_group_2,
    uint32_t num_hw_blocks_per_core_group_2
){
    auto input_shape = input_tensor.shape();
    auto output_shape = output_tensor.shape();

    uint32_t W = input_shape[3], H = input_shape[2], NC = input_shape[1]*input_shape[0];
    uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_compute_args = GetRuntimeArgs(program, compute_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    for (uint32_t i = 0, num_sticks_read = 0, num_sticks_write = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_hw_blocks_per_core;

        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_hw_blocks_per_core = num_hw_blocks_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_hw_blocks_per_core = num_hw_blocks_per_core_group_2;
        } else {
            //noop
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
                }
            );

            tt::tt_metal::SetRuntimeArgs(
                program,
                compute_kernel_id,
                core,
                { num_hw_blocks_per_core }
            );

            tt::tt_metal::SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {
                    output_tensor.buffer()->address(),
                    num_sticks_write,
                    num_hw_blocks_per_core,
                }
            );
        }
        else {
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


operation::ProgramWithCallbacks transpose_wh_multi_core(const Tensor &a, Tensor &output) {

    uint32_t num_tensor_tiles = a.volume() / TILE_HW;
    uint32_t W = a.shape()[3], H = a.shape()[2], C = a.shape()[1], N = a.shape()[0], NC = a.shape()[1] * a.shape()[0];
    bool row_major = a.get_layout() == Layout::ROW_MAJOR;

    uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt::tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt::tt_metal::detail::TileSize(dst_cb_data_format);

    tt::tt_metal::Buffer *src0_buffer = a.buffer();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::Device *device = a.device();

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x*num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x-1, num_cores_y-1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, row_major ? NC : num_tensor_tiles);

    tt::tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = row_major ? wt * 2 : 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config = tt::tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
		.set_page_size(src0_cb_index, src0_single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = row_major ? ht * 2 : 2;
    tt::tt_metal::CircularBufferConfig cb_output_config = tt::tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
		.set_page_size(output_cb_index, dst_single_tile_size);
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_output_config);

    if (row_major) {
        // tilize cb
        uint32_t im_cb_index = 24;
        uint32_t num_im_tiles = ht * wt;
        tt::tt_metal::CircularBufferConfig cb_im_config = tt::tt_metal::CircularBufferConfig(num_im_tiles * src0_single_tile_size, {{im_cb_index, src0_cb_data_format}})
            .set_page_size(im_cb_index, src0_single_tile_size);
        auto cb_im = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_im_config);

        // untilize cb
        uint32_t im2_cb_index = 25;
        uint32_t num_im2_tiles = ht;
        tt::tt_metal::CircularBufferConfig cb_im2_config = tt::tt_metal::CircularBufferConfig(num_im2_tiles * dst_single_tile_size, {{im2_cb_index, dst_cb_data_format}})
            .set_page_size(im2_cb_index, dst_single_tile_size);
        auto cb_im2 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_im2_config);
    }

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram,
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
        reader_compile_time_args.push_back((std::uint32_t) stick_size_is_power_of_two);
        if (stick_size_is_power_of_two) {
            uint32_t log2_stick_size = (std::uint32_t)log2(stick_size);
            reader_compile_time_args.push_back((std::uint32_t) log2_stick_size);
        } else {
            reader_compile_time_args.push_back(stick_size);
        }
    }

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) dst_is_dram
    };
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
        writer_compile_time_args.push_back((std::uint32_t) stick_size_is_power_of_two);
        if (stick_size_is_power_of_two) {
            uint32_t log2_stick_size = (std::uint32_t)log2(stick_size);
            writer_compile_time_args.push_back((std::uint32_t) log2_stick_size);
        } else {
            writer_compile_time_args.push_back(stick_size);
        }
    }

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        row_major ?
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/reader_unary_transpose_wh_interleaved_start_id_rm.cpp" :
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/reader_unary_transpose_wh_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        row_major ?
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/writer_unary_transpose_wh_interleaved_start_id_rm.cpp" :
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
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
        row_major ?
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_rm.cpp" :
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh.cpp",
        total_cores,
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en=fp32_dest_acc_en, .compile_args = compute_kernel_args,}
    );

    if (row_major) {
        override_runtime_args_wh_rm<true>(
            program, reader_kernel_id, compute_kernel_id, writer_kernel_id,
            a, output, num_cores_total, num_cores, num_cores_y,
            core_group_1, num_tiles_per_core_group_1, core_group_2,
            num_tiles_per_core_group_2);
    }
    else {
        override_runtime_args_wh<true>(
            program, reader_kernel_id, compute_kernel_id, writer_kernel_id,
            a, output, num_cores_total, num_cores, num_cores_y,
            core_group_1, num_tiles_per_core_group_1, core_group_2,
            num_tiles_per_core_group_2);
    }

    auto override_runtime_args_callback = [
            reader_kernel_id,
            compute_kernel_id,
            writer_kernel_id,
            compute_with_storage_grid_size
        ]
    (
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_tensor = input_tensors.at(0);
        auto dst_tensor = output_tensors.at(0);

        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        uint32_t num_cores_total = num_cores_x*num_cores_y;
        uint32_t num_tensor_tiles = src_tensor.volume() / TILE_HW;
        uint32_t NC = src_tensor.shape()[1] * src_tensor.shape()[0];
        bool row_major = src_tensor.get_layout() == Layout::ROW_MAJOR;

        auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, row_major ? NC : num_tensor_tiles);

        if (row_major) {
            override_runtime_args_wh_rm<false>(
                program, reader_kernel_id, compute_kernel_id, writer_kernel_id,
                src_tensor, dst_tensor, num_cores_total, num_cores, num_cores_y,
                core_group_1, num_tiles_per_core_group_1, core_group_2,
                num_tiles_per_core_group_2);
        }
        else {
            override_runtime_args_wh<false>(
                program, reader_kernel_id, compute_kernel_id, writer_kernel_id,
                src_tensor, dst_tensor, num_cores_total, num_cores, num_cores_y,
                core_group_1, num_tiles_per_core_group_1, core_group_2,
                num_tiles_per_core_group_2);
        }
    };

   return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

operation::ProgramWithCallbacks transpose_wh_multi_core_sharded(const Tensor &a, Tensor &output) {

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt::tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt::tt_metal::detail::TileSize(dst_cb_data_format);

    tt::tt_metal::Buffer *src0_buffer = a.buffer();

    int32_t num_tiles = a.volume()/TILE_HW;

    tt::tt_metal::Device *device = a.device();

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x-1, num_cores_y-1});

    auto shard_spec = a.shard_spec().value();

    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    auto& all_cores = shard_spec.grid;
    uint32_t num_cores = all_cores.num_cores();
    uint32_t num_tiles_per_shard = shard_spec.numel() / TILE_HW;

    tt::tt_metal::Shape output_shape = output.get_legacy_shape();

    tt::tt_metal::Buffer *dst_buffer = output.buffer();

    uint32_t src0_cb_index = tt::CB::c_in0;
    uint32_t num_input_tiles = num_tiles_per_shard;
    tt::tt_metal::CircularBufferConfig cb_src0_config = tt::tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
		.set_page_size(src0_cb_index, src0_single_tile_size).set_globally_allocated_address(*a.buffer());
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CB::c_out0; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_shard;
    tt::tt_metal::CircularBufferConfig cb_output_config = tt::tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
		.set_page_size(output_cb_index, dst_single_tile_size).set_globally_allocated_address(*output.buffer());
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_output_config);

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_cb_index,
    };

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
    };

    std::vector<uint32_t> compute_compile_time_args = {
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) output_cb_index,
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
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en=fp32_dest_acc_en, .compile_args = compute_compile_time_args}
    );

    uint32_t Wt = shard_spec.shape[1] / TILE_WIDTH;
    uint32_t Ht = a.get_legacy_shape()[-2] / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;
    uint32_t N = shard_spec.shape[0] / a.get_legacy_shape()[-2];
    uint32_t NHtWt = N * HtWt;

    auto bbox = all_cores.bounding_box();
    vector<CoreCoord> cores = grid_to_cores_with_noop(bbox.end_coord.x, bbox.end_coord.y, num_cores_x, num_cores_y, row_major);

    std::vector< std::vector<uint32_t> > unary_reader_args = { cores.size(), std::vector<uint32_t>(1) };
    std::vector< std::vector<uint32_t> > unary_compute_args = { cores.size(), std::vector<uint32_t>(5) };
    std::vector< std::vector<uint32_t> > unary_writer_args = { cores.size(), std::vector<uint32_t>(1) };
    std::fill(unary_reader_args.begin(), unary_reader_args.begin() + all_cores.num_cores(), std::vector<uint32_t>{NHtWt});
    std::fill(unary_compute_args.begin(), unary_compute_args.begin() + all_cores.num_cores(), std::vector<uint32_t>{NHtWt, HtWt, N, Ht, Wt});
    std::fill(unary_writer_args.begin(), unary_writer_args.begin() + all_cores.num_cores(), std::vector<uint32_t>{NHtWt});

    tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, cores, unary_reader_args);
    tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, cores, unary_compute_args);
    tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, cores, unary_writer_args);


    auto override_runtime_args_callback = [
            reader_kernel_id,
            compute_kernel_id,
            writer_kernel_id,
            cb_src0,
            cb_output,
            src0_single_tile_size,
            dst_single_tile_size,
            num_cores_x,
            num_cores_y
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        const auto& src_tensor = input_tensors.at(0);
        const auto& dst_tensor = output_tensors.at(0);

        const auto src_buffer = src_tensor.buffer();
        const auto dst_buffer = dst_tensor.buffer();

        bool src0_sharded = src_tensor.is_sharded();
        bool out_sharded = dst_tensor.is_sharded();

        auto shard_spec = src_tensor.shard_spec().value();

        uint32_t num_tiles_per_shard = shard_spec.numel() / TILE_HW;

        if (src0_sharded) {
            UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
            UpdateCircularBufferTotalSize(program, cb_src0, num_tiles_per_shard * src0_single_tile_size);
        }

        if (out_sharded) {
            UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
            UpdateCircularBufferTotalSize(program, cb_output, num_tiles_per_shard * dst_single_tile_size);
        }

        uint32_t Wt = shard_spec.shape[1] / TILE_WIDTH;
        uint32_t Ht = src_tensor.get_legacy_shape()[-2] / TILE_HEIGHT;
        uint32_t HtWt = Ht * Wt;
        uint32_t N = shard_spec.shape[0] / src_tensor.get_legacy_shape()[-2];
        uint32_t NHtWt = N * HtWt;

        const auto& all_cores = shard_spec.grid;
        bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

        auto bbox = all_cores.bounding_box();
        vector<CoreCoord> cores = grid_to_cores_with_noop(bbox.end_coord.x, bbox.end_coord.y, num_cores_x, num_cores_y, row_major);
        std::vector< std::vector<uint32_t> > unary_reader_args = { cores.size(), std::vector<uint32_t>(1) };
        std::vector< std::vector<uint32_t> > unary_compute_args = { cores.size(), std::vector<uint32_t>(5) };
        std::vector< std::vector<uint32_t> > unary_writer_args = { cores.size(), std::vector<uint32_t>(1) };
        std::fill(unary_reader_args.begin(), unary_reader_args.begin() + all_cores.num_cores(), std::vector<uint32_t>{NHtWt});
        std::fill(unary_compute_args.begin(), unary_compute_args.begin() + all_cores.num_cores(), std::vector<uint32_t>{NHtWt, HtWt, N, Ht, Wt});
        std::fill(unary_writer_args.begin(), unary_writer_args.begin() + all_cores.num_cores(), std::vector<uint32_t>{NHtWt});

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, cores, unary_reader_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, cores, unary_compute_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, cores, unary_writer_args);
    };

   return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}


operation::ProgramWithCallbacks transpose_wh_multi_core_sharded_rm(const Tensor &a, Tensor &output) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt::tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt::tt_metal::detail::TileSize(dst_cb_data_format);

    tt::tt_metal::Buffer *src0_buffer = a.buffer();

    const auto shape = a.get_legacy_shape();
    uint32_t W = a.shape()[3], H = a.shape()[2], C = a.shape()[1], N = a.shape()[0];
    uint32_t total_height = N * C * H;
    uint32_t stick_size_bytes = W * a.element_size();
    uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    auto shard_spec = a.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    uint32_t shard_width = shard_spec.shape[1];
    uint32_t num_hw_blocks_per_core = shard_height / H;

    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    tt::tt_metal::Device *device = a.device();

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;

    auto& all_cores = shard_spec.grid;
    uint32_t num_cores = shard_spec.num_cores();
    auto bbox = shard_spec.grid.bounding_box();
    CoreCoord grid_size = {bbox.end_coord.x + 1, bbox.end_coord.y+1};
    uint32_t num_cores_x = grid_size.x;
    uint32_t num_cores_y = grid_size.y;

    tt::log_debug("all_cores: {}", all_cores);
    tt::log_debug("num_cores: {}", num_cores);

    tt::tt_metal::Shape output_shape = output.get_legacy_shape();

    // sharded cb
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_src0_config = tt::tt_metal::CircularBufferConfig(shard_height * stick_size_bytes, {{src0_cb_index, src0_cb_data_format}})
        .set_page_size(src0_cb_index, stick_size_bytes).set_globally_allocated_address(*a.buffer());
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    // sharded cb
    uint32_t output_cb_index = tt::CB::c_out0; // output operands start at index 16
    tt::tt_metal::CircularBufferConfig cb_output_config = tt::tt_metal::CircularBufferConfig(shard_height * stick_size_bytes, {{output_cb_index, dst_cb_data_format}})
        .set_page_size(output_cb_index, stick_size_bytes).set_globally_allocated_address(*output.buffer());
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    // cb_in
    uint32_t in_cb_index = tt::CB::c_intermed0;
    uint32_t num_in_tiles = wt * 2; // double buffer
    tt::tt_metal::CircularBufferConfig cb_in_config = tt::tt_metal::CircularBufferConfig(num_in_tiles * src0_single_tile_size, {{in_cb_index, src0_cb_data_format}})
        .set_page_size(in_cb_index, src0_single_tile_size);
    auto cb_in = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_in_config);

    // tilize cb
    uint32_t im_cb_index = tt::CB::c_intermed1;
    uint32_t num_im_tiles = ht * wt;
    tt::tt_metal::CircularBufferConfig cb_im_config = tt::tt_metal::CircularBufferConfig(num_im_tiles * src0_single_tile_size, {{im_cb_index, src0_cb_data_format}})
        .set_page_size(im_cb_index, src0_single_tile_size);
    auto cb_im = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im_config);

    // untilize cb
    if (ht > 8) {
        uint32_t im2_cb_index = tt::CB::c_intermed2;
        uint32_t num_im2_tiles = ht;
        tt::tt_metal::CircularBufferConfig cb_im2_config = tt::tt_metal::CircularBufferConfig(num_im2_tiles * dst_single_tile_size, {{im2_cb_index, dst_cb_data_format}})
            .set_page_size(im2_cb_index, dst_single_tile_size);
        auto cb_im2 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im2_config);
    }

    // output_cb
    uint32_t out_cb_index = tt::CB::c_intermed3;
    uint32_t num_out_tiles = ht * 2; // double buffer
    tt::tt_metal::CircularBufferConfig cb_out_config = tt::tt_metal::CircularBufferConfig(num_out_tiles * dst_single_tile_size, {{out_cb_index, dst_cb_data_format}})
        .set_page_size(out_cb_index, dst_single_tile_size);
    auto cb_out = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) num_hw_blocks_per_core,
        (std::uint32_t) ht,
        (std::uint32_t) H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT,
        (std::uint32_t) H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT,
        (std::uint32_t) wt,
        (std::uint32_t) stick_size_bytes,
        (std::uint32_t) wt * a.element_size() * TILE_WIDTH,
    };
    reader_compile_time_args.push_back(H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT);
        reader_compile_time_args.push_back(H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT);

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/reader_unary_transpose_wh_sharded_rm.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) num_hw_blocks_per_core,
        (std::uint32_t) ht,
        (std::uint32_t) wt,
        (std::uint32_t) W > TILE_WIDTH ? TILE_WIDTH : W % TILE_WIDTH,
        (std::uint32_t) W % TILE_WIDTH == 0 ? TILE_WIDTH : W % TILE_WIDTH,
        (std::uint32_t) H * output.element_size(),
        (std::uint32_t) ht * output.element_size() * TILE_HEIGHT,
    };

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/writer_unary_transpose_wh_sharded_rm.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_compile_time_args = {
        (std::uint32_t) ht,
        (std::uint32_t) wt,
        (std::uint32_t) ht * wt,
        (std::uint32_t) num_hw_blocks_per_core,
    };

    std::map<string, string> compute_defines;
    compute_defines["SHARDED"] = "1";

    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_rm.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en=fp32_dest_acc_en, .compile_args = compute_compile_time_args, .defines = compute_defines}
    );

    auto override_runtime_args_callback = [
            reader_kernel_id,
            cb_src0,
            cb_output,
            src0_single_tile_size,
            dst_single_tile_size,
            num_cores_x,
            num_cores_y
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        const auto& src_tensor = input_tensors.at(0);
        const auto& dst_tensor = output_tensors.at(0);

        const auto src_buffer = src_tensor.buffer();
        const auto dst_buffer = dst_tensor.buffer();

        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
    };

   return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};

}

} // namespace ttnn::operations::reduction::detail
