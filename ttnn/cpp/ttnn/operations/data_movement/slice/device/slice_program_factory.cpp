// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "optional"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

#include "slice_op.hpp"
using namespace tt::constants;


namespace ttnn::operations::data_movement::detail {

inline std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_slice_runtime_args_rm(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    uint32_t num_cores_total,
    uint32_t num_cores,
    uint32_t num_cores_y,
    CoreRangeSet core_group_1,
    CoreRangeSet core_group_2,
    uint32_t num_sticks_per_core_group_1,
    uint32_t num_sticks_per_core_group_2,
    uint32_t max_read_size) {
    tt::tt_metal::Device* device = input_tensor.device();

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();

    uint32_t padded_row_size_bytes = input_shape[-1] * input_tensor.element_size();
    uint32_t unpadded_row_size_bytes = output_shape[-1] * input_tensor.element_size();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_unpadded_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_padded_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_unpadded_sticks_per_dim[0] = 1;
    num_padded_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_sticks_per_dim[i] = num_unpadded_dim;
        num_padded_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    uint32_t unpadded_row_size_bytes_offset = tt::round_up(unpadded_row_size_bytes, TILE_WIDTH / 2);

    vector<uint32_t> common_reader_kernel_args = {
        input_tensor.buffer()->address() + output_tensor_start[-1] * output_tensor.element_size(),
        padded_row_size_bytes,
        unpadded_row_size_bytes,
        unpadded_row_size_bytes_offset,
        num_dims,
        0,
        0,
        0,
        0};
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_unpadded_sticks_per_dim.begin(), num_unpadded_sticks_per_dim.end());
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_padded_sticks_per_dim.begin(), num_padded_sticks_per_dim.end());

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_total);

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(input_tensor, ttnn::Shape(output_tensor_start));
    for (uint32_t i = 0, num_sticks_written = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_sticks_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        } else {
            // no-op
            num_sticks_per_core = 0;
        }

        // issue more reads before calling barrier
        uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
        if (num_sticks_per_core != 0) {
            auto num_sticks_per_core_pad32 = num_sticks_per_core + (32 - num_sticks_per_core % 32) % 32;
            num_sticks_per_core_read = tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core_pad32, unpadded_row_size_bytes_offset, max_read_size);
            num_read_per_barrier = num_sticks_per_core_pad32 / num_sticks_per_core_read;
        }

        id_per_dim[0] = num_sticks_written % num_unpadded_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_written / num_unpadded_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;

        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_unpadded_sticks_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }
        vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        //
        uint32_t addr_offset = 5;  // input buffer addr, padded_row_size_bytes, unpadded_row_size_bytes, num_dims
        reader_kernel_args[addr_offset++] = start_id;
        reader_kernel_args[addr_offset++] = num_sticks_per_core;
        reader_kernel_args[addr_offset++] = num_sticks_per_core_read;
        reader_kernel_args[addr_offset] = num_read_per_barrier;
        reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        vector<uint32_t> writer_kernel_args = {
            output_buffer->address(), unpadded_row_size_bytes, unpadded_row_size_bytes_offset, num_sticks_per_core, num_sticks_per_core_read, num_read_per_barrier, num_sticks_written, 0};
        num_sticks_written += num_sticks_per_core;
        ret_val[i] = {reader_kernel_args, writer_kernel_args};
    }

    return ret_val;
}

operation::ProgramWithCallbacks slice_rm_multi_core(
    const Tensor& a, Tensor& output, const tt::tt_metal::Shape& output_tensor_start, const tt::tt_metal::Shape& output_tensor_end) {
    const tt::tt_metal::Shape output_shape = output.get_legacy_shape();

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::Device* device = a.device();

    uint32_t num_unpadded_sticks = output.volume() / output.get_legacy_shape()[-1];

    auto compute_with_storage_grid_size = DeviceComputeWithStorageGridSize(device);
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    uint32_t padded_row_size_bytes = a.get_legacy_shape()[-1] * a.element_size();
    uint32_t unpadded_row_size_bytes = output_shape[-1] * a.element_size();

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args_vec = {(std::uint32_t)src0_is_dram};
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t src_stick_size = padded_row_size_bytes;
    uint32_t dst_stick_size = unpadded_row_size_bytes;

    uint32_t src0_cb_index = 0;
    uint32_t max_read_size = 4096;
    uint32_t cb_page_size = dst_is_dram ? tt::round_up(unpadded_row_size_bytes, TILE_WIDTH) : tt::round_up(unpadded_row_size_bytes, TILE_WIDTH / 2);
    uint32_t num_input_pages = num_sticks_per_core_group_1 > num_sticks_per_core_group_2 ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;
    uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
    if (num_input_pages != 0) {
        auto num_sticks_per_core_pad32 = num_input_pages + (32 - num_input_pages % 32) % 32;
        num_sticks_per_core_read = tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core_pad32, cb_page_size, max_read_size);
        num_read_per_barrier = num_sticks_per_core_pad32 / num_sticks_per_core_read;
    }
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_read_per_barrier * 2 * cb_page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, cb_page_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);


    std::vector<uint32_t> writer_compile_time_args_vec = {(std::uint32_t)src0_cb_index, (std::uint32_t)dst_is_dram};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args_vec));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/slice_writer_unary_stick_layout_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_vec));

    auto all_runtime_args = get_slice_runtime_args_rm(
        a,
        output,
        output_tensor_start,
        num_cores_total,
        num_cores,
        num_cores_y,
        core_group_1,
        core_group_2,
        num_sticks_per_core_group_1,
        num_sticks_per_core_group_2,
        max_read_size);

    for (uint32_t i = 0, num_sticks_written = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);

        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
    }

    auto override_runtime_args_callback =
        [unary_reader_kernel_id, unary_writer_kernel_id, compute_with_storage_grid_size, max_read_size](
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
            uint32_t num_unpadded_sticks = dst_tensor.volume() / dst_tensor.get_legacy_shape()[-1];
            auto
                [num_cores,
                 all_cores,
                 core_group_1,
                 core_group_2,
                 num_sticks_per_core_group_1,
                 num_sticks_per_core_group_2] =
                    tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);

            const auto tensor_start = static_cast<const ttnn::operations::data_movement::SliceDeviceOperation *>(operation)->slice_start;
            auto all_runtime_args = get_slice_runtime_args_rm(
                src_tensor,
                dst_tensor,
                tensor_start,
                num_cores_total,
                num_cores,
                num_cores_y,
                core_group_1,
                core_group_2,
                num_sticks_per_core_group_1,
                num_sticks_per_core_group_2,
                max_read_size);

            for (uint32_t i = 0, num_tiles_written = 0; i < num_cores_total; i++) {
                CoreCoord core = {i / num_cores_y, i % num_cores_y};

                { SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first); }

                { SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second); }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

operation::ProgramWithCallbacks slice_rm_strided_single_core(const Tensor& a, Tensor& output, const tt::tt_metal::Shape& output_tensor_start, const tt::tt_metal::Shape& output_tensor_end, const tt::tt_metal::Shape& step) {
    // TODO: multi core implementation - work division is not trivial as we need to determine the N/C/H/W start and end points for each split, and base that off stride
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    const tt::tt_metal::Shape output_shape = output.get_legacy_shape();
    const tt::tt_metal::Shape input_shape = a.get_legacy_shape();
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    uint32_t src_is_dram = a.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t dst_is_dram = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;


    uint32_t page_size_output = dst_is_dram ? tt::round_up(output_shape[-1] * a.element_size(), TILE_WIDTH) : tt::round_up(output_shape[-1] * a.element_size(), TILE_WIDTH / 2);
    uint32_t page_size_input = src_is_dram ? tt::round_up(input_shape[-1] * a.element_size(), TILE_WIDTH) : tt::round_up(input_shape[-1] * a.element_size(), TILE_WIDTH / 2);


    tt::tt_metal::CircularBufferConfig cb_src0_config =
    tt::tt_metal::CircularBufferConfig(1*page_size_input, {{tt::CB::c_in0, cb_data_format}})
        .set_page_size(tt::CB::c_in0, page_size_input);

    tt::tt_metal::CircularBufferConfig cb_dst0_config =
    tt::tt_metal::CircularBufferConfig(2*page_size_output, {{tt::CB::c_intermed0, cb_data_format}})
        .set_page_size(tt::CB::c_intermed0, page_size_output);

    CoreRange core({0, 0}, {0, 0});
    auto cb_input_tensor = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);
    auto cb_output_tensor = tt::tt_metal::CreateCircularBuffer(program, core, cb_dst0_config);


    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/strided_slice_reader_rm_interleaved.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(
            {
                src_is_dram,
                input_shape[3],
                input_shape[2],
                input_shape[1],
                input_shape[0],
                step[3],
                step[2],
                step[1],
                step[0],
                (uint32_t) page_size_input,
            }

        ));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(
            {
                dst_is_dram,
                (uint32_t) page_size_output,
            }
        ));

    tt::tt_metal::SetRuntimeArgs(
    program, unary_reader_kernel_id, core,
    {
        a.buffer()->address(),
        output_tensor_start[3],
        output_tensor_start[2],
        output_tensor_start[1],
        output_tensor_start[0],
        output_tensor_end[3] + 1,
        output_tensor_end[2] + 1,
        output_tensor_end[1] + 1,
        output_tensor_end[0] + 1,

    });

    tt::tt_metal::SetRuntimeArgs(
    program, unary_writer_kernel_id, core,
    {
        output.buffer()->address(),
        output_shape[0]*output_shape[1]*output_shape[2],
    });

    auto override_address_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto input_buffer = input_buffers.at(0);
        auto output_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            reader_runtime_args[0] = input_buffer->address();

            auto &writer_runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            writer_runtime_args[0] = output_buffer->address();
        }

    };

    return {.program=std::move(program), .override_addresses_callback=override_address_callback};

}

inline std::vector<std::vector<uint32_t>> group_contiguous_values(std::vector<uint32_t>& values) {
    std::vector<std::vector<uint32_t>> chunks;
    if (values.empty()) return chunks;

    // Initialize the first chunk
    std::vector<uint32_t> current_chunk;
    current_chunk.push_back(values[0]);

    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] == values[i - 1] + 1) {
            current_chunk.push_back(values[i]);
        } else {
            chunks.push_back(current_chunk);
            current_chunk.clear();
            current_chunk.push_back(values[i]);
        }
    }
    // Add the last chunk
    chunks.push_back(current_chunk);
    return chunks;
}

inline std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_slice_runtime_args_rm_sharded(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    uint32_t num_cores_unpadded,
    bool row_major,
    uint32_t num_cores_x_unpadded,
    uint32_t num_cores_y_unpadded,
    uint32_t shard_height_unpadded,
    uint32_t shard_height_padded,
    uint32_t num_cores_x_padded,
    uint32_t num_cores_y_padded) {

    tt::tt_metal::Device* device = input_tensor.device();

    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();

    uint32_t padded_row_size_bytes = input_shape[-1] * input_tensor.element_size();
    uint32_t unpadded_row_size_bytes = output_shape[-1] * input_tensor.element_size();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_unpadded_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_padded_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_unpadded_sticks_per_dim[0] = 1;
    num_padded_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_sticks_per_dim[i] = num_unpadded_dim;
        num_padded_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    uint32_t unpadded_row_size_bytes_offset = tt::round_up(unpadded_row_size_bytes, TILE_WIDTH / 2);

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_unpadded);

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(input_tensor, ttnn::Shape(output_tensor_start));
    for (uint32_t i = 0, num_sticks_written = 0; i < num_cores_unpadded; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_unpadded, i / num_cores_x_unpadded};
        } else {
            core = {i / num_cores_y_unpadded, i % num_cores_y_unpadded};
        }
        uint32_t num_sticks_per_core_unpadded = shard_height_unpadded;
        uint32_t num_sticks_per_core_padded = shard_height_padded;

        // figure out the start read stick id for each core, and the start id for each dim
        id_per_dim[0] = num_sticks_written % num_unpadded_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_written / num_unpadded_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;

        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_unpadded_sticks_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }

        num_sticks_written += num_sticks_per_core_unpadded;

        // stores all sticks id for a core
        std::vector<uint32_t> stick_ids_per_core;
        uint32_t src_stick_id = start_id;
        for (uint32_t i = 0; i < num_sticks_per_core_unpadded; ++i) {
            stick_ids_per_core.push_back(src_stick_id);
            src_stick_id++;
            for(uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == num_unpadded_sticks_per_dim[j]) {
                    id_per_dim[j] = 0;
                    src_stick_id += num_padded_sticks_per_dim[j];
                } else {
                    break;
                }
            }
        }

        // figure out the stick id in a shard, and the core id for the stick.
        std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> core_stick_map;
        for (uint32_t i = 0; i < num_sticks_per_core_unpadded; ++i) {
            uint32_t stick_id = stick_ids_per_core[i];
            uint32_t shard_id = stick_id / num_sticks_per_core_padded;
            uint32_t stick_id_in_shard = stick_id - (shard_id * num_sticks_per_core_padded);

            uint32_t shard_grid_inner_dim = row_major ? num_cores_x_padded : num_cores_y_padded;
            uint32_t shard_grid_outer_dim_id = shard_id / shard_grid_inner_dim;
            uint32_t shard_grid_inner_dim_id = shard_id - (shard_grid_outer_dim_id * shard_grid_inner_dim);

            uint32_t worker_y_logical = row_major ? shard_grid_outer_dim_id : shard_grid_inner_dim_id;
            uint32_t worker_x_logical = row_major ? shard_grid_inner_dim_id : shard_grid_outer_dim_id;

            if (worker_x_logical < num_cores_x_padded and worker_y_logical < num_cores_y_padded) {
                auto core_physical = DeviceWorkerCoreFromLogicalCore(device, CoreCoord{worker_x_logical, worker_y_logical});
                // save stick id in a shard, and core coord into a map
                std::pair<uint32_t, uint32_t> xy_pair = row_major ? std::make_pair(core_physical.y, core_physical.x)
                                                                    : std::make_pair(core_physical.x, core_physical.y);
                core_stick_map[xy_pair].push_back(stick_id_in_shard);
            }
        }

        // reader rt args
        vector<uint32_t> reader_kernel_args;
        reader_kernel_args.push_back(core_stick_map.size()); // num_cores

        tt::log_debug("num_cores: {}", core_stick_map.size());

        for (auto core_stick_pair : core_stick_map) {
            auto xy_pair = core_stick_pair.first;
            if (row_major) {
                reader_kernel_args.push_back(xy_pair.second); // noc x
                reader_kernel_args.push_back(xy_pair.first); // noc y
            } else {
                reader_kernel_args.push_back(xy_pair.first); // noc x
                reader_kernel_args.push_back(xy_pair.second); // noc y
            }

            tt::log_debug("xy_pair.first: {}", xy_pair.first);
            tt::log_debug("xy_pair.second: {}", xy_pair.second);
        }

        // coalesce the sticks into chunks
        vector<std::vector<std::vector<uint32_t>>> stick_chunks_per_core;
        for (auto core_stick_pair : core_stick_map) {
            auto stick_chunks = group_contiguous_values(core_stick_pair.second);
            stick_chunks_per_core.push_back(stick_chunks);

            reader_kernel_args.push_back(stick_chunks.size()); // num_chunks for current core
            tt::log_debug("chunk_size: {}", stick_chunks.size());
        }
        for (auto stick_chunks : stick_chunks_per_core) {
            for (auto chunk : stick_chunks) {
                reader_kernel_args.push_back(chunk[0]); // start id of a chunk
                tt::log_debug("chunk_start_id: {}", chunk[0]);

                reader_kernel_args.push_back(chunk.size()); // length of a chunk
                tt::log_debug("chunk_length: {}", chunk.size());
            }
        }

        vector<uint32_t> writer_kernel_args;
        ret_val[i] = {reader_kernel_args, writer_kernel_args};
    }

    return ret_val;
}

operation::ProgramWithCallbacks slice_rm_multi_core_sharded(
    const Tensor& a, Tensor& output, const tt::tt_metal::Shape& output_tensor_start, const tt::tt_metal::Shape& output_tensor_end) {
    const tt::tt_metal::Shape output_shape = output.get_legacy_shape();

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::Device* device = a.device();

    uint32_t num_padded_sticks = a.volume() / a.get_legacy_shape()[-1];
    uint32_t num_unpadded_sticks = output.volume() / output.get_legacy_shape()[-1];

    // stick sizes
    uint32_t W_padded = a.shape()[-1];
    uint32_t W_unpadded = output.shape()[-1];
    auto stick_size_padded = W_padded * a.element_size();
    auto stick_size_unpadded = W_unpadded * output.element_size();

    // input shard spec
    auto shard_spec_padded = a.shard_spec().value();
    uint32_t shard_height_padded = shard_spec_padded.shape[0];
    uint32_t shard_width_padded = shard_spec_padded.shape[1];

    auto& all_cores_padded = shard_spec_padded.grid;
    uint32_t num_cores_padded = shard_spec_padded.num_cores();
    auto bbox_padded = shard_spec_padded.grid.bounding_box();
    CoreCoord grid_size_padded = {bbox_padded.end_coord.x + 1, bbox_padded.end_coord.y+1};
    uint32_t num_cores_x_padded = grid_size_padded.x;
    uint32_t num_cores_y_padded = grid_size_padded.y;

    tt::log_debug("num_padded_sticks: {}", num_padded_sticks);
    tt::log_debug("shard_height_padded: {}", shard_height_padded);
    tt::log_debug("all_cores_padded: {}", all_cores_padded);
    tt::log_debug("num_cores_padded: {}", num_cores_padded);

    // output shard spec
    auto shard_spec_unpadded = output.shard_spec().value();
    uint32_t shard_height_unpadded = shard_spec_unpadded.shape[0];
    uint32_t shard_width_unpadded = shard_spec_unpadded.shape[1];
    bool row_major = shard_spec_unpadded.orientation == ShardOrientation::ROW_MAJOR;

    auto& all_cores_unpadded = shard_spec_unpadded.grid;
    uint32_t num_cores_unpadded = shard_spec_unpadded.num_cores();
    auto bbox_unpadded = shard_spec_unpadded.grid.bounding_box();
    CoreCoord grid_size_unpadded = {bbox_unpadded.end_coord.x + 1, bbox_unpadded.end_coord.y+1};
    uint32_t num_cores_x_unpadded = grid_size_unpadded.x;
    uint32_t num_cores_y_unpadded = grid_size_unpadded.y;

    tt::log_debug("num_unpadded_sticks: {}", num_unpadded_sticks);
    tt::log_debug("shard_height_unpadded: {}", shard_height_unpadded);
    tt::log_debug("all_cores_unpadded: {}", all_cores_unpadded);
    tt::log_debug("num_cores_unpadded: {}", num_cores_unpadded);

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    auto compute_with_storage_grid_size = DeviceComputeWithStorageGridSize(device);
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    uint32_t src0_cb_index = 0;
    tt::tt_metal::CircularBufferConfig cb_src0_config = tt::tt_metal::CircularBufferConfig(shard_height_padded * stick_size_padded, {{src0_cb_index, cb_data_format}})
        .set_page_size(src0_cb_index, stick_size_padded).set_globally_allocated_address(*a.buffer());
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CB::c_out0; // output operands start at index 16
    tt::tt_metal::CircularBufferConfig cb_output_config = tt::tt_metal::CircularBufferConfig(shard_height_unpadded * stick_size_unpadded, {{output_cb_index, dst_cb_data_format}})
        .set_page_size(output_cb_index, stick_size_unpadded).set_globally_allocated_address(*output.buffer());
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_output_config);

    std::vector<uint32_t> reader_ct_args = {(std::uint32_t) stick_size_padded,
                                            (std::uint32_t) stick_size_unpadded,
                                            (std::uint32_t) shard_height_unpadded};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp",
        all_cores_unpadded,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    auto all_runtime_args = get_slice_runtime_args_rm_sharded(
        a,
        output,
        output_tensor_start,
        num_cores_unpadded,
        row_major,
        num_cores_x_unpadded,
        num_cores_y_unpadded,
        shard_height_unpadded,
        shard_height_padded,
        num_cores_x_padded,
        num_cores_y_padded
        );

    for (uint32_t i = 0; i < num_cores_unpadded; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_unpadded, i / num_cores_x_unpadded};
        } else {
            core = {i / num_cores_y_unpadded, i % num_cores_y_unpadded};
        }
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
    }

    auto override_runtime_args_callback = [
            cb_src0,
            cb_output
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_buffer_a = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer_a);
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

template <bool initialize_args>
inline __attribute__((always_inline)) void set_slice_runtime_args_tile(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const uint32_t& num_cores_total,
    const uint32_t& num_cores,
    const std::vector<CoreCoord>& cores,
    const uint32_t& num_cores_group_1,
    const uint32_t& num_cores_group_2,
    const uint32_t& num_tiles_per_core_group_1,
    const uint32_t& num_tiles_per_core_group_2,
    const Program& program,
    const tt::tt_metal::KernelHandle& unary_reader_kernel_id,
    const tt::tt_metal::KernelHandle& unary_writer_kernel_id,
    std::vector<uint32_t>& accumulated_total_per_dim) {
    const auto input_buffer = input_tensor.buffer();
    const auto output_buffer = output_tensor.buffer();
    const auto& input_shape = input_tensor.get_legacy_shape();
    const auto& output_shape = output_tensor.get_legacy_shape();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    const auto set_common_reader_args = [&](
        uint32_t* reader_common_args,
        uint32_t* num_unpadded_tiles_per_dim,
        uint32_t* num_padded_tiles_per_dim) __attribute__((always_inline)) {
        reader_common_args[0] = input_buffer->address();
        num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
        num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
        num_padded_tiles_per_dim[0] = num_padded_Xt;
        num_padded_tiles_per_dim[1] = num_padded_Yt;
        accumulated_total_per_dim[0] = num_total_Xt;
        accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;
        for (int32_t i = 2; i < num_dims; ++i) {
            uint32_t num_unpadded_dim = output_shape[-(i + 1)];
            uint32_t num_total_dim = input_shape[-(i + 1)];
            uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
            num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
            num_padded_tiles_per_dim[i] = num_padded_dim;
            accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
        }
    };

    const auto set_reader_rt_args = [&](
        uint32_t* reader_rt_args,
        const uint32_t* num_unpadded_tiles_per_dim,
        const uint32_t* num_padded_tiles_per_dim,
        const uint32_t& num_tiles_per_core,
        const uint32_t& start_offset,
        const uint32_t& num_tiles_written) __attribute__((always_inline)) {
        reader_rt_args[2] = num_tiles_written % num_unpadded_tiles_per_dim[0];
        uint32_t unpadded_written = num_tiles_written / num_unpadded_tiles_per_dim[0];
        uint32_t start_id = reader_rt_args[2] + start_offset;
        for (uint32_t j = 1; j < num_dims; ++j) {
            reader_rt_args[2 + j] = unpadded_written % num_unpadded_tiles_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_tiles_per_dim[j];
            start_id += reader_rt_args[2 + j] * accumulated_total_per_dim[j - 1];
        }
        reader_rt_args[0] = start_id;
        reader_rt_args[1] = num_tiles_per_core;
    };

    if constexpr (initialize_args) {
        std::vector<uint32_t> reader_common_args(1 + num_dims * 2);
        uint32_t* num_unpadded_tiles_per_dim = reader_common_args.data() + 1;
        uint32_t* num_padded_tiles_per_dim = num_unpadded_tiles_per_dim + num_dims;
        set_common_reader_args(reader_common_args.data(), num_unpadded_tiles_per_dim, num_padded_tiles_per_dim);
        SetCommonRuntimeArgs(program, unary_reader_kernel_id, reader_common_args);
    }
    auto& reader_common_args = GetCommonRuntimeArgs(program, unary_reader_kernel_id);
    uint32_t* num_unpadded_tiles_per_dim = reader_common_args.data() + 1;
    uint32_t* num_padded_tiles_per_dim = num_unpadded_tiles_per_dim + num_dims;
    if constexpr (!initialize_args) {
        set_common_reader_args(reader_common_args.data(), num_unpadded_tiles_per_dim, num_padded_tiles_per_dim);
    }

    uint32_t start_offset = ttnn::operations::data_movement::get_tiled_start_offset(input_tensor, ttnn::Shape(output_tensor_start));

    auto& reader_kernel_args_by_core = GetRuntimeArgs(program, unary_reader_kernel_id);
    auto& writer_kernel_args_by_core = GetRuntimeArgs(program, unary_writer_kernel_id);
    const uint32_t num_used_cores = num_cores_group_1 + num_cores_group_2;
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores_total; ++i) {
        const CoreCoord& core = cores[i];
        uint32_t num_tiles_per_core;
        if (i < num_cores_group_1) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (i < num_used_cores) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op
            if constexpr (initialize_args) {
                std::vector<uint32_t> reader_kernel_args(2 + num_dims, 0);
                std::vector<uint32_t> writer_kernel_args(3, 0);
                tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);
                tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);
            } else {
                auto& reader_kernel_args = reader_kernel_args_by_core[core.x][core.y];
                reader_kernel_args[1] = 0;
                auto& writer_kernel_args = writer_kernel_args_by_core[core.x][core.y];
                writer_kernel_args[1] = 0;
            }
            continue;
        }

        if constexpr (initialize_args) {
            std::vector<uint32_t> reader_kernel_args(2 + num_dims);
            set_reader_rt_args(
                reader_kernel_args.data(),
                num_unpadded_tiles_per_dim,
                num_padded_tiles_per_dim,
                num_tiles_per_core,
                start_offset,
                num_tiles_written);
            SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);
        } else {
            auto& reader_kernel_args = reader_kernel_args_by_core[core.x][core.y];
            set_reader_rt_args(
                reader_kernel_args.data(),
                num_unpadded_tiles_per_dim,
                num_padded_tiles_per_dim,
                num_tiles_per_core,
                start_offset,
                num_tiles_written);
        }

        if constexpr (initialize_args) {
            vector<uint32_t> writer_kernel_args = {output_buffer->address(), num_tiles_per_core, num_tiles_written};
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);
        } else {
            auto& writer_kernel_args = writer_kernel_args_by_core[core.x][core.y];
            writer_kernel_args[0] = output_buffer->address();
            writer_kernel_args[1] = num_tiles_per_core;
            writer_kernel_args[2] = num_tiles_written;
        }
        num_tiles_written += num_tiles_per_core;
    }
}

operation::ProgramWithCallbacks slice_tile_multi_core(
    const Tensor& a, Tensor& output, const tt::tt_metal::Shape& output_tensor_start, const tt::tt_metal::Shape& output_tensor_end) {
    const tt::tt_metal::Shape output_shape = output.get_legacy_shape();

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::Device* device = a.device();

    uint32_t num_unpadded_tiles = output.volume() / TILE_HW;

    auto compute_with_storage_grid_size = DeviceComputeWithStorageGridSize(device);
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    std::uint32_t num_dims = static_cast<std::uint32_t>(a.get_legacy_shape().rank());

    // Reader compile-time args
    // Data is 32 byte aligned
    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(src0_cb_index),
        static_cast<uint32_t>(num_dims),
        static_cast<uint32_t>(src0_is_dram),
    };
    std::vector<uint32_t> writer_compile_time_args = {
        static_cast<uint32_t>(src0_cb_index), static_cast<uint32_t>(dst_is_dram)};

    // Tilized reader
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    const auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, false);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    set_slice_runtime_args_tile<true>(
        a,
        output,
        output_tensor_start,
        num_cores_total,
        num_cores,
        cores,
        core_group_1.num_cores(),
        core_group_2.num_cores(),
        num_tiles_per_core_group_1,
        num_tiles_per_core_group_2,
        program,
        unary_reader_kernel_id,
        unary_writer_kernel_id,
        accumulated_total_per_dim);

    auto override_runtime_args_callback = [unary_reader_kernel_id,
                                           unary_writer_kernel_id,
                                           compute_with_storage_grid_size,
                                           cores,
                                           accumulated_total_per_dim](
                                              const void* operation,
                                              const Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) mutable {
        const Tensor& src_tensor = input_tensors[0];
        const Tensor& dst_tensor = output_tensors[0];
        uint32_t num_unpadded_tiles = dst_tensor.volume() / TILE_HW;

        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        uint32_t num_cores_total = cores.size();

        auto
            [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
                tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

        const auto& tensor_start = static_cast<const ttnn::operations::data_movement::SliceDeviceOperation *>(operation)->slice_start;
        set_slice_runtime_args_tile<false>(
            src_tensor,
            dst_tensor,
            tensor_start,
            num_cores_total,
            num_cores,
            cores,
            core_group_1.num_cores(),
            core_group_2.num_cores(),
            num_tiles_per_core_group_1,
            num_tiles_per_core_group_2,
            program,
            unary_reader_kernel_id,
            unary_writer_kernel_id,
            accumulated_total_per_dim);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

operation::ProgramWithCallbacks slice_multi_core(
    const Tensor& a, Tensor& output, const tt::tt_metal::Shape& output_tensor_start, const tt::tt_metal::Shape& output_tensor_end, const std::optional<tt::tt_metal::Shape>& step) {
    const std::optional<tt::tt_metal::Shape> step_modified = step.has_value() ? (std::all_of(step->begin(), step->end(), [](int32_t i) { return i == 1; }) ? std::nullopt : step ): std::nullopt;
    switch (a.get_layout()) {
        case Layout::ROW_MAJOR: return a.is_sharded() ?
            slice_rm_multi_core_sharded(a, output, output_tensor_start, output_tensor_end) :
            (step_modified.has_value() ?
                slice_rm_strided_single_core(a, output, output_tensor_start, output_tensor_end, step_modified.value()) :
                slice_rm_multi_core(a, output, output_tensor_start, output_tensor_end));
        case Layout::TILE: return slice_tile_multi_core(a, output, output_tensor_start, output_tensor_end);
        default: TT_ASSERT(false, "Unsupported Layout");
    }
    return {};
}

}  // namespace ttnn::operations::data_movement::detail
