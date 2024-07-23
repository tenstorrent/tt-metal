// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/transpose/transpose_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"


using uint32_t = std::uint32_t;
using namespace tt::constants;


namespace tt {

namespace tt_metal {

std::vector< std::array< std::vector<uint32_t>, 3 > > get_runtime_args(const Tensor &input_tensor,
                                                       Tensor &output_tensor,
                                                       uint32_t num_cores_total,
                                                       uint32_t num_cores,
                                                       uint32_t num_cores_y,
                                                       CoreRangeSet core_group_1,
                                                       uint32_t num_tiles_per_core_group_1,
                                                       CoreRangeSet core_group_2,
                                                       uint32_t num_tiles_per_core_group_2
                                                        )
{

    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();

    uint32_t W = input_shape[3], H = input_shape[2], NC = input_shape[1]*input_shape[0];
    uint32_t HW = H*W;

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = input_tensor.volume() / TILE_HW;

    auto HtWt = Ht * Wt;
    std::vector< std::array< std::vector<uint32_t>, 3 > > ret_val(num_cores_total);


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
        }
        uint32_t h = num_tiles_read % Ht;
        uint32_t w = num_tiles_read / Ht % Wt;

        std::vector<uint32_t> compute_runtime_args = {num_tiles_per_core};


        std::vector<uint32_t> reader_runtime_args = {
                input_tensor.buffer()->address(),
                num_tiles_per_core,
                round_down(num_tiles_read, HtWt) + h * Wt + w,
                h,
                w,
                Ht,
                Wt,
                HtWt
        };



        std::vector<uint32_t> writer_runtime_args = {
                output_tensor.buffer()->address(),
                num_tiles_per_core,
                num_tiles_read
        };
        num_tiles_read += num_tiles_per_core;
        ret_val[i] = {reader_runtime_args, compute_runtime_args, writer_runtime_args};
    }

    return ret_val;
}


std::vector< std::array< std::vector<uint32_t>, 3 > > get_runtime_args_rm(const Tensor &input_tensor,
                                                       Tensor &output_tensor,
                                                       uint32_t num_cores_total,
                                                       uint32_t num_cores,
                                                       uint32_t num_cores_y,
                                                       CoreRangeSet core_group_1,
                                                       uint32_t num_hw_blocks_per_core_group_1,
                                                       CoreRangeSet core_group_2,
                                                       uint32_t num_hw_blocks_per_core_group_2
                                                        )
{

    auto input_shape = input_tensor.shape();
    auto output_shape = output_tensor.shape();

    uint32_t W = input_shape[3], H = input_shape[2], NC = input_shape[1]*input_shape[0];
    uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    std::vector< std::array< std::vector<uint32_t>, 3 > > ret_val(num_cores_total);

    for(uint32_t i = 0, num_sticks_read = 0, num_sticks_write = 0; i < num_cores_total; i++) {
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
        // compute
        std::vector<uint32_t> compute_runtime_args = {num_hw_blocks_per_core};

        // reader
        std::vector<uint32_t> reader_runtime_args = {
                input_tensor.buffer()->address(),
                num_sticks_read,
                num_hw_blocks_per_core,
        };

        // writer
        std::vector<uint32_t> writer_runtime_args = {
                output_tensor.buffer()->address(),
                num_sticks_write,
                num_hw_blocks_per_core,
        };

        num_sticks_read += num_hw_blocks_per_core * H;
        num_sticks_write += num_hw_blocks_per_core * W;
        ret_val[i] = {reader_runtime_args, compute_runtime_args, writer_runtime_args};
    }

    return ret_val;
}


operation::ProgramWithCallbacks transpose_wh_multi_core(const Tensor &a, Tensor &output) {

    uint32_t num_tensor_tiles = a.volume() / TILE_HW;
    uint32_t W = a.shape()[3], H = a.shape()[2], C = a.shape()[1], N = a.shape()[0], NC = a.shape()[1] * a.shape()[0];
    bool row_major = a.get_layout() == Layout::ROW_MAJOR;

    uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    tt_metal::Program program = tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x*num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x-1, num_cores_y-1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, row_major ? NC : num_tensor_tiles);

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = row_major ? wt * 2 : 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
		.set_page_size(src0_cb_index, src0_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = row_major ? ht * 2 : 2;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
		.set_page_size(output_cb_index, dst_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, total_cores, cb_output_config);

    if (row_major) {
        // tilize cb
        uint32_t im_cb_index = 24;
        uint32_t num_im_tiles = ht * wt;
        tt_metal::CircularBufferConfig cb_im_config = tt_metal::CircularBufferConfig(num_im_tiles * src0_single_tile_size, {{im_cb_index, src0_cb_data_format}})
            .set_page_size(im_cb_index, src0_single_tile_size);
        auto cb_im = tt_metal::CreateCircularBuffer(program, total_cores, cb_im_config);

        // untilize cb
        uint32_t im2_cb_index = 25;
        uint32_t num_im2_tiles = ht * wt;
        tt_metal::CircularBufferConfig cb_im2_config = tt_metal::CircularBufferConfig(num_im2_tiles * dst_single_tile_size, {{im2_cb_index, dst_cb_data_format}})
            .set_page_size(im2_cb_index, dst_single_tile_size);
        auto cb_im2 = tt_metal::CreateCircularBuffer(program, total_cores, cb_im2_config);
    }

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
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

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
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

    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        row_major ?
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/reader_unary_transpose_wh_interleaved_start_id_rm.cpp" :
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/reader_unary_transpose_wh_interleaved_start_id.cpp",
        total_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        row_major ?
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/writer_unary_tranpose_wh_interleaved_start_id_rm.cpp" :
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        total_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_kernel_args = {};
    if (row_major) {
        compute_kernel_args.push_back(ht);
        compute_kernel_args.push_back(wt);
        compute_kernel_args.push_back(ht * wt);
    }
    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        row_major ?
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_rm.cpp" :
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh.cpp",
        total_cores,
        tt_metal::ComputeConfig{.fp32_dest_acc_en=fp32_dest_acc_en, .compile_args = compute_kernel_args,}
    );

    auto all_runtime_args = row_major ? get_runtime_args_rm(a, output, num_cores_total, num_cores, num_cores_y,
                                            core_group_1, num_tiles_per_core_group_1, core_group_2,
                                            num_tiles_per_core_group_2) :
                                        get_runtime_args(a, output, num_cores_total, num_cores, num_cores_y,
                                            core_group_1, num_tiles_per_core_group_1, core_group_2,
                                            num_tiles_per_core_group_2);

    for(uint32_t i = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            all_runtime_args[i][0]

        );

        tt_metal::SetRuntimeArgs(
            program,
            compute_kernel_id,
            core,
            all_runtime_args[i][1]

        );

        tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            all_runtime_args[i][2]
        );
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
        auto all_runtime_args = row_major ? get_runtime_args_rm(src_tensor, dst_tensor, num_cores_total, num_cores, num_cores_y,
                                                core_group_1, num_tiles_per_core_group_1, core_group_2,
                                                num_tiles_per_core_group_2) :
                                            get_runtime_args(src_tensor, dst_tensor, num_cores_total, num_cores, num_cores_y,
                                                core_group_1, num_tiles_per_core_group_1, core_group_2,
                                                num_tiles_per_core_group_2);

        for(uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args[i][0]);
            }

            {
                SetRuntimeArgs(program, compute_kernel_id, core, all_runtime_args[i][1]);
            }

            {
                SetRuntimeArgs(program, writer_kernel_id, core, all_runtime_args[i][2]);
            }

        }
    };

   return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

operation::ProgramWithCallbacks transpose_wh_multi_core_sharded(const Tensor &a, Tensor &output) {


    tt_metal::Program program = tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();

    int32_t num_tiles = a.volume()/TILE_HW;

    tt_metal::Device *device = a.device();

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

    Shape output_shape = output.get_legacy_shape();

    tt_metal::Buffer *dst_buffer = output.buffer();

    uint32_t src0_cb_index = CB::c_in0;
    uint32_t num_input_tiles = num_tiles_per_shard;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
		.set_page_size(src0_cb_index, src0_single_tile_size).set_globally_allocated_address(*a.buffer());
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = CB::c_out0; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_shard;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
		.set_page_size(output_cb_index, dst_single_tile_size).set_globally_allocated_address(*output.buffer());;
    auto cb_output = tt_metal::CreateCircularBuffer(program, total_cores, cb_output_config);

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

    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        total_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded.cpp",
        total_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));


    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_sharded.cpp",
        total_cores,
        tt_metal::ComputeConfig{.fp32_dest_acc_en=fp32_dest_acc_en, .compile_args = compute_compile_time_args}
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

    tt_metal::SetRuntimeArgs(program, reader_kernel_id, cores, unary_reader_args);
    tt_metal::SetRuntimeArgs(program, compute_kernel_id, cores, unary_compute_args);
    tt_metal::SetRuntimeArgs(program, writer_kernel_id, cores, unary_writer_args);


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

        tt_metal::SetRuntimeArgs(program, reader_kernel_id, cores, unary_reader_args);
        tt_metal::SetRuntimeArgs(program, compute_kernel_id, cores, unary_compute_args);
        tt_metal::SetRuntimeArgs(program, writer_kernel_id, cores, unary_writer_args);
    };

   return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
