// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_program_factory.hpp"

#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks untilize_with_unpadding_single_core(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en) {
    const auto& input_shape = a.get_legacy_shape();
    const auto& output_shape = output.get_legacy_shape();

    tt::tt_metal::Program program{};

    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    tt::log_debug("untilize_with_unpadding_single_core");
    tt::log_debug("input_cb_data_format: {}", input_cb_data_format);
    tt::log_debug("output_cb_data_format: {}", output_cb_data_format);

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    int32_t num_tiles = a.volume() / TILE_HW;

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::Device* device = a.device();

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    auto input_w = input_shape.rank() >= 4 ? input_shape[-4] : 1;
    auto input_z = input_shape.rank() >= 3 ? input_shape[-3] : 1;
    auto input_y = input_shape.rank() >= 2 ? input_shape[-2] : 1;
    auto input_x = input_shape[-1];

    auto output_w = output_shape.rank() >= 4 ? output_shape[-4] : 1;
    auto output_z = output_shape.rank() >= 3 ? output_shape[-3] : 1;
    auto output_y = output_shape.rank() >= 2 ? output_shape[-2] : 1;
    auto output_x = output_shape[-1];

    uint32_t num_padded_sticks = input_w * input_z * input_y;
    uint32_t num_unpadded_sticks = input_w * input_z * output_y;
    uint32_t padded_stick_size = input_x * output.element_size();  // Assuming bfloat16 dataformat
    uint32_t unpadded_stick_size = output_x * output.element_size();

    constexpr uint32_t alignment = 32;

    uint32_t num_tiles_in_row = input_x / TILE_WIDTH;
    // Ensure we don't intrude into storage space
    uint32_t max_l1_size = a.device()->l1_size_per_core() / 2 - a.device()->get_base_allocator_addr(HalMemType::L1);
    // Memory usage is 2 CBs of width W, plus buffer of size alignment + (W * datum size)
    uint32_t max_X = (max_l1_size - alignment) / (output.element_size() * TILE_HEIGHT * 2 + output.element_size());
    uint32_t max_tiles = max_X / TILE_WIDTH;

    // Currently need the number of tiles in a row to be divisible by tiles in a block
    uint32_t num_tiles_per_block = 1;
    if (num_tiles_in_row <= max_tiles) {
        num_tiles_per_block = num_tiles_in_row;
    } else {
        for (uint32_t n_t = max_tiles; n_t > 0; n_t--) {
            if (num_tiles_in_row % n_t == 0) {
                num_tiles_per_block = n_t;
                break;
            }
        }
    }
    uint32_t block_width = num_tiles_per_block * TILE_WIDTH;
    uint32_t block_row_size = block_width * output.element_size();
    uint32_t num_blocks_w_output = unpadded_stick_size / block_row_size;
    uint32_t num_blocks_w_input = padded_stick_size / block_row_size;
    uint32_t block_row_leftover_size = unpadded_stick_size - num_blocks_w_output * block_row_size;

    // Number of blocks that differ between input and output
    const uint32_t num_blocks_w_diff = num_blocks_w_input - num_blocks_w_output - (block_row_leftover_size > 0 ? 1 : 0);

    const uint32_t padded_Y_diff_blocks = (input_y - output_y) / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t padded_Z_diff_blocks = (input_z - output_z) * input_y / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t padded_W_diff_blocks = (input_w - output_w) * input_z * input_y / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t num_leftover_Y = output_y - output_y / TILE_HEIGHT * TILE_HEIGHT;

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;
    auto cb_src0_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * input_single_tile_size, {{src0_cb_index, input_cb_data_format}})
                              .set_page_size(src0_cb_index, input_single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_block;
    auto cb_output_config = tt::tt_metal::CircularBufferConfig(
                                num_output_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
                                .set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    vector<uint32_t> writer_kernel_args = {
        dst_buffer->address(),
        output_w,
        padded_W_diff_blocks,
        output_z,
        padded_Z_diff_blocks,
        output_y,
        padded_Y_diff_blocks,
        num_leftover_Y,
        output_x,
        unpadded_stick_size,
        padded_stick_size,
        num_blocks_w_input,
        num_blocks_w_output,
        num_blocks_w_diff,
        block_row_size,
        block_row_leftover_size};

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_is_dram};

    bool out_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t stick_size = unpadded_stick_size;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)std::log2(stick_size) : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)out_is_dram,
        (std::uint32_t)stick_size_is_power_of_two,
        (std::uint32_t)log2_stick_size,
        (std::uint32_t)(input_cb_data_format == tt::DataFormat::Float32)};

    // Tilized reader
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Untilized writer
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
        "writer_unary_unpad_dims_split_rows.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    vector<uint32_t> compute_args = {uint32_t(num_tiles / num_tiles_per_block), uint32_t(num_tiles_per_block)};

    std::string compute_kernel(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp");
    if (num_tiles_per_block > MAX_PACK_UNTILIZE_WIDTH || !use_pack_untilize) {
        log_debug(tt::LogOp, "Using slow untilize.");
        compute_kernel = "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp";
    } else {
        log_debug(tt::LogOp, "Using fast pack untilize.");
    }

    auto untilize_kernel_id = tt::tt_metal::CreateKernel(
        program,
        compute_kernel,
        core,
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_args});

    tt::tt_metal::SetRuntimeArgs(
        program, unary_reader_kernel_id, core, {src0_buffer->address(), uint32_t(num_tiles), 0});

    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);

    auto override_runtime_args_callback = [reader_kernel_id = unary_reader_kernel_id,
                                           writer_kernel_id = unary_writer_kernel_id](
                                              const Program& program,
                                              const std::vector<Buffer*>& input_buffers,
                                              const std::vector<Buffer*>& output_buffers) {
        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks untilize_with_unpadding_multi_core_interleaved(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en) {
    tt::tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    const auto& input_shape = a.get_padded_shape();
    const auto& output_shape = output.get_padded_shape();

    Device* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();

    uint32_t num_blocks = a.volume() / input_shape[-1] / TILE_HEIGHT;
    uint32_t num_tiles_per_row = a.get_legacy_shape()[-1] / TILE_WIDTH;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(grid_size, num_blocks);

    bool has_cliff = core_range_cliff.size() > 0;

    uint32_t padded_row_size_bytes = input_shape[-1] * a.element_size();     // Assuming bfloat16 dataformat
    uint32_t unpadded_row_size_bytes = output_shape[-1] * a.element_size();  // Assuming bfloat16 dataformat

    create_cb(tt::CB::c_in0, program, all_cores, input_single_tile_size, num_tiles_per_row, input_cb_data_format);
    create_cb(tt::CB::c_out0, program, all_cores, output_single_tile_size, num_tiles_per_row, output_cb_data_format);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    uint32_t src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;

    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        ReaderDataMovementConfig({src0_is_dram}));

    /** writer
     */
    uint32_t out_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t stick_size = unpadded_row_size_bytes;
    uint32_t stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)std::log2(stick_size) : 0;

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_multicore.cpp",
        all_cores,
        WriterDataMovementConfig(
            {out_is_dram,
             stick_size_is_power_of_two,
             log2_stick_size,
             input_cb_data_format == tt::DataFormat::Float32}));

    /** compute
     */
    std::string compute_kernel(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp");
    if (num_tiles_per_row > MAX_PACK_UNTILIZE_WIDTH || !use_pack_untilize) {
        compute_kernel = "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp";
    }

    if (core_range.size() > 0) {
        auto tilize_kernel_id = CreateKernel(
            program,
            compute_kernel,
            core_range,
            ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = {nblocks_per_core, num_tiles_per_row}});
    }
    if (has_cliff) {
        auto tilize_cliff_kernel_id = CreateKernel(
            program,
            compute_kernel,
            core_range_cliff,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = {nblocks_per_core_cliff, num_tiles_per_row}});
    }

    auto input_w = input_shape.rank() >= 4 ? input_shape[-4] : 1;
    auto input_z = input_shape.rank() >= 3 ? input_shape[-3] : 1;
    auto input_y = input_shape.rank() >= 2 ? input_shape[-2] : 1;
    auto input_x = input_shape[-1];

    auto output_w = output_shape.rank() >= 4 ? output_shape[-4] : 1;
    auto output_z = output_shape.rank() >= 3 ? output_shape[-3] : 1;
    auto output_y = output_shape.rank() >= 2 ? output_shape[-2] : 1;
    auto output_x = output_shape[-1];

    Padding padding(
        {{0, input_w - output_w}, {0, input_z - output_z}, {0, input_y - output_y}, {0, input_x - output_x}},
        Padding::PadValue::Any);
    auto core_assignments =
        ttnn::distribute_work(output_shape, padding, ncores, nblocks_per_core, has_cliff, nblocks_per_core_cliff);

    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;
    uint32_t ncores_x = grid_size.x;

    const auto& cores = grid_to_cores(ncores, grid_size.x, grid_size.y, true);
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const std::vector<BlockRep>& assignment = core_assignments.at(i);

        // writer runtime args
        vector<uint32_t> writer_rt_args = {
            dst_buffer->address(),
            unpadded_row_size_bytes,
            padded_row_size_bytes,
            row_start_id,
            static_cast<unsigned int>(assignment.size()),
        };

        uint32_t nblocks_per_core = 0;

        for (const auto& el : assignment) {
            nblocks_per_core += el.block_count();
            row_start_id += el.data_row_count();
            writer_rt_args.push_back(el.n_data);
            writer_rt_args.push_back(el.n_mixed);
            writer_rt_args.push_back(el.n_pads);
            writer_rt_args.push_back(el.times);
        }

        uint32_t num_tiles_per_core = num_tiles_per_row * nblocks_per_core;

        // reader runtime args
        vector<uint32_t> reader_rt_args = {src0_buffer->address(), num_tiles_per_core, tile_start_id};

        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);

        tile_start_id += num_tiles_per_core;
    }

    auto override_runtime_args_callback =
        [reader_kernel_id = unary_reader_kernel_id, writer_kernel_id = unary_writer_kernel_id, cores = cores](
            const Program& program,
            const std::vector<Buffer*>& input_buffers,
            const std::vector<Buffer*>& output_buffers) {
            auto src_buffer = input_buffers.at(0);
            auto dst_buffer = output_buffers.at(0);

            auto& reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            auto& writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);

            for (const auto& core : cores) {
                {
                    auto& runtime_args = reader_runtime_args_by_core[core.x][core.y];
                    runtime_args[0] = src_buffer->address();
                }
                {
                    auto& runtime_args = writer_runtime_args_by_core[core.x][core.y];
                    runtime_args[0] = dst_buffer->address();
                }
            }
        };

    return {std::move(program), override_runtime_args_callback};
}

// This purely supports input block shard -> output interleaved for now
operation::ProgramWithCallbacks untilize_with_unpadding_multi_core_sharded(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en) {
    tt::tt_metal::Program program{};

    bool src_sharded = a.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();
    // Special handling for tensors of W=16 and H%32==0
    // In this case skip untilizing on compute and in writer kernel just copy face0 and face2,
    // and skip face1 and face3.
    bool unpad_tensor_w_16 = output.get_legacy_shape()[-1] == 16 && output.get_legacy_shape()[-2] % TILE_HEIGHT == 0;
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    Device* device = a.device();

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t num_rows_block = 0, block_row_size = 0, output_row_size = 0, last_block_row_size_unpadded = 0,
             num_output_rows_unpadded = 0;
    CoreCoord end_core;
    uint32_t last_idx = 0;
    auto shard_spec = a.shard_spec().value();

    // I am not sure it is correct to ever use the shard_spec here.
    auto out_shard_spec = output.shard_spec().has_value() ? output.shard_spec().value() : shard_spec;

    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto grid = *shard_spec.grid.ranges().begin();
    uint32_t ncores_x = grid.end_coord.x + 1;
    uint32_t ncores_y = grid.end_coord.y + 1;
    auto all_cores = shard_spec.grid;
    uint32_t ncores = all_cores.num_cores();
    uint32_t ntiles_per_block = shard_spec.shape[1] / TILE_WIDTH;
    uint32_t nblocks_per_core = shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t batch = a.volume() / (a.get_legacy_shape()[-2] * a.get_legacy_shape()[-1]);
    uint32_t ntiles_per_batch = ntiles_per_block * nblocks_per_core / batch;

    num_rows_block = out_shard_spec.shape[0];
    block_row_size = out_shard_spec.shape[1] * output.element_size();         // in0_block_w * TILE_WIDTH * dtype_nbytes
    output_row_size = output.get_legacy_shape()[-1] * output.element_size();  // output row size bytes
    last_block_row_size_unpadded =
        block_row_size -
        (tt::round_up(output.get_legacy_shape()[-1], out_shard_spec.shape[1]) - output.get_legacy_shape()[-1]) *
            output.element_size();
    uint32_t num_output_rows = output.volume() / output.get_legacy_shape()[-1];
    num_output_rows_unpadded =
        num_rows_block - (tt::round_up(num_output_rows, out_shard_spec.shape[0]) - num_output_rows);
    if (a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        last_idx = tt::div_up(output.get_legacy_shape()[-1], out_shard_spec.shape[1]) - 1;
    } else if (a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        last_idx = tt::div_up(num_output_rows, out_shard_spec.shape[0]) - 1;
    } else {
        end_core = {
            tt::div_up(output.get_legacy_shape()[-1], out_shard_spec.shape[1]) - 1,
            tt::div_up(num_output_rows, out_shard_spec.shape[0]) - 1};
    }
    if (!row_major) {
        std::swap(end_core.x, end_core.y);
    }

    uint32_t num_input_tiles = ntiles_per_block * nblocks_per_core;
    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CB::c_in0,
        program,
        all_cores,
        input_single_tile_size,
        num_input_tiles,
        input_cb_data_format,
        src_sharded ? a.buffer() : nullptr);

    uint32_t num_output_tiles = out_sharded ? (unpad_tensor_w_16 ? 16 : ntiles_per_batch * 2) : ntiles_per_block * 2;
    auto [output_cb_index, cb_output] =
        create_cb(tt::CB::c_out0, program, all_cores, output_single_tile_size, num_output_tiles, output_cb_data_format);

    auto [sharded_output_cb_index, cb_sharded_output] = out_sharded ? create_cb(
                                                                          tt::CB::c_out1,
                                                                          program,
                                                                          all_cores,
                                                                          block_row_size,
                                                                          num_output_rows_unpadded,
                                                                          output_cb_data_format,
                                                                          output.buffer())
                                                                    : std::make_tuple(tt::CB::c_out1, CBHandle{});

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    KernelHandle unary_reader_kernel_id;
    std::vector<uint32_t> reader_ct_args = {(std::uint32_t)src0_cb_index};

    unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    /** writer
     */
    KernelHandle unary_writer_kernel_id;
    if (out_sharded) {
        vector<uint32_t> writer_ct_args = {(uint32_t)output_cb_index, (uint32_t)sharded_output_cb_index};
        unary_writer_kernel_id = CreateKernel(
            program,
            unpad_tensor_w_16 ? "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
                           "writer_unary_unpad_width_16_sharded.cpp"
                         : "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
                           "writer_unary_unpad_batch_rows_sharded.cpp",
            all_cores,
            WriterDataMovementConfig(writer_ct_args));
    } else {
        bool out_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
        vector<uint32_t> writer_ct_args = {
            (uint32_t)out_is_dram, (uint32_t)(input_cb_data_format == tt::DataFormat::Float32)};
        unary_writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp",
            all_cores,
            WriterDataMovementConfig(writer_ct_args));
    }

    /** compute
     */
    vector<uint32_t> compute_args = {
        (uint32_t)nblocks_per_core,  // per_core_block_cnt
        (uint32_t)ntiles_per_block,  // per_block_ntiles
    };

    std::string compute_kernel(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp");
    if (unpad_tensor_w_16) {
        // Use copy compute kernel just for a potential data type conversion.
        compute_kernel = "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/eltwise_copy.cpp";
        compute_args[0] = (uint32_t)num_input_tiles;  // per_core_tile_cnt
    } else if (ntiles_per_block > MAX_PACK_UNTILIZE_WIDTH || !use_pack_untilize) {
        log_debug(tt::LogOp, "Using slow untilize.");
        compute_kernel = "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp";
    } else {
        log_debug(tt::LogOp, "Using fast pack untilize.");
    }

    auto untilize_kernel_id = CreateKernel(
        program,
        compute_kernel,
        all_cores,
        ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_args});

    // reader runtime args
    vector<uint32_t> reader_rt_args = {
        ntiles_per_block * nblocks_per_core  // ntiles
    };
    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, all_cores, reader_rt_args);
    std::vector<CoreCoord> cores;

    if (out_sharded) {
        vector<uint32_t> writer_rt_args;
        if (unpad_tensor_w_16) {
            writer_rt_args = {num_output_rows_unpadded, num_input_tiles};
        } else {
            writer_rt_args = {
                num_output_rows_unpadded,
                ntiles_per_batch,
                out_shard_spec.shape[0] / batch,
                shard_spec.shape[1] * output.element_size(),
                block_row_size,
                batch};
        }
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, all_cores, writer_rt_args);
    } else {
        uint32_t tile_start_id = 0;
        uint32_t row_start_id = 0;
        cores = grid_to_cores(ncores, ncores_x, ncores_y, row_major);
        for (uint32_t i = 0; i < cores.size(); ++i) {
            CoreCoord& core = cores[i];

            // writer runtime args
            vector<uint32_t> writer_rt_args;
            uint32_t block_start_row_offset;
            uint32_t block_start_row_id_offset;
            uint32_t row_size_unpadded = block_row_size;
            uint32_t num_rows_unpadded = num_rows_block;
            if (a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
                block_start_row_offset = i * block_row_size;
                block_start_row_id_offset = 0;
                if (i > last_idx) {
                    row_size_unpadded = 0;
                    num_rows_unpadded = 0;
                } else {
                    num_rows_unpadded = num_output_rows_unpadded;
                    if (i == last_idx) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                }
            } else if (a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                block_start_row_offset = 0;
                block_start_row_id_offset = i * num_rows_block;
                if (i > last_idx) {
                    row_size_unpadded = 0;
                    num_rows_unpadded = 0;
                } else {
                    if (i == last_idx) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                    row_size_unpadded = last_block_row_size_unpadded;
                }
            } else {
                if (row_major) {
                    block_start_row_offset = core.x * block_row_size;
                    block_start_row_id_offset = core.y * num_rows_block;
                    if (core.x == end_core.x) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.y == end_core.y) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                } else {
                    block_start_row_offset = core.y * block_row_size;
                    block_start_row_id_offset = core.x * num_rows_block;
                    if (core.y == end_core.y) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.x == end_core.x) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                }
                if (core.x > end_core.x || core.y > end_core.y) {
                    row_size_unpadded = 0;
                    num_rows_unpadded = 0;
                }
            }

            writer_rt_args = {
                dst_buffer->address(),  // dst_addr
                num_rows_block,
                block_row_size,
                1,
                1,
                1,
                output_row_size,
                row_size_unpadded,
                num_rows_unpadded,
                block_start_row_id_offset,
                block_start_row_offset};

            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);
        }
    }

    auto override_runtime_arguments_callback = [reader_kernel_id = unary_reader_kernel_id,
                                                writer_kernel_id = unary_writer_kernel_id,
                                                cb_src0 = cb_src0,
                                                cb_sharded_output = cb_sharded_output,
                                                cores](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        bool src_sharded = input_tensors.at(0).memory_config().is_sharded();
        bool out_sharded = output_tensors.at(0).memory_config().is_sharded();

        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);

        if (out_sharded) {
            UpdateDynamicCircularBufferAddress(program, cb_sharded_output, *dst_buffer);
        } else {
            auto& runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
            for (const CoreCoord& core : cores) {
                auto& runtime_args = runtime_args_by_core[core.x][core.y];
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks untilize_with_unpadding_multi_core(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en) {
    if (a.memory_config().is_sharded()) {
        return untilize_with_unpadding_multi_core_sharded(a, output, use_pack_untilize, fp32_dest_acc_en);
    } else {
        return untilize_with_unpadding_multi_core_interleaved(a, output, use_pack_untilize, fp32_dest_acc_en);
    }
}

}  // namespace ttnn::operations::data_movement::detail
