// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_sharded_rm_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

TransposeWHShardedRMProgramFactory::cached_program_t TransposeWHShardedRMProgramFactory::create(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_wh needs to be allocated in a buffer on device!");

    Program program = CreateProgram();

    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];
    uint32_t stick_size_bytes = W * input_tensor.element_size();
    uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    uint32_t output_page_size, pack_num_pages, pack_num_pages_last_col, pack_num_pages_last_row,
        pack_num_pages_last_row_col;
    if ((W % TILE_WIDTH) != 0 and (H % TILE_HEIGHT) != 0) {
        output_page_size = (W % TILE_WIDTH) * (H % TILE_HEIGHT) * output_tensor.element_size();
        pack_num_pages = dst_single_tile_size / output_page_size;
        auto output_page_size_last_col = TILE_WIDTH * (H % TILE_HEIGHT) * output_tensor.element_size();
        pack_num_pages_last_col = dst_single_tile_size / output_page_size_last_col;
        auto output_page_size_last_row = TILE_HEIGHT * (W % TILE_WIDTH) * output_tensor.element_size();
        pack_num_pages_last_row = dst_single_tile_size / output_page_size_last_row;
        pack_num_pages_last_row_col = 1;
    } else if ((W % TILE_WIDTH) != 0 and (H % TILE_HEIGHT) == 0) {
        output_page_size = (W % TILE_WIDTH) * (TILE_HEIGHT)*output_tensor.element_size();
        pack_num_pages = dst_single_tile_size / output_page_size;
        pack_num_pages_last_col = pack_num_pages;
        pack_num_pages_last_row = 1;
        pack_num_pages_last_row_col = 1;
    } else if ((W % TILE_WIDTH) == 0 and (H % TILE_HEIGHT) != 0) {
        output_page_size = (TILE_WIDTH) * (H % TILE_HEIGHT) * output_tensor.element_size();
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

    log_debug(tt::LogOp, "output_page_size: {}", output_page_size);
    log_debug(tt::LogOp, "pack_num_pages: {}", pack_num_pages);
    log_debug(tt::LogOp, "pack_num_pages_last_col: {}", pack_num_pages_last_col);
    log_debug(tt::LogOp, "pack_num_pages_last_row: {}", pack_num_pages_last_row);
    log_debug(tt::LogOp, "pack_num_pages_last_row_col: {}", pack_num_pages_last_row_col);

    auto shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    uint32_t num_hw_blocks_per_core = shard_height / H;

    log_debug(tt::LogOp, "shard_height: {}", shard_height);
    log_debug(tt::LogOp, "dst_single_tile_size: {}", dst_single_tile_size);

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;

    auto& all_cores = shard_spec.grid;
    [[maybe_unused]] uint32_t num_cores = shard_spec.num_cores();
    auto bbox = shard_spec.grid.bounding_box();
    CoreCoord grid_size = {bbox.end_coord.x + 1, bbox.end_coord.y + 1};
    uint32_t num_cores_x = grid_size.x;
    uint32_t num_cores_y = grid_size.y;

    log_debug(tt::LogOp, "all_cores: {}", all_cores);
    log_debug(tt::LogOp, "num_cores: {}", num_cores);

    // sharded cb
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(shard_height * stick_size_bytes, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, stick_size_bytes)
            .set_globally_allocated_address(*input_tensor.buffer());
    auto cb_src0 = CreateCircularBuffer(program, all_cores, cb_src0_config);

    // sharded cb
    uint32_t output_cb_index = tt::CBIndex::c_16;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(stick_size_bytes * shard_height, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, output_page_size)
            .set_globally_allocated_address(*output_tensor.buffer());
    auto cb_output = CreateCircularBuffer(program, all_cores, cb_output_config);

    // cb_in
    uint32_t in_cb_index = tt::CBIndex::c_24;
    uint32_t num_in_tiles = wt * 2;  // double buffer
    CircularBufferConfig cb_in_config =
        CircularBufferConfig(num_in_tiles * src0_single_tile_size, {{in_cb_index, src0_cb_data_format}})
            .set_page_size(in_cb_index, src0_single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_in_config);

    // tilize cb
    uint32_t im_cb_index = tt::CBIndex::c_25;
    uint32_t num_im_tiles = ht * wt;
    CircularBufferConfig cb_im_config =
        CircularBufferConfig(num_im_tiles * src0_single_tile_size, {{im_cb_index, src0_cb_data_format}})
            .set_page_size(im_cb_index, src0_single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_im_config);

    // untilize cb
    if (ht > 8) {
        uint32_t im2_cb_index = tt::CBIndex::c_26;
        uint32_t num_im2_tiles = ht;
        CircularBufferConfig cb_im2_config =
            CircularBufferConfig(num_im2_tiles * dst_single_tile_size, {{im2_cb_index, dst_cb_data_format}})
                .set_page_size(im2_cb_index, dst_single_tile_size);
        CreateCircularBuffer(program, all_cores, cb_im2_config);

        // compute_output_cb
        uint32_t out_cb_index = tt::CBIndex::c_27;
        uint32_t num_out_tiles = ht * 2;  // double buffer
        CircularBufferConfig cb_out_config =
            CircularBufferConfig(num_out_tiles * dst_single_tile_size, {{out_cb_index, dst_cb_data_format}})
                .set_page_size(out_cb_index, dst_single_tile_size);
        CreateCircularBuffer(program, all_cores, cb_out_config);
    }

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)num_hw_blocks_per_core,
        (std::uint32_t)ht,
        (std::uint32_t)H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT,
        (std::uint32_t)H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT,
        (std::uint32_t)wt,
        (std::uint32_t)stick_size_bytes,
        (std::uint32_t)wt * input_tensor.element_size() * TILE_WIDTH,
    };
    reader_compile_time_args.push_back(H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT);
    reader_compile_time_args.push_back(H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_wh_sharded_rm.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)num_hw_blocks_per_core,
        (std::uint32_t)ht,
        (std::uint32_t)wt,
        (std::uint32_t)W > TILE_WIDTH ? TILE_WIDTH : W % TILE_WIDTH,
        (std::uint32_t)W % TILE_WIDTH == 0 ? TILE_WIDTH : W % TILE_WIDTH,
        (std::uint32_t)H * output_tensor.element_size(),
        (std::uint32_t)ht * output_tensor.element_size() * TILE_HEIGHT,
    };

    CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "writer_unary_transpose_wh_sharded_rm.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

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

    std::map<std::string, std::string> compute_defines;
    compute_defines["SHARDED"] = "1";

    CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_rm.cpp",
        all_cores,
        ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_compile_time_args,
            .defines = compute_defines});

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .cb_src0 = cb_src0,
         .cb_output = cb_output,
         .num_cores_x = num_cores_x,
         .num_cores_y = num_cores_y}};
}

void TransposeWHShardedRMProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TransposeParams& /*operation_attributes*/,
    const TransposeInputs& tensor_args,
    Tensor& output_tensor) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    const auto& src_tensor = tensor_args.input;

    auto* const src_buffer = src_tensor.buffer();
    auto* const dst_buffer = output_tensor.buffer();

    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_src0, *src_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_output, *dst_buffer);
}

}  // namespace ttnn::prim
