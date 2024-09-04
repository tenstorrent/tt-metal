// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/cb_utils.hpp"
#include "paged_cache_operation.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "ttnn/operations/experimental/paged_cache/device/paged_fill_cache_program_factory.hpp"

namespace ttnn::operations::experimental::paged_cache::detail {

using namespace tt::constants;
using namespace tt;


operation::ProgramWithCallbacks paged_fill_cache_multi_core(const Tensor& cache_tensor, const Tensor &input_tensor, const Tensor &page_table_tensor, const uint32_t batch_idx) {
    Program program{};

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    // input_tensor: [1, num_heads, input_seq_len, head_dim]
    // cache_tensor: [max_num_blocks, 1, block_size, head_dim]
    // page_table_tensor: [b, max_num_blocks_per_seq]
    const uint32_t num_heads = input_tensor.get_legacy_shape()[1];
    const uint32_t input_seq_len = input_tensor.get_legacy_shape()[2];

    const uint32_t max_num_blocks = cache_tensor.get_legacy_shape()[0];
    const uint32_t block_size = cache_tensor.get_legacy_shape()[2];
    const uint32_t head_dim = cache_tensor.get_legacy_shape()[3];

    const uint32_t batch = page_table_tensor.get_legacy_shape()[0];
    const uint32_t max_num_blocks_per_seq = page_table_tensor.get_legacy_shape()[1];

    const uint32_t input_seq_len_t = input_seq_len / TILE_HEIGHT;
    const uint32_t Wt = head_dim / TILE_WIDTH;
    const uint32_t block_size_t = block_size / TILE_HEIGHT;

    uint32_t num_blocks_of_work = num_heads * input_seq_len_t;

    // Pagetable-specific parameters
    uint32_t page_table_stick_size_B = page_table_tensor.buffer()->aligned_page_size();
    TT_FATAL(page_table_stick_size_B % 32 == 0, "page table page size in bytes must be a multiple of 32 due to address alignment");
    uint32_t log2_page_table_stick_size_B = std::log2(page_table_stick_size_B);
    tt::DataFormat page_table_data_format = tt_metal::datatype_to_dataformat_converter(page_table_tensor.get_dtype());

    TT_FATAL(1 << log2_page_table_stick_size_B == page_table_stick_size_B, "page_table_stick_size_B must be a power of 2");

    tt_metal::Device *device = input_tensor.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major;
    uint32_t num_cores, num_blocks_per_core_group_1, num_blocks_per_core_group_2;

    CoreRangeSet all_cores({}), core_group_1({}), core_group_2({});

    row_major = true;
    std::tie(num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2) = split_work_to_cores(compute_with_storage_grid_size,  num_blocks_of_work, row_major);
    uint32_t num_input_tiles = Wt * 2; // double buffered

    tt::CB src0_cb_index = tt::CB::c_in0;
    tt::CB page_table_cb_index = tt::CB::c_in1;
    create_cb(src0_cb_index, program, all_cores, single_tile_size, num_input_tiles, cb_data_format);
    create_cb(page_table_cb_index, program, all_cores, page_table_stick_size_B, 1, page_table_data_format);


    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = cache_tensor.buffer();
    auto page_table_buffer = page_table_tensor.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    bool page_table_is_dram = page_table_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;



    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t) src_is_dram,
        (uint32_t) src0_cb_index,
        Wt
    };

    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t) dst_is_dram,
        (uint32_t) page_table_is_dram,
        (uint32_t) src0_cb_index,
        (uint32_t) page_table_cb_index,
        num_heads,
        block_size_t,
        Wt,
        log2_page_table_stick_size_B,
        page_table_stick_size_B
    };

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/reader_fill_cache_interleaved.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_fill_cache_interleaved.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++){
        const CoreCoord &core = cores.at(i);
        uint32_t num_blocks_per_core = 0;
        if (i < g1_numcores) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (i < g1_numcores + g2_numcores) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            num_blocks_per_core = 0;
        }

        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                src_buffer->address(),
                num_blocks_written*Wt, // start_tile_id
                num_blocks_per_core, // num_rows
            }
        );

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                page_table_buffer->address(),
                num_blocks_written, // start_row_num
                num_blocks_per_core, // num_rows
                batch_idx, // batch_idx
            }
        );
        num_blocks_written+=num_blocks_per_core;
    }

    auto override_runtime_args_callback = [
            unary_reader_kernel_id,
            unary_writer_kernel_id,
            cores,
            g1_numcores,
            g2_numcores,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2,
            Wt
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
        const auto batch_idx = static_cast<const PagedUpdateCacheDeviceOperation*>(operation)->batch_idx;

        auto dst_addr = input_tensors.at(0).buffer()->address();
        auto src_addr = input_tensors.at(1).buffer()->address();
        auto page_table_addr = input_tensors.at(2).buffer()->address();

        auto& reader_args_by_core = GetRuntimeArgs(program, unary_reader_kernel_id);
        auto& writer_args_by_core = GetRuntimeArgs(program, unary_writer_kernel_id);

        for (uint32_t i = 0, num_blocks_written = 0; i < cores.size(); i++){
            const CoreCoord &core = cores.at(i);
            uint32_t num_blocks_per_core = 0;
            if (i < g1_numcores) {
                num_blocks_per_core = num_blocks_per_core_group_1;
            } else if (i < g1_numcores + g2_numcores) {
                num_blocks_per_core = num_blocks_per_core_group_2;
            } else {
                num_blocks_per_core = 0;
            }

            auto& reader_args = reader_args_by_core.at(core.x).at(core.y);
            reader_args[0] = src_addr;
            reader_args[1] = num_blocks_written*Wt; // start_tile_id
            reader_args[2] = num_blocks_per_core; // num_rows

            auto& writer_args = writer_args_by_core.at(core.x).at(core.y);
            writer_args[0] = dst_addr;
            writer_args[1] = page_table_addr;
            writer_args[2] = num_blocks_written; // start_row_num
            writer_args[3] = num_blocks_per_core; // num_rows
            writer_args[4] = batch_idx; // batch_idx

            num_blocks_written += num_blocks_per_core;
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

}  // ttnn::operations::experimental::paged_cache::detail
