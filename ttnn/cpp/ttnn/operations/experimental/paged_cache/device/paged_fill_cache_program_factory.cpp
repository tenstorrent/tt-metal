// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "paged_cache_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/experimental/paged_cache/device/paged_fill_cache_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::paged_cache::detail {

using namespace tt::constants;
using namespace tt;

operation::ProgramWithCallbacks paged_fill_cache_multi_core(
    const Tensor& cache_tensor,
    const Tensor& input_tensor,
    const Tensor& page_table_tensor,
    std::optional<const Tensor> batch_idx_tensor,
    const uint32_t batch_idx_fallback) {
    Program program{};

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    // input_tensor: [1, num_heads, input_seq_len, head_dim]
    // cache_tensor: [max_num_blocks, 1, block_size, head_dim]
    // page_table_tensor: [b, max_num_blocks_per_seq]
    const uint32_t num_heads = input_tensor.padded_shape()[1];
    const uint32_t input_seq_len = input_tensor.padded_shape()[2];

    const uint32_t max_num_blocks = cache_tensor.padded_shape()[0];
    const uint32_t block_size = cache_tensor.padded_shape()[2];
    const uint32_t head_dim = cache_tensor.padded_shape()[3];

    const uint32_t batch = page_table_tensor.padded_shape()[0];
    const uint32_t max_num_blocks_per_seq = page_table_tensor.padded_shape()[1];

    const uint32_t input_seq_len_t = input_seq_len / TILE_HEIGHT;
    const uint32_t Wt = head_dim / TILE_WIDTH;
    const uint32_t block_size_t = block_size / TILE_HEIGHT;

    uint32_t num_blocks_of_work = num_heads * input_seq_len_t;
    uint32_t num_blocks_of_work_per_head = input_seq_len_t;

    // Pagetable-specific parameters
    uint32_t page_table_stick_size_B = page_table_tensor.buffer()->aligned_page_size();
    TT_FATAL(
        page_table_stick_size_B % 32 == 0,
        "page table page size in bytes must be a multiple of 32 due to address alignment");
    uint32_t log2_page_table_stick_size_B = std::log2(page_table_stick_size_B);
    tt::DataFormat page_table_data_format = tt_metal::datatype_to_dataformat_converter(page_table_tensor.dtype());

    // batch_idx_tensor specific parameters
    bool use_batch_idx_tensor = batch_idx_tensor.has_value();
    uint32_t batch_idx_buffer_addr = 0;
    tt::DataFormat batch_idx_data_format = tt::DataFormat::UInt32;  // Assuming batch_idx is uint32
    uint32_t batch_idx_stick_size_B = 4;                            // Assuming scalar uint32
    bool batch_idx_is_dram = false;

    if (use_batch_idx_tensor) {
        const auto& tensor = batch_idx_tensor.value();
        batch_idx_buffer_addr = tensor.buffer()->address();
        batch_idx_data_format = tt_metal::datatype_to_dataformat_converter(tensor.dtype());
        batch_idx_stick_size_B = tensor.element_size();
        TT_FATAL(tensor.physical_volume() == 1, "batch_idx_tensor must contain a single element.");
        batch_idx_is_dram = tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM;
    }

    tt_metal::IDevice* device = input_tensor.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major;
    uint32_t num_cores, num_blocks_per_core_group_1, num_blocks_per_core_group_2;

    CoreRangeSet all_cores, core_group_1, core_group_2;

    row_major = true;
    std::tie(
        num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2) =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks_of_work, row_major);
    uint32_t num_input_tiles = Wt * 2;  // double buffered

    tt::CBIndex src0_cb_index = tt::CBIndex::c_0;
    tt::CBIndex page_table_cb_index = tt::CBIndex::c_1;
    tt::CBIndex cb_batch_idx_id = tt::CBIndex::c_2;  // New CB for batch_idx_tensor

    create_cb(src0_cb_index, program, all_cores, single_tile_size, num_input_tiles, cb_data_format);
    create_cb(page_table_cb_index, program, all_cores, page_table_stick_size_B, 1, page_table_data_format);
    if (use_batch_idx_tensor) {
        create_cb(cb_batch_idx_id, program, all_cores, batch_idx_stick_size_B, 1, batch_idx_data_format);
    }

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = cache_tensor.buffer();
    auto page_table_buffer = page_table_tensor.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool page_table_is_dram = page_table_buffer->buffer_type() == tt_metal::BufferType::DRAM;

    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram, (uint32_t)src0_cb_index, Wt};

    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t)dst_is_dram,
        (uint32_t)page_table_is_dram,
        (uint32_t)src0_cb_index,
        (uint32_t)page_table_cb_index,
        num_heads,
        num_blocks_of_work_per_head,
        block_size_t,
        Wt,
        log2_page_table_stick_size_B,
        page_table_stick_size_B,
        // New compile-time args for batch_idx_tensor
        (uint32_t)use_batch_idx_tensor,
        cb_batch_idx_id,              // Meaningful only if use_batch_idx_tensor is true
        (uint32_t)batch_idx_is_dram,  // Meaningful only if use_batch_idx_tensor is true
        batch_idx_stick_size_B        // Meaningful only if use_batch_idx_tensor is true
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

    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        const CoreCoord& core = cores.at(i);
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
                num_blocks_written * Wt,  // start_tile_id
                num_blocks_per_core,      // num_rows
            });

        uint32_t writer_batch_arg = use_batch_idx_tensor ? batch_idx_buffer_addr : batch_idx_fallback;

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                page_table_buffer->address(),
                num_blocks_written,   // start_row_num
                num_blocks_per_core,  // num_rows
                writer_batch_arg,     // batch_idx_tensor_addr or batch_idx_fallback
            });
        num_blocks_written += num_blocks_per_core;
    }

    auto override_runtime_args_callback =
        [unary_reader_kernel_id,
         unary_writer_kernel_id,
         cores,
         g1_numcores,
         g2_numcores,
         num_blocks_per_core_group_1,
         num_blocks_per_core_group_2,
         Wt,
         use_batch_idx_tensor  // Capture this
    ](const void* operation,   // Should be PagedUpdateCacheDeviceOperation
        Program& program,
        const std::vector<Tensor>& input_tensors,                                // cache, input, page_table
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,  // batch_idx_tensor if used
        const std::vector<Tensor>& output_tensors) {                             // output_tensors not used by fill op
            auto dst_addr = input_tensors.at(0).buffer()->address();         // cache_tensor
            auto src_addr = input_tensors.at(1).buffer()->address();         // input_tensor
            auto page_table_addr = input_tensors.at(2).buffer()->address();  // page_table_tensor

            uint32_t current_kernel_batch_arg;
            const auto op_specific = static_cast<const PagedUpdateCacheDeviceOperation*>(operation);

            if (use_batch_idx_tensor) {

                TT_FATAL(
                    op_specific->batch_idx_tensor_opt.has_value(),
                    "batch_idx_tensor_opt is expected in PagedUpdateCacheDeviceOperation but not provided for callback "
                    "when use_batch_idx_tensor is true.");
                current_kernel_batch_arg = op_specific->batch_idx_tensor_opt.value().buffer()->address();
            } else {
                // Fallback to scalar batch_idx from the operation struct
                current_kernel_batch_arg =
                    op_specific->batch_idx_fallback;  // Assumes PagedUpdateCacheDeviceOperation has batch_idx_fallback
            }

            auto& reader_args_by_core = GetRuntimeArgs(program, unary_reader_kernel_id);
            auto& writer_args_by_core = GetRuntimeArgs(program, unary_writer_kernel_id);

            for (uint32_t i = 0, num_blocks_written = 0; i < cores.size(); i++) {
                const CoreCoord& core = cores.at(i);
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
                reader_args[1] = num_blocks_written * Wt;  // start_tile_id
                reader_args[2] = num_blocks_per_core;      // num_rows

                auto& writer_args = writer_args_by_core.at(core.x).at(core.y);
                writer_args[0] = dst_addr;
                writer_args[1] = page_table_addr;
                writer_args[2] = num_blocks_written;        // start_row_num
                writer_args[3] = num_blocks_per_core;       // num_rows
                writer_args[4] = current_kernel_batch_arg;  // batch_idx_tensor_addr or batch_idx_fallback

                num_blocks_written += num_blocks_per_core;
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::experimental::paged_cache::detail
