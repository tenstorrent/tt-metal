// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_default_program_factory.hpp"
#include "tilize_multi_core_block_program_factory.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

TilizeMultiCoreDefaultProgramFactory::cached_program_t TilizeMultiCoreDefaultProgramFactory::create(
    const ttnn::prim::TilizeParams& operation_attributes,
    const ttnn::prim::TilizeInputs& tensor_args,
    const Tensor& output_tensor) {
    auto a = tensor_args.input_tensor;
    const auto& output = output_tensor;
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    auto sub_core_grids = operation_attributes.sub_core_grids;
    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);
    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32;

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    auto logical_shape = a.logical_shape();
    uint32_t logical_width = logical_shape[-1];
    uint32_t ntiles_per_row = tt::div_up(logical_width, TILE_WIDTH);
    uint32_t ntiles = dst_buffer->num_pages();
    uint32_t nblocks = tt::div_up(ntiles, ntiles_per_row);
    auto* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, nblocks);

    uint32_t page_size = src0_buffer->page_size();
    uint32_t aligned_page_size = src0_buffer->aligned_page_size();
    uint32_t total_pages_per_row = 1;
    uint32_t shard_width = 0;
    uint32_t size_of_valid_data_in_last_page = page_size;
    if (a.is_sharded()) {
        shard_width =
            a.shard_spec().has_value() ? a.shard_spec().value().shape[1] : a.nd_shard_spec().value().shard_shape[-1];
        total_pages_per_row = tt::div_up(logical_width, shard_width);
        uint32_t padding_size = (total_pages_per_row * page_size) - (a.logical_shape()[-1] * a.element_size());
        size_of_valid_data_in_last_page = page_size - padding_size;
    }

    uint32_t num_blocks_in_row = 1;
    uint32_t cb_ntiles = ntiles_per_row;
    uint32_t pages_per_block = total_pages_per_row;

    uint32_t max_l1_space = operations::data_movement::get_max_l1_space(a);
    uint32_t max_cb_tiles = max_l1_space / (input_single_tile_size + output_single_tile_size);

    if (ntiles_per_row > max_cb_tiles && a.is_sharded() && total_pages_per_row > 1) {
        uint32_t tiles_per_page = shard_width / TILE_WIDTH;
        uint32_t max_pages_per_block = std::max(max_cb_tiles / tiles_per_page, 1u);

        pages_per_block = max_pages_per_block;
        for (uint32_t p = max_pages_per_block; p >= 1; --p) {
            if (total_pages_per_row % p == 0) {
                pages_per_block = p;
                break;
            }
        }

        num_blocks_in_row = total_pages_per_row / pages_per_block;
        cb_ntiles = pages_per_block * tiles_per_page;
        size_of_valid_data_in_last_page = page_size;
    }

    create_cb(tt::CBIndex::c_0, program, all_cores, input_single_tile_size, cb_ntiles, input_cb_data_format);

    auto [output_cb_index, _] =
        create_cb(tt::CBIndex::c_16, program, all_cores, output_single_tile_size, cb_ntiles, output_cb_data_format);

    /** reader
     */
    std::vector<uint32_t> reader_ct_args = {
        aligned_page_size, pages_per_block, size_of_valid_data_in_last_page, total_pages_per_row};
    TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args);
    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
        "reader_unary_stick_layout_split_rows_multicore.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args));

    /** writer
     */
    std::vector<uint32_t> writer_ct_args = {output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    /** compute
     */
    std::vector<uint32_t> compute_args = {nblocks_per_core * num_blocks_in_row, cb_ntiles};
    std::vector<uint32_t> compute_args_cliff = {nblocks_per_core_cliff * num_blocks_in_row, cb_ntiles};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_llk_acc) {
        unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    }

    if (!core_range.ranges().empty()) {
        CreateKernel(
            program,
            "ttnn/cpp/ttnn/kernel/compute/tilize.cpp",
            core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_llk_acc,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .compile_args = compute_args,
            });
    }
    if (!core_range_cliff.empty()) {
        CreateKernel(
            program,
            "ttnn/cpp/ttnn/kernel/compute/tilize.cpp",
            core_range_cliff,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_llk_acc,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .compile_args = compute_args_cliff,
            });
    }

    // 1D distribution of blocks across cores
    bool has_cliff = !core_range_cliff.empty();

    uint32_t ncores_full = ncores - has_cliff;
    uint32_t tile_start_id = 0;
    uint32_t page_start_id = 0;
    const auto& cores = corerange_to_cores(available_grid);
    for (uint32_t i = 0; i < ncores_full; ++i) {
        const CoreCoord& core = cores[i];

        // reader runtime args
        const std::array reader_rt_args = {
            src0_buffer->address(),
            nblocks_per_core * TILE_HEIGHT,
            page_size,
            cb_ntiles,
            page_size,
            num_blocks_in_row,
            std::uint32_t{0},  // num leftover tiles
            std::uint32_t{0},  // leftover width in row
            page_start_id};

        // writer runtime args
        const std::array writer_rt_args = {
            dst_buffer->address(),
            ntiles_per_row * nblocks_per_core,  // ntiles per core
            tile_start_id                       // start id
        };

        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);

        tile_start_id += ntiles_per_row * nblocks_per_core;
        page_start_id += TILE_HEIGHT * nblocks_per_core * total_pages_per_row;
    }
    if (has_cliff) {
        const CoreCoord& core = cores[ncores_full];

        // reader runtime args
        const std::array reader_rt_args = {
            src0_buffer->address(),
            nblocks_per_core_cliff * TILE_HEIGHT,
            page_size,
            cb_ntiles,
            page_size,
            num_blocks_in_row,
            std::uint32_t{0},  // num leftover tiles
            std::uint32_t{0},  // leftover width in row
            page_start_id};

        // writer runtime args
        const std::array writer_rt_args = {
            dst_buffer->address(),
            ntiles_per_row * nblocks_per_core_cliff,  // ntiles per core
            tile_start_id                             // start id
        };

        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);
    }

    shared_variables_t shared_vars{unary_reader_kernel_id, unary_writer_kernel_id, cores, ncores};
    return cached_program_t{std::move(program), std::move(shared_vars)};
}

void TilizeMultiCoreDefaultProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ttnn::prim::TilizeParams& /*operation_attributes*/,
    const ttnn::prim::TilizeInputs& tensor_args,
    const Tensor& output_tensor) {
    auto* src_buffer = tensor_args.input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& cores = cached_program.shared_variables.cores;
    auto ncores = cached_program.shared_variables.ncores;

    auto& reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
    auto& writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
    for (uint32_t i = 0; i < ncores; ++i) {
        const CoreCoord& core = cores[i];
        {
            auto& runtime_args = reader_runtime_args_by_core[core.x][core.y];
            runtime_args[0] = src_buffer->address();
        }
        {
            auto& runtime_args = writer_runtime_args_by_core[core.x][core.y];
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::prim
