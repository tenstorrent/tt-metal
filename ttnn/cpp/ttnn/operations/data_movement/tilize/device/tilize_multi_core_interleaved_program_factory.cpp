// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_interleaved_program_factory.hpp"
#include "tilize_multi_core_block_program_factory.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

TilizeMultiCoreInterleavedProgramFactory::cached_program_t TilizeMultiCoreInterleavedProgramFactory::create(
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

    int32_t ntiles = a.physical_volume() / TILE_HW;
    uint32_t ntiles_per_block = a.padded_shape()[-1] / TILE_WIDTH;
    uint32_t nblocks = std::ceil((float)ntiles / ntiles_per_block);
    uint32_t block_size_nbytes = a.padded_shape()[-1] * a.element_size();

    IDevice* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, nblocks);

    create_cb(tt::CBIndex::c_0, program, all_cores, input_single_tile_size, ntiles_per_block, input_cb_data_format);

    auto [output_cb_index, _] = create_cb(
        tt::CBIndex::c_16, program, all_cores, output_single_tile_size, ntiles_per_block, output_cb_data_format);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    std::vector<uint32_t> reader_ct_args = {block_size_nbytes};
    TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args);
    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
        "reader_unary_stick_layout_split_rows_interleaved.cpp",
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
    std::vector<uint32_t> compute_args = {nblocks_per_core, ntiles_per_block};
    std::vector<uint32_t> compute_args_cliff = {nblocks_per_core_cliff, ntiles_per_block};

    if (!core_range.ranges().empty()) {
        CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
            core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_llk_acc,
                .compile_args = compute_args,
            });
    }
    if (!core_range_cliff.empty()) {
        CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
            core_range_cliff,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_llk_acc,
                .compile_args = compute_args_cliff,
            });
    }

    // 1D distribution of blocks across cores
    bool has_cliff = !core_range_cliff.empty();

    uint32_t ncores_full = ncores - has_cliff;
    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;
    const auto& cores = corerange_to_cores(available_grid);
    for (uint32_t i = 0; i < ncores_full; ++i) {
        const CoreCoord& core = cores[i];

        // reader runtime args
        const std::array reader_rt_args = {
            src0_buffer->address(),
            nblocks_per_core * TILE_HEIGHT,
            block_size_nbytes,
            ntiles_per_block,
            block_size_nbytes,
            std::uint32_t{1},  // full blocks in row
            std::uint32_t{0},  // num leftover tiles
            std::uint32_t{0},  // leftover width in row
            row_start_id};

        // writer runtime args
        const std::array writer_rt_args = {
            dst_buffer->address(),
            ntiles_per_block * nblocks_per_core,  // ntiles per core
            tile_start_id                         // start id
        };

        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);

        tile_start_id += ntiles_per_block * nblocks_per_core;
        row_start_id += TILE_HEIGHT * nblocks_per_core;
    }
    if (has_cliff) {
        // the last core is a cliff core with nblocks_per_core_cliff blocks
        const CoreCoord& core = cores[ncores_full];

        // reader runtime args
        const std::array reader_rt_args = {
            src0_buffer->address(),
            nblocks_per_core_cliff * TILE_HEIGHT,
            block_size_nbytes,
            ntiles_per_block,
            block_size_nbytes,
            std::uint32_t{1},  // full blocks in row
            std::uint32_t{0},  // num leftover tiles
            std::uint32_t{0},  // leftover width in row
            row_start_id};

        // writer runtime args
        const std::array writer_rt_args = {
            dst_buffer->address(),
            ntiles_per_block * nblocks_per_core_cliff,  // ntiles per core
            tile_start_id                               // start id
        };

        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);
    }

    shared_variables_t shared_vars{unary_reader_kernel_id, unary_writer_kernel_id, cores, ncores};
    return cached_program_t{std::move(program), std::move(shared_vars)};
}

void TilizeMultiCoreInterleavedProgramFactory::override_runtime_arguments(
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
