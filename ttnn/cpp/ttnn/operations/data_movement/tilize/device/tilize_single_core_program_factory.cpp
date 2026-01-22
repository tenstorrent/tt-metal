// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_single_core_program_factory.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {
TilizeSingleCoreProgramFactory::cached_program_t TilizeSingleCoreProgramFactory::create(
    const ttnn::prim::TilizeParams& operation_attributes,
    const ttnn::prim::TilizeInputs& tensor_args,
    const Tensor& output_tensor) {
    tt::tt_metal::Program program{};

    auto a = tensor_args.input_tensor;
    const auto& output = output_tensor;
    auto sub_core_grids = operation_attributes.sub_core_grids;

    CoreRange default_core({0, 0}, {0, 0});
    CoreRange core = sub_core_grids.has_value() ? corerange_to_cores(sub_core_grids.value()).at(0) : default_core;

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    // This should allocate a DRAM buffer on the device

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32;

    uint32_t num_tiles = a.physical_volume() / TILE_HW;

    auto width = a.padded_shape()[-1];
    uint32_t stick_s = width;
    uint32_t num_sticks = a.physical_volume() / width;
    uint32_t stick_size = stick_s * a.element_size();  // Assuming bfloat16 dataformat

    uint32_t num_tiles_in_row = stick_s / TILE_WIDTH;
    uint32_t num_tiles_per_block = 1;

    if (!operation_attributes.use_low_perf) {
        // Ensure we don't intrude into storage space
        uint32_t max_l1_size =
            (a.device()->l1_size_per_core() / 2) - a.device()->allocator()->get_base_allocator_addr(HalMemType::L1);
        uint32_t max_tiles = max_l1_size / (input_single_tile_size + output_single_tile_size);  // 2 CBs
        // Currently need the number of tiles in a row to be divisible by tiles in a block
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
    }

    uint32_t block_width_size = num_tiles_per_block * TILE_WIDTH * a.element_size();
    uint32_t num_full_blocks_in_row = num_tiles_in_row / num_tiles_per_block;
    uint32_t num_leftover_tiles = num_tiles_in_row % num_tiles_per_block;
    uint32_t leftover_width_in_row = num_leftover_tiles * a.element_size();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;

    auto src0_cb_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * input_single_tile_size, {{src0_cb_index, input_cb_data_format}})
                              .set_page_size(src0_cb_index, input_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core, src0_cb_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = num_tiles_per_block;
    auto cb_output_config = tt::tt_metal::CircularBufferConfig(
                                num_output_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
                                .set_page_size(output_cb_index, output_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    const std::array reader_kernel_args = {
        src0_buffer->address(),
        num_sticks,
        stick_size,
        num_tiles_per_block,
        block_width_size,
        num_full_blocks_in_row,
        num_leftover_tiles,
        leftover_width_in_row,
        std::uint32_t{0},  // row_start_id
    };

    // Reader compile-time args
    std::vector<uint32_t> reader_compile_time_args = {stick_size};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Tilized reader
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
        "reader_unary_stick_layout_split_rows_interleaved.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Tilized writer
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_args = {
        num_tiles / num_tiles_per_block,  // per_core_block_cnt
        num_tiles_per_block               // per_core_block_tile_cnt
    };

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
        core,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_llk_acc,
            .compile_args = compute_args,
        });

    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);

    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_buffer->address(), num_tiles, 0});
    return cached_program_t{std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id, core}};
}

void TilizeSingleCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ttnn::prim::TilizeParams& /*operation_attributes*/,
    const ttnn::prim::TilizeInputs& tensor_args,
    const Tensor& output_tensor) {
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& core = cached_program.shared_variables.core;
    auto* src_buffer = tensor_args.input_tensor.buffer();
    auto& program = cached_program.program;
    auto* dst_buffer = output_tensor.buffer();
    CoreCoord core_0 = corerange_to_cores(core).at(0);
    {
        auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core_0);
        runtime_args[0] = src_buffer->address();
    }
    {
        auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core_0);
        runtime_args[0] = dst_buffer->address();
    }
}
}  // namespace ttnn::prim
