// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "argmax_single_core_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::reduction::argmax::program {

using namespace tt::constants;

static std::tuple<uint32_t, uint32_t> get_page_sizes_single_core(
    const Tensor& input, const Tensor& output, bool keepdim, bool reduce_all) {
    const auto& input_shape = input.padded_shape();
    const uint32_t rank = input_shape.size();

    switch (input.layout()) {
        case Layout::ROW_MAJOR: {
            const uint32_t red_dim_units = input_shape[rank - 1];
            const uint32_t input_unit_size = input.element_size();
            const uint32_t output_unit_size = output.element_size();
            const uint32_t output_last_dim = reduce_all or keepdim or (rank < 2) ? 1 : input_shape[rank - 2];

            return {red_dim_units * input_unit_size, output_last_dim * output_unit_size};
        }
        case Layout::TILE: {
            TT_FATAL(
                output.layout() == Layout::ROW_MAJOR,
                "For TILE input layout, only ROW_MAJOR output is supported, got output layout: {}",
                output.layout());

            return {input.tensor_spec().compute_page_size_bytes(), output.tensor_spec().compute_page_size_bytes()};
        }
        default:
            TT_FATAL(
                false,
                "Unsupported input layout {} for argmax single-core. Supported: ROW_MAJOR, TILE",
                input.layout());
    }
}

static void create_circular_buffers_single_core(
    tt::tt_metal::Program& program,
    auto& all_cores,
    uint32_t src_cb_index,
    uint32_t dst_cb_index,
    uint32_t src_page_size,
    uint32_t dst_page_size,
    tt::DataFormat input_format,
    tt::DataFormat output_format) {
    // Create input CB
    auto src_cb_config = tt::tt_metal::CircularBufferConfig(src_page_size, {{src_cb_index, input_format}})
                             .set_page_size(src_cb_index, src_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, src_cb_config);

    // Create output CB
    auto dst_cb_config = tt::tt_metal::CircularBufferConfig(dst_page_size, {{dst_cb_index, output_format}})
                             .set_page_size(dst_cb_index, dst_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, dst_cb_config);
}

static std::vector<uint32_t> get_ctime_args_single_core(
    const Tensor& input,
    uint32_t src_page_size,
    uint32_t dst_page_size,
    uint32_t src_cb_index,
    uint32_t dst_cb_index,
    bool keepdim,
    bool reduce_all) {
    const auto& input_shape = input.padded_shape();
    const uint32_t rank = input_shape.size();

    switch (input.layout()) {
        case Layout::ROW_MAJOR: {
            const uint32_t red_dim_units = input_shape[rank - 1];
            const uint32_t output_last_dim = reduce_all or keepdim or (rank < 2) ? 1 : input_shape[rank - 2];
            const uint32_t inner_dim_units = output_last_dim;
            const uint32_t outer_dim_units = input.logical_volume() / inner_dim_units / red_dim_units;

            return {
                src_cb_index,
                dst_cb_index,
                src_page_size,
                dst_page_size,
                outer_dim_units,
                inner_dim_units,
                red_dim_units,
                (uint32_t)(reduce_all),
            };
        }
        case Layout::TILE: {
            const uint32_t logical_rank = input.logical_shape().size();
            const uint32_t w_tiles = input_shape[rank - 1] / TILE_WIDTH;
            const uint32_t h_tiles = input_shape[rank - 2] / TILE_HEIGHT;
            const uint32_t w_logical = input.logical_shape()[logical_rank - 1];
            const uint32_t h_logical = logical_rank > 1 ? input.logical_shape()[logical_rank - 2] : 1;
            const uint32_t outer_dim_units = input.logical_volume() / (h_logical * w_logical);

            return {
                src_cb_index,
                dst_cb_index,
                src_page_size,
                dst_page_size,
                TILE_HEIGHT,
                TILE_WIDTH,
                h_tiles,
                w_tiles,
                h_logical,
                w_logical,
                outer_dim_units,
                (uint32_t)(reduce_all),
                (uint32_t)(keepdim),
            };
        }
        default:
            TT_FATAL(
                false,
                "Unsupported input layout {} for argmax single-core. Supported: ROW_MAJOR, TILE",
                input.layout());
    }
}

ArgMaxSingleCoreProgramFactory::cached_program_t ArgMaxSingleCoreProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& dim = operation_attributes.dim;
    const bool keepdim = operation_attributes.keepdim;

    tt::tt_metal::Program program{};
    const tt::tt_metal::IDevice* device = output.device();
    const bool reduce_all = not dim.has_value();

    // Circular buffers
    const auto src_cb_index = tt::CBIndex::c_0;
    const auto dst_cb_index = tt::CBIndex::c_1;

    const auto grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_units = 1;  // single-core
    auto [num_cores, all_cores, unused_1, unused_2, unused_3, unused_4] =
        tt::tt_metal::split_work_to_cores(grid_size, num_units);

    const tt::DataFormat input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const auto [src_page_size, dst_page_size] = get_page_sizes_single_core(input, output, keepdim, reduce_all);
    create_circular_buffers_single_core(
        program,
        all_cores,
        src_cb_index,
        dst_cb_index,
        src_page_size,
        dst_page_size,
        input_data_format,
        output_data_format);

    // Compile-time args
    std::vector<uint32_t> ctime_args = get_ctime_args_single_core(
        input, src_page_size, dst_page_size, src_cb_index, dst_cb_index, keepdim, reduce_all);

    auto* const src_buffer = input.buffer();
    auto* const dst_buffer = output.buffer();
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(ctime_args);
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(ctime_args);

    // Kernel
    std::string kernel_path =
        input.layout() == Layout::ROW_MAJOR
            ? "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved.cpp"
            : "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_tile_layout.cpp";

    const std::map<std::string, std::string> kernel_defines;
    const tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program, kernel_path, all_cores, tt::tt_metal::ReaderDataMovementConfig(ctime_args, kernel_defines));

    // Runtime args
    const auto cores = grid_to_cores(num_cores, grid_size.x, grid_size.y, false);
    for (const auto& core : cores) {
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, {src_buffer->address(), dst_buffer->address()});
    }

    return {std::move(program), {reader_kernel_id, cores}};
}

void ArgMaxSingleCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    for (const auto& core : cores) {
        auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
        runtime_args[0] = src_buffer->address();
        runtime_args[1] = dst_buffer->address();
    }
}

}  // namespace ttnn::operations::reduction::argmax::program
