// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "argmax_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::prim {

using namespace tt::tt_metal;

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
                static_cast<uint32_t>(reduce_all),
            };
        }
        case Layout::TILE: {
            const uint32_t logical_rank = input.logical_shape().size();
            const uint32_t tile_width = input.tensor_spec().tile().get_width();
            const uint32_t tile_height = input.tensor_spec().tile().get_height();
            const uint32_t w_tiles = input_shape[rank - 1] / tile_width;
            const uint32_t h_tiles = input_shape[rank - 2] / tile_height;
            const uint32_t w_logical = input.logical_shape()[logical_rank - 1];
            const uint32_t h_logical = logical_rank > 1 ? input.logical_shape()[logical_rank - 2] : 1;
            const uint32_t outer_dim_units = input.logical_volume() / (h_logical * w_logical);

            return {
                src_cb_index,
                dst_cb_index,
                src_page_size,
                dst_page_size,
                tile_height,
                tile_width,
                h_tiles,
                w_tiles,
                h_logical,
                w_logical,
                outer_dim_units,
                static_cast<uint32_t>(reduce_all),
                static_cast<uint32_t>(keepdim),
            };
        }
        default:
            TT_FATAL(
                false,
                "Unsupported input layout {} for argmax single-core. Supported: ROW_MAJOR, TILE",
                input.layout());
    }
}

ProgramDescriptor ArgMaxSingleCoreProgramFactory::create_descriptor(
    const ArgmaxParams& operation_attributes, const ArgmaxInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& dim = operation_attributes.dim;
    const bool keepdim = operation_attributes.keepdim;

    ProgramDescriptor desc;
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

    // Create input CB
    desc.cbs.push_back(CBDescriptor{
        .total_size = src_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src_cb_index),
            .data_format = input_data_format,
            .page_size = src_page_size,
        }}},
    });

    // Create output CB
    desc.cbs.push_back(CBDescriptor{
        .total_size = dst_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(dst_cb_index),
            .data_format = output_data_format,
            .page_size = dst_page_size,
        }}},
    });

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

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kernel_path;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(ctime_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // Runtime args
    const auto cores = grid_to_cores(num_cores, grid_size.x, grid_size.y, false);
    for (const auto& core : cores) {
        reader_desc.emplace_runtime_args(core, {src_buffer, dst_buffer});
    }

    desc.kernels.push_back(std::move(reader_desc));

    return desc;
}

}  // namespace ttnn::prim
