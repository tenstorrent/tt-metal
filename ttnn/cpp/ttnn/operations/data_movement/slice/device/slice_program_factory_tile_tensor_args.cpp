// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile_tensor_args.hpp"

#include <optional>
#include <span>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor SliceTileTensorArgsProgramFactory::create_descriptor(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    const auto& start_tensor = tensor_args.start_tensor.value();
    const auto& end_tensor = tensor_args.end_tensor.value();
    tt::tt_metal::IDevice* device = input_tensor.device();
    ProgramDescriptor desc;

    uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    tt::tt_metal::Buffer* src_buffer = input_tensor.buffer();
    tt::tt_metal::Buffer* start_buffer = start_tensor.buffer();
    tt::tt_metal::Buffer* end_buffer = end_tensor.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();

    TT_FATAL(src_buffer != nullptr, "Input buffer should be allocated on device!");
    TT_FATAL(start_buffer != nullptr, "Start buffer should be allocated on device!");
    TT_FATAL(end_buffer != nullptr, "End buffer should be allocated on device!");
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    constexpr uint8_t src0_cb_index = 0;
    constexpr uint8_t tensor_cb_index = 1;
    constexpr uint32_t num_input_tiles = 2;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tensor_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_tensor.padded_shape().rank());
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    uint32_t tile_width = tile_shape[1];
    uint32_t tile_height = tile_shape[0];

    std::vector<uint32_t> reader_compile_time_args = {
        src0_cb_index, tensor_cb_index, num_dims, tile_width, tile_height};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*start_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*end_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {src0_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Reader common runtime args layout (matches kernel):
    //   [src_addr, start_buf_addr, end_buf_addr,
    //    num_unpadded_per_dim..., num_padded_per_dim..., input_shape...]
    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output.padded_shape();
    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    accumulated_total_per_dim[0] = num_total_Xt;
    accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;

    // src/start/end buffer bindings are registered separately below; this vector holds only the per-dim values.
    std::vector<uint32_t> reader_common_dims(num_dims * 3);
    std::span<uint32_t> reader_common_dims_view{reader_common_dims};
    auto num_unpadded_tiles_per_dim = reader_common_dims_view.subspan(0, num_dims);
    auto num_padded_tiles_per_dim = reader_common_dims_view.subspan(num_dims, num_dims);
    auto input_shape_args = reader_common_dims_view.subspan(num_dims * 2, num_dims);
    num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
    num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
    num_padded_tiles_per_dim[0] = num_padded_Xt;
    num_padded_tiles_per_dim[1] = num_padded_Yt;
    for (int32_t i = 2; i < static_cast<int32_t>(num_dims); ++i) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
        num_padded_tiles_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }
    for (int32_t i = 0; i < static_cast<int32_t>(num_dims); ++i) {
        input_shape_args[i] = input_shape[i];
    }

    // Reader per-core runtime args: [start_id, num_tiles, id_per_dim...]
    // Writer per-core runtime args: [dst_addr, num_tiles, start_id]
    constexpr uint32_t start_offset = 0;

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor::RuntimeArgs reader_runtime_args;
    uint32_t num_tiles_written = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op core (num_tiles == 0 → address unused, but bound for a uniform arg layout)
            std::vector<uint32_t> reader_args(2 + num_dims, 0);
            reader_runtime_args.emplace_back(core, std::move(reader_args));
            writer_desc.emplace_runtime_args(core, {dst_buffer, 0u, 0u});
            continue;
        }

        std::vector<uint32_t> reader_args(2 + num_dims);
        reader_args[2] = num_tiles_written % num_unpadded_tiles_per_dim[0];
        uint32_t unpadded_written = num_tiles_written / num_unpadded_tiles_per_dim[0];
        uint32_t start_id = reader_args[2] + start_offset;
        for (uint32_t j = 1; j < num_dims; ++j) {
            reader_args[2 + j] = unpadded_written % num_unpadded_tiles_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_tiles_per_dim[j];
            start_id += reader_args[2 + j] * accumulated_total_per_dim[j - 1];
        }
        reader_args[0] = start_id;
        reader_args[1] = num_tiles_per_core;

        reader_runtime_args.emplace_back(core, std::move(reader_args));
        writer_desc.emplace_runtime_args(core, {dst_buffer, num_tiles_per_core, num_tiles_written});
        num_tiles_written += num_tiles_per_core;
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.runtime_args = std::move(reader_runtime_args);
    KernelDescriptor::RTArgList reader_common;
    reader_common.reserve(3 + (num_dims * 3));
    reader_common.push_back(src_buffer);
    reader_common.push_back(start_buffer);
    reader_common.push_back(end_buffer);
    reader_common.append(reader_common_dims);
    reader_desc.emplace_common_runtime_args(reader_common);
    reader_desc.config = ReaderConfigDescriptor{};
    desc.kernels.push_back(std::move(reader_desc));

    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
