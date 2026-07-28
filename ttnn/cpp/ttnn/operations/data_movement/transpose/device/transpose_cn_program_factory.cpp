// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_cn_program_factory.hpp"
#include "transpose_utils.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor TransposeCNProgramFactory::create_descriptor(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    auto input_shape = input_tensor.padded_shape();
    bool row_major = input_tensor.layout() == Layout::ROW_MAJOR;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_cn needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_cn needs to be allocated in a buffer on device!");

    ProgramDescriptor desc;

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t page_shape[2] = {TILE_WIDTH, TILE_HEIGHT};
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        page_shape[0] = 1;
        page_shape[1] = input_shape[-1];
    }
    uint32_t page_size = page_shape[0] * page_shape[1];
    uint32_t stick_size = (row_major) ? page_shape[1] * input_tensor.element_size() : tt::tile_size(cb_data_format);

    Buffer* src0_buffer = input_tensor.buffer();
    IDevice* device = input_tensor.device();

    uint32_t num_tensor_pages = input_tensor.physical_volume() / page_size;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_pages);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_pages = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_pages * stick_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = stick_size,
        }}},
    });

    KernelDescriptor::Defines reader_defines;
    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(src0_cb_index), src0_buffer->aligned_page_size(), stick_size};
    std::vector<uint32_t> reader_common_runtime_args;
    TensorAccessorArgs(*src0_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(reader_compile_time_args, reader_common_runtime_args);
    KernelDescriptor::Defines writer_defines;
    std::vector<uint32_t> writer_compile_time_args = {
        static_cast<uint32_t>(src0_cb_index), dst_buffer->aligned_page_size(), stick_size};
    std::vector<uint32_t> writer_common_runtime_args;
    TensorAccessorArgs(*dst_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_compile_time_args, writer_common_runtime_args);

    if (row_major) {
        reader_defines.emplace_back("CN_RM", "1");
        writer_defines.emplace_back("CN_RM", "1");
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_cn_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = total_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.common_runtime_args = std::move(reader_common_runtime_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "writer_unary_transpose_cn_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = total_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = std::move(writer_defines);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.common_runtime_args = std::move(writer_common_runtime_args);

    // Set runtime arguments for each core
    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t Wt = W / page_shape[1];
    uint32_t Ht = H / page_shape[0];
    uint32_t HtWt = Ht * Wt;
    uint32_t CHtWt = C * HtWt;
    uint32_t NCHtWt = num_tensor_pages;
    uint32_t batch_step = CHtWt - HtWt;
    uint32_t channel_step = NCHtWt - HtWt;

    reader_desc.runtime_args.reserve(num_cores_total);
    writer_desc.runtime_args.reserve(num_cores_total);
    for (uint32_t i = 0, num_pages_read = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_pages_per_core = 0;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        }

        uint32_t hw = num_pages_read % HtWt;
        uint32_t curr_c = num_pages_read / HtWt;
        uint32_t n = curr_c % N;
        uint32_t start_tile = num_pages_read + (curr_c * batch_step) - (curr_c / N * channel_step);

        reader_desc.emplace_runtime_args(
            core, {src0_buffer, N, C, HtWt, batch_step, channel_step, num_pages_per_core, start_tile, hw, n});
        writer_desc.emplace_runtime_args(core, {dst_buffer, num_pages_per_core, num_pages_read});

        num_pages_read += num_pages_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
