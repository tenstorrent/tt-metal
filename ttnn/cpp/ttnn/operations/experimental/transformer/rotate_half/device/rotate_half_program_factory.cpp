// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotate_half_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor RotateHalfProgramFactory::create_descriptor(
    const RotateHalfParams& /*operation_attributes*/, const Tensor& input, Tensor& tensor_return_value) {
    using namespace tt::constants;

    ProgramDescriptor desc;

    const CoreCoord core({0, 0});
    CoreRange core_range(core, core);
    CoreRangeSet core_set{core_range};

    Tensor& output = tensor_return_value;

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);

    tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
    const uint32_t scalar_single_tile_size = tt::tile_size(scalar_cb_data_format);

    const uint32_t num_tiles = input.physical_volume() / TILE_HW;
    const uint32_t num_rows = input.physical_volume() / input.padded_shape()[-1] / TILE_HEIGHT;
    const uint32_t half_row_size = input.padded_shape()[-1] / TILE_WIDTH / 2;

    // Used for half of tensor that is multiplied
    constexpr uint8_t src_mul_cb_index = 0;
    const uint32_t num_input_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src_mul_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    // Used for bcast scalar
    constexpr uint8_t src_scalar_cb_index = 1;
    const uint32_t num_scalar_tiles = 1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_scalar_tiles * scalar_single_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src_scalar_cb_index,
            .data_format = cb_data_format,
            .page_size = scalar_single_tile_size,
        }}},
    });

    // Used for half of tensor that is not multiplied
    constexpr uint8_t src_no_mul_cb_index = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src_no_mul_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    constexpr uint8_t output_mul_cb_index = tt::CBIndex::c_16;
    const uint32_t num_output_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * single_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_mul_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });
    constexpr uint8_t output_no_mul_cb_index = src_no_mul_cb_index;

    const uint16_t bfloat16_scalar = std::bit_cast<uint16_t>(bfloat16(-1.0f));

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args = {
        src_no_mul_cb_index, src_mul_cb_index, src_scalar_cb_index, static_cast<uint32_t>(bfloat16_scalar)};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {output_no_mul_cb_index, output_mul_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotate_half/device/kernels/dataflow/"
        "reader_rotate_half_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_set;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.emplace_runtime_args(core, {src_buffer, num_rows, half_row_size, static_cast<uint32_t>(0)});

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotate_half/device/kernels/dataflow/"
        "writer_rotate_half_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_set;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.emplace_runtime_args(core, {dst_buffer, num_rows, half_row_size, static_cast<uint32_t>(0)});

    std::map<std::string, std::string> bcast_compute_defines = {
        {"BCAST_OP", "mul_tiles_bcast"},
        {"BCAST_LLKOP", "EltwiseBinaryType::ELWMUL"},
        {"BCAST_DIM", "BroadcastType::SCALAR"},
        {"BCAST_SCALAR", "1"}};

    KernelDescriptor bcast_desc;
    bcast_desc.kernel_source = "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_hw.cpp";
    bcast_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    bcast_desc.core_ranges = core_set;
    bcast_desc.defines = KernelDescriptor::Defines{bcast_compute_defines.begin(), bcast_compute_defines.end()};
    bcast_desc.config = ComputeConfigDescriptor{};
    bcast_desc.runtime_args.emplace_back(
        core,
        std::vector<uint32_t>{
            1,             // B
            1,             // Ht
            num_tiles / 2  // Wt
        });

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(bcast_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
