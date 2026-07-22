// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gelu_bw_program_factory.hpp"
#include "gelu_bw_device_operation_types.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::unary_backward::gelu_bw {

using namespace tt::constants;

tt::tt_metal::ProgramDescriptor GeluBwProgramFactory::create_descriptor(
    const GeluBwParams& args, const GeluBwInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& grad_output = tensor_args.grad_output;

    ProgramDescriptor desc;

    DataFormat src0_cb_data_format = datatype_to_dataformat_converter(grad_output.dtype());
    uint32_t src0_single_tile_size = tile_size(src0_cb_data_format);
    DataFormat src1_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t src1_single_tile_size = tile_size(src1_cb_data_format);
    DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tile_size(dst_cb_data_format);

    uint32_t num_tiles = input.physical_volume() / TILE_HW;

    IDevice* device = input.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    constexpr uint32_t num_input_tiles = 2;
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    constexpr uint32_t src1_cb_index = CBIndex::c_1;
    constexpr uint32_t num_output_tiles = 2;
    constexpr uint32_t output_cb_index = CBIndex::c_2;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * src0_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * src1_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = src1_cb_data_format,
            .page_size = src1_single_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * dst_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }}},
    });

    auto* src0_buffer = grad_output.buffer();
    auto* src1_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args = {0};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/"
        "reader_binary_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    bool fp32_dest_acc_en = (src0_cb_data_format == DataFormat::Float32) ||
                            (src1_cb_data_format == DataFormat::Float32) || (dst_cb_data_format == DataFormat::Float32);

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[src0_cb_index] =
        (grad_output.dtype() == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;
    unpack_to_dest_mode[src1_cb_index] =
        (input.dtype() == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;

    std::string compute_kernel_path;
    if (args.approximate) {
        // For bfloat16, we have 8 DST tiles available in DstSync::SyncHalf.
        // For float32, we have 4 DST tiles available in DstSync::SyncHalf.
        compute_kernel_path = fp32_dest_acc_en ? "ttnn/cpp/ttnn/operations/eltwise/unary_backward/gelu_bw/device/"
                                                 "kernels/compute/eltwise_bw_gelu_tanh_fp32.cpp"
                                               : "ttnn/cpp/ttnn/operations/eltwise/unary_backward/gelu_bw/device/"
                                                 "kernels/compute/eltwise_bw_gelu_tanh.cpp";
    } else {
        compute_kernel_path =
            "ttnn/cpp/ttnn/operations/eltwise/unary_backward/gelu_bw/device/"
            "kernels/compute/eltwise_bw_gelu_poly.cpp";
    }
    std::map<std::string, std::string> compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines["COPY_DEST_VALUES"] = "copy_dest_values<DataFormat::Float32>";
    } else {
        compute_defines["COPY_DEST_VALUES"] = "copy_dest_values<DataFormat::Float16_b>";
    }
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = compute_kernel_path;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = {unpack_to_dest_mode.begin(), unpack_to_dest_mode.end()},
    };
    compute_desc.defines = {compute_defines.begin(), compute_defines.end()};

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        reader_desc.emplace_runtime_args(
            core, {src0_buffer, src1_buffer, num_tiles_per_core, num_tiles_written, 0u, 0u, num_cores_y});

        compute_desc.emplace_runtime_args(core, {num_tiles_per_core});

        writer_desc.emplace_runtime_args(core, {dst_buffer, num_tiles_per_core, num_tiles_written});

        num_tiles_written += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::unary_backward::gelu_bw
