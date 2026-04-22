// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanh_bw_program_factory.hpp"
#include "tanh_bw_device_operation_types.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::unary_backward::tanh_bw {

using namespace tt::constants;
using namespace tt::tt_metal;

ProgramDescriptor TanhBwProgramFactory::create_descriptor(
    const TanhBwParams& /*args*/, const TanhBwInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;              // src0
    const auto& grad_output = tensor_args.grad_output;  // src1

    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat src1_cb_data_format = datatype_to_dataformat_converter(grad_output.dtype());
    uint32_t src1_single_tile_size = tt::tile_size(src1_cb_data_format);
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;

    IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    uint32_t num_input_tiles = 2;
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t num_output_tiles = 2;
    uint32_t output_cb_index = tt::CBIndex::c_2;

    auto* src0_buffer = grad_output.buffer();
    auto* src1_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    ProgramDescriptor desc;

    // ---- Circular buffers ----

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

    // ---- Kernel compile-time args ----

    KernelDescriptor::CompileTimeArgs reader_compile_time_args = {0};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    KernelDescriptor::CompileTimeArgs writer_compile_time_args = {static_cast<uint32_t>(output_cb_index)};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    bool fp32_dest_acc_en = (dst_cb_data_format == tt::DataFormat::Float32) ||
                            (dst_cb_data_format == tt::DataFormat::Int32) ||
                            (dst_cb_data_format == tt::DataFormat::UInt32);

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[src1_cb_index] = UnpackToDestMode::UnpackToDestFp32;

    // ---- Reader kernel ----

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    // ---- Writer kernel ----

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    // ---- Compute kernel ----

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/"
        "kernels/compute/eltwise_bw_tanh_deriv.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = {unpack_to_dest_mode.begin(), unpack_to_dest_mode.end()},
    };

    // ---- Per-core runtime args ----

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

        reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                src0_buffer->address(),
                src1_buffer->address(),
                num_tiles_per_core,
                num_tiles_written,
                0,
                0,
                num_cores_y});

        compute_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{num_tiles_per_core, 1});

        writer_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{dst_buffer->address(), num_tiles_per_core, num_tiles_written});

        num_tiles_written += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::unary_backward::tanh_bw
