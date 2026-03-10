// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::examples {

using namespace tt;
using namespace tt::tt_metal;

ProgramDescriptor ExampleDeviceOperation::SingleCore::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tile_size(cb_data_format);
    tt::DataFormat cb_data_format_output = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t single_tile_size_output = tt::tile_size(cb_data_format_output);

    uint32_t num_tiles = input_tensor.physical_volume() / constants::TILE_HW;

    CoreCoord compute_with_storage_grid_size = {1, 1};
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    // ---- Build the ProgramDescriptor ----

    ProgramDescriptor desc;

    // Circular buffers
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
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

    constexpr uint32_t output_cb_index = CBIndex::c_2;
    constexpr uint32_t num_output_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * single_tile_size_output,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = cb_data_format_output,
            .page_size = single_tile_size_output,
        }}},
    });

    // Reader kernel
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer kernel
    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    // Compute kernel
    std::vector<uint32_t> compute_kernel_args_group_1 = {
        num_tiles_per_core_group_1,  // per_core_block_cnt
        1                            // per_core_block_size
    };

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_group_1;
    compute_desc.compile_time_args = compute_kernel_args_group_1;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
    };

    // Runtime args per core
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        reader_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{src_buffer->address(), num_tiles_per_core, num_tiles_written});

        writer_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{dst_buffer->address(), num_tiles_per_core, num_tiles_written});

        num_tiles_written += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

// NOTE: No override_nondeterministic_runtime_args needed here!
// Buffer addresses are automatically patched by the framework on cache hits.
// Only implement override_nondeterministic_runtime_args(Program&, ...) if you
// have truly dynamic parameters (e.g. random seeds) that change every call.

}  // namespace ttnn::operations::examples
