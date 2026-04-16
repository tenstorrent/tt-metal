// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include "clone_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/math.hpp"

namespace ttnn::operations::data_movement::clone {

using namespace tt::constants;
using namespace tt::tt_metal;

ProgramDescriptor CloneOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    auto input_data_format = datatype_to_dataformat_converter(input.dtype());
    auto output_data_format = datatype_to_dataformat_converter(output.dtype());
    bool convert_dtype = input_data_format != output_data_format;
    bool tilized = output.layout() == Layout::TILE;
    auto compute_unit_size = [&](const auto& tensor, const auto& data_format) {
        return tilized ? tt::tile_size(data_format) : tensor.logical_shape()[-1] * tensor.element_size();
    };
    uint32_t input_unit_size = compute_unit_size(input, input_data_format);
    uint32_t output_unit_size = compute_unit_size(output, output_data_format);
    uint32_t num_units =
        tilized ? output.physical_volume() / TILE_HW : output.physical_volume() / output.logical_shape()[-1];

    auto output_memory_layout = output.memory_config().memory_layout();
    bool is_sharded = output_memory_layout != TensorMemoryLayout::INTERLEAVED;

    uint32_t num_cores;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_units_per_core_group_1;
    uint32_t num_units_per_core_group_2;
    uint32_t num_cores_x;
    uint32_t num_cores_y;

    if (is_sharded) {
        auto shard_spec = output.buffer()->shard_spec();
        all_cores = shard_spec.grid();
        num_cores = all_cores.num_cores();

        auto shard_shape = shard_spec.shape();
        uint32_t shard_height = shard_shape[0];
        uint32_t shard_width = shard_shape[1];

        // For row-major sharded, the unit (stick) size must be the shard width, not the
        // full tensor width. Using tensor width causes OOB reads past the shard boundary.
        if (!tilized) {
            input_unit_size = shard_width * input.element_size();
            output_unit_size = shard_width * output.element_size();
        }

        if (tilized) {
            num_units_per_core_group_1 = (shard_height * shard_width) / TILE_HW;
        } else {
            num_units_per_core_group_1 = shard_height;
        }

        num_units_per_core_group_2 = 0;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();

        auto grid_size = all_cores.bounding_box();
        num_cores_x = grid_size.end_coord.x + 1;
        num_cores_y = grid_size.end_coord.y + 1;
    } else {
        auto compute_with_storage_grid_size = output.device()->compute_with_storage_grid_size();
        num_cores_x = compute_with_storage_grid_size.x;
        num_cores_y = compute_with_storage_grid_size.y;
        auto
            [num_cores_result,
             all_cores_result,
             core_group_1_result,
             core_group_2_result,
             num_units_per_core_group_1_result,
             num_units_per_core_group_2_result] = split_work_to_cores(compute_with_storage_grid_size, num_units);
        num_cores = num_cores_result;
        all_cores = all_cores_result;
        core_group_1 = core_group_1_result;
        core_group_2 = core_group_2_result;
        num_units_per_core_group_1 = num_units_per_core_group_1_result;
        num_units_per_core_group_2 = num_units_per_core_group_2_result;
    }

    auto alignment = input.buffer()->alignment();

    uint32_t src_cb_id = tt::CBIndex::c_4;
    uint32_t aligned_input_unit_size = tt::align(input_unit_size, alignment);

    ProgramDescriptor desc;

    // Source CB
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * aligned_input_unit_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src_cb_id,
            .data_format = input_data_format,
            .page_size = aligned_input_unit_size,
        }}},
    });

    uint32_t dst_cb_id = src_cb_id;
    if (convert_dtype) {
        dst_cb_id = tt::CBIndex::c_20;
        uint32_t aligned_output_unit_size = tt::align(output_unit_size, alignment);
        desc.cbs.push_back(CBDescriptor{
            .total_size = 2 * aligned_output_unit_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = dst_cb_id,
                .data_format = output_data_format,
                .page_size = aligned_output_unit_size,
            }}},
        });
    }

    auto* input_buffer = input.buffer();
    auto* output_buffer = output.buffer();

    // Compile-time args differ for tilized vs row-major
    KernelDescriptor::CompileTimeArgs reader_ct_args;
    KernelDescriptor::CompileTimeArgs writer_ct_args;
    if (tilized) {
        reader_ct_args = {src_cb_id};
        TensorAccessorArgs(*input_buffer).append_to(reader_ct_args);
        writer_ct_args = {dst_cb_id};
        TensorAccessorArgs(*output_buffer).append_to(writer_ct_args);
    } else {
        reader_ct_args = {src_cb_id, input_unit_size};
        TensorAccessorArgs(*input_buffer).append_to(reader_ct_args);
        writer_ct_args = {dst_cb_id, output_unit_size};
        TensorAccessorArgs(*output_buffer).append_to(writer_ct_args);
    }

    // Kernel paths depend on sharded vs interleaved and tilized vs RM
    const char* read_kernel_path;
    const char* write_kernel_path;

    if (is_sharded) {
        read_kernel_path =
            tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel_sharded.cpp"
                    : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel_rm_sharded.cpp";
        write_kernel_path =
            tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_sharded.cpp"
                    : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_rm_sharded.cpp";
    } else {
        read_kernel_path = tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel.cpp"
                                   : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel_rm.cpp";
        write_kernel_path = tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel.cpp"
                                    : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_rm.cpp";
    }

    // Reader kernel
    KernelDescriptor reader_desc;
    reader_desc.kernel_source = read_kernel_path;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer kernel
    KernelDescriptor writer_desc;
    writer_desc.kernel_source = write_kernel_path;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Compute kernel for dtype conversion (dual core groups)
    if (convert_dtype) {
        auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
            get_compute_kernel_config_args(input.device()->arch(), operation_attributes.compute_kernel_config);

        ComputeConfigDescriptor compute_config{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
        };

        if (!core_group_1.ranges().empty()) {
            KernelDescriptor compute_desc_1;
            compute_desc_1.kernel_source =
                "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/compute_kernel.cpp";
            compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
            compute_desc_1.core_ranges = core_group_1;
            compute_desc_1.compile_time_args = {src_cb_id, dst_cb_id, num_units_per_core_group_1};
            compute_desc_1.config = compute_config;
            desc.kernels.push_back(std::move(compute_desc_1));
        }
        if (!core_group_2.ranges().empty()) {
            KernelDescriptor compute_desc_2;
            compute_desc_2.kernel_source =
                "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/compute_kernel.cpp";
            compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
            compute_desc_2.core_ranges = core_group_2;
            compute_desc_2.compile_time_args = {src_cb_id, dst_cb_id, num_units_per_core_group_2};
            compute_desc_2.config = compute_config;
            desc.kernels.push_back(std::move(compute_desc_2));
        }
    }

    // Runtime args per core
    uint32_t start_id = 0;
    uint32_t num_cores_group_1 = core_group_1.num_cores();
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);
    reader_desc.runtime_args.reserve(cores.size());
    writer_desc.runtime_args.reserve(cores.size());

    for (size_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        uint32_t num_units_per_core = i < num_cores_group_1 ? num_units_per_core_group_1 : num_units_per_core_group_2;

        if (is_sharded) {
            if (tilized) {
                reader_desc.runtime_args.emplace_back(
                    core,
                    KernelDescriptor::CoreRuntimeArgs{
                        static_cast<uint32_t>(input_buffer->address()), num_units_per_core});
                writer_desc.runtime_args.emplace_back(
                    core,
                    KernelDescriptor::CoreRuntimeArgs{
                        static_cast<uint32_t>(output_buffer->address()), num_units_per_core});
            } else {
                reader_desc.runtime_args.emplace_back(
                    core,
                    KernelDescriptor::CoreRuntimeArgs{
                        static_cast<uint32_t>(input_buffer->address()), input_unit_size, num_units_per_core});
                writer_desc.runtime_args.emplace_back(
                    core,
                    KernelDescriptor::CoreRuntimeArgs{
                        static_cast<uint32_t>(output_buffer->address()), output_unit_size, num_units_per_core});
            }
        } else {
            if (tilized) {
                reader_desc.runtime_args.emplace_back(
                    core,
                    KernelDescriptor::CoreRuntimeArgs{
                        static_cast<uint32_t>(input_buffer->address()), num_units_per_core, start_id});
                writer_desc.runtime_args.emplace_back(
                    core,
                    KernelDescriptor::CoreRuntimeArgs{
                        static_cast<uint32_t>(output_buffer->address()), num_units_per_core, start_id});
            } else {
                reader_desc.runtime_args.emplace_back(
                    core,
                    KernelDescriptor::CoreRuntimeArgs{
                        static_cast<uint32_t>(input_buffer->address()), input_unit_size, num_units_per_core, start_id});
                writer_desc.runtime_args.emplace_back(
                    core,
                    KernelDescriptor::CoreRuntimeArgs{
                        static_cast<uint32_t>(output_buffer->address()),
                        output_unit_size,
                        num_units_per_core,
                        start_id});
            }
            start_id += num_units_per_core;
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::operations::data_movement::clone
