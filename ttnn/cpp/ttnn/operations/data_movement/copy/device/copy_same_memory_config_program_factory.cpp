// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_device_operation.hpp"

#include <cmath>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

constexpr const char* KERNEL_READER_UNARY_START_ID =
    "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/reader_unary_start_id.cpp";
constexpr const char* KERNEL_READER_RM_SHARDED =
    "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/reader_unary_stick_start_id.cpp";
constexpr const char* KERNEL_READER_RM_INTERLEAVED =
    "ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp";
constexpr const char* KERNEL_WRITER_UNARY_START_ID =
    "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/writer_unary_start_id.cpp";
constexpr const char* KERNEL_WRITER_RM_SHARDED =
    "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/writer_unary_stick_start_id.cpp";
constexpr const char* KERNEL_WRITER_RM_INTERLEAVED =
    "ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp";
constexpr const char* KERNEL_COMPUTE_ELTWISE_COPY = "ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp";

void append_defines(KernelDescriptor& kd, const std::map<std::string, std::string>& kernel_defines) {
    kd.defines.reserve(kd.defines.size() + kernel_defines.size());
    for (const auto& [k, v] : kernel_defines) {
        kd.defines.emplace_back(k, v);
    }
}

}  // namespace

ProgramDescriptor CopyDeviceOperation::SameMemoryConfig::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const Tensor& input = tensor_args.input;
    const bool backwards = operation_attributes.backwards;

    ProgramDescriptor desc;

    const bool tilized = output.layout() == Layout::TILE;
    const bool sharded = input.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED;
    // The tilized reader/writer kernels use the unified TensorAccessor, which addresses both
    // interleaved and sharded tensors, so they no longer need the ShardedAddrGen path. The
    // row-major (stick) kernels still rely on ShardedAddrGen for sharded tensors.
    const bool use_sharded_addrgen = sharded && !tilized;
    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const auto& input_tile = input.tensor_spec().tile();
    const auto& output_tile = output.tensor_spec().tile();
    uint32_t input_unit_size =
        tilized ? input_tile.get_tile_size(input_cb_data_format) : input.padded_shape()[-1] * input.element_size();
    const uint32_t full_input_row = input_unit_size;
    if (sharded && !tilized) {
        input_unit_size = input.memory_config().shard_spec()->shape[1] * input.element_size();
    }
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_unit_size =
        tilized ? output_tile.get_tile_size(output_cb_data_format) : output.padded_shape()[-1] * output.element_size();
    const uint32_t full_output_row = output_unit_size;
    if (sharded && !tilized) {
        output_unit_size = output.memory_config().shard_spec()->shape[1] * output.element_size();
    }

    const bool convert_dtype = input_cb_data_format != output_cb_data_format;

    const uint32_t num_units = tilized ? output.physical_volume() / input_tile.get_tile_hw()
                                       : output.physical_volume() / output.padded_shape()[-1];

    IDevice* device = output.device();

    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
            split_work_to_cores(compute_with_storage_grid_size, num_units);

    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    const uint32_t num_input_units = 2;
    const uint32_t input_alignment = input.buffer()->alignment();
    const uint32_t aligned_input_unit_size = tt::align(input_unit_size, input_alignment);
    {
        CBFormatDescriptor format_desc{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = input_cb_data_format,
            .page_size = aligned_input_unit_size,
        };
        // Attach TileDescriptor on TILE-layout CBs so JIT get_tile_size(cb) matches tiny tiles.
        if (tilized) {
            format_desc.tile = TileDescriptor(input_tile);
        }
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_input_units * aligned_input_unit_size,
            .core_ranges = all_cores,
            .format_descriptors = {{std::move(format_desc)}},
        });
    }

    uint32_t output_cb_index = src0_cb_index;  // same as input cb
    if (convert_dtype) {
        output_cb_index = tt::CBIndex::c_16;
        const uint32_t num_output_units = 2;
        const uint32_t output_alignment = output.buffer()->alignment();
        const uint32_t aligned_output_unit_size = tt::align(output_unit_size, output_alignment);
        CBFormatDescriptor format_desc{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_cb_data_format,
            .page_size = aligned_output_unit_size,
        };
        if (tilized) {
            format_desc.tile = TileDescriptor(output_tile);
        }
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_output_units * aligned_output_unit_size,
            .core_ranges = all_cores,
            .format_descriptors = {{std::move(format_desc)}},
        });
    }

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    KernelDescriptor::CompileTimeArgs reader_compile_time_args;
    KernelDescriptor::CompileTimeArgs writer_compile_time_args;
    if (tilized) {
        writer_compile_time_args = {static_cast<uint32_t>(output_cb_index)};
    } else {
        reader_compile_time_args = {static_cast<uint32_t>(src0_cb_index), static_cast<uint32_t>(input_unit_size)};
        writer_compile_time_args = {static_cast<uint32_t>(output_cb_index), static_cast<uint32_t>(output_unit_size)};
    }
    std::map<std::string, std::string> kernel_defines;
    if (use_sharded_addrgen) {
        kernel_defines["SHARDED"] = "1";
        shard_builder::extend_sharding_compile_time_args(input, writer_compile_time_args);
        shard_builder::extend_sharding_compile_time_args(input, reader_compile_time_args);
    } else {
        TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
    }
    if (backwards) {
        kernel_defines["BACKWARDS"] = "1";
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        tilized ? KERNEL_READER_UNARY_START_ID : (sharded ? KERNEL_READER_RM_SHARDED : KERNEL_READER_RM_INTERLEAVED);
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    append_defines(reader_desc, kernel_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        tilized ? KERNEL_WRITER_UNARY_START_ID : (sharded ? KERNEL_WRITER_RM_SHARDED : KERNEL_WRITER_RM_INTERLEAVED);
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    append_defines(writer_desc, kernel_defines);
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t start_id = backwards ? num_units - 1 : 0;

    const uint32_t g1_numcores = core_group_1.num_cores();
    const std::vector<CoreCoord> cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        const uint32_t num_units_per_core = i < g1_numcores ? num_units_per_core_group_1 : num_units_per_core_group_2;

        if (tilized) {
            KernelDescriptor::RTArgList reader_ra;
            reader_ra.push_back(src_buffer);
            reader_ra.push_back(num_units_per_core);
            reader_ra.push_back(start_id);
            KernelDescriptor::RTArgList writer_ra;
            writer_ra.push_back(dst_buffer);
            writer_ra.push_back(num_units_per_core);
            writer_ra.push_back(start_id);
            if (use_sharded_addrgen) {
                std::vector<uint32_t> reader_tail;
                std::vector<uint32_t> writer_tail;
                shard_builder::extend_sharding_run_time_args(input, reader_tail);
                shard_builder::extend_sharding_run_time_args(input, writer_tail);
                reader_ra.append(reader_tail);
                writer_ra.append(writer_tail);
            }
            reader_desc.emplace_runtime_args(core, reader_ra);
            writer_desc.emplace_runtime_args(core, writer_ra);
        } else {
            KernelDescriptor::RTArgList reader_ra;
            reader_ra.push_back(src_buffer);
            reader_ra.push_back(input_unit_size);
            reader_ra.push_back(num_units_per_core);
            reader_ra.push_back(start_id);
            reader_ra.push_back(full_input_row / input_unit_size);
            KernelDescriptor::RTArgList writer_ra;
            writer_ra.push_back(dst_buffer);
            writer_ra.push_back(output_unit_size);
            writer_ra.push_back(num_units_per_core);
            writer_ra.push_back(start_id);
            writer_ra.push_back(full_output_row / output_unit_size);
            if (use_sharded_addrgen) {
                std::vector<uint32_t> reader_tail;
                std::vector<uint32_t> writer_tail;
                shard_builder::extend_sharding_run_time_args(input, reader_tail);
                shard_builder::extend_sharding_run_time_args(input, writer_tail);
                reader_ra.append(reader_tail);
                writer_ra.append(writer_tail);
            }
            reader_desc.emplace_runtime_args(core, reader_ra);
            writer_desc.emplace_runtime_args(core, writer_ra);
        }
        if (backwards) {
            start_id -= num_units_per_core;
        } else {
            start_id += num_units_per_core;
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    if (convert_dtype) {
        KernelDescriptor compute_g1;
        compute_g1.kernel_source = KERNEL_COMPUTE_ELTWISE_COPY;
        compute_g1.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_g1.core_ranges = core_group_1;
        compute_g1.compile_time_args = {num_units_per_core_group_1};
        compute_g1.config = ComputeConfigDescriptor{};
        desc.kernels.push_back(std::move(compute_g1));

        if (!core_group_2.ranges().empty()) {
            KernelDescriptor compute_g2;
            compute_g2.kernel_source = KERNEL_COMPUTE_ELTWISE_COPY;
            compute_g2.source_type = KernelDescriptor::SourceType::FILE_PATH;
            compute_g2.core_ranges = core_group_2;
            compute_g2.compile_time_args = {num_units_per_core_group_2};
            compute_g2.config = ComputeConfigDescriptor{};
            desc.kernels.push_back(std::move(compute_g2));
        }
    }

    return desc;
}

}  // namespace ttnn::prim
