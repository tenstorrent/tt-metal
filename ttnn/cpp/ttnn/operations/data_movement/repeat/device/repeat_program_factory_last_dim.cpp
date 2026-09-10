// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat/device/repeat_program_factory_last_dim.hpp"

#include <cstdint>
#include <filesystem>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/repeat/device/repeat_program_factory_common.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts RepeatProgramFactoryLastDim::create_program_artifacts(
    const RepeatParams& operation_attributes, const RepeatInputs& tensor_args, Tensor& tensor_return_value) {
    // We are repeating the last dim on a 2D shape
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const uint32_t num_repeats = operation_attributes.m_num_repeats;
    // get datum size
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t data_size = input.element_size();
    IDevice* device = input.device();
    // Multi device pre-computation
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_cores_total = num_cores_x * num_cores_y;
    const CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    const CoreRangeSet total_core_ranges{total_cores};

    ttnn::Shape input_log_shape = ttnn::Shape(input.logical_shape().view());
    ttnn::Shape output_log_shape = ttnn::Shape(output.logical_shape().view());
    const uint32_t source_page_size_bytes = input_log_shape[-1] * data_size;
    const uint32_t dest_page_size_bytes = source_page_size_bytes * num_repeats;
    TT_FATAL(
        dest_page_size_bytes == output_log_shape[-1] * data_size,
        "Data size of output does not match requirement for repeat last dim");
    uint32_t read_start_page = 0;
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    // Per-core page count so read/write start on page boundaries.
    const uint32_t number_of_pages = input_log_shape[-2];
    const uint32_t responsibility = ((number_of_pages - 1) / num_cores_total) + 1;
    const uint32_t cb_size_bytes = (READ_ALIGNMENT * 2) + ((source_page_size_bytes & 0xF) == 0 ? source_page_size_bytes
                                   : (source_page_size_bytes & 0x7) == 0                       ? source_page_size_bytes * 2
                                   : (source_page_size_bytes & 0x3) == 0                       ? source_page_size_bytes * 4
                                   : (source_page_size_bytes & 0x1) == 0                       ? source_page_size_bytes * 8
                                                                           : source_page_size_bytes * 16);

    // RM sharded -> rm_sharded; RM interleaved -> rm_interleaved (TILE uses higher-dim factory).
    const bool src_sharded = src_buffer->buffer_distribution_spec().has_value();
    const bool dst_sharded = dst_buffer->buffer_distribution_spec().has_value();
    const bool needs_alignment_cb = !src_sharded && !dst_sharded;

    // Metal 2.0 named resource ids. Declared function-local so the unity build (both repeat factory
    // .cpp files land in one translation unit) sees no duplicate anonymous-namespace symbols.
    const KernelSpecName READER{"reader"};
    const DFBSpecName SRC0{"src0"};
    const DFBSpecName SRC1{"src1"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    // Dataflow buffers: one page each. Each is a single-toucher scratchpad the reader fills and drains
    // itself, so it self-loops (reader bound PRODUCER + CONSUMER).
    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SRC0,
        .entry_size = cb_size_bytes,
        .num_entries = 1,
        .data_format_metadata = cb_data_format,
    });
    // Second buffer only for interleaved RM.
    if (needs_alignment_cb) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = SRC1,
            .entry_size = cb_size_bytes,
            .num_entries = 1,
            .data_format_metadata = cb_data_format,
        });
    }

    // Self-loop the DFBs: bind the reader as both PRODUCER and CONSUMER of each (one accessor name each).
    Group<DFBBinding> dfb_bindings = {
        DFBBinding{.dfb_spec_name = SRC0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = SRC0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER}};
    if (needs_alignment_cb) {
        dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = SRC1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER});
        dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = SRC1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER});
    }

    std::filesystem::path kernel_source;
    if (src_sharded || dst_sharded) {
        kernel_source = "ttnn/cpp/ttnn/operations/data_movement/repeat/device/kernels/repeat_last_dim_rm_sharded.cpp";
    } else {
        kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/repeat/device/kernels/repeat_last_dim_rm_interleaved.cpp";
    }

    KernelSpec reader{
        .unique_id = READER,
        .source = kernel_source,
        .dfb_bindings = dfb_bindings,
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"},
             TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .compile_time_args = {{"original_page_size_bytes", source_page_size_bytes}, {"num_repeats", num_repeats}},
        .runtime_arg_schema = {.runtime_arg_names = {"page_start", "page_end", "nop"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    uint32_t done = 0;
    for (uint32_t core_x = 0; core_x < num_cores_x; core_x++) {
        for (uint32_t core_y = 0; core_y < num_cores_y; core_y++) {
            const CoreCoord core = {core_x, core_y};
            if (done == 1) {
                // Idle core: early exit.
                AddRuntimeArgsForNode(
                    reader_run_args.runtime_arg_values,
                    core,
                    {{"page_start", uint32_t{0}}, {"page_end", uint32_t{0}}, {"nop", uint32_t{1}}});
            } else {
                const uint32_t start_of_read = read_start_page;
                uint32_t end_of_read = read_start_page + responsibility;
                end_of_read = end_of_read < number_of_pages ? end_of_read : number_of_pages;

                AddRuntimeArgsForNode(
                    reader_run_args.runtime_arg_values,
                    core,
                    {{"page_start", start_of_read}, {"page_end", end_of_read}, {"nop", uint32_t{0}}});
                read_start_page = end_of_read;
                done = (end_of_read == input_log_shape[-2]) ? 1 : 0;
            }
        }
    }

    ProgramSpec spec{
        .name = "repeat_last_dim",
        .kernels = {reader},
        .dataflow_buffers = dataflow_buffers,
        .tensor_parameters =
            {TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
             TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()}},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {READER}, .target_nodes = total_core_ranges}},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args)};
    run_args.tensor_args = {{INPUT, input.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
