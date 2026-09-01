// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_device_operation.hpp"

#include <cmath>
#include <filesystem>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <cstdint>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

namespace m2 = tt::tt_metal::experimental;

namespace {

constexpr const char* KERNEL_READER =
    "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/redistribute_pages_row_major_reader.cpp";
constexpr const char* KERNEL_WRITER =
    "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/redistribute_pages_row_major_writer.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts CopyDeviceOperation::DefaultRowMajor::create_program_artifacts(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input = tensor_args.input;
    const auto& output = output_tensor;

    const auto bytes_per_element = input.element_size();
    const auto elements_per_tensor_row = input.logical_shape()[-1];
    std::uint32_t num_input_pages_in_row = 1;
    std::uint32_t num_output_pages_in_row = 1;
    std::uint32_t elements_per_output_page = output.logical_shape()[-1];
    std::uint32_t elements_per_input_page = input.logical_shape()[-1];

    if (input.is_sharded() && input.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        std::uint32_t input_shard_width =
            (input.shard_spec().has_value() ? input.shard_spec().value().shape[1]
                                            : input.nd_shard_spec().value().shard_shape[-1]);
        num_input_pages_in_row = tt::div_up(elements_per_tensor_row, input_shard_width);
        elements_per_input_page = input_shard_width;
    }
    if (output.is_sharded() && output.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        std::uint32_t output_shard_width =
            (output.shard_spec().has_value() ? output.shard_spec().value().shape[1]
                                             : output.nd_shard_spec().value().shard_shape[-1]);
        num_output_pages_in_row = tt::div_up(elements_per_tensor_row, output_shard_width);
        elements_per_output_page = output_shard_width;
    }

    auto* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    const std::uint32_t total_logical_rows = input.logical_volume() / input.logical_shape()[-1];
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_logical_rows);
    std::vector<CoreCoord> ordered_cores = corerange_to_cores(all_cores, num_cores, true);

    constexpr std::uint32_t MAX_SUBBLOCK_SIZE_BYTES =
        65536 * 4;  // Chosen empirically to prevent large row OOM DFB error
    std::uint32_t input_page_size = input.buffer()->page_size();
    std::uint32_t aligned_output_page_size =
        output.buffer()->aligned_page_size();  // Since we are double buffering, the output page_size must be aligned so
    // the noc_write reads from an aligned address in the DFB
    std::uint32_t input_subblock_size_bytes = elements_per_input_page * bytes_per_element;
    std::uint32_t output_subblock_size_bytes =
        elements_per_output_page *
        bytes_per_element;  // If the input/output row size is not too large, we can just set the subblock to be the
    // page and reduce the number of NoC reads/writes from/to pages.

    if (input_page_size >
        MAX_SUBBLOCK_SIZE_BYTES) {  // If the input/output row size is too large, the page size will be too large for
        // the DFB, so we process data in subblock units of MAX_SUBBLOCK_SIZE_BYTES instead
        input_page_size = MAX_SUBBLOCK_SIZE_BYTES;
        input_subblock_size_bytes = MAX_SUBBLOCK_SIZE_BYTES;
    }
    if (aligned_output_page_size > MAX_SUBBLOCK_SIZE_BYTES) {
        aligned_output_page_size = MAX_SUBBLOCK_SIZE_BYTES;
        output_subblock_size_bytes = MAX_SUBBLOCK_SIZE_BYTES;
    }

    // Dataflow buffer identities and tensor parameters.
    const m2::DFBSpecName INPUT_PAGES{"input_pages"};  // legacy c_0 — reader-private L1 scratchpad (self-loop)
    const m2::DFBSpecName OUTPUT_PAGE{"output_page"};  // legacy c_1 — reader->writer output-page FIFO (double buffered)
    const m2::TensorParamName INPUT{"input"};
    const m2::TensorParamName OUTPUT{"output"};
    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};

    // The DFB that stores input pages (a reader-private scratchpad, single entry).
    const auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    m2::DataflowBufferSpec input_pages_dfb{
        .unique_id = INPUT_PAGES,
        .entry_size = input_page_size,
        .num_entries = 1,
        .data_format_metadata = input_data_format,
    };

    // The DFB that stores output pages. This one is double buffered, since it is shared between the reader
    // and writer kernels.
    const auto output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    m2::DataflowBufferSpec output_page_dfb{
        .unique_id = OUTPUT_PAGE,
        .entry_size = aligned_output_page_size,
        .num_entries = 2,
        .data_format_metadata = output_data_format,
    };

    const auto arch = device->arch();

    m2::KernelSpec reader{
        .unique_id = READER,
        .source = KERNEL_READER,
        // input_pages is touched by the reader alone (fill + drain) — self-loop, one accessor name.
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = INPUT_PAGES,
                    .accessor_name = "in0",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
                m2::DFBBinding{
                    .dfb_spec_name = INPUT_PAGES,
                    .accessor_name = "in0",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
                m2::DFBBinding{
                    .dfb_spec_name = OUTPUT_PAGE,
                    .accessor_name = "in1",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"},
            },
        .compile_time_args =
            {
                {"num_output_pages_in_row", num_output_pages_in_row},
                {"num_input_pages_in_row", num_input_pages_in_row},
                {"elements_per_output_page", elements_per_output_page},
                {"bytes_per_element", bytes_per_element},
                {"elements_per_input_page", elements_per_input_page},
                {"elements_per_tensor_row", elements_per_tensor_row},
                {"bytes_per_input_subblock", input_subblock_size_bytes},
                {"bytes_per_output_subblock", output_subblock_size_bytes},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"start_row", "num_rows_to_process"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = KERNEL_WRITER,
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = OUTPUT_PAGE,
                    .accessor_name = "in1",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"},
            },
        .compile_time_args =
            {
                {"num_output_pages_in_row", num_output_pages_in_row},
                {"elements_per_output_page", elements_per_output_page},
                {"bytes_per_element", bytes_per_element},
                {"elements_per_tensor_row", elements_per_tensor_row},
                {"bytes_per_output_subblock", output_subblock_size_bytes},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"start_row", "num_rows_to_process"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    // Runtime args: each core owns a contiguous span of logical rows.
    m2::KernelRunArgs reader_run_args{.kernel = READER};
    m2::KernelRunArgs writer_run_args{.kernel = WRITER};
    std::uint32_t start_row_id = 0;
    for (const auto& core : ordered_cores) {
        std::uint32_t num_rows_to_process = num_rows_per_core_group_1;
        if (core_group_2.contains(core)) {
            num_rows_to_process = num_rows_per_core_group_2;
        }
        m2::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"start_row", start_row_id}, {"num_rows_to_process", num_rows_to_process}});
        m2::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"start_row", start_row_id}, {"num_rows_to_process", num_rows_to_process}});
        start_row_id += num_rows_to_process;
    }

    m2::ProgramSpec spec{
        .name = "copy_default_row_major",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(input_pages_dfb), std::move(output_page_dfb)},
        .tensor_parameters =
            {
                m2::TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units =
            {
                m2::WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER},
                    .target_nodes = all_cores,
                },
            },
    };

    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args.push_back(std::move(reader_run_args));
    run_params.kernel_run_args.push_back(std::move(writer_run_args));
    run_params.tensor_args.emplace(INPUT, m2::TensorArgument{input.mesh_tensor()});
    run_params.tensor_args.emplace(OUTPUT, m2::TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::prim
