// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_device_operation.hpp"

#include <cmath>
#include <filesystem>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
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

// The interleaved reader/writer are borrowed from eltwise/unary; bind their pre-existing Metal 2.0
// forks (do not re-fork). The compute kernel is the in-family sharded eltwise_copy; this port created
// its Metal 2.0 fork beside the original.
constexpr const char* KERNEL_READER_INTERLEAVED =
    "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp";
constexpr const char* KERNEL_WRITER_INTERLEAVED =
    "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp";
constexpr const char* KERNEL_COMPUTE_ELTWISE_COPY =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy_metal2.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts CopyDeviceOperation::DefaultTilized::create_program_artifacts(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input = tensor_args.input;
    const auto& output = output_tensor;

    auto* device = input.device();
    auto compute_with_storage_grid_size =
        device->compute_with_storage_grid_size();  // This can be replaced with get_worker_cores in subdevices

    const auto& logical_shape = input.logical_shape();
    const auto& tile = input.tensor_spec().tile();
    const std::uint32_t tile_height = tile.get_height();
    const std::uint32_t tile_width = tile.get_width();
    const std::uint32_t rank = logical_shape.rank();
    std::uint32_t total_tiles = 1;
    if (rank >= 1) {
        total_tiles = tt::div_up(logical_shape[-1], tile_width);
    }
    if (rank >= 2) {
        total_tiles *= tt::div_up(logical_shape[-2], tile_height);
    }
    for (std::uint32_t i = 0; i + 2 < rank; ++i) {
        total_tiles *= logical_shape[i];
    }
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_tiles);
    std::vector<CoreCoord> ordered_cores = corerange_to_cores(all_cores, num_cores, true);

    const auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const bool convert_df = input_data_format != output_data_format;

    // Dataflow buffer identities and tensor parameters.
    const m2::DFBSpecName IN{"in"};    // legacy c_0 — input pages (double buffered)
    const m2::DFBSpecName OUT{"out"};  // legacy c_16 — output pages (only when converting data format)
    const m2::TensorParamName INPUT{"input"};
    const m2::TensorParamName OUTPUT{"output"};
    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};

    const auto aligned_input_page_size = input.buffer()->aligned_page_size();
    m2::DataflowBufferSpec in_dfb{
        .unique_id = IN,
        .entry_size = aligned_input_page_size,
        .num_entries = 2,
        .data_format_metadata = input_data_format,
    };

    // When converting data formats through the compute kernel, output pages land in a separate DFB.
    // Double buffered, and the output page_size is aligned so the noc_write reads from an aligned
    // address in the DFB.
    const auto aligned_output_page_size = output.buffer()->aligned_page_size();
    m2::DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = aligned_output_page_size,
        .num_entries = 2,
        .data_format_metadata = output_data_format,
    };

    // The writer drains the output-format DFB: the compute-produced OUT when converting, otherwise the
    // reader-produced IN directly.
    const m2::DFBSpecName& writer_dfb = convert_df ? OUT : IN;

    m2::KernelSpec reader{
        .unique_id = READER,
        .source = KERNEL_READER_INTERLEAVED,
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = IN,
                    .accessor_name = "in",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_pages", "start_id"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = KERNEL_WRITER_INTERLEAVED,
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = writer_dfb,
                    .accessor_name = "out",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_pages", "start_id"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    // Compute kernel (only present when converting data format): consumes IN, produces OUT.
    // ComputeGen1Config defaults match the legacy ComputeConfigDescriptor{} defaults; opt_level is set
    // to O3 explicitly (legacy compute defaults to O3; Metal 2.0 CompilerOptions defaults to O2).
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = KERNEL_COMPUTE_ELTWISE_COPY,
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = IN,
                    .accessor_name = "in",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
                m2::DFBBinding{
                    .dfb_spec_name = OUT,
                    .accessor_name = "out",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"per_core_tile_cnt"},
            },
        .hw_config = m2::ComputeHardwareConfig{},
    };

    // Runtime args: each core owns a contiguous span of tiles.
    m2::KernelRunArgs reader_run_args{.kernel = READER};
    m2::KernelRunArgs writer_run_args{.kernel = WRITER};
    m2::KernelRunArgs compute_run_args{.kernel = COMPUTE};
    std::uint32_t start_tile_id = 0;
    for (const auto& core : ordered_cores) {
        std::uint32_t num_tiles_to_process = num_tiles_per_core_group_1;
        if (core_group_2.contains(core)) {
            num_tiles_to_process = num_tiles_per_core_group_2;
        }
        m2::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_pages", num_tiles_to_process}, {"start_id", start_tile_id}});
        m2::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_pages", num_tiles_to_process}, {"start_id", start_tile_id}});
        if (convert_df) {
            m2::AddRuntimeArgsForNode(
                compute_run_args.runtime_arg_values, core, {{"per_core_tile_cnt", num_tiles_to_process}});
        }
        start_tile_id += num_tiles_to_process;
    }

    m2::Group<m2::KernelSpec> kernels{std::move(reader), std::move(writer)};
    m2::Group<m2::DataflowBufferSpec> dataflow_buffers{std::move(in_dfb)};
    m2::Group<m2::KernelSpecName> work_unit_kernels{READER, WRITER};
    if (convert_df) {
        kernels.push_back(std::move(compute));
        dataflow_buffers.push_back(std::move(out_dfb));
        work_unit_kernels.push_back(COMPUTE);
    }

    m2::ProgramSpec spec{
        .name = "copy_default_tilized",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                m2::TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units =
            {
                m2::WorkUnitSpec{
                    .name = "main",
                    .kernels = std::move(work_unit_kernels),
                    .target_nodes = all_cores,
                },
            },
    };

    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args.push_back(std::move(reader_run_args));
    run_params.kernel_run_args.push_back(std::move(writer_run_args));
    if (convert_df) {
        run_params.kernel_run_args.push_back(std::move(compute_run_args));
    }
    run_params.tensor_args.emplace(INPUT, m2::TensorArgument{input.mesh_tensor()});
    run_params.tensor_args.emplace(OUTPUT, m2::TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::prim
