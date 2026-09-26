// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include "full_program_factory_common.hpp"
#include "full_program_factory_interleaved.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::full {

namespace m2 = tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts FullInterleavedProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    [[maybe_unused]] const tensor_args_t&,
    tensor_return_value_t& output) {
    auto dtype = operation_attributes.dtype;
    auto fill_value = operation_attributes.fill_value;

    auto grid = operation_attributes.mesh_device->compute_with_storage_grid_size();
    auto num_pages = (uint32_t)output.buffer()->num_pages();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(grid, num_pages);

    uint32_t page_size = output.buffer()->page_size();
    TT_FATAL(page_size % output.element_size() == 0, "Page size must be divisible by element size");
    uint32_t elems_per_page = page_size / output.element_size();

    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(dtype);

    // Two instances of one kernel source split the output pages between the two data-movement RISCs.
    // Each instance owns a *private* single-entry buffer for its own copy of the fill-value page, so
    // neither instance is an endpoint of the other's buffer.
    const m2::DFBSpecName FILL_VALUE_WRITER{"fill_value_writer"};
    const m2::DFBSpecName FILL_VALUE_READER{"fill_value_reader"};
    const m2::TensorParamName OUTPUT{"output"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName READER{"reader"};

    const std::filesystem::path kernel_source{"ttnn/cpp/ttnn/operations/full/device/kernels/writer_full.cpp"};

    // One page, sized to the output's page: the kernel builds a single filled page and writes it to
    // every output page it owns.
    auto make_fill_value_dfb = [&](const m2::DFBSpecName& name) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = page_size,
            .num_entries = 1,
            .data_format_metadata = data_format,
        };
    };

    // Both endpoints of a fill-value buffer land on the single instance that touches it: that instance
    // fills the page (reserve_back / push_back) and drains it (wait_front / pop_front) itself. One
    // accessor name gives the kernel one DataflowBuffer object driving both directions.
    auto bind_fill_value_dfb = [](const m2::DFBSpecName& name) {
        return m2::Group<m2::DFBBinding>{
            m2::DFBBinding{
                .dfb_spec_name = name,
                .accessor_name = "value",
                .endpoint_type = m2::DFBEndpointType::PRODUCER,
            },
            m2::DFBBinding{
                .dfb_spec_name = name,
                .accessor_name = "value",
                .endpoint_type = m2::DFBEndpointType::CONSUMER,
            },
        };
    };

    // Exactly one OUTPUT_DTYPE_* define reaches the kernel. Without it the fill loop compiles out
    // entirely and the buffer holds whatever was already in SRAM.
    const m2::KernelSpec::CompilerOptions::Defines writer_defines(get_writer_defines(dtype));
    auto u = encode_fill_value(fill_value, dtype);

    const m2::KernelSpec::CompileTimeArgs compile_time_args{
        {"elems_per_page", elems_per_page},
        {"page_size", page_size},
    };
    const m2::KernelSpec::RuntimeArgSchema runtime_arg_schema{
        .runtime_arg_names = {"fill_value", "num_pages_per_core", "start_id"},
    };

    m2::Group<m2::KernelSpec> kernels;
    m2::Group<m2::DataflowBufferSpec> dataflow_buffers;
    m2::Group<m2::KernelSpecName> work_unit_kernels;

    kernels.push_back(m2::KernelSpec{
        .unique_id = WRITER,
        .source = kernel_source,
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings = bind_fill_value_dfb(FILL_VALUE_WRITER),
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args = compile_time_args,
        .runtime_arg_schema = runtime_arg_schema,
        .hw_config = ttnn::create_writer_datamovement_config(),
    });
    dataflow_buffers.push_back(make_fill_value_dfb(FILL_VALUE_WRITER));
    work_unit_kernels.push_back(WRITER);

    auto cores = corerange_to_cores(all_cores, std::nullopt);

    const bool has_reader = num_pages > num_cores;
    if (has_reader) {
        kernels.push_back(m2::KernelSpec{
            .unique_id = READER,
            .source = kernel_source,
            .compiler_options = {.defines = writer_defines},
            .dfb_bindings = bind_fill_value_dfb(FILL_VALUE_READER),
            .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
            .compile_time_args = compile_time_args,
            .runtime_arg_schema = runtime_arg_schema,
            .hw_config = ttnn::create_reader_datamovement_config(),
        });
        dataflow_buffers.push_back(make_fill_value_dfb(FILL_VALUE_READER));
        work_unit_kernels.push_back(READER);
    }

    m2::KernelRunArgs writer_run_args{.kernel = WRITER};
    m2::KernelRunArgs reader_run_args{.kernel = READER};

    uint32_t page_offset = 0;

    for (const auto& core : cores) {
        uint32_t num_pages_per_core;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        if (has_reader) {
            uint32_t reader_page_start = page_offset;
            uint32_t num_pages_per_reader = num_pages_per_core / 2;
            m2::AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"fill_value", u.u32}, {"num_pages_per_core", num_pages_per_reader}, {"start_id", reader_page_start}});

            uint32_t writer_page_start = reader_page_start + num_pages_per_reader;
            uint32_t num_pages_per_writer = num_pages_per_core - num_pages_per_reader;
            m2::AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"fill_value", u.u32}, {"num_pages_per_core", num_pages_per_writer}, {"start_id", writer_page_start}});
        } else {
            m2::AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"fill_value", u.u32}, {"num_pages_per_core", num_pages_per_core}, {"start_id", page_offset}});
        }
        page_offset += num_pages_per_core;
    }

    m2::ProgramSpec spec{
        .name = "full_interleaved",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = {m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()}},
        .work_units = {m2::WorkUnitSpec{
            .name = "main",
            .kernels = std::move(work_unit_kernels),
            .target_nodes = all_cores,
        }},
    };

    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args.push_back(std::move(writer_run_args));
    if (has_reader) {
        run_params.kernel_run_args.push_back(std::move(reader_run_args));
    }
    run_params.tensor_args.emplace(OUTPUT, m2::TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::operations::full
