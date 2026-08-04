// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <filesystem>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include "full_program_factory_common.hpp"
#include "full_program_factory_sharded.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::full {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace m2 = tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts FullShardedProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output) {
    auto fill_value = operation_attributes.fill_value;
    DataType dtype{operation_attributes.dtype};

    auto data_format = datatype_to_dataformat_converter(dtype);

    uint32_t tensor_width_in_pages = output.buffer()->shard_spec().tensor2d_shape_in_pages[1];

    std::vector<CoreCoord> runtime_cores = get_optimal_worker_cores_for_sharded_tensor(output);
    const auto& compute_core_range = CoreRangeSet(ttsl::Span<const CoreCoord>(runtime_cores));

    const auto& aligned_page_size = output.buffer()->aligned_page_size();
    const auto& page_size = output.buffer()->page_size();

    const m2::DFBSpecName FILL_VALUE{"fill_value"};
    const m2::TensorParamName OUTPUT{"output"};
    const m2::KernelSpecName WRITER{"writer"};

    // One page, sized to the output's page: the kernel builds a single filled page and writes it to
    // every page of its shard.
    m2::DataflowBufferSpec fill_value_dfb{
        .unique_id = FILL_VALUE,
        .entry_size = static_cast<uint32_t>(page_size),
        .num_entries = 1,
        .data_format_metadata = data_format,
    };

    // Exactly one OUTPUT_DTYPE_* define reaches the kernel. Without it the fill loop compiles out
    // entirely and the buffer holds whatever was already in SRAM.
    const m2::KernelSpec::CompilerOptions::Defines writer_defines(get_writer_defines(dtype));
    auto u = encode_fill_value(fill_value, dtype);

    uint32_t elems_per_page = page_size / datum_size(data_format);

    m2::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/full/device/kernels/writer_full_sharded.cpp"},
        .compiler_options = {.defines = writer_defines},
        // The one instance that touches this buffer both fills the page (reserve_back / push_back) and
        // drains it (wait_front / pop_front), so it holds both endpoints under one accessor name.
        .dfb_bindings =
            {m2::DFBBinding{
                 .dfb_spec_name = FILL_VALUE,
                 .accessor_name = "value",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER,
             },
             m2::DFBBinding{
                 .dfb_spec_name = FILL_VALUE,
                 .accessor_name = "value",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER,
             }},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args =
            {{"elems_per_page", elems_per_page},
             {"page_size", static_cast<uint32_t>(page_size)},
             {"aligned_page_size", static_cast<uint32_t>(aligned_page_size)},
             {"tensor_width_in_pages", tensor_width_in_pages}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"fill_value", "start_page_id", "num_pages_per_shard_row", "num_pages_per_shard_col"}},
        .hw_config = ttnn::create_writer_datamovement_config(operation_attributes.mesh_device->arch()),
    };

    uint32_t shard_height_in_pages = output.buffer()->shard_spec().shape_in_pages()[0];
    uint32_t shard_width_in_pages = output.buffer()->shard_spec().shape_in_pages()[1];
    uint32_t tensor_height_in_pages = output.buffer()->shard_spec().tensor2d_shape_in_pages[0];
    uint32_t num_shards_across_width = tt::div_up(tensor_width_in_pages, shard_width_in_pages);
    uint32_t num_shards_across_height = tt::div_up(tensor_height_in_pages, shard_height_in_pages);

    m2::KernelRunArgs writer_run_args{.kernel = WRITER};

    for (uint32_t i = 0; i < runtime_cores.size(); i++) {
        const auto& core = runtime_cores[i];

        uint32_t shard_row_idx = i / num_shards_across_width;
        uint32_t shard_col_idx = i % num_shards_across_width;

        uint32_t first_page_id =
            (shard_row_idx * shard_height_in_pages * tensor_width_in_pages) + (shard_col_idx * shard_width_in_pages);

        uint32_t valid_pages_width = (shard_col_idx == num_shards_across_width - 1)
                                         ? (tensor_width_in_pages - (shard_col_idx * shard_width_in_pages))
                                         : shard_width_in_pages;

        uint32_t valid_pages_height = (shard_row_idx == num_shards_across_height - 1)
                                          ? (tensor_height_in_pages - (shard_row_idx * shard_height_in_pages))
                                          : shard_height_in_pages;
        m2::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"fill_value", u.u32},
             {"start_page_id", first_page_id},
             {"num_pages_per_shard_row", valid_pages_width},
             {"num_pages_per_shard_col", valid_pages_height}});
    }

    m2::ProgramSpec spec{
        .name = "full_sharded",
        .kernels = {std::move(writer_spec)},
        .dataflow_buffers = {std::move(fill_value_dfb)},
        .tensor_parameters = {m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()}},
        .work_units = {m2::WorkUnitSpec{
            .name = "main",
            .kernels = {WRITER},
            .target_nodes = compute_core_range,
        }},
    };

    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args.push_back(std::move(writer_run_args));
    run_params.tensor_args.emplace(OUTPUT, m2::TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::operations::full
