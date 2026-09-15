// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "pack_convolution_carry_program_factory.hpp"

#include <limits>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim {

ttnn::device_operation::ProgramArtifacts PackConvolutionCarryProgramFactory::create_program_artifacts(
    const PackConvolutionCarryParams& attrs, const PackConvolutionCarryInputs& in, std::vector<Tensor>& outputs) {
    namespace m2 = tt::tt_metal::experimental;
    const auto& input = in.input.mesh_tensor();
    const auto& wrap_indicator = in.wrap_indicator.mesh_tensor();
    const auto& output = outputs[0].mesh_tensor();
    const auto& device = input.device();
    const auto arch = device.arch();
    const uint32_t channel_tiles = attrs.channels / tt::constants::TILE_WIDTH;
    auto dist = kda_factory_detail::distribute_prep(
        device.compute_with_storage_grid_size(), channel_tiles, std::numeric_limits<uint32_t>::max());

    const m2::KernelSpecName reader_name{"reader"};
    const m2::KernelSpecName compute_name{"compute"};
    const m2::KernelSpecName writer_name{"writer"};
    const m2::DFBSpecName packed_rm_name{"packed_rm"};
    const m2::DFBSpecName packed_tile_name{"packed_tile"};
    const m2::TensorParamName input_name{"input"};
    const m2::TensorParamName wrap_indicator_name{"wrap_indicator"};
    const m2::TensorParamName output_name{"output"};

    const auto format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_size = tt::tile_size(format);
    auto make_dfb = [format, tile_size](const m2::DFBSpecName& name) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tile_size,
            .num_entries = 2,
            .data_format_metadata = format,
        };
    };

    m2::KernelSpec reader{
        .unique_id = reader_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/"
            "reader_pack_convolution_carry.cpp",
        .dfb_bindings = {m2::DFBBinding{packed_rm_name, "packed_rm", m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings =
            {
                m2::TensorBinding{input_name, "input"},
                m2::TensorBinding{wrap_indicator_name, "wrap_indicator"},
            },
        .compile_time_args = {{"history_rows", attrs.history_rows}},
        .runtime_arg_schema = {.runtime_arg_names = {"ct_start", "ct_count", "sequence", "wrap_row"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    m2::KernelSpec compute{
        .unique_id = compute_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/compute/"
            "pack_convolution_carry.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                m2::DFBBinding{packed_rm_name, "packed_rm", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{packed_tile_name, "packed_tile", m2::DFBEndpointType::PRODUCER},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"ct_count"}},
        .hw_config = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config),
    };

    m2::KernelSpec writer{
        .unique_id = writer_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/"
            "writer_pack_convolution_carry.cpp",
        .dfb_bindings = {m2::DFBBinding{packed_tile_name, "packed_tile", m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {m2::TensorBinding{output_name, "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"ct_start", "ct_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    m2::KernelRunArgs reader_args{.kernel = reader_name};
    m2::KernelRunArgs compute_args{.kernel = compute_name};
    m2::KernelRunArgs writer_args{.kernel = writer_name};
    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        const auto& core = dist.cores[i];
        m2::AddRuntimeArgsForNode(
            reader_args.runtime_arg_values,
            core,
            {{"ct_start", dist.wi_start[i]},
             {"ct_count", dist.wi_count[i]},
             {"sequence", attrs.sequence},
             {"wrap_row", attrs.wrap_row}});
        m2::AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"ct_count", dist.wi_count[i]}});
        m2::AddRuntimeArgsForNode(
            writer_args.runtime_arg_values, core, {{"ct_start", dist.wi_start[i]}, {"ct_count", dist.wi_count[i]}});
    }

    m2::ProgramSpec spec{
        .name = "pack_convolution_carry",
        .kernels = {std::move(reader), std::move(compute), std::move(writer)},
        .dataflow_buffers = {make_dfb(packed_rm_name), make_dfb(packed_tile_name)},
        .tensor_parameters =
            {
                m2::TensorParameter{.unique_id = input_name, .spec = input.tensor_spec()},
                m2::TensorParameter{.unique_id = wrap_indicator_name, .spec = wrap_indicator.tensor_spec()},
                m2::TensorParameter{.unique_id = output_name, .spec = output.tensor_spec()},
            },
        .work_units =
            {
                m2::WorkUnitSpec{
                    .name = "main",
                    .kernels = {reader_name, compute_name, writer_name},
                    .target_nodes = dist.core_set,
                },
            },
    };
    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_args), std::move(compute_args), std::move(writer_args)};
    run_args.tensor_args = {{input_name, input}, {wrap_indicator_name, wrap_indicator}, {output_name, output}};
    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
