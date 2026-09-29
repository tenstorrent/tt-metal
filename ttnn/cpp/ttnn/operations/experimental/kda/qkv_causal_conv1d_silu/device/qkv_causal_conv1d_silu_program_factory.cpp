// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_program_factory.hpp"

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

namespace {

// Kimi-K3 supplies four learned causal-convolution taps; one channel block of weights is queued per tap.
constexpr uint32_t tap_count = 4;

}  // namespace

ttnn::device_operation::ProgramArtifacts QkvCausalConv1dSiluProgramFactory::create_program_artifacts(
    const QkvCausalConv1dSiluParams& attrs, const QkvCausalConv1dSiluInputs& in, std::vector<Tensor>& outputs) {
    const auto& input = in.input.mesh_tensor();
    const auto& history = in.history.mesh_tensor();
    const auto& tap0 = in.tap0.mesh_tensor();
    const auto& tap1 = in.tap1.mesh_tensor();
    const auto& tap2 = in.tap2.mesh_tensor();
    const auto& tap3 = in.tap3.mesh_tensor();
    const auto& q = outputs[0].mesh_tensor();
    const auto& k = outputs[1].mesh_tensor();
    const auto& v = outputs[2].mesh_tensor();
    const auto& device = input.device();
    const auto arch = device.arch();

    const uint32_t Mt = attrs.sequence / tt::constants::TILE_HEIGHT;
    const uint32_t Qt = attrs.q_width / tt::constants::TILE_WIDTH;
    const uint32_t Kt = attrs.k_width / tt::constants::TILE_WIDTH;
    const uint32_t Vt = attrs.v_width / tt::constants::TILE_WIDTH;
    const uint32_t Ct = Qt + Kt + Vt;
    const uint32_t block_ct = attrs.channel_chunk_size / tt::constants::TILE_WIDTH;
    const uint32_t num_blocks = Ct / block_ct;
    auto dist = kda_factory_detail::distribute_prep(
        device.compute_with_storage_grid_size(), Mt * num_blocks, std::numeric_limits<uint32_t>::max());
    const auto& cores = dist.core_set;

    const tt::tt_metal::experimental::KernelSpecName reader_kernel_name{"reader"};
    const tt::tt_metal::experimental::KernelSpecName writer_kernel_name{"writer"};
    const tt::tt_metal::experimental::KernelSpecName compute_kernel_name{"compute"};

    const tt::tt_metal::experimental::DFBSpecName act_rm_dfb_name{"act_rm"};
    const tt::tt_metal::experimental::DFBSpecName act_tile_dfb_name{"act_tile"};
    const tt::tt_metal::experimental::DFBSpecName weights_dfb_name{"weights"};
    const tt::tt_metal::experimental::DFBSpecName partial_dfb_name{"partial"};
    const tt::tt_metal::experimental::DFBSpecName output_dfb_name{"output"};

    const tt::tt_metal::experimental::TensorParamName input_tensor_name{"input"};
    const tt::tt_metal::experimental::TensorParamName history_tensor_name{"history"};
    const tt::tt_metal::experimental::TensorParamName tap0_tensor_name{"tap0"};
    const tt::tt_metal::experimental::TensorParamName tap1_tensor_name{"tap1"};
    const tt::tt_metal::experimental::TensorParamName tap2_tensor_name{"tap2"};
    const tt::tt_metal::experimental::TensorParamName tap3_tensor_name{"tap3"};
    const tt::tt_metal::experimental::TensorParamName q_tensor_name{"q"};
    const tt::tt_metal::experimental::TensorParamName k_tensor_name{"k"};
    const tt::tt_metal::experimental::TensorParamName v_tensor_name{"v"};

    const auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_size = tt::tile_size(input_data_format);
    auto make_dfb = [input_data_format, tile_size](
                        const tt::tt_metal::experimental::DFBSpecName& name, uint32_t tiles) {
        return tt::tt_metal::experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tile_size,
            .num_entries = tiles,
            .data_format_metadata = input_data_format,
        };
    };

    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::DataflowBufferSpec> dfbs = {
        make_dfb(act_rm_dfb_name, 2 * block_ct),
        make_dfb(act_tile_dfb_name, block_ct),
        make_dfb(weights_dfb_name, tap_count * block_ct),
        make_dfb(partial_dfb_name, 2 * block_ct),
        make_dfb(output_dfb_name, 2 * block_ct),
    };

    tt::tt_metal::experimental::KernelSpec reader{
        .unique_id = reader_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/"
            "reader_qkv_causal_conv1d_silu.cpp",
        .dfb_bindings =
            {
                tt::tt_metal::experimental::DFBBinding{
                    act_rm_dfb_name, "act_rm", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    weights_dfb_name, "weights", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                tt::tt_metal::experimental::TensorBinding{input_tensor_name, "input"},
                tt::tt_metal::experimental::TensorBinding{history_tensor_name, "history"},
                tt::tt_metal::experimental::TensorBinding{tap0_tensor_name, "tap0"},
                tt::tt_metal::experimental::TensorBinding{tap1_tensor_name, "tap1"},
                tt::tt_metal::experimental::TensorBinding{tap2_tensor_name, "tap2"},
                tt::tt_metal::experimental::TensorBinding{tap3_tensor_name, "tap3"},
            },
        .compile_time_args = {{"block_ct", block_ct}, {"num_blocks", num_blocks}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    tt::tt_metal::experimental::KernelSpec writer{
        .unique_id = writer_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/"
            "writer_qkv_causal_conv1d_silu.cpp",
        .dfb_bindings = {tt::tt_metal::experimental::DFBBinding{
            output_dfb_name, "output", tt::tt_metal::experimental::DFBEndpointType::CONSUMER}},
        .tensor_bindings =
            {
                tt::tt_metal::experimental::TensorBinding{q_tensor_name, "q"},
                tt::tt_metal::experimental::TensorBinding{k_tensor_name, "k"},
                tt::tt_metal::experimental::TensorBinding{v_tensor_name, "v"},
            },
        .compile_time_args = {{"Qt", Qt}, {"Kt", Kt}, {"Vt", Vt}, {"block_ct", block_ct}, {"num_blocks", num_blocks}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    tt::tt_metal::experimental::KernelSpec compute{
        .unique_id = compute_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/compute/"
            "qkv_causal_conv1d_silu.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                tt::tt_metal::experimental::DFBBinding{
                    act_rm_dfb_name, "act_rm", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    act_tile_dfb_name, "act_tile", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    act_tile_dfb_name, "act_tile", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    weights_dfb_name, "weights", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    partial_dfb_name, "partial", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    partial_dfb_name, "partial", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    output_dfb_name, "output", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
            },
        .compile_time_args = {{"block_ct", block_ct}, {"num_blocks", num_blocks}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_count"}},
        .hw_config = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config),
    };

    tt::tt_metal::experimental::KernelRunArgs reader_run_args{.kernel = reader_kernel_name};
    tt::tt_metal::experimental::KernelRunArgs writer_run_args{.kernel = writer_kernel_name};
    tt::tt_metal::experimental::KernelRunArgs compute_run_args{.kernel = compute_kernel_name};
    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        const auto& core = dist.cores[i];
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"wi_start", dist.wi_start[i]}, {"wi_count", dist.wi_count[i]}});
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"wi_start", dist.wi_start[i]}, {"wi_count", dist.wi_count[i]}});
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values, core, {{"wi_count", dist.wi_count[i]}});
    }

    tt::tt_metal::experimental::ProgramSpec spec{
        .name = "qkv_causal_conv1d_silu",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {
                tt::tt_metal::experimental::TensorParameter{
                    .unique_id = input_tensor_name, .spec = input.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{
                    .unique_id = history_tensor_name, .spec = history.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{.unique_id = tap0_tensor_name, .spec = tap0.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{.unique_id = tap1_tensor_name, .spec = tap1.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{.unique_id = tap2_tensor_name, .spec = tap2.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{.unique_id = tap3_tensor_name, .spec = tap3.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{.unique_id = q_tensor_name, .spec = q.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{.unique_id = k_tensor_name, .spec = k.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{.unique_id = v_tensor_name, .spec = v.tensor_spec()},
            },
        .work_units =
            {
                tt::tt_metal::experimental::WorkUnitSpec{
                    .name = "main",
                    .kernels = {reader_kernel_name, writer_kernel_name, compute_kernel_name},
                    .target_nodes = cores,
                },
            },
    };

    tt::tt_metal::experimental::ProgramRunArgs run_args;
    run_args.kernel_run_args.reserve(3);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args = {
        {input_tensor_name, input},
        {history_tensor_name, history},
        {tap0_tensor_name, tap0},
        {tap1_tensor_name, tap1},
        {tap2_tensor_name, tap2},
        {tap3_tensor_name, tap3},
        {q_tensor_name, q},
        {k_tensor_name, k},
        {v_tensor_name, v},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
