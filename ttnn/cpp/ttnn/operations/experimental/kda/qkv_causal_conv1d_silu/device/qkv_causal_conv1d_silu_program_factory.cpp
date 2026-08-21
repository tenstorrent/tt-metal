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

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace m2 = tt::tt_metal::experimental;

namespace {

uint32_t choose_channel_block_tiles(uint32_t channel_tiles) {
    constexpr uint32_t max_single_block_channel_tiles = 48;
    constexpr uint32_t wide_channel_block_tiles = 24;
    uint32_t block_tiles = channel_tiles <= max_single_block_channel_tiles ? channel_tiles : wide_channel_block_tiles;
    while (channel_tiles % block_tiles != 0) {
        --block_tiles;
    }
    return block_tiles;
}

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

    const uint32_t Mt = attrs.sequence / TILE_HEIGHT;
    const uint32_t Qt = attrs.q_width / TILE_WIDTH;
    const uint32_t Kt = attrs.k_width / TILE_WIDTH;
    const uint32_t Vt = attrs.v_width / TILE_WIDTH;
    const uint32_t Ct = Qt + Kt + Vt;
    const uint32_t block_ct = choose_channel_block_tiles(Ct);
    const uint32_t num_blocks = Ct / block_ct;
    auto dist = kda_factory_detail::distribute_prep(
        device.compute_with_storage_grid_size(), Mt * num_blocks, std::numeric_limits<uint32_t>::max());
    const auto& cores = dist.core_set;

    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};

    const m2::DFBSpecName ACT_RM_DFB{"act_rm"};
    const m2::DFBSpecName ACT_TILE_DFB{"act_tile"};
    const m2::DFBSpecName WEIGHTS_DFB{"weights"};
    const m2::DFBSpecName PARTIAL_DFB{"partial"};
    const m2::DFBSpecName OUTPUT_DFB{"output"};

    const m2::TensorParamName INPUT{"input"};
    const m2::TensorParamName HISTORY{"history"};
    const m2::TensorParamName TAP0{"tap0"};
    const m2::TensorParamName TAP1{"tap1"};
    const m2::TensorParamName TAP2{"tap2"};
    const m2::TensorParamName TAP3{"tap3"};
    const m2::TensorParamName Q{"q"};
    const m2::TensorParamName K{"k"};
    const m2::TensorParamName V{"v"};

    const auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_size = tt::tile_size(input_data_format);
    auto make_dfb = [input_data_format, tile_size](const m2::DFBSpecName& name, uint32_t tiles) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tile_size,
            .num_entries = tiles,
            .data_format_metadata = input_data_format,
        };
    };

    m2::Group<m2::DataflowBufferSpec> dfbs = {
        make_dfb(ACT_RM_DFB, 2 * block_ct),
        make_dfb(ACT_TILE_DFB, block_ct),
        make_dfb(WEIGHTS_DFB, 4 * block_ct),
        make_dfb(PARTIAL_DFB, 2 * block_ct),
        make_dfb(OUTPUT_DFB, 2 * block_ct),
    };

    m2::KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/"
            "reader_qkv_causal_conv1d_silu.cpp",
        .dfb_bindings =
            {
                m2::DFBBinding{ACT_RM_DFB, "act_rm", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{WEIGHTS_DFB, "weights", m2::DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                m2::TensorBinding{INPUT, "input"},
                m2::TensorBinding{HISTORY, "history"},
                m2::TensorBinding{TAP0, "tap0"},
                m2::TensorBinding{TAP1, "tap1"},
                m2::TensorBinding{TAP2, "tap2"},
                m2::TensorBinding{TAP3, "tap3"},
            },
        .compile_time_args = {{"block_ct", block_ct}, {"num_blocks", num_blocks}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/"
            "writer_qkv_causal_conv1d_silu.cpp",
        .dfb_bindings = {m2::DFBBinding{OUTPUT_DFB, "output", m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings =
            {
                m2::TensorBinding{Q, "q"},
                m2::TensorBinding{K, "k"},
                m2::TensorBinding{V, "v"},
            },
        .compile_time_args = {{"Qt", Qt}, {"Kt", Kt}, {"Vt", Vt}, {"block_ct", block_ct}, {"num_blocks", num_blocks}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/compute/"
            "qkv_causal_conv1d_silu.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                m2::DFBBinding{ACT_RM_DFB, "act_rm", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{ACT_TILE_DFB, "act_tile", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{ACT_TILE_DFB, "act_tile", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{WEIGHTS_DFB, "weights", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{PARTIAL_DFB, "partial", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{PARTIAL_DFB, "partial", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{OUTPUT_DFB, "output", m2::DFBEndpointType::PRODUCER},
            },
        .compile_time_args = {{"block_ct", block_ct}, {"num_blocks", num_blocks}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_count"}},
        .hw_config = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config),
    };

    m2::KernelRunArgs reader_run_args{.kernel = READER};
    m2::KernelRunArgs writer_run_args{.kernel = WRITER};
    m2::KernelRunArgs compute_run_args{.kernel = COMPUTE};
    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        const auto& core = dist.cores[i];
        m2::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"wi_start", dist.wi_start[i]}, {"wi_count", dist.wi_count[i]}});
        m2::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"wi_start", dist.wi_start[i]}, {"wi_count", dist.wi_count[i]}});
        m2::AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"wi_count", dist.wi_count[i]}});
    }

    m2::ProgramSpec spec{
        .name = "qkv_causal_conv1d_silu",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {
                m2::TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                m2::TensorParameter{.unique_id = HISTORY, .spec = history.tensor_spec()},
                m2::TensorParameter{.unique_id = TAP0, .spec = tap0.tensor_spec()},
                m2::TensorParameter{.unique_id = TAP1, .spec = tap1.tensor_spec()},
                m2::TensorParameter{.unique_id = TAP2, .spec = tap2.tensor_spec()},
                m2::TensorParameter{.unique_id = TAP3, .spec = tap3.tensor_spec()},
                m2::TensorParameter{.unique_id = Q, .spec = q.tensor_spec()},
                m2::TensorParameter{.unique_id = K, .spec = k.tensor_spec()},
                m2::TensorParameter{.unique_id = V, .spec = v.tensor_spec()},
            },
        .work_units =
            {
                m2::WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER, COMPUTE},
                    .target_nodes = cores,
                },
            },
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args.reserve(3);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args = {
        {INPUT, input},
        {HISTORY, history},
        {TAP0, tap0},
        {TAP1, tap1},
        {TAP2, tap2},
        {TAP3, tap3},
        {Q, q},
        {K, k},
        {V, v},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
