// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_program_factory.hpp"

#include "ttnn/operations/experimental/kda/factory/chronology_binding.hpp"
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

ttnn::device_operation::MeshWorkloadArtifacts QkvCausalConv1dSiluTiledProgramFactory::create_mesh_workload_artifacts(
    const QkvCausalConv1dSiluParams& attrs,
    const QkvCausalConv1dSiluInputs& in,
    std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    using namespace tt::tt_metal::experimental;
    const auto& input = in.input.mesh_tensor();
    const auto& device = input.device();
    const bool has_table = in.history_table.has_value();
    const auto& predecessor = has_table ? in.history_table->mesh_tensor() : in.predecessor_carry->mesh_tensor();

    const uint32_t Mt = attrs.sequence / tt::constants::TILE_HEIGHT;
    const uint32_t Qt = attrs.q_width / tt::constants::TILE_WIDTH;
    const uint32_t Kt = attrs.k_width / tt::constants::TILE_WIDTH;
    const uint32_t Vt = attrs.v_width / tt::constants::TILE_WIDTH;
    const uint32_t Ct = Qt + Kt + Vt;
    const uint32_t block_ct = attrs.channel_chunk_size / tt::constants::TILE_WIDTH;
    const uint32_t num_blocks = Ct / block_ct;
    const uint32_t group = qkv_causal_conv1d_silu_tiled_group(block_ct);
    const uint32_t input_wt = input.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t input_ct_offset = attrs.input_column_offset / tt::constants::TILE_WIDTH;
    auto dist = kda_factory_detail::distribute_prep(
        device.compute_with_storage_grid_size(), Mt * num_blocks, std::numeric_limits<uint32_t>::max());
    const auto& cores = dist.core_set;

    const KernelSpecName reader_name{"reader"};
    const KernelSpecName writer_name{"writer"};
    const KernelSpecName compute_name{"compute"};
    const DFBSpecName act_name{"act_tile"};
    const DFBSpecName weights_name{"weights"};
    const DFBSpecName taps_name{"taps"};
    const DFBSpecName eye_name{"eye"};
    const DFBSpecName output_name{"output"};
    const ScratchpadSpecName scratch_name{"act_window"};

    const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_size = tt::tile_size(data_format);
    auto make_dfb = [data_format, tile_size](const DFBSpecName& name, uint32_t tiles) {
        return DataflowBufferSpec{
            .unique_id = name, .entry_size = tile_size, .num_entries = tiles, .data_format_metadata = data_format};
    };
    Group<DataflowBufferSpec> dfbs = {
        make_dfb(act_name, 2 * qkv_causal_conv1d_silu_tap_count * group),
        make_dfb(weights_name, qkv_causal_conv1d_silu_tap_count * block_ct),
        make_dfb(taps_name, qkv_causal_conv1d_silu_tap_count * block_ct),
        make_dfb(eye_name, 1),
        make_dfb(output_name, 2 * group),
    };

    const TensorParamName input_tensor{"input"};
    const TensorParamName history_tensor{"history"};
    const TensorParamName predecessor_tensor{"predecessor_carry"};
    const TensorParamName tap_tensors[4] = {
        TensorParamName{"tap0"}, TensorParamName{"tap1"}, TensorParamName{"tap2"}, TensorParamName{"tap3"}};
    const TensorParamName q_tensor{"q"};
    const TensorParamName k_tensor{"k"};
    const TensorParamName v_tensor{"v"};
    const TensorParamName actual_start_tensor{"actual_start"};
    const TensorParamName final_history_tensor{"final_history"};

    KernelSpec reader{
        .unique_id = reader_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/"
            "reader_qkv_causal_conv1d_silu_tiled.cpp",
        .dfb_bindings =
            {
                DFBBinding{act_name, "act_tile", DFBEndpointType::PRODUCER},
                DFBBinding{taps_name, "taps", DFBEndpointType::PRODUCER},
                DFBBinding{eye_name, "eye", DFBEndpointType::PRODUCER},
            },
        .scratchpad_bindings = {{scratch_name, "act_window"}},
        .tensor_bindings =
            {
                TensorBinding{input_tensor, "input"},
                TensorBinding{history_tensor, "history"},
                TensorBinding{predecessor_tensor, "predecessor_carry"},
                TensorBinding{tap_tensors[0], "tap0"},
                TensorBinding{tap_tensors[1], "tap1"},
                TensorBinding{tap_tensors[2], "tap2"},
                TensorBinding{tap_tensors[3], "tap3"},
                TensorBinding{actual_start_tensor, "actual_start"},
            },
        .compile_time_args =
            {{"block_ct", block_ct},
             {"Mt", Mt},
             {"group", group},
             {"input_wt", input_wt},
             {"input_ct_offset", input_ct_offset},
             {"has_table", uint32_t(has_table)}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    KernelSpec writer{
        .unique_id = writer_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/"
            "writer_qkv_causal_conv1d_silu.cpp",
        .dfb_bindings = {DFBBinding{output_name, "output", DFBEndpointType::CONSUMER}},
        .tensor_bindings =
            {
                TensorBinding{q_tensor, "q"},
                TensorBinding{k_tensor, "k"},
                TensorBinding{v_tensor, "v"},
            },
        .compile_time_args = {{"Qt", Qt}, {"Kt", Kt}, {"Vt", Vt}, {"block_ct", block_ct}, {"Mt", Mt}, {"group", group}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    KernelSpec compute{
        .unique_id = compute_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/compute/"
            "qkv_causal_conv1d_silu_tiled.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{act_name, "act_tile", DFBEndpointType::CONSUMER},
                DFBBinding{taps_name, "taps", DFBEndpointType::CONSUMER},
                DFBBinding{eye_name, "eye", DFBEndpointType::CONSUMER},
                DFBBinding{weights_name, "weights", DFBEndpointType::PRODUCER},
                DFBBinding{weights_name, "weights", DFBEndpointType::CONSUMER},
                DFBBinding{output_name, "output", DFBEndpointType::PRODUCER},
            },
        .compile_time_args = {{"block_ct", block_ct}, {"Mt", Mt}, {"group", group}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = ttnn::to_compute_hardware_config(attrs.compute_kernel_config),
    };

    KernelRunArgs reader_run_args{.kernel = reader_name};
    KernelRunArgs writer_run_args{.kernel = writer_name};
    KernelRunArgs compute_run_args{.kernel = compute_name};
    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        const auto& core = dist.cores[i];
        for (auto* args : {&reader_run_args, &writer_run_args, &compute_run_args}) {
            AddRuntimeArgsForNode(
                args->runtime_arg_values, core, {{"wi_start", dist.wi_start[i]}, {"wi_count", dist.wi_count[i]}});
        }
    }

    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = input_tensor, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = history_tensor, .spec = in.history.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = predecessor_tensor, .spec = predecessor.tensor_spec()},
        TensorParameter{.unique_id = tap_tensors[0], .spec = in.tap0.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = tap_tensors[1], .spec = in.tap1.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = tap_tensors[2], .spec = in.tap2.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = tap_tensors[3], .spec = in.tap3.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = q_tensor, .spec = outputs[0].mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = k_tensor, .spec = outputs[1].mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = v_tensor, .spec = outputs[2].mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = actual_start_tensor, .spec = in.actual_start.mesh_tensor().tensor_spec()},
    };

    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {input_tensor, input},
        {history_tensor, in.history.mesh_tensor()},
        {predecessor_tensor, predecessor},
        {tap_tensors[0], in.tap0.mesh_tensor()},
        {tap_tensors[1], in.tap1.mesh_tensor()},
        {tap_tensors[2], in.tap2.mesh_tensor()},
        {tap_tensors[3], in.tap3.mesh_tensor()},
        {q_tensor, outputs[0].mesh_tensor()},
        {k_tensor, outputs[1].mesh_tensor()},
        {v_tensor, outputs[2].mesh_tensor()},
        {actual_start_tensor, in.actual_start.mesh_tensor()},
    };
    if (has_table) {
        const auto& final_history = outputs[3].mesh_tensor();
        tensor_parameters.push_back({.unique_id = final_history_tensor, .spec = final_history.tensor_spec()});
        run_args.tensor_args.emplace(final_history_tensor, final_history);
        reader.tensor_bindings.push_back({final_history_tensor, "final_history"});
    }

    const uint32_t scratch_bytes =
        static_cast<uint32_t>(qkv_causal_conv1d_silu_tiled_scratch_bytes(block_ct, tile_size));
    ProgramSpec spec{
        .name = "qkv_causal_conv1d_silu_tiled",
        .kernels = {},
        .dataflow_buffers = std::move(dfbs),
        .scratchpads = {{.unique_id = scratch_name, .size_per_node = scratch_bytes}},
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {reader_name, writer_name, compute_name},
                    .target_nodes = cores,
                },
            },
    };
    kda_factory_detail::bind_actual_end(spec, run_args, in.actual_end, reader);
    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};

    run_args.kernel_run_args.reserve(3);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));

    return kda_factory_detail::chronology_workload(
        ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)},
        tensor_coords,
        device,
        attrs.sequence_parallel_axis,
        attrs.sequence,
        reader_name);
}

}  // namespace ttnn::experimental::prim
