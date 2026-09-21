// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/factory/chronology_binding.hpp"

#include "ttnn/operations/experimental/kda/affine_exclusive_scan/device/affine_exclusive_scan_program_factory.hpp"

#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim {

ttnn::device_operation::MeshWorkloadArtifacts AffineExclusiveScanProgramFactory::create_mesh_workload_artifacts(
    const AffineExclusiveScanParams& attrs,
    const AffineExclusiveScanInputs& in,
    std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const auto& a = in.a.mesh_tensor();
    const auto& b = in.b.mesh_tensor();
    const auto& initial_state = in.initial_state.mesh_tensor();
    const auto& tail_a = in.tail_a.mesh_tensor();
    const auto& tail_b = in.tail_b.mesh_tensor();
    const auto& tail_entry_states = in.tail_entry_states.mesh_tensor();
    const auto& output = outputs[0].mesh_tensor();
    const auto& device = a.device();
    const auto arch = device.arch();

    const uint32_t key_tiles = attrs.key_dim / tt::constants::TILE_WIDTH;
    const uint32_t value_tiles = attrs.value_dim / tt::constants::TILE_WIDTH;
    const uint32_t groups_per_head = attrs.groups_per_head;
    const uint32_t group_heads = attrs.batch_heads * groups_per_head;
    const uint32_t key_matrix_tiles = key_tiles * key_tiles;
    const uint32_t state_matrix_tiles = key_tiles * value_tiles;

    const auto grid = device.compute_with_storage_grid_size();
    auto distribution = kda_factory_detail::distribute_prep(grid, group_heads, group_heads);
    const auto& cores = distribution.core_set;

    const tt::tt_metal::experimental::KernelSpecName dataflow_kernel_name{"dataflow"};
    const tt::tt_metal::experimental::KernelSpecName compute_kernel_name{"compute"};

    const tt::tt_metal::experimental::DFBSpecName initial_a_dfb_name{"initial_a"};
    const tt::tt_metal::experimental::DFBSpecName initial_b_dfb_name{"initial_b"};
    const tt::tt_metal::experimental::DFBSpecName local_a_dfb_name{"local_a"};
    const tt::tt_metal::experimental::DFBSpecName local_b_dfb_name{"local_b"};
    const tt::tt_metal::experimental::DFBSpecName to_remote_a_dfb_name{"to_remote_a"};
    const tt::tt_metal::experimental::DFBSpecName to_remote_b_dfb_name{"to_remote_b"};
    const tt::tt_metal::experimental::DFBSpecName from_remote_affine_dfb_name{"from_remote_affine"};
    const tt::tt_metal::experimental::DFBSpecName initial_state_dfb_name{"initial_state"};
    const tt::tt_metal::experimental::DFBSpecName final_dfb_name{"final"};
    const tt::tt_metal::experimental::DFBSpecName tail_affine_dfb_name{"tail_affine"};
    const tt::tt_metal::experimental::DFBSpecName tail_entry_states_dfb_name{"tail_entry_states"};
    const tt::tt_metal::experimental::DFBSpecName reset_b_dfb_name{"reset_b"};

    const tt::tt_metal::experimental::SemaphoreSpecName ready_semaphore_name{"ready"};
    const tt::tt_metal::experimental::SemaphoreSpecName arrival_semaphore_name{"arrival"};
    const tt::tt_metal::experimental::SemaphoreSpecName release_semaphore_name{"release"};

    const tt::tt_metal::experimental::TensorParamName a_tensor_name{"a"};
    const tt::tt_metal::experimental::TensorParamName b_tensor_name{"b"};
    const tt::tt_metal::experimental::TensorParamName initial_state_tensor_name{"initial_state"};
    const tt::tt_metal::experimental::TensorParamName output_tensor_name{"output"};
    const tt::tt_metal::experimental::TensorParamName tail_a_tensor_name{"tail_a"};
    const tt::tt_metal::experimental::TensorParamName tail_b_tensor_name{"tail_b"};
    const tt::tt_metal::experimental::TensorParamName tail_entry_states_tensor_name{"tail_entry_states"};

    auto make_dfb = [](const tt::tt_metal::experimental::DFBSpecName& name, uint32_t tiles, tt::DataFormat format) {
        return tt::tt_metal::experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::tile_size(format),
            .num_entries = tiles,
            .data_format_metadata = format,
        };
    };
    const auto summary_format = tt::tt_metal::datatype_to_dataformat_converter(in.a.dtype());
    const uint32_t segmented_affine_tiles = key_matrix_tiles + state_matrix_tiles;
    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::DataflowBufferSpec> dataflow_buffers = {
        make_dfb(initial_a_dfb_name, key_matrix_tiles, summary_format),
        make_dfb(initial_b_dfb_name, state_matrix_tiles, summary_format),
        make_dfb(local_a_dfb_name, 2 * key_matrix_tiles, tt::DataFormat::Float32),
        make_dfb(local_b_dfb_name, 2 * state_matrix_tiles, tt::DataFormat::Float32),
        make_dfb(to_remote_a_dfb_name, key_matrix_tiles, tt::DataFormat::Float32),
        make_dfb(to_remote_b_dfb_name, state_matrix_tiles, tt::DataFormat::Float32),
        make_dfb(from_remote_affine_dfb_name, key_matrix_tiles + state_matrix_tiles, tt::DataFormat::Float32),
        make_dfb(initial_state_dfb_name, state_matrix_tiles, tt::DataFormat::Float32),
        make_dfb(final_dfb_name, state_matrix_tiles, tt::DataFormat::Float32),
        make_dfb(tail_affine_dfb_name, segmented_affine_tiles, summary_format),
        make_dfb(tail_entry_states_dfb_name, state_matrix_tiles, tt::DataFormat::Float32),
        make_dfb(reset_b_dfb_name, state_matrix_tiles, tt::DataFormat::Float32),
    };
    // Initial inputs/state and final output are one-shot transfers. TO_REMOTE stays single-slot because dataflow
    // releases the current block before the remote input that makes compute runnable.
    // LOCAL is the only unconstrained cross-kernel path with overlapping current/next lifetimes, so it is depth two.
    // The sender addresses a peer's inbound mailbox with its own local write pointer, which is only
    // valid while these buffers hold one phase-independent slot. Any additional depth lets sender and
    // receiver select different halves and silently corrupts the scan.
    for (const auto& dataflow_buffer : dataflow_buffers) {
        if (dataflow_buffer.unique_id == from_remote_affine_dfb_name) {
            TT_FATAL(
                dataflow_buffer.num_entries == key_matrix_tiles + state_matrix_tiles,
                "affine_exclusive_scan: remotely addressed mailbox {} must hold exactly one block",
                *dataflow_buffer.unique_id);
        }
    }

    tt::tt_metal::experimental::KernelSpec dataflow{
        .unique_id = dataflow_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/affine_exclusive_scan/device/kernels/dataflow/"
            "reader_writer_affine_exclusive_scan.cpp",
        .dfb_bindings =
            {
                tt::tt_metal::experimental::DFBBinding{
                    initial_a_dfb_name, "initial_a", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    initial_b_dfb_name, "initial_b", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    local_a_dfb_name, "local_a", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    local_b_dfb_name, "local_b", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    to_remote_a_dfb_name, "to_remote_a", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    to_remote_b_dfb_name, "to_remote_b", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    from_remote_affine_dfb_name,
                    "from_remote_affine",
                    tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    initial_state_dfb_name, "initial_state", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    final_dfb_name, "final", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::ProducerOf(tail_affine_dfb_name, "tail_affine"),
                tt::tt_metal::experimental::ProducerOf(tail_entry_states_dfb_name, "tail_entry_states"),
            },
        .semaphore_bindings =
            {
                tt::tt_metal::experimental::SemaphoreBinding{ready_semaphore_name, "ready"},
                tt::tt_metal::experimental::SemaphoreBinding{arrival_semaphore_name, "arrival"},
                tt::tt_metal::experimental::SemaphoreBinding{release_semaphore_name, "release"},
            },
        .tensor_bindings =
            {
                tt::tt_metal::experimental::TensorBinding{a_tensor_name, "a"},
                tt::tt_metal::experimental::TensorBinding{b_tensor_name, "b"},
                tt::tt_metal::experimental::TensorBinding{initial_state_tensor_name, "initial_state"},
                tt::tt_metal::experimental::TensorBinding{output_tensor_name, "output"},
                tt::tt_metal::experimental::TensorBinding{tail_a_tensor_name, "tail_a"},
                tt::tt_metal::experimental::TensorBinding{tail_b_tensor_name, "tail_b"},
                tt::tt_metal::experimental::TensorBinding{tail_entry_states_tensor_name, "tail_entry_states"},
            },
        .compile_time_args =
            {{"Kt", key_tiles}, {"Vt", value_tiles}, {"BH", attrs.batch_heads}, {"G", groups_per_head}},
        .runtime_arg_schema = {.runtime_arg_names = {"worker_index", "group"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
        .advanced_options = {.num_common_runtime_varargs = 2 * group_heads},
    };

    auto compute_hardware_config = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config);
    auto& unpack_modes = tt::tt_metal::experimental::unpack_modes(compute_hardware_config);
    for (const auto& name :
         {local_a_dfb_name,
          local_b_dfb_name,
          from_remote_affine_dfb_name,
          initial_state_dfb_name,
          tail_entry_states_dfb_name,
          reset_b_dfb_name}) {
        unpack_modes[name] = tt::tt_metal::UnpackMode::UnpackToSrc;
    }
    if (summary_format == tt::DataFormat::Float32) {
        unpack_modes[initial_a_dfb_name] = tt::tt_metal::UnpackMode::UnpackToSrc;
        unpack_modes[initial_b_dfb_name] = tt::tt_metal::UnpackMode::UnpackToSrc;
        unpack_modes[tail_affine_dfb_name] = tt::tt_metal::UnpackMode::UnpackToSrc;
    }

    tt::tt_metal::experimental::KernelSpec compute{
        .unique_id = compute_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/affine_exclusive_scan/device/kernels/compute/"
            "affine_exclusive_scan.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                tt::tt_metal::experimental::DFBBinding{
                    initial_a_dfb_name, "initial_a", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    initial_b_dfb_name, "initial_b", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    local_a_dfb_name, "local_a", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    local_b_dfb_name, "local_b", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    to_remote_a_dfb_name, "to_remote_a", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    to_remote_b_dfb_name, "to_remote_b", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    from_remote_affine_dfb_name,
                    "from_remote_affine",
                    tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    initial_state_dfb_name, "initial_state", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    final_dfb_name, "final", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::ConsumerOf(tail_affine_dfb_name, "tail_affine"),
                tt::tt_metal::experimental::ConsumerOf(tail_entry_states_dfb_name, "tail_entry_states"),
                tt::tt_metal::experimental::ProducerOf(reset_b_dfb_name, "reset_b"),
                tt::tt_metal::experimental::ConsumerOf(reset_b_dfb_name, "reset_b"),
            },
        .compile_time_args = {{"Kt", key_tiles}, {"Vt", value_tiles}, {"G", groups_per_head}},
        .runtime_arg_schema = {.runtime_arg_names = {"group"}},
        .hw_config = std::move(compute_hardware_config),
    };

    std::vector<uint32_t> worker_coordinates;
    worker_coordinates.reserve(2 * group_heads);
    for (const auto& worker : distribution.cores) {
        const auto physical = device.worker_core_from_logical_core(worker);
        worker_coordinates.push_back(physical.x);
        worker_coordinates.push_back(physical.y);
    }
    tt::tt_metal::experimental::KernelRunArgs dataflow_run{.kernel = dataflow_kernel_name};
    dataflow_run.advanced_options.common_runtime_varargs = std::move(worker_coordinates);
    tt::tt_metal::experimental::KernelRunArgs compute_run{.kernel = compute_kernel_name};
    for (uint32_t worker_index = 0; worker_index < group_heads; ++worker_index) {
        const auto& core = distribution.cores[worker_index];
        const uint32_t group = worker_index % groups_per_head;
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            dataflow_run.runtime_arg_values, core, {{"worker_index", worker_index}, {"group", group}});
        tt::tt_metal::experimental::AddRuntimeArgsForNode(compute_run.runtime_arg_values, core, {{"group", group}});
    }

    tt::tt_metal::experimental::ProgramSpec program_spec{
        .name = "affine_exclusive_scan",
        .dataflow_buffers = std::move(dataflow_buffers),
        .semaphores =
            {
                tt::tt_metal::experimental::SemaphoreSpec{.unique_id = ready_semaphore_name, .target_nodes = cores},
                tt::tt_metal::experimental::SemaphoreSpec{.unique_id = arrival_semaphore_name, .target_nodes = cores},
                tt::tt_metal::experimental::SemaphoreSpec{.unique_id = release_semaphore_name, .target_nodes = cores},
            },
        .tensor_parameters =
            {
                tt::tt_metal::experimental::TensorParameter{.unique_id = a_tensor_name, .spec = a.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{.unique_id = b_tensor_name, .spec = b.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{
                    .unique_id = initial_state_tensor_name, .spec = initial_state.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{
                    .unique_id = output_tensor_name, .spec = output.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{
                    .unique_id = tail_a_tensor_name, .spec = tail_a.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{
                    .unique_id = tail_b_tensor_name, .spec = tail_b.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{
                    .unique_id = tail_entry_states_tensor_name, .spec = tail_entry_states.tensor_spec()},
            },
        .work_units =
            {
                tt::tt_metal::experimental::WorkUnitSpec{
                    .name = "main",
                    .kernels = {dataflow_kernel_name, compute_kernel_name},
                    .target_nodes = cores,
                },
            },
    };

    tt::tt_metal::experimental::ProgramRunArgs program_run_args;
    program_run_args.kernel_run_args = {std::move(dataflow_run), std::move(compute_run)};
    program_run_args.tensor_args = {
        {a_tensor_name, a},
        {b_tensor_name, b},
        {initial_state_tensor_name, initial_state},
        {output_tensor_name, output},
        {tail_a_tensor_name, tail_a},
        {tail_b_tensor_name, tail_b},
        {tail_entry_states_tensor_name, tail_entry_states},
    };

    kda_factory_detail::bind_chronology(program_spec, program_run_args, in.actual_start, dataflow, compute);
    kda_factory_detail::bind_actual_end(program_spec, program_run_args, in.actual_end, in.actual_start, dataflow);
    program_spec.kernels = {std::move(dataflow), std::move(compute)};
    return kda_factory_detail::chronology_workload(
        ttnn::device_operation::ProgramArtifacts{
            .spec = std::move(program_spec),
            .run_params = std::move(program_run_args),
        },
        tensor_coords,
        device,
        attrs.sequence_parallel_axis,
        attrs.local_rows,
        dataflow_kernel_name);
}

}  // namespace ttnn::experimental::prim
