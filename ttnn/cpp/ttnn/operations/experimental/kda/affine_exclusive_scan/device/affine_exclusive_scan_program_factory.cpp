// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/factory/chronology_binding.hpp"

#include "ttnn/operations/experimental/kda/affine_exclusive_scan/device/affine_exclusive_scan_program_factory.hpp"

#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/workload/specification/dataflow_buffer_spec.hpp>
#include <tt-metalium/workload/specification/kernel_spec.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/workload/specification/semaphore_spec.hpp>
#include <tt-metalium/workload/specification/tensor_parameter.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim {

ttnn::device_operation::MeshWorkloadArtifacts AffineExclusiveScanProgramFactory::create_mesh_workload_artifacts(
    const AffineExclusiveScanParams& attrs,
    const AffineExclusiveScanInputs& in,
    std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const auto& summaries = in.summaries.mesh_tensor();
    const auto& initial_state = in.initial_state.mesh_tensor();
    const auto& output = outputs[0].mesh_tensor();
    const auto& device = summaries.device();

    const uint32_t key_tiles = attrs.key_dim / tt::constants::TILE_WIDTH;
    const uint32_t value_tiles = attrs.value_dim / tt::constants::TILE_WIDTH;
    const uint32_t groups_per_head = attrs.groups_per_head;
    const uint32_t group_heads = attrs.batch_heads * groups_per_head;
    const uint32_t key_matrix_tiles = key_tiles * key_tiles;
    const uint32_t state_matrix_tiles = key_tiles * value_tiles;

    const auto grid = device.compute_with_storage_grid_size();
    auto distribution = kda_factory_detail::distribute_prep(grid, group_heads, group_heads);
    const auto& cores = distribution.core_set;

    const tt::tt_metal::KernelSpecName dataflow_kernel_name{"dataflow"};
    const tt::tt_metal::KernelSpecName compute_kernel_name{"compute"};

    const tt::tt_metal::DFBSpecName initial_a_dfb_name{"initial_a"};
    const tt::tt_metal::DFBSpecName initial_b_dfb_name{"initial_b"};
    const tt::tt_metal::DFBSpecName local_a_dfb_name{"local_a"};
    const tt::tt_metal::DFBSpecName local_b_dfb_name{"local_b"};
    const tt::tt_metal::DFBSpecName to_remote_a_dfb_name{"to_remote_a"};
    const tt::tt_metal::DFBSpecName to_remote_b_dfb_name{"to_remote_b"};
    const tt::tt_metal::DFBSpecName from_remote_affine_dfb_name{"from_remote_affine"};
    const tt::tt_metal::DFBSpecName initial_state_dfb_name{"initial_state"};
    const tt::tt_metal::DFBSpecName final_dfb_name{"final"};
    const tt::tt_metal::DFBSpecName tail_affine_dfb_name{"tail_affine"};
    const tt::tt_metal::DFBSpecName tail_entry_states_dfb_name{"tail_entry_states"};
    const tt::tt_metal::DFBSpecName reset_b_dfb_name{"reset_b"};

    const tt::tt_metal::SemaphoreSpecName ready_semaphore_name{"ready"};
    const tt::tt_metal::SemaphoreSpecName arrival_semaphore_name{"arrival"};
    const tt::tt_metal::SemaphoreSpecName release_semaphore_name{"release"};

    const tt::tt_metal::TensorParamName summaries_tensor_name{"summaries"};
    const tt::tt_metal::TensorParamName initial_state_tensor_name{"initial_state"};
    const tt::tt_metal::TensorParamName output_tensor_name{"output"};
    const tt::tt_metal::TensorParamName tail_entry_states_tensor_name{"tail_entry_states"};

    auto make_dfb = [](const tt::tt_metal::DFBSpecName& name, uint32_t tiles, tt::DataFormat format) {
        return tt::tt_metal::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::tile_size(format),
            .num_entries = tiles,
            .data_format_metadata = format,
        };
    };
    const auto summary_format = tt::tt_metal::datatype_to_dataformat_converter(in.summaries.dtype());
    const uint32_t segmented_affine_tiles = key_matrix_tiles + state_matrix_tiles;
    tt::tt_metal::Group<tt::tt_metal::DataflowBufferSpec> dataflow_buffers = {
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

    tt::tt_metal::KernelSpec dataflow{
        .unique_id = dataflow_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/affine_exclusive_scan/device/kernels/dataflow/"
            "reader_writer_affine_exclusive_scan.cpp",
        .dfb_bindings =
            {
                tt::tt_metal::DFBBinding{initial_a_dfb_name, "initial_a", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{initial_b_dfb_name, "initial_b", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{local_a_dfb_name, "local_a", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{local_b_dfb_name, "local_b", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{to_remote_a_dfb_name, "to_remote_a", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{to_remote_b_dfb_name, "to_remote_b", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{
                    from_remote_affine_dfb_name, "from_remote_affine", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{
                    initial_state_dfb_name, "initial_state", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{final_dfb_name, "final", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::ProducerOf(tail_affine_dfb_name, "tail_affine"),
                tt::tt_metal::ProducerOf(tail_entry_states_dfb_name, "tail_entry_states"),
            },
        .semaphore_bindings =
            {
                tt::tt_metal::SemaphoreBinding{ready_semaphore_name, "ready"},
                tt::tt_metal::SemaphoreBinding{arrival_semaphore_name, "arrival"},
                tt::tt_metal::SemaphoreBinding{release_semaphore_name, "release"},
            },
        .tensor_bindings =
            {
                tt::tt_metal::TensorBinding{summaries_tensor_name, "summaries"},
                tt::tt_metal::TensorBinding{initial_state_tensor_name, "initial_state"},
                tt::tt_metal::TensorBinding{output_tensor_name, "output"},
            },
        .compile_time_args =
            {{"Kt", key_tiles},
             {"Vt", value_tiles},
             {"BH", attrs.batch_heads},
             {"G", groups_per_head},
             {"has_tail", static_cast<uint32_t>(in.tail_entry_states.has_value())}},
        .runtime_arg_schema = {.runtime_arg_names = {"worker_index", "group"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
        .advanced_options = {.num_common_runtime_varargs = 2 * group_heads},
    };

    auto compute_hardware_config = ttnn::to_compute_hardware_config(attrs.compute_kernel_config);
    auto& unpack_modes = compute_hardware_config.unpack_modes;
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

    tt::tt_metal::KernelSpec compute{
        .unique_id = compute_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/affine_exclusive_scan/device/kernels/compute/"
            "affine_exclusive_scan.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                tt::tt_metal::DFBBinding{initial_a_dfb_name, "initial_a", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{initial_b_dfb_name, "initial_b", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{local_a_dfb_name, "local_a", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{local_b_dfb_name, "local_b", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{to_remote_a_dfb_name, "to_remote_a", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{to_remote_b_dfb_name, "to_remote_b", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{
                    from_remote_affine_dfb_name, "from_remote_affine", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{
                    initial_state_dfb_name, "initial_state", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{final_dfb_name, "final", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::ConsumerOf(tail_affine_dfb_name, "tail_affine"),
                tt::tt_metal::ConsumerOf(tail_entry_states_dfb_name, "tail_entry_states"),
                tt::tt_metal::ProducerOf(reset_b_dfb_name, "reset_b"),
                tt::tt_metal::ConsumerOf(reset_b_dfb_name, "reset_b"),
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
    tt::tt_metal::KernelRunArgs dataflow_run{.kernel = dataflow_kernel_name};
    dataflow_run.advanced_options.common_runtime_varargs = std::move(worker_coordinates);
    tt::tt_metal::KernelRunArgs compute_run{.kernel = compute_kernel_name};
    for (uint32_t worker_index = 0; worker_index < group_heads; ++worker_index) {
        const auto& core = distribution.cores[worker_index];
        const uint32_t group = worker_index % groups_per_head;
        tt::tt_metal::AddRuntimeArgsForNode(
            dataflow_run.runtime_arg_values, core, {{"worker_index", worker_index}, {"group", group}});
        tt::tt_metal::AddRuntimeArgsForNode(compute_run.runtime_arg_values, core, {{"group", group}});
    }

    tt::tt_metal::ProgramSpec program_spec{
        .name = "affine_exclusive_scan",
        .dataflow_buffers = std::move(dataflow_buffers),
        .semaphores =
            {
                tt::tt_metal::SemaphoreSpec{.unique_id = ready_semaphore_name, .target_nodes = cores},
                tt::tt_metal::SemaphoreSpec{.unique_id = arrival_semaphore_name, .target_nodes = cores},
                tt::tt_metal::SemaphoreSpec{.unique_id = release_semaphore_name, .target_nodes = cores},
            },
        .tensor_parameters =
            {
                tt::tt_metal::TensorParameter{.unique_id = summaries_tensor_name, .spec = summaries.tensor_spec()},
                tt::tt_metal::TensorParameter{
                    .unique_id = initial_state_tensor_name, .spec = initial_state.tensor_spec()},
                tt::tt_metal::TensorParameter{.unique_id = output_tensor_name, .spec = output.tensor_spec()},
            },
        .work_units =
            {
                tt::tt_metal::WorkUnitSpec{
                    .name = "main",
                    .kernels = {dataflow_kernel_name, compute_kernel_name},
                    .target_nodes = cores,
                },
            },
    };

    tt::tt_metal::ProgramRunArgs program_run_args;
    program_run_args.kernel_run_args = {std::move(dataflow_run), std::move(compute_run)};
    program_run_args.tensor_args = {
        {summaries_tensor_name, summaries},
        {initial_state_tensor_name, initial_state},
        {output_tensor_name, output},
    };
    // The tail seed exists only with tail transforms; kernels resolve it with get_token_if_present.
    if (in.tail_entry_states) {
        const auto& tail_entry_states = in.tail_entry_states->mesh_tensor();
        program_spec.tensor_parameters.push_back(
            {.unique_id = tail_entry_states_tensor_name, .spec = tail_entry_states.tensor_spec()});
        program_run_args.tensor_args.emplace(tail_entry_states_tensor_name, tail_entry_states);
        dataflow.tensor_bindings.push_back({tail_entry_states_tensor_name, "tail_entry_states"});
    }

    kda_factory_detail::bind_chronology(program_spec, program_run_args, in.actual_start, dataflow, compute);
    kda_factory_detail::bind_actual_end(program_spec, program_run_args, in.actual_end, dataflow);
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
