// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/affine_exclusive_scan/device/affine_exclusive_scan_program_factory.hpp"

#include <algorithm>
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

ttnn::device_operation::ProgramArtifacts AffineExclusiveScanProgramFactory::create_program_artifacts(
    const AffineExclusiveScanParams& attrs, const AffineExclusiveScanInputs& in, std::vector<Tensor>& outputs) {
    const auto& a = in.a.mesh_tensor();
    const auto& b = in.b.mesh_tensor();
    const auto& initial_state = in.initial_state.mesh_tensor();
    const auto& output = outputs[0].mesh_tensor();
    const auto& device = a.device();
    const auto arch = device.arch();

    const uint32_t Kt = attrs.key_dim / tt::constants::TILE_WIDTH;
    const uint32_t Vt = attrs.value_dim / tt::constants::TILE_WIDTH;
    const uint32_t G = attrs.groups_per_head;
    const uint32_t group_heads = attrs.batch_heads * G;
    const uint32_t kk = Kt * Kt;
    const uint32_t kv = Kt * Vt;

    const auto grid = device.compute_with_storage_grid_size();
    constexpr uint32_t kMaxAffineScanWorkers = 128;
    TT_FATAL(
        group_heads <= std::min<uint32_t>(grid.x * grid.y, kMaxAffineScanWorkers),
        "affine_exclusive_scan supports at most {} group workers, got {}",
        kMaxAffineScanWorkers,
        group_heads);
    auto dist = kda_factory_detail::distribute_prep(grid, group_heads, group_heads);
    const auto& cores = dist.core_set;

    const tt::tt_metal::experimental::KernelSpecName DATAFLOW{"dataflow"};
    const tt::tt_metal::experimental::KernelSpecName COMPUTE{"compute"};

    const tt::tt_metal::experimental::DFBSpecName INITIAL_A{"initial_a"};
    const tt::tt_metal::experimental::DFBSpecName INITIAL_B{"initial_b"};
    const tt::tt_metal::experimental::DFBSpecName LOCAL_A{"local_a"};
    const tt::tt_metal::experimental::DFBSpecName LOCAL_B{"local_b"};
    const tt::tt_metal::experimental::DFBSpecName TO_REMOTE_A{"to_remote_a"};
    const tt::tt_metal::experimental::DFBSpecName TO_REMOTE_B{"to_remote_b"};
    const tt::tt_metal::experimental::DFBSpecName FROM_REMOTE_A{"from_remote_a"};
    const tt::tt_metal::experimental::DFBSpecName FROM_REMOTE_B{"from_remote_b"};
    const tt::tt_metal::experimental::DFBSpecName INITIAL_STATE{"initial_state"};
    const tt::tt_metal::experimental::DFBSpecName FINAL{"final"};
    const tt::tt_metal::experimental::DFBSpecName SCRATCH{"scratch"};

    const tt::tt_metal::experimental::SemaphoreSpecName READY{"ready"};
    const tt::tt_metal::experimental::SemaphoreSpecName ARRIVAL{"arrival"};
    const tt::tt_metal::experimental::SemaphoreSpecName RELEASE{"release"};

    const tt::tt_metal::experimental::TensorParamName A{"a"};
    const tt::tt_metal::experimental::TensorParamName B{"b"};
    const tt::tt_metal::experimental::TensorParamName INITIAL_STATE_TENSOR{"initial_state"};
    const tt::tt_metal::experimental::TensorParamName OUTPUT{"output"};

    auto make_dfb = [](const tt::tt_metal::experimental::DFBSpecName& name, uint32_t tiles, tt::DataFormat format) {
        return tt::tt_metal::experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::tile_size(format),
            .num_entries = tiles,
            .data_format_metadata = format,
        };
    };
    const auto summary_format = tt::tt_metal::datatype_to_dataformat_converter(in.a.dtype());
    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::DataflowBufferSpec> dfbs = {
        make_dfb(INITIAL_A, kk, summary_format),
        make_dfb(INITIAL_B, kv, summary_format),
        make_dfb(LOCAL_A, 2 * kk, tt::DataFormat::Float32),
        make_dfb(LOCAL_B, 2 * kv, tt::DataFormat::Float32),
        make_dfb(TO_REMOTE_A, kk, tt::DataFormat::Float32),
        make_dfb(TO_REMOTE_B, kv, tt::DataFormat::Float32),
        make_dfb(FROM_REMOTE_A, kk, tt::DataFormat::Float32),
        make_dfb(FROM_REMOTE_B, kv, tt::DataFormat::Float32),
        make_dfb(INITIAL_STATE, kv, tt::DataFormat::Float32),
        make_dfb(FINAL, kv, tt::DataFormat::Float32),
        make_dfb(SCRATCH, kv, tt::DataFormat::Float32),
    };
    // Initial inputs/state and final output are one-shot transfers; scratch is compute-local. TO_REMOTE stays
    // single-slot because dataflow releases the current block before the remote input that makes compute runnable.
    // LOCAL is the only unconstrained cross-kernel path with overlapping current/next lifetimes, so it is depth two.
    // The sender addresses a peer's inbound mailbox with its own local write pointer, which is only
    // valid while these buffers hold one phase-independent slot. Any additional depth lets sender and
    // receiver select different halves and silently corrupts the scan.
    for (const auto& dfb : dfbs) {
        if (dfb.unique_id == FROM_REMOTE_A || dfb.unique_id == FROM_REMOTE_B) {
            TT_FATAL(
                dfb.num_entries == (dfb.unique_id == FROM_REMOTE_A ? kk : kv),
                "affine_exclusive_scan: remotely addressed mailbox {} must hold exactly one block",
                *dfb.unique_id);
        }
    }

    tt::tt_metal::experimental::KernelSpec dataflow{
        .unique_id = DATAFLOW,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/affine_exclusive_scan/device/kernels/dataflow/"
            "reader_writer_affine_exclusive_scan.cpp",
        .dfb_bindings =
            {
                tt::tt_metal::experimental::DFBBinding{
                    INITIAL_A, "initial_a", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    INITIAL_B, "initial_b", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    LOCAL_A, "local_a", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    LOCAL_B, "local_b", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    TO_REMOTE_A, "to_remote_a", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    TO_REMOTE_B, "to_remote_b", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    FROM_REMOTE_A, "from_remote_a", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    FROM_REMOTE_B, "from_remote_b", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    INITIAL_STATE, "initial_state", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    FINAL, "final", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
            },
        .semaphore_bindings =
            {
                tt::tt_metal::experimental::SemaphoreBinding{READY, "ready"},
                tt::tt_metal::experimental::SemaphoreBinding{ARRIVAL, "arrival"},
                tt::tt_metal::experimental::SemaphoreBinding{RELEASE, "release"},
            },
        .tensor_bindings =
            {
                tt::tt_metal::experimental::TensorBinding{A, "a"},
                tt::tt_metal::experimental::TensorBinding{B, "b"},
                tt::tt_metal::experimental::TensorBinding{INITIAL_STATE_TENSOR, "initial_state"},
                tt::tt_metal::experimental::TensorBinding{OUTPUT, "output"},
            },
        .compile_time_args = {{"Kt", Kt}, {"Vt", Vt}, {"BH", attrs.batch_heads}, {"G", G}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"worker_index", "group"},
             .common_runtime_arg_names = {"coordinator_x", "coordinator_y"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
        .advanced_options = {.num_common_runtime_varargs = 2 * group_heads},
    };

    auto compute_hw = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config);
    auto& unpack_modes = tt::tt_metal::experimental::unpack_modes(compute_hw);
    for (const auto& name : {LOCAL_A, LOCAL_B, FROM_REMOTE_A, FROM_REMOTE_B, INITIAL_STATE, SCRATCH}) {
        unpack_modes[name] = tt::tt_metal::UnpackMode::UnpackToSrc;
    }
    if (summary_format == tt::DataFormat::Float32) {
        unpack_modes[INITIAL_A] = tt::tt_metal::UnpackMode::UnpackToSrc;
        unpack_modes[INITIAL_B] = tt::tt_metal::UnpackMode::UnpackToSrc;
    }

    tt::tt_metal::experimental::KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/affine_exclusive_scan/device/kernels/compute/"
            "affine_exclusive_scan.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                tt::tt_metal::experimental::DFBBinding{
                    INITIAL_A, "initial_a", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    INITIAL_B, "initial_b", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    LOCAL_A, "local_a", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    LOCAL_B, "local_b", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    TO_REMOTE_A, "to_remote_a", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    TO_REMOTE_B, "to_remote_b", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    FROM_REMOTE_A, "from_remote_a", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    FROM_REMOTE_B, "from_remote_b", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    INITIAL_STATE, "initial_state", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    FINAL, "final", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    SCRATCH, "scratch", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    SCRATCH, "scratch", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
            },
        .compile_time_args = {{"Kt", Kt}, {"Vt", Vt}, {"G", G}},
        .runtime_arg_schema = {.runtime_arg_names = {"group"}},
        .hw_config = std::move(compute_hw),
    };

    std::vector<uint32_t> worker_coordinates;
    worker_coordinates.reserve(2 * group_heads);
    for (const auto& worker : dist.cores) {
        const auto physical = device.worker_core_from_logical_core(worker);
        worker_coordinates.push_back(physical.x);
        worker_coordinates.push_back(physical.y);
    }
    const auto coordinator = device.worker_core_from_logical_core(dist.cores[0]);
    tt::tt_metal::experimental::KernelRunArgs dataflow_run{
        .kernel = DATAFLOW,
        .common_runtime_arg_values =
            {{"coordinator_x", static_cast<uint32_t>(coordinator.x)},
             {"coordinator_y", static_cast<uint32_t>(coordinator.y)}},
    };
    dataflow_run.advanced_options.common_runtime_varargs = std::move(worker_coordinates);
    tt::tt_metal::experimental::KernelRunArgs compute_run{.kernel = COMPUTE};
    for (uint32_t flat = 0; flat < group_heads; ++flat) {
        const auto& core = dist.cores[flat];
        const uint32_t group = flat % G;
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            dataflow_run.runtime_arg_values, core, {{"worker_index", flat}, {"group", group}});
        tt::tt_metal::experimental::AddRuntimeArgsForNode(compute_run.runtime_arg_values, core, {{"group", group}});
    }

    tt::tt_metal::experimental::ProgramSpec spec{
        .name = "affine_exclusive_scan",
        .kernels = {std::move(dataflow), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .semaphores =
            {
                tt::tt_metal::experimental::SemaphoreSpec{.unique_id = READY, .target_nodes = cores},
                tt::tt_metal::experimental::SemaphoreSpec{.unique_id = ARRIVAL, .target_nodes = cores},
                tt::tt_metal::experimental::SemaphoreSpec{.unique_id = RELEASE, .target_nodes = cores},
            },
        .tensor_parameters =
            {
                tt::tt_metal::experimental::TensorParameter{.unique_id = A, .spec = a.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{.unique_id = B, .spec = b.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{
                    .unique_id = INITIAL_STATE_TENSOR, .spec = initial_state.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units =
            {
                tt::tt_metal::experimental::WorkUnitSpec{
                    .name = "main",
                    .kernels = {DATAFLOW, COMPUTE},
                    .target_nodes = cores,
                },
            },
    };

    tt::tt_metal::experimental::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(dataflow_run), std::move(compute_run)};
    run_args.tensor_args = {
        {A, a},
        {B, b},
        {INITIAL_STATE_TENSOR, initial_state},
        {OUTPUT, output},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
