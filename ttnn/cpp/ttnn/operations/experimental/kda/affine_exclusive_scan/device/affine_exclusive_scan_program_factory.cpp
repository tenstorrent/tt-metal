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

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace m2 = tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts AffineExclusiveScanProgramFactory::create_program_artifacts(
    const AffineExclusiveScanParams& attrs, const AffineExclusiveScanInputs& in, std::vector<Tensor>& outputs) {
    const auto& a = in.a.mesh_tensor();
    const auto& b = in.b.mesh_tensor();
    const auto& initial_state = in.initial_state.mesh_tensor();
    const auto& output = outputs[0].mesh_tensor();
    const auto& device = a.device();
    const auto arch = device.arch();

    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt = attrs.value_dim / TILE_WIDTH;
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

    const m2::KernelSpecName DATAFLOW{"dataflow"};
    const m2::KernelSpecName COMPUTE{"compute"};

    const m2::DFBSpecName INITIAL_A{"initial_a"};
    const m2::DFBSpecName INITIAL_B{"initial_b"};
    const m2::DFBSpecName STAGE_A_PING{"stage_a_ping"};
    const m2::DFBSpecName STAGE_B_PING{"stage_b_ping"};
    const m2::DFBSpecName STAGE_A_PONG{"stage_a_pong"};
    const m2::DFBSpecName STAGE_B_PONG{"stage_b_pong"};
    const m2::DFBSpecName SEND_A_PING{"send_a_ping"};
    const m2::DFBSpecName SEND_B_PING{"send_b_ping"};
    const m2::DFBSpecName SEND_A_PONG{"send_a_pong"};
    const m2::DFBSpecName SEND_B_PONG{"send_b_pong"};
    const m2::DFBSpecName REMOTE_A{"remote_a"};
    const m2::DFBSpecName REMOTE_B{"remote_b"};
    const m2::DFBSpecName INITIAL_STATE{"initial_state"};
    const m2::DFBSpecName FINAL{"final"};
    const m2::DFBSpecName SCRATCH{"scratch"};
    const m2::DFBSpecName STAGE_TOKEN{"stage_token"};

    const m2::SemaphoreSpecName READY{"ready"};
    const m2::SemaphoreSpecName ARRIVAL{"arrival"};
    const m2::SemaphoreSpecName RELEASE{"release"};

    const m2::TensorParamName A{"a"};
    const m2::TensorParamName B{"b"};
    const m2::TensorParamName INITIAL_STATE_TENSOR{"initial_state"};
    const m2::TensorParamName OUTPUT{"output"};

    auto make_dfb = [](const m2::DFBSpecName& name,
                       uint32_t tiles,
                       tt::DataFormat format,
                       m2::Group<m2::DFBSpecName> aliases = {}) {
        auto dfb = m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::tile_size(format),
            .num_entries = tiles,
            .data_format_metadata = format,
        };
        dfb.advanced_options.alias_with = std::move(aliases);
        return dfb;
    };
    const auto summary_format = datatype_to_dataformat_converter(in.a.dtype());
    m2::Group<m2::DataflowBufferSpec> dfbs = {
        make_dfb(INITIAL_A, kk, summary_format),
        make_dfb(INITIAL_B, kv, summary_format),
        make_dfb(STAGE_A_PING, kk, tt::DataFormat::Float32, {SEND_A_PING}),
        make_dfb(STAGE_B_PING, kv, tt::DataFormat::Float32, {SEND_B_PING}),
        make_dfb(STAGE_A_PONG, kk, tt::DataFormat::Float32, {SEND_A_PONG}),
        make_dfb(STAGE_B_PONG, kv, tt::DataFormat::Float32, {SEND_B_PONG}),
        make_dfb(SEND_A_PING, kk, tt::DataFormat::Float32, {STAGE_A_PING}),
        make_dfb(SEND_B_PING, kv, tt::DataFormat::Float32, {STAGE_B_PING}),
        make_dfb(SEND_A_PONG, kk, tt::DataFormat::Float32, {STAGE_A_PONG}),
        make_dfb(SEND_B_PONG, kv, tt::DataFormat::Float32, {STAGE_B_PONG}),
        make_dfb(REMOTE_A, kk, tt::DataFormat::Float32),
        make_dfb(REMOTE_B, kv, tt::DataFormat::Float32),
        make_dfb(INITIAL_STATE, kv, tt::DataFormat::Float32),
        make_dfb(FINAL, kv, tt::DataFormat::Float32),
        make_dfb(SCRATCH, kv, tt::DataFormat::Float32),
        make_dfb(STAGE_TOKEN, 1, tt::DataFormat::Float32),
    };

    m2::KernelSpec dataflow{
        .unique_id = DATAFLOW,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/affine_exclusive_scan/device/kernels/dataflow/"
            "reader_writer_affine_exclusive_scan.cpp",
        .dfb_bindings =
            {
                m2::DFBBinding{INITIAL_A, "initial_a", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{INITIAL_B, "initial_b", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{SEND_A_PING, "send_a_ping", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{SEND_B_PING, "send_b_ping", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{SEND_A_PONG, "send_a_pong", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{SEND_B_PONG, "send_b_pong", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{REMOTE_A, "remote_a", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{REMOTE_B, "remote_b", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{INITIAL_STATE, "initial_state", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{FINAL, "final", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{STAGE_TOKEN, "stage_token", m2::DFBEndpointType::PRODUCER},
            },
        .semaphore_bindings =
            {
                m2::SemaphoreBinding{READY, "ready"},
                m2::SemaphoreBinding{ARRIVAL, "arrival"},
                m2::SemaphoreBinding{RELEASE, "release"},
            },
        .tensor_bindings =
            {
                m2::TensorBinding{A, "a"},
                m2::TensorBinding{B, "b"},
                m2::TensorBinding{INITIAL_STATE_TENSOR, "initial_state"},
                m2::TensorBinding{OUTPUT, "output"},
            },
        .compile_time_args = {{"Kt", Kt}, {"Vt", Vt}, {"BH", attrs.batch_heads}, {"G", G}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"worker_index", "group"},
             .common_runtime_arg_names = {"coordinator_x", "coordinator_y"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
        .advanced_options = {.num_common_runtime_varargs = 2 * group_heads},
    };

    auto compute_hw = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config);
    auto& unpack_modes = m2::unpack_modes(compute_hw);
    for (const auto& name :
         {STAGE_A_PING,
          STAGE_B_PING,
          STAGE_A_PONG,
          STAGE_B_PONG,
          REMOTE_A,
          REMOTE_B,
          INITIAL_STATE,
          SCRATCH,
          STAGE_TOKEN}) {
        unpack_modes[name] = UnpackMode::UnpackToSrc;
    }
    if (summary_format == tt::DataFormat::Float32) {
        unpack_modes[INITIAL_A] = UnpackMode::UnpackToSrc;
        unpack_modes[INITIAL_B] = UnpackMode::UnpackToSrc;
    }

    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/affine_exclusive_scan/device/kernels/compute/"
            "affine_exclusive_scan.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                m2::DFBBinding{INITIAL_A, "initial_a", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{INITIAL_B, "initial_b", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{STAGE_A_PING, "stage_a_ping", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{STAGE_A_PING, "stage_a_ping", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{STAGE_B_PING, "stage_b_ping", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{STAGE_B_PING, "stage_b_ping", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{STAGE_A_PONG, "stage_a_pong", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{STAGE_A_PONG, "stage_a_pong", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{STAGE_B_PONG, "stage_b_pong", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{STAGE_B_PONG, "stage_b_pong", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{SEND_A_PING, "send_a_ping", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{SEND_B_PING, "send_b_ping", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{SEND_A_PONG, "send_a_pong", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{SEND_B_PONG, "send_b_pong", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{REMOTE_A, "remote_a", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{REMOTE_B, "remote_b", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{INITIAL_STATE, "initial_state", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{FINAL, "final", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{SCRATCH, "scratch", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{SCRATCH, "scratch", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{STAGE_TOKEN, "stage_token", m2::DFBEndpointType::CONSUMER},
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
    m2::KernelRunArgs dataflow_run{
        .kernel = DATAFLOW,
        .common_runtime_arg_values =
            {{"coordinator_x", static_cast<uint32_t>(coordinator.x)},
             {"coordinator_y", static_cast<uint32_t>(coordinator.y)}},
    };
    dataflow_run.advanced_options.common_runtime_varargs = std::move(worker_coordinates);
    m2::KernelRunArgs compute_run{.kernel = COMPUTE};
    for (uint32_t flat = 0; flat < group_heads; ++flat) {
        const auto& core = dist.cores[flat];
        const uint32_t group = flat % G;
        m2::AddRuntimeArgsForNode(dataflow_run.runtime_arg_values, core, {{"worker_index", flat}, {"group", group}});
        m2::AddRuntimeArgsForNode(compute_run.runtime_arg_values, core, {{"group", group}});
    }

    m2::ProgramSpec spec{
        .name = "affine_exclusive_scan",
        .kernels = {std::move(dataflow), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .semaphores =
            {
                m2::SemaphoreSpec{.unique_id = READY, .target_nodes = cores},
                m2::SemaphoreSpec{.unique_id = ARRIVAL, .target_nodes = cores},
                m2::SemaphoreSpec{.unique_id = RELEASE, .target_nodes = cores},
            },
        .tensor_parameters =
            {
                m2::TensorParameter{.unique_id = A, .spec = a.tensor_spec()},
                m2::TensorParameter{.unique_id = B, .spec = b.tensor_spec()},
                m2::TensorParameter{.unique_id = INITIAL_STATE_TENSOR, .spec = initial_state.tensor_spec()},
                m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units =
            {
                m2::WorkUnitSpec{
                    .name = "main",
                    .kernels = {DATAFLOW, COMPUTE},
                    .target_nodes = cores,
                },
            },
    };

    m2::ProgramRunArgs run_args;
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
