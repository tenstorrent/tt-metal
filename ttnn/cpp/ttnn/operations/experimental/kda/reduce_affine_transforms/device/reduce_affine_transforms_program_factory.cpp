// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/reduce_affine_transforms/device/reduce_affine_transforms_program_factory.hpp"

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

ttnn::device_operation::ProgramArtifacts ReduceAffineTransformsProgramFactory::create_program_artifacts(
    const ReduceAffineTransformsParams& attrs,
    const ReduceAffineTransformsInputs& in,
    std::vector<ttnn::Tensor>& outputs) {
    const auto& a = in.a.mesh_tensor();
    const auto& b = in.b.mesh_tensor();
    const auto& output_a = outputs[0].mesh_tensor();
    const auto& output_b = outputs[1].mesh_tensor();
    const auto& device = a.device();
    const auto arch = device.arch();

    const uint32_t Kt = attrs.key_dim / tt::constants::TILE_WIDTH;
    const uint32_t Vt = attrs.value_dim / tt::constants::TILE_WIDTH;
    const uint32_t G = attrs.groups_per_head;
    const uint32_t group_heads = attrs.batch_heads * G;
    const uint32_t a_tiles = Kt * Kt;
    const uint32_t b_tiles = Kt * Vt;

    const auto grid = device.compute_with_storage_grid_size();
    auto dist = kda_factory_detail::distribute_prep(grid, group_heads, group_heads);
    const auto& cores = dist.core_set;

    const tt::tt_metal::experimental::KernelSpecName dataflow_kernel_name{"dataflow"};
    const tt::tt_metal::experimental::KernelSpecName compute_kernel_name{"compute"};

    const tt::tt_metal::experimental::DFBSpecName initial_a_dfb_name{"initial_a"};
    const tt::tt_metal::experimental::DFBSpecName initial_b_dfb_name{"initial_b"};
    const tt::tt_metal::experimental::DFBSpecName stage_a_dfb_name{"stage_a"};
    const tt::tt_metal::experimental::DFBSpecName stage_b_dfb_name{"stage_b"};
    const tt::tt_metal::experimental::DFBSpecName send_a_dfb_name{"send_a"};
    const tt::tt_metal::experimental::DFBSpecName send_b_dfb_name{"send_b"};
    const tt::tt_metal::experimental::DFBSpecName remote_a_dfb_name{"remote_a"};
    const tt::tt_metal::experimental::DFBSpecName remote_b_dfb_name{"remote_b"};
    const tt::tt_metal::experimental::DFBSpecName scratch_dfb_name{"scratch"};

    const tt::tt_metal::experimental::SemaphoreSpecName ready_semaphore_name{"ready"};
    const tt::tt_metal::experimental::SemaphoreSpecName arrival_semaphore_name{"arrival"};
    const tt::tt_metal::experimental::SemaphoreSpecName release_semaphore_name{"release"};

    const tt::tt_metal::experimental::TensorParamName a_tensor_name{"a"};
    const tt::tt_metal::experimental::TensorParamName b_tensor_name{"b"};
    const tt::tt_metal::experimental::TensorParamName output_a_tensor_name{"output_a"};
    const tt::tt_metal::experimental::TensorParamName output_b_tensor_name{"output_b"};

    auto make_dfb = [](const tt::tt_metal::experimental::DFBSpecName& name, uint32_t tiles, tt::DataFormat format) {
        return tt::tt_metal::experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::tile_size(format),
            .num_entries = tiles,
            .data_format_metadata = format,
        };
    };
    const auto summary_format = tt::tt_metal::datatype_to_dataformat_converter(in.a.dtype());
    constexpr auto internal_format = tt::DataFormat::Float32;
    // Per-core DFB storage is 6*Kt^2 + 7*Kt*Vt tiles. Production uses K=V=128 (Kt=Vt=4); supporting larger shapes
    // may require a separately designed chunked protocol rather than silently assuming this footprint fits every
    // architecture's usable L1.
    // Stage and send buffers are double-buffered so compute can produce the next prefix while the current one is used.
    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::DataflowBufferSpec> dfbs = {
        make_dfb(initial_a_dfb_name, a_tiles, summary_format),
        make_dfb(initial_b_dfb_name, b_tiles, summary_format),
        make_dfb(stage_a_dfb_name, 2 * a_tiles, internal_format),
        make_dfb(stage_b_dfb_name, 2 * b_tiles, internal_format),
        make_dfb(send_a_dfb_name, 2 * a_tiles, internal_format),
        make_dfb(send_b_dfb_name, 2 * b_tiles, internal_format),
        make_dfb(remote_a_dfb_name, a_tiles, internal_format),
        make_dfb(remote_b_dfb_name, b_tiles, internal_format),
        make_dfb(scratch_dfb_name, b_tiles, internal_format),
    };
    // The sender addresses a peer's inbound mailbox with its own local write pointer, which is only
    // valid while these buffers hold one phase-independent slot. Any additional depth lets sender and
    // receiver select different halves and silently corrupts the reduction.
    for (const auto& dfb : dfbs) {
        if (dfb.unique_id == remote_a_dfb_name || dfb.unique_id == remote_b_dfb_name) {
            TT_FATAL(
                dfb.num_entries == (dfb.unique_id == remote_a_dfb_name ? a_tiles : b_tiles),
                "reduce_affine_transforms: remotely addressed mailbox {} must hold exactly one block",
                *dfb.unique_id);
        }
    }

    tt::tt_metal::experimental::KernelSpec dataflow{
        .unique_id = dataflow_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/reduce_affine_transforms/device/kernels/dataflow/"
            "reader_writer_reduce_affine_transforms.cpp",
        .dfb_bindings =
            {
                tt::tt_metal::experimental::DFBBinding{
                    initial_a_dfb_name, "initial_a", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    initial_b_dfb_name, "initial_b", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    send_a_dfb_name, "send_a", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    send_b_dfb_name, "send_b", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    remote_a_dfb_name, "remote_a", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    remote_b_dfb_name, "remote_b", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
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
                tt::tt_metal::experimental::TensorBinding{output_a_tensor_name, "output_a"},
                tt::tt_metal::experimental::TensorBinding{output_b_tensor_name, "output_b"},
            },
        .compile_time_args = {{"Kt", Kt}, {"Vt", Vt}, {"G", G}},
        .runtime_arg_schema = {.runtime_arg_names = {"worker_index", "group"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
        .advanced_options = {.num_common_runtime_varargs = 2 * group_heads},
    };

    auto compute_hw = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config);
    auto& unpack_modes = tt::tt_metal::experimental::unpack_modes(compute_hw);
    for (const auto& name :
         {stage_a_dfb_name, stage_b_dfb_name, remote_a_dfb_name, remote_b_dfb_name, scratch_dfb_name}) {
        unpack_modes[name] = tt::tt_metal::UnpackMode::UnpackToSrc;
    }
    if (summary_format == tt::DataFormat::Float32) {
        unpack_modes[initial_a_dfb_name] = tt::tt_metal::UnpackMode::UnpackToSrc;
        unpack_modes[initial_b_dfb_name] = tt::tt_metal::UnpackMode::UnpackToSrc;
    }

    tt::tt_metal::experimental::KernelSpec compute{
        .unique_id = compute_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/reduce_affine_transforms/device/kernels/compute/"
            "reduce_affine_transforms.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                tt::tt_metal::experimental::DFBBinding{
                    initial_a_dfb_name, "initial_a", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    initial_b_dfb_name, "initial_b", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    stage_a_dfb_name, "stage_a", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    stage_a_dfb_name, "stage_a", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    stage_b_dfb_name, "stage_b", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    stage_b_dfb_name, "stage_b", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    send_a_dfb_name, "send_a", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    send_b_dfb_name, "send_b", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    remote_a_dfb_name, "remote_a", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    remote_b_dfb_name, "remote_b", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    scratch_dfb_name, "scratch", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    scratch_dfb_name, "scratch", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
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
    tt::tt_metal::experimental::KernelRunArgs dataflow_run{.kernel = dataflow_kernel_name};
    dataflow_run.advanced_options.common_runtime_varargs = std::move(worker_coordinates);
    tt::tt_metal::experimental::KernelRunArgs compute_run{.kernel = compute_kernel_name};
    for (uint32_t worker_index = 0; worker_index < group_heads; ++worker_index) {
        const auto& core = dist.cores[worker_index];
        const uint32_t group = worker_index % G;
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            dataflow_run.runtime_arg_values, core, {{"worker_index", worker_index}, {"group", group}});
        tt::tt_metal::experimental::AddRuntimeArgsForNode(compute_run.runtime_arg_values, core, {{"group", group}});
    }

    tt::tt_metal::experimental::ProgramSpec spec{
        .name = "reduce_affine_transforms",
        .kernels = {std::move(dataflow), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
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
                    .unique_id = output_a_tensor_name, .spec = output_a.tensor_spec()},
                tt::tt_metal::experimental::TensorParameter{
                    .unique_id = output_b_tensor_name, .spec = output_b.tensor_spec()},
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

    tt::tt_metal::experimental::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(dataflow_run), std::move(compute_run)};
    run_args.tensor_args = {
        {a_tensor_name, a},
        {b_tensor_name, b},
        {output_a_tensor_name, output_a},
        {output_b_tensor_name, output_b},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
