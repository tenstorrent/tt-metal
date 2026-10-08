// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/factory/chronology_binding.hpp"

#include "ttnn/operations/experimental/kda/chain_affine_transforms/device/chain_affine_transforms_program_factory.hpp"

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

ttnn::device_operation::MeshWorkloadArtifacts ChainAffineTransformsProgramFactory::create_mesh_workload_artifacts(
    const ChainAffineTransformsParams& attrs,
    const ChainAffineTransformsInputs& in,
    std::vector<ttnn::Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    namespace m2 = tt::tt_metal::experimental;
    const auto& transforms = in.transforms.mesh_tensor();
    const auto& initial_state = in.initial_state.mesh_tensor();
    const auto& entry_state = outputs[0].mesh_tensor();
    const auto& final_state = outputs[1].mesh_tensor();
    const auto& device = transforms.device();

    const uint32_t BH = attrs.batch_heads;
    const uint32_t Kt = attrs.key_dim / tt::constants::TILE_WIDTH;
    const uint32_t Vt_full = attrs.value_dim / tt::constants::TILE_WIDTH;

    // Each step's update S = A S + B is independent per value column, so a head's columns are split across value
    // block cores. A is value-independent: value block 0 reads it once per step and multicasts it to the head's
    // other blocks, so DRAM traffic does not scale with the blocks (re-reading A per block was DRAM-bound).
    const auto dist = kda_factory_detail::distribute_value_blocks(device.compute_with_storage_grid_size(), BH, Vt_full);
    const uint32_t Vt = dist.value_tiles_per_core;
    const uint32_t value_blocks = dist.value_blocks;
    const bool mcast_shared = value_blocks > 1;

    const m2::KernelSpecName dataflow_kernel_name{"dataflow"};
    const m2::KernelSpecName compute_kernel_name{"compute"};
    const m2::DFBSpecName initial_dfb{"initial"};
    const m2::DFBSpecName a_dfb{"a"};
    const m2::DFBSpecName b_dfb{"b"};
    const m2::DFBSpecName product_dfb{"product"};
    const m2::DFBSpecName state_dfb{"state"};
    const m2::DFBSpecName out_dfb{"out"};
    const m2::TensorParamName transforms_name{"transforms"};
    const m2::TensorParamName initial_state_name{"initial_state"};
    const m2::TensorParamName entry_state_name{"entry_state"};
    const m2::TensorParamName final_state_name{"final_state"};
    const m2::SemaphoreSpecName ready_semaphore_name{"ready"};
    const m2::SemaphoreSpecName valid_semaphore_name{"valid"};

    auto make_dfb = [](const m2::DFBSpecName& name, uint32_t tiles, tt::DataFormat format) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::tile_size(format),
            .num_entries = tiles,
            .data_format_metadata = format,
        };
    };
    const auto transform_format = tt::tt_metal::datatype_to_dataformat_converter(in.transforms.dtype());
    constexpr auto fp32 = tt::DataFormat::Float32;
    const uint32_t a_tiles = Kt * Kt;
    const uint32_t state_tiles = Kt * Vt;
    // 384 KiB per core at the production K = V = 128; validation checks chain_affine_transforms_l1_bytes against L1.
    m2::Group<m2::DataflowBufferSpec> dfbs = {
        make_dfb(initial_dfb, state_tiles, fp32),
        make_dfb(a_dfb, chain_affine_transforms_transform_buffers * a_tiles, transform_format),
        make_dfb(b_dfb, chain_affine_transforms_transform_buffers * state_tiles, transform_format),
        make_dfb(product_dfb, state_tiles, fp32),
        make_dfb(state_dfb, state_tiles, fp32),
        make_dfb(out_dfb, state_tiles, fp32),
    };

    m2::KernelSpec dataflow{
        .unique_id = dataflow_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/chain_affine_transforms/device/kernels/dataflow/"
            "reader_writer_chain_affine_transforms.cpp",
        .dfb_bindings =
            {
                m2::ProducerOf(initial_dfb, "initial"),
                m2::ProducerOf(a_dfb, "a"),
                m2::ProducerOf(b_dfb, "b"),
                m2::ConsumerOf(out_dfb, "out"),
            },
        .tensor_bindings =
            {
                m2::TensorBinding{transforms_name, "transforms"},
                m2::TensorBinding{initial_state_name, "initial_state"},
                m2::TensorBinding{entry_state_name, "entry_state"},
                m2::TensorBinding{final_state_name, "final_state"},
            },
        .compile_time_args =
            {{"Kt", Kt},
             {"Vt", Vt},
             {"Vt_full", Vt_full},
             {"BH", BH},
             {"mcast_shared", static_cast<uint32_t>(mcast_shared)}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"head", "value_block", "peer_x0", "peer_y0", "peer_x1", "peer_y1", "receivers"}},
        // The kernel manages every DFB credit explicitly, including the 4-byte actual_start read.
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    // Matmul operands unpack to source registers; the product unpacks losslessly to DST for the FP32 SFPU add.
    auto compute_hw = ttnn::to_compute_hardware_config(attrs.compute_kernel_config);
    compute_hw.unpack_modes[initial_dfb] = tt::tt_metal::UnpackMode::UnpackToSrc;
    compute_hw.unpack_modes[state_dfb] = tt::tt_metal::UnpackMode::UnpackToSrc;
    compute_hw.unpack_modes[product_dfb] = tt::tt_metal::UnpackMode::UnpackToDest;
    m2::KernelSpec compute{
        .unique_id = compute_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/chain_affine_transforms/device/kernels/compute/"
            "chain_affine_transforms.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                m2::ConsumerOf(initial_dfb, "initial"),
                m2::ConsumerOf(a_dfb, "a"),
                m2::ConsumerOf(b_dfb, "b"),
                m2::ProducerOf(product_dfb, "product"),
                m2::ConsumerOf(product_dfb, "product"),
                m2::ProducerOf(state_dfb, "state"),
                m2::ConsumerOf(state_dfb, "state"),
                m2::ProducerOf(out_dfb, "out"),
            },
        .compile_time_args = {{"Kt", Kt}, {"Vt", Vt}, {"steps", attrs.steps}},
        .hw_config = std::move(compute_hw),
    };

    dataflow.semaphore_bindings.push_back(m2::SemaphoreBinding{ready_semaphore_name, "ready"});
    dataflow.semaphore_bindings.push_back(m2::SemaphoreBinding{valid_semaphore_name, "valid"});

    m2::KernelRunArgs dataflow_run{.kernel = dataflow_kernel_name};
    for (uint32_t index = 0; index < dist.cores.size(); ++index) {
        const uint32_t value_block = dist.value_block[index];
        // The sender (value block 0) addresses its siblings' row segment; receivers address the sender. The
        // dataflow kernel runs on NoC 0, so the segment starts at its lowest coordinate.
        uint32_t peer_x0 = 0;
        uint32_t peer_y0 = 0;
        uint32_t peer_x1 = 0;
        uint32_t peer_y1 = 0;
        if (mcast_shared) {
            const uint32_t sender_index = index - value_block;
            const auto first =
                device.worker_core_from_logical_core(dist.cores[value_block == 0 ? sender_index + 1 : sender_index]);
            const auto last = device.worker_core_from_logical_core(dist.cores[sender_index + value_blocks - 1]);
            peer_x0 = first.x;
            peer_y0 = first.y;
            peer_x1 = last.x;
            peer_y1 = last.y;
        }
        m2::AddRuntimeArgsForNode(
            dataflow_run.runtime_arg_values,
            dist.cores[index],
            {{"head", dist.head[index]},
             {"value_block", value_block},
             {"peer_x0", peer_x0},
             {"peer_y0", peer_y0},
             {"peer_x1", peer_x1},
             {"peer_y1", peer_y1},
             {"receivers", value_blocks - 1}});
    }
    m2::KernelRunArgs compute_run{.kernel = compute_kernel_name};

    m2::ProgramSpec spec{
        .name = "chain_affine_transforms",
        .dataflow_buffers = std::move(dfbs),
        .semaphores =
            {
                m2::SemaphoreSpec{.unique_id = ready_semaphore_name, .target_nodes = dist.core_set},
                m2::SemaphoreSpec{.unique_id = valid_semaphore_name, .target_nodes = dist.core_set},
            },
        .tensor_parameters =
            {
                m2::TensorParameter{.unique_id = transforms_name, .spec = transforms.tensor_spec()},
                m2::TensorParameter{.unique_id = initial_state_name, .spec = initial_state.tensor_spec()},
                m2::TensorParameter{.unique_id = entry_state_name, .spec = entry_state.tensor_spec()},
                m2::TensorParameter{.unique_id = final_state_name, .spec = final_state.tensor_spec()},
            },
        .work_units =
            {
                m2::WorkUnitSpec{
                    .name = "main",
                    .kernels = {dataflow_kernel_name, compute_kernel_name},
                    .target_nodes = dist.core_set,
                },
            },
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(dataflow_run), std::move(compute_run)};
    run_args.tensor_args = {
        {transforms_name, transforms},
        {initial_state_name, initial_state},
        {entry_state_name, entry_state},
        {final_state_name, final_state},
    };

    kda_factory_detail::bind_chronology(spec, run_args, in.actual_start, dataflow, compute);
    spec.kernels = {std::move(dataflow), std::move(compute)};
    return kda_factory_detail::chronology_workload(
        ttnn::device_operation::ProgramArtifacts{
            .spec = std::move(spec),
            .run_params = std::move(run_args),
        },
        tensor_coords,
        device,
        attrs.sequence_parallel_axis,
        attrs.local_rows,
        dataflow_kernel_name);
}

}  // namespace ttnn::experimental::prim
