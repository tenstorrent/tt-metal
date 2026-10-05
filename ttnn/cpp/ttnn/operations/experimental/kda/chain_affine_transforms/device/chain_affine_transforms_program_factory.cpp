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
    const uint32_t Vt = attrs.value_dim / tt::constants::TILE_WIDTH;

    // One core per head. Splitting a head's value columns across cores re-reads the whole A block per core
    // (DRAM-bound at 4 blocks: 137 us vs 78 us at 1 block for Galaxy SP8xTP4); at 1 or 2 blocks the kernel is
    // bound by the step-to-step dependency chain instead.
    const auto grid = device.compute_with_storage_grid_size();
    const auto dist = kda_factory_detail::distribute_prep(grid, BH, BH);

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
        .compile_time_args = {{"Kt", Kt}, {"Vt", Vt}, {"BH", BH}},
        .runtime_arg_schema = {.runtime_arg_names = {"head"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
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

    m2::KernelRunArgs dataflow_run{.kernel = dataflow_kernel_name};
    for (uint32_t head = 0; head < BH; ++head) {
        m2::AddRuntimeArgsForNode(dataflow_run.runtime_arg_values, dist.cores[head], {{"head", head}});
    }
    m2::KernelRunArgs compute_run{.kernel = compute_kernel_name};

    m2::ProgramSpec spec{
        .name = "chain_affine_transforms",
        .dataflow_buffers = std::move(dfbs),
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
