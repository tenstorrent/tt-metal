// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_sharded_program_factory.hpp"

#include <filesystem>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

// Metal 2.0 program factory: builds the immutable ProgramSpec and its mutable ProgramRunArgs.
// Behavior-preserving port of the legacy ProgramDescriptor multi-core (height-)sharded tilize factory.
ttnn::device_operation::ProgramArtifacts TilizeMultiCoreShardedProgramFactory::create_program_spec(
    const TilizeParams& /*operation_attributes*/, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);
    bool fp32_llk_acc = input.dtype() == DataType::FLOAT32 || input.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    auto shard_spec = input.shard_spec().value();
    uint32_t num_tiles_per_shard = shard_spec.shape[0] * shard_spec.shape[1] / TILE_HW;
    uint32_t num_tiles_per_row = shard_spec.shape[1] / TILE_WIDTH;
    const CoreRangeSet& all_cores = shard_spec.grid;

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "tilize_multi_core_sharded";

    // "src0" (legacy CB c_0): borrowed-memory DFB on the sharded input buffer (io is L1) —
    // backing address resolves at runtime from the "input" TensorArgument.
    // "output" (legacy CB c_16): borrowed-memory DFB on the sharded output buffer.
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"src0"},
            .entry_size = input_single_tile_size,
            .num_entries = num_tiles_per_shard,
            .data_format_metadata = input_cb_data_format,
            .borrowed_from = m2::TensorParamName{"input"},
        },
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"output"},
            .entry_size = output_single_tile_size,
            .num_entries = num_tiles_per_shard,
            .data_format_metadata = output_cb_data_format,
            .borrowed_from = m2::TensorParamName{"output"},
        },
    };

    // Reader (forked from eltwise/unary reader_unary_sharded.cpp). Just pushes the borrowed
    // src0 DFB by num_tiles_per_core; the legacy CT was the src0 CB index (now dfb::src0).
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
                                        "reader_unary_sharded_m2.cpp"},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"src0"},
                    .accessor_name = "src0",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_tiles_per_core"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };

    // Writer (forked from data_movement/sharded writer_unary_sharded.cpp). Waits on the borrowed
    // output DFB for num_units entries; the legacy CT was the output CB index (now dfb::output).
    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
                                        "writer_unary_sharded_m2.cpp"},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"output"},
                    .accessor_name = "output",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_units"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    // Compute (reuses the single-core port's fork of ttnn/cpp/ttnn/kernel/compute/tilize.cpp).
    // The legacy sharded compute used the same shared source with CTs
    // {cb_in0, cb_out, per_core_block_cnt, per_core_block_tile_cnt}; the first two collapse to
    // dfb::src0 / dfb::output bindings, the latter two stay CTAs.
    //   per_core_block_cnt      = num_tiles_per_shard / num_tiles_per_row
    //   per_core_block_tile_cnt = num_tiles_per_row
    m2::KernelSpec compute{
        .unique_id = m2::KernelSpecName{"compute"},
        .source =
            std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_m2.cpp"},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"src0"},
                    .accessor_name = "src0",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"output"},
                    .accessor_name = "output",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args =
            {
                {"per_core_block_cnt", num_tiles_per_shard / num_tiles_per_row},
                {"per_core_block_tile_cnt", num_tiles_per_row},
            },
        .hw_config =
            m2::ComputeHardwareConfig{
                .fp32_dest_acc_en = fp32_llk_acc,
            },
    };
    // Legacy set unpack_to_dest_mode[c_0]=UnpackToDestFp32 when fp32_llk_acc; preserve that for
    // the src0 DFB (the compute consumer of the fp32 input).
    if (fp32_llk_acc) {
        std::get<m2::ComputeHardwareConfig>(compute.hw_config).unpack_to_dest_mode = {
            {m2::DFBSpecName{"src0"}, UnpackToDestMode::UnpackToDestFp32}};
    }

    spec.kernels = {reader, writer, compute};
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()},
    };

    // All three kernels run on every shard core. Both DFBs are borrowed onto io tensors and
    // produced/consumed across reader/compute/writer — they share the single WorkUnitSpec on
    // all_cores so every node hosting a DFB hosts both its producer and consumer (Local-DFB rule).
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "tilize_multi_core_sharded",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute"}},
            .target_nodes = all_cores,
        },
    };

    // ---- ProgramRunArgs (mutable) ----
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};

    // Sharded readers/writers consume only the (constant per-launch) num_tiles_per_shard count.
    for (const auto& core : corerange_to_cores(all_cores)) {
        reader_run.runtime_arg_values.push_back({core, {{"num_tiles_per_core", num_tiles_per_shard}}});
        writer_run.runtime_arg_values.push_back({core, {{"num_units", num_tiles_per_shard}}});
    }

    run.kernel_run_args = {reader_run, writer_run};
    run.tensor_args = {
        {m2::TensorParamName{"input"}, input.mesh_tensor()},
        {m2::TensorParamName{"output"}, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
