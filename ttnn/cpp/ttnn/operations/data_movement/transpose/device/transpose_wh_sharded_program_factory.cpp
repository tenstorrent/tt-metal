// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_sharded_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <algorithm>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TransposeWHShardedProgramFactory::create_program_spec(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_wh needs to be allocated in a buffer on device!");

    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    const auto tile = input_tensor.tensor_spec().tile();
    const uint32_t tile_hw = tile.get_tile_hw();

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;

    auto shard_spec = input_tensor.shard_spec().value();

    auto& all_cores = shard_spec.grid;
    uint32_t num_tiles_per_shard = shard_spec.numel() / tile_hw;

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "transpose_wh_sharded";

    // Sharded CBs (legacy c_0 / c_16): borrowed-memory DFBs on the sharded input/output buffers. The
    // backing L1 address resolves at runtime from the input/output TensorArguments (the legacy factory
    // set CBDescriptor .buffer + relied on UpdateDynamicCircularBufferAddress on cache hit). total_size
    // tracked num_tiles_per_shard (which can vary across cache hits); the DFB's num_entries plays the
    // same role.
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"src0"},
            .entry_size = src0_single_tile_size,
            .num_entries = num_tiles_per_shard,
            .data_format_metadata = src0_cb_data_format,
            .borrowed_from = m2::TensorParamName{"input"},
        },
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"out"},
            .entry_size = dst_single_tile_size,
            .num_entries = num_tiles_per_shard,
            .data_format_metadata = dst_cb_data_format,
            .borrowed_from = m2::TensorParamName{"output"},
        },
    };

    // Reader (forked from eltwise/unary reader_unary_sharded.cpp): pushes the borrowed src0 DFB by
    // num_tiles_per_core; the legacy CT was the src0 CB index (now dfb::src0).
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
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

    // Writer (forked from data_movement/sharded writer_unary_sharded.cpp): waits on the borrowed out
    // DFB for num_units entries; the legacy CT was the output CB index (now dfb::out).
    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                                        "writer_unary_sharded_m2.cpp"},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"out"},
                    .accessor_name = "out",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_units"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    // Compute (forked from transpose_wh_sharded.cpp). The legacy CTs {cb_in, cb_out} collapse to the
    // dfb::src0 / dfb::out bindings; the work counts stay RTAs.
    m2::ComputeHardwareConfig compute_hw{.fp32_dest_acc_en = fp32_dest_acc_en};
    if (src0_cb_data_format == tt::DataFormat::Float32) {
        // Legacy set unpack_to_dest_mode[c_0]=UnpackToDestFp32 when fp32; preserve for the src0 DFB.
        compute_hw.unpack_to_dest_mode.insert(
            {m2::DFBSpecName{"src0"}, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32});
    }

    m2::KernelSpec compute{
        .unique_id = m2::KernelSpecName{"compute"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/"
                                        "transpose_wh_sharded_m2.cpp"},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"src0"},
                    .accessor_name = "src0",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"out"},
                    .accessor_name = "out",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"NHtWt", "HtWt", "N", "Ht", "Wt"},
            },
        .hw_config = compute_hw,
    };

    spec.kernels = {reader, writer, compute};
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output_tensor.tensor_spec()},
    };
    // All three kernels run on every shard core. Both DFBs are borrowed onto io tensors and
    // produced/consumed across reader/compute/writer — they share the single WorkUnitSpec on all_cores
    // (Local-DFB rule). The legacy factory additionally launched the kernels on the full grid with
    // zeroed args on the trailing no-op cores; those cores did no work, so targeting the active shard
    // cores is behavior-preserving.
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "transpose_wh_sharded",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute"}},
            .target_nodes = all_cores,
        },
    };

    // ---- ProgramRunArgs (mutable) ----
    auto padded_shape = input_tensor.padded_shape();
    auto shard_shape = shard_spec.shape;

    uint32_t H = padded_shape[2];
    uint32_t Hs = shard_shape[0], Ws = shard_shape[1];

    uint32_t Hts = Hs / tile.get_height();
    uint32_t Wts = Ws / tile.get_width();

    uint32_t Ht = H / tile.get_height();
    uint32_t Ht_per_shard = std::min(Ht, Hts);

    uint32_t num_hw_blocks_per_shard = Hts > Ht ? Hts / Ht : 1;

    uint32_t HtWt_tile_size = Ht_per_shard * Wts;
    uint32_t num_blocks = num_hw_blocks_per_shard * HtWt_tile_size;

    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};
    m2::KernelRunArgs compute_run{.kernel = m2::KernelSpecName{"compute"}};

    // Active shard cores get the real arg values (matching the legacy per-active-core args; the legacy
    // no-op tail cores are not part of this work unit).
    for (const auto& core : corerange_to_cores(all_cores)) {
        reader_run.runtime_arg_values.push_back({core, {{"num_tiles_per_core", num_blocks}}});
        writer_run.runtime_arg_values.push_back({core, {{"num_units", num_blocks}}});
        compute_run.runtime_arg_values.push_back(
            {core,
             {{"NHtWt", num_blocks},
              {"HtWt", HtWt_tile_size},
              {"N", num_hw_blocks_per_shard},
              {"Ht", Ht_per_shard},
              {"Wt", Wts}}});
    }

    run.kernel_run_args = {reader_run, writer_run, compute_run};
    run.tensor_args = {
        {m2::TensorParamName{"input"}, input_tensor.mesh_tensor()},
        {m2::TensorParamName{"output"}, output_tensor.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
