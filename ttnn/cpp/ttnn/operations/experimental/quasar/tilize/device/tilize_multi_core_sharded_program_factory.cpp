// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_sharded_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
const TensorParamName INPUT_TENSOR{"input"};
const TensorParamName OUTPUT_TENSOR{"output"};
const DFBSpecName INPUT_DFB{"in"};
const DFBSpecName OUTPUT_DFB{"out"};
const KernelSpecName READER_KERNEL{"reader"};
const KernelSpecName WRITER_KERNEL{"writer"};
const KernelSpecName COMPUTE_KERNEL{"compute"};
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts TilizeMultiCoreShardedProgramFactory::create_program_artifacts(
    const TilizeParams& /*operation_attributes*/, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE;  // resolve the file-local ids below
    const Tensor& input = tensor_args.input_tensor;
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

    // -- Tensor parameters --
    ProgramSpec spec;
    spec.name = "tilize_multi_core_sharded";
    spec.tensor_parameters = {
        TensorParameter{.unique_id = INPUT_TENSOR, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()},
    };

    // -- Dataflow buffers (borrowed memory: input buffer backs c_0, output buffer backs c_16) --
    spec.dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = INPUT_DFB,
            .entry_size = input_single_tile_size,
            .num_entries = num_tiles_per_shard,
            .data_format_metadata = input_cb_data_format,
            .borrowed_from = INPUT_TENSOR,
        },
        DataflowBufferSpec{
            .unique_id = OUTPUT_DFB,
            .entry_size = output_single_tile_size,
            .num_entries = num_tiles_per_shard,
            .data_format_metadata = output_cb_data_format,
            .borrowed_from = OUTPUT_TENSOR,
        },
    };

    // -- Reader kernel --
    KernelSpec reader{
        .unique_id = READER_KERNEL,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/tilize/device/kernels/dataflow/reader_unary_sharded.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = INPUT_DFB,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core"}},
        .hw_config = ttnn::create_reader_datamovement_config(input.device()->arch()),
    };

    // -- Writer kernel --
    KernelSpec writer{
        .unique_id = WRITER_KERNEL,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/tilize/device/kernels/dataflow/writer_unary_sharded.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUTPUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"num_units"}},
        .hw_config = ttnn::create_writer_datamovement_config(input.device()->arch()),
    };

    // -- Compute kernel --
    ttnn::ComputeKernelConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_llk_acc};
    ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(input.device()->arch(), compute_config);
    if (fp32_llk_acc) {
        std::visit([&](auto& c) { c.unpack_modes.emplace(INPUT_DFB, UnpackMode::UnpackToDest); }, compute_hw);
    }
    KernelSpec compute{
        .unique_id = COMPUTE_KERNEL,
        .source = "ttnn/cpp/ttnn/operations/experimental/quasar/tilize/device/kernels/compute/tilize.cpp",
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = INPUT_DFB,
                 .accessor_name = "in",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = OUTPUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .compile_time_args =
            {{"per_core_block_cnt", num_tiles_per_shard / num_tiles_per_row},
             {"per_core_block_tile_cnt", num_tiles_per_row}},
        .hw_config = compute_hw,
    };

    spec.kernels = {reader, writer, compute};
    spec.work_units = {WorkUnitSpec{
        .name = "tilize_sharded_wu",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
        .target_nodes = all_cores,
    }};

    // -- Run args --
    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    for (const auto& core : corerange_to_cores(all_cores)) {
        reader_run.runtime_arg_values["num_tiles_per_core"][core] = num_tiles_per_shard;
        writer_run.runtime_arg_values["num_units"][core] = num_tiles_per_shard;
    }
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{input.mesh_tensor()}},
        {OUTPUT_TENSOR, TensorArgument{output.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
