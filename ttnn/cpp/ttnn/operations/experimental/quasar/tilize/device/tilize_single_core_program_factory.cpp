// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_single_core_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
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
const TensorParamName SC_INPUT_TENSOR{"input"};
const TensorParamName SC_OUTPUT_TENSOR{"output"};
const DFBSpecName SC_INPUT_DFB{"in"};
const DFBSpecName SC_OUTPUT_DFB{"out"};
const KernelSpecName SC_READER_KERNEL{"reader"};
const KernelSpecName SC_WRITER_KERNEL{"writer"};
const KernelSpecName SC_COMPUTE_KERNEL{"compute"};
}  // namespace

ttnn::device_operation::ProgramArtifacts TilizeSingleCoreProgramFactory::create_program_artifacts(
    const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    CoreRange default_core({0, 0}, {0, 0});
    CoreRange core = sub_core_grids.has_value() ? corerange_to_cores(sub_core_grids.value()).at(0) : default_core;

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    uint32_t num_tiles = a.physical_volume() / TILE_HW;

    auto width = a.padded_shape()[-1];
    uint32_t stick_s = width;
    uint32_t num_sticks = a.physical_volume() / width;

    uint32_t num_tiles_in_row = stick_s / TILE_WIDTH;
    uint32_t num_tiles_per_block = 1;

    if (!operation_attributes.use_low_perf) {
        // Ensure we don't intrude into storage space
        uint32_t max_l1_size =
            (a.device()->l1_size_per_core() / 2) - a.device()->allocator()->get_base_allocator_addr(HalMemType::L1);
        uint32_t max_tiles = max_l1_size / (input_single_tile_size + output_single_tile_size);  // 2 CBs
        // Currently need the number of tiles in a row to be divisible by tiles in a block
        if (num_tiles_in_row <= max_tiles) {
            num_tiles_per_block = num_tiles_in_row;
        } else {
            for (uint32_t n_t = max_tiles; n_t > 0; n_t--) {
                if (num_tiles_in_row % n_t == 0) {
                    num_tiles_per_block = n_t;
                    break;
                }
            }
        }
    }

    uint32_t block_width_size = num_tiles_per_block * TILE_WIDTH * a.element_size();
    uint32_t num_full_blocks_in_row = num_tiles_in_row / num_tiles_per_block;

    const uint32_t num_input_tiles = num_tiles_per_block;
    const uint32_t num_output_tiles = num_tiles_per_block;

    // -- Spec --
    ProgramSpec spec;
    spec.name = "tilize_single_core";
    spec.tensor_parameters = {
        TensorParameter{.unique_id = SC_INPUT_TENSOR, .spec = a.tensor_spec()},
        TensorParameter{.unique_id = SC_OUTPUT_TENSOR, .spec = output.tensor_spec()},
    };

    spec.dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = SC_INPUT_DFB,
            .entry_size = input_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = input_cb_data_format,
        },
        DataflowBufferSpec{
            .unique_id = SC_OUTPUT_DFB,
            .entry_size = output_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = output_cb_data_format,
        },
    };

    // -- Reader kernel --
    KernelSpec reader{
        .unique_id = SC_READER_KERNEL,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/tilize/device/kernels/dataflow/"
            "reader_unary_stick_layout_split_rows_singlecore.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SC_INPUT_DFB,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = SC_INPUT_TENSOR, .accessor_name = "src"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_sticks", "num_tiles_per_block", "block_width_size", "num_full_blocks_in_row", "start_stick_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(a.device()->arch()),
    };

    // -- Writer kernel --
    KernelSpec writer{
        .unique_id = SC_WRITER_KERNEL,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/tilize/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SC_OUTPUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = SC_OUTPUT_TENSOR, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(a.device()->arch()),
    };

    // -- Compute kernel --
    ttnn::ComputeKernelConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_llk_acc};
    ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(a.device()->arch(), compute_config);
    if (fp32_llk_acc) {
        std::visit(
            [&](auto& c) { c.unpack_to_dest_mode.emplace(SC_INPUT_DFB, UnpackToDestMode::UnpackToDestFp32); },
            compute_hw);
    }
    KernelSpec compute{
        .unique_id = SC_COMPUTE_KERNEL,
        .source = "ttnn/cpp/ttnn/operations/experimental/quasar/tilize/device/kernels/compute/tilize.cpp",
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = SC_INPUT_DFB,
                 .accessor_name = "in",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = SC_OUTPUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .compile_time_args =
            {{"per_core_block_cnt", num_tiles / num_tiles_per_block}, {"per_core_block_tile_cnt", num_tiles_per_block}},
        .hw_config = compute_hw,
    };

    spec.kernels = {reader, writer, compute};
    spec.work_units = {WorkUnitSpec{
        .name = "tilize_single_core_wu",
        .kernels = {SC_READER_KERNEL, SC_WRITER_KERNEL, SC_COMPUTE_KERNEL},
        .target_nodes = core.start_coord,
    }};

    // -- Run args --
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{
            .kernel = SC_READER_KERNEL,
            .runtime_arg_values = {KernelRunArgs::NodeRuntimeArgs{
                .node = core.start_coord,
                .args =
                    {{"num_sticks", num_sticks},
                     {"num_tiles_per_block", num_tiles_per_block},
                     {"block_width_size", block_width_size},
                     {"num_full_blocks_in_row", num_full_blocks_in_row},
                     {"start_stick_id", 0u}}}}},
        KernelRunArgs{
            .kernel = SC_WRITER_KERNEL,
            .runtime_arg_values = {KernelRunArgs::NodeRuntimeArgs{
                .node = core.start_coord, .args = {{"num_pages", num_tiles}, {"start_id", 0u}}}}},
    };
    run_args.tensor_args = {
        {SC_INPUT_TENSOR, TensorArgument{a.mesh_tensor()}},
        {SC_OUTPUT_TENSOR, TensorArgument{output.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
