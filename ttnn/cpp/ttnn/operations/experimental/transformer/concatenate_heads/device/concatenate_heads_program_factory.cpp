// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concatenate_heads_device_operation.hpp"

#include "concatenate_heads_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using namespace tt;

namespace {
const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};
const DFBSpecName IN0{"in0"};
const TensorParamName INPUT{"input"};
const TensorParamName OUTPUT{"output"};
}  // namespace

ttnn::device_operation::ProgramArtifacts
ConcatenateHeadsDeviceOperation::ConcatenateHeadsProgramFactory::create_program_artifacts(
    const ConcatenateHeadsParams& operation_attributes, const ConcatenateHeadsInputs& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input.mesh_tensor();
    const auto& out = output.mesh_tensor();
    const auto& ashape = a.padded_shape();
    const auto& compute_with_storage_grid_size = operation_attributes.compute_with_storage_grid_size;

    tt::DataFormat dfb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt::tile_size(dfb_data_format);
    TT_ASSERT(tensor_args.input.buffer()->size() % single_tile_size == 0);

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // Output shape is: [B, 1, 384, 1024]
    uint32_t per_core_tiles = (ashape[1] * ashape[3]) / TILE_WIDTH;
    uint32_t in0_h_tiles = ashape[2] / TILE_HEIGHT;

    // These parameters are identical to out_* in multi_core_create_qkv_heads
    uint32_t in0_w = 64;
    uint32_t in0_w_tiles = in0_w / TILE_WIDTH;
    uint32_t in0_c = per_core_tiles / in0_w_tiles;
    uint32_t in0_HtWt = in0_h_tiles * in0_w_tiles;
    uint32_t in0_CHtWt = in0_c * in0_HtWt;

    // Parallelize ashape[2] (384 / 32 = 12 tiles) across columns
    // Parallelize ashape[0] (B) across rows
    uint32_t num_cores_x = ashape[2] / TILE_HEIGHT;
    uint32_t num_cores_y = ashape[0];
    TT_ASSERT(num_cores_x <= compute_with_storage_grid_size.x);
    TT_ASSERT(num_cores_y <= compute_with_storage_grid_size.y);
    CoreCoord core_range = {num_cores_x, num_cores_y};

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    ProgramSpec spec;
    spec.name = "concatenate_heads";

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    CoreRangeSet all_cores(CoreRange(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1}));

    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = INPUT,
        .spec = a.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT,
        .spec = out.tensor_spec(),
    });

    // Create dataflow buffers
    uint32_t dfb0_tiles = per_core_tiles * 2;  // double buffer
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN0,
        .entry_size = single_tile_size,
        .num_entries = dfb0_tiles,
        .data_format_metadata = dfb_data_format,
    });

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/concatenate_heads/device/kernels/dataflow/"
            "reader_tm_tile_layout_concat_heads.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN0,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT,
                    .accessor_name = "input",
                },
            },
        .compile_time_args =
            {
                // READER COMPILE TIME ARGS
                {"in0_w_tiles", in0_w_tiles},
                {"in0_c", in0_c},
                {"in0_HtWt", in0_HtWt},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"in0_tensor_tile_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/concatenate_heads/device/kernels/dataflow/"
            "writer_tm_tile_layout_concat_heads.cpp",
        .dfb_bindings =
            {
                // The writer drains the same buffer the reader fills.
                DFBBinding{
                    .dfb_spec_name = IN0,
                    .accessor_name = "out0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = OUTPUT,
                    .accessor_name = "output",
                },
            },
        .compile_time_args =
            {
                // WRITER COMPILE TIME ARGS
                {"in0_w_tiles", in0_w_tiles},
                {"in0_c", in0_c},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"out_tensor_tile_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (uint32_t core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for (uint32_t core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};
            uint32_t in0_tensor_tile_id = (core_idx_x * in0_w_tiles) + (core_idx_y * in0_CHtWt);

            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {
                    {"in0_tensor_tile_id", in0_tensor_tile_id},
                });
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {
                    {"out_tensor_tile_id", (core_idx_x + core_idx_y * num_cores_c) * per_core_tiles},
                });
        }
    }

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));

    spec.work_units.push_back(WorkUnitSpec{
        .name = "concatenate_heads",
        .kernels = {READER, WRITER},
        .target_nodes = all_cores,
    });

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args = {
        {INPUT, a},
        {OUTPUT, out},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
