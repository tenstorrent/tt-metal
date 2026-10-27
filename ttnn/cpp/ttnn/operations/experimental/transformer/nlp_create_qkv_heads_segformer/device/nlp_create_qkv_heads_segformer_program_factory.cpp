// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer/device/nlp_create_qkv_heads_segformer_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts
NlpCreateHeadsSegformerDeviceOperation::NlpCreateQkvHeadsSegformerProgramFactory::create_program_artifacts(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& a = tensor_args.input_tensor;
    const auto& ashape = a.padded_shape();

    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt::tile_size(data_format);
    TT_ASSERT(a.buffer()->size() % single_tile_size == 0);

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t per_tensor_tiles = ashape[3] / TILE_WIDTH;
    const uint32_t q_num_tiles_per_tensor = per_tensor_tiles;
    const uint32_t num_q_heads = q_num_tiles_per_tensor;  // hard-coding the head_dim = 32

    // Per output tensor args
    // Output shape for Q/K/V is: [B, head_num, s, 32] # Needs shuffling from [B, 1, s, hidden_dim]
    uint32_t q_out_h_tiles = ashape[2] / TILE_WIDTH;
    uint32_t q_out_w_tiles = 1;                                 // hard-coding the head_dim = 32
    uint32_t q_out_c = q_num_tiles_per_tensor / q_out_w_tiles;  // num_heads
    uint32_t q_out_HtWt = q_out_h_tiles * q_out_w_tiles;
    uint32_t q_out_CHtWt = q_out_c * q_out_HtWt;
    uint32_t q_num_tiles = num_q_heads * q_out_w_tiles;

    auto* device = a.device();
    CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of per_tensor_tiles per core
    uint32_t num_blocks = ashape[0] * ashape[1] * ashape[2] / TILE_HEIGHT;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    ttnn::Tensor& q = std::get<0>(output);

    TT_ASSERT(q.buffer() != nullptr, "Output q buffer should be allocated on device!");

    // The Metal 2.0 binding layer works with the Metalium tensor type; extract once.
    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& q_mesh_tensor = q.mesh_tensor();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName QV{"qv"};  // Q head tiles, reader -> writer
    const TensorParamName INPUT{"input"};
    const TensorParamName Q{"q"};

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_create_qkv_heads.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = QV,
                    .accessor_name = "qv",
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
                {"q_num_tiles", q_num_tiles},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_blocks", "in0_tensor_tile_id", "in1_tensor_tile_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer/device/kernels/dataflow/"
            "writer_tm_tile_layout_nlp_create_qkv_heads.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = QV,
                    .accessor_name = "qv",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = Q,
                    .accessor_name = "q",
                },
            },
        .compile_time_args =
            {
                {"q_out_h_tiles", q_out_h_tiles},
                {"q_out_w_tiles", q_out_w_tiles},
                {"q_out_HtWt", q_out_HtWt},
                {"q_out_c", num_q_heads},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_blocks", "q_out_h_dim", "q_out_tensor_tile_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    // Create dataflow buffers
    uint32_t qv_num_tiles = per_tensor_tiles * 2;  // double buffer
    DataflowBufferSpec qv_dfb{
        .unique_id = QV,
        .entry_size = single_tile_size,
        .num_entries = qv_num_tiles,
        .data_format_metadata = data_format,
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_blocks_per_core = 0;
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {
                {"num_blocks", num_blocks_per_core},
                {"in0_tensor_tile_id", num_blocks_written * per_tensor_tiles},
                {"in1_tensor_tile_id", 0u},
            });

        uint32_t q_out_h_dim = num_blocks_written % q_out_h_tiles;
        uint32_t q_out_tensor_tile_id =
            (num_blocks_written / q_out_h_tiles * q_out_CHtWt) + (q_out_h_dim * q_out_w_tiles);

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"num_blocks", num_blocks_per_core},
                {"q_out_h_dim", q_out_h_dim},
                {"q_out_tensor_tile_id", q_out_tensor_tile_id},
            });
        num_blocks_written += num_blocks_per_core;
    }

    ProgramSpec spec{
        .name = "nlp_create_qkv_heads_segformer",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(qv_dfb)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = Q, .spec = q_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "all_cores",
                    .kernels = {READER, WRITER},
                    .target_nodes = all_cores,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {INPUT, input_mesh_tensor},
        {Q, q_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
