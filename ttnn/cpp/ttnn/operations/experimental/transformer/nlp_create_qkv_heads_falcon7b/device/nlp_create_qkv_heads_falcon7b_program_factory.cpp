// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_falcon7b_device_operation.hpp"

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
NlpCreateHeadsFalcon7BDeviceOperation::NlpCreateQkvHeadsFalcon7BProgramFactory::create_program_artifacts(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& a = tensor_args;
    const auto& ashape = a.padded_shape();

    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt::tile_size(data_format);
    TT_ASSERT(a.buffer()->size() % single_tile_size == 0);

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t per_tensor_tiles = ashape[3] / TILE_WIDTH;  // 146
    uint32_t q_num_tiles_per_tensor = 142;
    uint32_t kv_num_tiles_per_tensor = 2;

    // Per output tensor args
    // Output shape for Q is: [B, 71, s, 64] # Needs shuffling from [B, 1, s, 4544]
    // Output shape for K/V is: [B, 1, s, 64] # Just split, no shuffling after
    uint32_t q_out_h_tiles = ashape[2] / TILE_HEIGHT;
    uint32_t q_out_w_tiles = 2;                                 // head_dim
    uint32_t q_out_c = q_num_tiles_per_tensor / q_out_w_tiles;  // num_heads
    uint32_t q_out_HtWt = q_out_h_tiles * q_out_w_tiles;
    uint32_t q_out_CHtWt = q_out_c * q_out_HtWt;

    CoreCoord compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of per_tensor_tiles per core
    uint32_t num_blocks = ashape[0] * ashape[1] * ashape[2] / TILE_HEIGHT;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    ttnn::Tensor& q = tensor_return_value.q;
    ttnn::Tensor& k = tensor_return_value.k;
    ttnn::Tensor& v = tensor_return_value.v;

    TT_ASSERT(q.buffer() != nullptr, "Output q buffer should be allocated on device!");
    TT_ASSERT(k.buffer() != nullptr, "Output k buffer should be allocated on device!");
    TT_ASSERT(v.buffer() != nullptr, "Output v buffer should be allocated on device!");

    // The Metal 2.0 binding layer works with the Metalium tensor type; extract once.
    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& q_mesh_tensor = q.mesh_tensor();
    const auto& k_mesh_tensor = k.mesh_tensor();
    const auto& v_mesh_tensor = v.mesh_tensor();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName QKV{"qkv"};  // fused QKV input tiles, reader -> writer
    const TensorParamName INPUT{"input"};
    const TensorParamName Q{"q"};
    const TensorParamName K{"k"};
    const TensorParamName V{"v"};

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "reader_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = QKV,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT,
                    .accessor_name = "src",
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_falcon7b/device/kernels/dataflow/"
            "writer_tm_tile_layout_nlp_create_qkv_heads_falcon7b.cpp",
        .dfb_bindings =
            {
                // Same buffer the reader fills: the writer drains the fused QKV tiles straight to Q/K/V.
                DFBBinding{
                    .dfb_spec_name = QKV,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = Q,
                    .accessor_name = "q",
                },
                TensorBinding{
                    .tensor_parameter_name = K,
                    .accessor_name = "k",
                },
                TensorBinding{
                    .tensor_parameter_name = V,
                    .accessor_name = "v",
                },
            },
        .compile_time_args =
            {
                {"q_num_tiles", q_num_tiles_per_tensor},
                {"kv_num_tiles", kv_num_tiles_per_tensor},
                {"q_out_h_tiles", q_out_h_tiles},
                {"q_out_w_tiles", q_out_w_tiles},
                {"q_out_c", q_out_c},
                {"q_out_HtWt", q_out_HtWt},
            },
        .runtime_arg_schema =
            {.runtime_arg_names = {"num_blocks", "q_out_h_dim", "q_out_tensor_tile_id", "kv_out_tensor_tile_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    // Create dataflow buffers
    uint32_t qkv_num_tiles = per_tensor_tiles * 2;  // double buffer
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = QKV,
            .entry_size = single_tile_size,
            .num_entries = qkv_num_tiles,
            .data_format_metadata = data_format,
        },
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
                {"num_pages", num_blocks_per_core * per_tensor_tiles},
                {"start_id", num_blocks_written * per_tensor_tiles},
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
                {"kv_out_tensor_tile_id", num_blocks_written * kv_num_tiles_per_tensor},
            });

        num_blocks_written += num_blocks_per_core;
    }

    // One work unit per work-split core group; together they cover all_cores, where the legacy reader and
    // writer ran.
    Group<WorkUnitSpec> work_units = {
        WorkUnitSpec{
            .name = "core_group_1",
            .kernels = {READER, WRITER},
            .target_nodes = core_group_1,
        },
    };
    if (core_group_2.num_cores() > 0) {
        work_units.push_back(WorkUnitSpec{
            .name = "core_group_2",
            .kernels = {READER, WRITER},
            .target_nodes = core_group_2,
        });
    }

    ProgramSpec spec{
        .name = "nlp_create_qkv_heads_falcon7b",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = Q, .spec = q_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = K, .spec = k_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = V, .spec = v_mesh_tensor.tensor_spec()},
            },
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {INPUT, input_mesh_tensor},
        {Q, q_mesh_tensor},
        {K, k_mesh_tensor},
        {V, v_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
