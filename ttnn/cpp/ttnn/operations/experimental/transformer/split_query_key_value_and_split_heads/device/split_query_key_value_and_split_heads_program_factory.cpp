// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "split_query_key_value_and_split_heads_device_operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts SplitFusedQKVAndSplitHeadsProgramFactory::create_program_artifacts(
    const SplitQueryKeyValueAndSplitHeadsParams& operation_attributes,
    const SplitQueryKeyValueAndSplitHeadsInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName K_PRE_TRANSPOSE{"k_pre_transpose"};
    const DFBSpecName QV{"qv"};
    const DFBSpecName K_TRANSPOSED{"k_transposed"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT_Q{"q"};
    const TensorParamName OUTPUT_K{"k"};
    const TensorParamName OUTPUT_V{"v"};

    const auto& a = tensor_args.input_tensor;
    auto& output = output_tensors;
    auto compute_with_storage_grid_size = operation_attributes.compute_with_storage_grid_size;

    const auto& ashape = a.padded_shape();

    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt::tile_size(data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t per_core_tiles = ashape[3] / TILE_WIDTH;  // 96
    uint32_t num_tensors = 3;
    uint32_t num_tiles_per_tensor = per_core_tiles / num_tensors;  // 32
    uint32_t block_size = 1;  // TODO: Play around with different reader and writer block_sizes?
    bool block_size_is_one = block_size == 1;
    uint32_t num_blocks_per_tensor = num_tiles_per_tensor / block_size;

    // Per output tensor args
    // Output shape is: [B, 16, 384, 64] (Q, V heads) or [B, 16, 64, 384] (K heads)
    // For K heads, we write "w_dim" to h_dim instead, but keep nomenclature the same
    uint32_t out_h_tiles = ashape[2] / TILE_HEIGHT;
    uint32_t out_w = 64;
    uint32_t out_w_tiles = out_w / TILE_WIDTH;
    uint32_t out_c = num_tiles_per_tensor / out_w_tiles;
    uint32_t out_HtWt = out_h_tiles * out_w_tiles;
    uint32_t out_CHtWt = out_c * out_HtWt;
    // If block_size_is_one, writer kernel waits differently
    uint32_t writer_num_blocks_per_tensor = block_size_is_one ? out_c : num_blocks_per_tensor;
    uint32_t num_c_per_block = block_size_is_one ? 1 : block_size / out_w_tiles;

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
    TT_ASSERT((output.size() == 3), "Output vector must be size 3 for split fused qkv!");
    ttnn::Tensor& q = output[0];
    ttnn::Tensor& k = output[1];
    ttnn::Tensor& v = output[2];

    tt_metal::Buffer* q_buffer = q.buffer();
    TT_ASSERT(q_buffer != nullptr, "Output q buffer should be allocated on device!");
    tt_metal::Buffer* k_buffer = k.buffer();
    TT_ASSERT(k_buffer != nullptr, "Output k buffer should be allocated on device!");
    tt_metal::Buffer* v_buffer = v.buffer();
    TT_ASSERT(v_buffer != nullptr, "Output v buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    ProgramSpec spec;
    spec.name = "split_query_key_value_and_split_heads";

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    CoreRangeSet all_cores(CoreRange(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1}));

    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = INPUT,
        .spec = a.mesh_tensor().tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT_Q,
        .spec = q.mesh_tensor().tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT_K,
        .spec = k.mesh_tensor().tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT_V,
        .spec = v.mesh_tensor().tensor_spec(),
    });

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
            "dataflow/reader_tm_tile_layout_create_qkv_heads.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = K_PRE_TRANSPOSE,
                    .accessor_name = "k",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
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
                // READER COMPILE TIME ARGS
                {"block_size", block_size},
                {"out_num_blocks_per_tensor", num_blocks_per_tensor},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"in0_tensor_tile_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
            "dataflow/writer_tm_tile_layout_create_qkv_heads.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = K_TRANSPOSED,
                    .accessor_name = "k",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = QV,
                    .accessor_name = "qv",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = OUTPUT_Q,
                    .accessor_name = "q",
                },
                TensorBinding{
                    .tensor_parameter_name = OUTPUT_K,
                    .accessor_name = "k",
                },
                TensorBinding{
                    .tensor_parameter_name = OUTPUT_V,
                    .accessor_name = "v",
                },
            },
        .compile_time_args =
            {
                // WRITER COMPILE TIME ARGS
                {"block_size_is_one", static_cast<uint32_t>(block_size_is_one)},
                {"block_size", block_size},
                {"out_num_blocks_per_tensor", writer_num_blocks_per_tensor},
                {"out_num_c_per_block", num_c_per_block},
                {"out_w_tiles", out_w_tiles},
                {"out_h_tiles", out_h_tiles},
                {"out_HtWt", out_HtWt},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"out_tensor_tile_id", "out_tensor_tile_id_with_transpose"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    // Dummy compute kernel
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/kernel/compute/transpose_wh_metal2.cpp",
        // The legacy compute descriptor resolved to O3; Metal 2.0 defaults every kernel to O2.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = K_PRE_TRANSPOSE,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = K_TRANSPOSED,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args = {{"NHtWt", num_tiles_per_tensor}},
        .hw_config = ComputeHardwareConfig{},
    };

    // Create dataflow buffers
    // Use k_pre_transpose and k_transposed for K heads, which uses compute for transpose_wh
    uint32_t k_pre_transpose_tiles = num_tiles_per_tensor * 2;  // double buffer
    uint32_t k_transposed_tiles = num_tiles_per_tensor;
    // Use qv as input and output for Q and V heads
    uint32_t qv_tiles = num_tiles_per_tensor * 4;  // 2 tensors + double buffer
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = K_PRE_TRANSPOSE,
        .entry_size = single_tile_size,
        .num_entries = k_pre_transpose_tiles,
        .data_format_metadata = data_format,
    });

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = QV,
        .entry_size = single_tile_size,
        .num_entries = qv_tiles,
        .data_format_metadata = data_format,
    });

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = K_TRANSPOSED,
        .entry_size = single_tile_size,
        .num_entries = k_transposed_tiles,
        .data_format_metadata = data_format,
    });

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (uint32_t core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for (uint32_t core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};

            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {
                    {"in0_tensor_tile_id", (core_idx_x + core_idx_y * num_cores_c) * per_core_tiles},
                });
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {
                    {"out_tensor_tile_id", (core_idx_x * out_w_tiles) + (core_idx_y * out_CHtWt)},
                    {"out_tensor_tile_id_with_transpose", core_idx_x + (core_idx_y * out_CHtWt)},
                });
        }
    }

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.kernels.push_back(std::move(compute));

    spec.work_units.push_back(WorkUnitSpec{
        .name = "split_query_key_value_and_split_heads",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = all_cores,
    });

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {INPUT, a.mesh_tensor()},
        {OUTPUT_Q, q.mesh_tensor()},
        {OUTPUT_K, k.mesh_tensor()},
        {OUTPUT_V, v.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
