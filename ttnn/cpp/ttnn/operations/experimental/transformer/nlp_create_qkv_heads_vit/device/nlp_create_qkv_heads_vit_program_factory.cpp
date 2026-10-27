// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/device/nlp_create_qkv_heads_vit_device_operation.hpp"

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
NlpCreateHeadsVitDeviceOperation::NlpCreateQkvHeadsVitProgramFactory::create_program_artifacts(
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
    uint32_t per_tensor_tiles = ashape[3] / TILE_WIDTH;  // 72
    const uint32_t q_num_tiles_per_tensor = 24;
    const uint32_t num_q_heads = 12;
    const uint32_t num_kv_heads = 12;

    // Per output tensor args
    // Output shape for Q,K,V is: [B, 12, s, 64] # Needs shuffling from [B, 1, s, 2304]
    uint32_t q_out_h_tiles = ashape[2] / TILE_HEIGHT;
    uint32_t q_out_w_tiles = 2;                                 // head_dim
    uint32_t q_out_c = q_num_tiles_per_tensor / q_out_w_tiles;  // num_heads
    uint32_t q_out_HtWt = q_out_h_tiles * q_out_w_tiles;
    uint32_t q_out_CHtWt = q_out_c * q_out_HtWt;
    uint32_t kv_out_CHtWt = num_kv_heads * q_out_HtWt;
    uint32_t q_num_tiles = num_q_heads * q_out_w_tiles;
    uint32_t kv_num_tiles = num_kv_heads * q_out_w_tiles;

    CoreCoord compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of per_tensor_tiles per core
    uint32_t num_blocks = ashape[0] * ashape[1] * ashape[2] / TILE_HEIGHT;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    TT_ASSERT((output.size() == 3), "Output vector must be size 3 for split fused qkv!");
    ttnn::Tensor& q = output[0];
    ttnn::Tensor& k = output[1];
    ttnn::Tensor& v = output[2];

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
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};
    const DFBSpecName QV{"qv"};        // Q and V head tiles, reader -> writer (K too unless transpose_k_heads)
    const DFBSpecName K_IN{"k_in"};    // K head tiles, reader -> compute (transpose_k_heads only)
    const DFBSpecName K_OUT{"k_out"};  // transposed K head tiles, compute -> writer (transpose_k_heads only)
    const TensorParamName INPUT{"input"};
    const TensorParamName Q{"q"};
    const TensorParamName K{"k"};
    const TensorParamName V{"v"};

    Group<KernelSpec> kernels;
    Group<KernelSpecName> work_unit_kernels_g1 = {READER, WRITER};
    Group<KernelSpecName> work_unit_kernels_g2 = {READER, WRITER};

    ///////////// K transpose ////////////////////
    const bool transpose_k_heads = false;
    KernelSpec::CompilerOptions::Defines reader_defines;
    KernelSpec::CompilerOptions::Defines writer_defines;
    if (transpose_k_heads) {
        // One compute spec per work-split core group: the block count is a compile-time argument, so
        // each group compiles its own instance.
        auto make_compute = [&](const KernelSpecName& unique_id, uint32_t NHtWt) {
            return KernelSpec{
                .unique_id = unique_id,
                .source = "ttnn/cpp/ttnn/kernel/compute/transpose_wh_metal2.cpp",
                // The legacy compute descriptor resolved to O3; Metal 2.0 defaults every kernel to O2.
                .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
                .dfb_bindings =
                    {
                        DFBBinding{
                            .dfb_spec_name = K_IN,
                            .accessor_name = "in",
                            .endpoint_type = DFBEndpointType::CONSUMER,
                        },
                        DFBBinding{
                            .dfb_spec_name = K_OUT,
                            .accessor_name = "out",
                            .endpoint_type = DFBEndpointType::PRODUCER,
                        },
                    },
                .compile_time_args = {{"NHtWt", NHtWt}},
                // The legacy compute descriptor left every field at its default; so does this one.
                .hw_config = ComputeHardwareConfig{},
            };
        };
        kernels.push_back(make_compute(COMPUTE_G1, num_blocks_per_core_group_1 * kv_num_tiles));
        work_unit_kernels_g1.push_back(COMPUTE_G1);

        if (core_group_2.num_cores() > 0) {
            kernels.push_back(make_compute(COMPUTE_G2, num_blocks_per_core_group_2 * kv_num_tiles));
            work_unit_kernels_g2.push_back(COMPUTE_G2);
        }
        reader_defines.emplace("TRANSPOSE_K_HEADS", "1");
        writer_defines.emplace("TRANSPOSE_K_HEADS", "1");
    }
    //////////////////////////////////////////////

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_create_qkv_heads.cpp",
        .compiler_options = {.defines = reader_defines},
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
                {"kv_num_tiles", kv_num_tiles},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_blocks", "in0_tensor_tile_id", "in1_tensor_tile_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/device/kernels/dataflow/"
            "writer_tm_tile_layout_nlp_create_qkv_heads.cpp",
        .compiler_options = {.defines = writer_defines},
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
                {"q_out_h_tiles", q_out_h_tiles},
                {"q_out_w_tiles", q_out_w_tiles},
                {"q_out_HtWt", q_out_HtWt},
                {"q_out_c", num_q_heads},
                {"kv_out_c", num_kv_heads},
            },
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_blocks", "q_out_h_dim", "q_out_tensor_tile_id", "k_out_tensor_tile_id", "v_out_tensor_tile_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    // Create dataflow buffers
    uint32_t qv_num_tiles = per_tensor_tiles * 2;  // double buffer
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = QV,
            .entry_size = single_tile_size,
            .num_entries = qv_num_tiles,
            .data_format_metadata = data_format,
        },
    };

    // If we transpose_k_heads:
    // - reader will write to k_in, instead of qv
    // - compute will wait on k_in and write to k_out
    // - writer will wait on k_out, instead of qv
    if (transpose_k_heads) {
        reader.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = K_IN,
            .accessor_name = "k",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        writer.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = K_OUT,
            .accessor_name = "k",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });

        uint32_t k_in_num_tiles = per_tensor_tiles * 2;  // double buffer
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = K_IN,
            .entry_size = single_tile_size,
            .num_entries = k_in_num_tiles,
            .data_format_metadata = data_format,
        });

        uint32_t k_out_num_tiles = per_tensor_tiles * 2;  // double buffer
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = K_OUT,
            .entry_size = single_tile_size,
            .num_entries = k_out_num_tiles,
            .data_format_metadata = data_format,
        });
    }

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
        uint32_t v_out_tensor_tile_id =
            (num_blocks_written / q_out_h_tiles * kv_out_CHtWt) + (q_out_h_dim * q_out_w_tiles);
        uint32_t k_out_tensor_tile_id = transpose_k_heads
                                            ? (num_blocks_written / q_out_h_tiles * kv_out_CHtWt) + q_out_h_dim
                                            : v_out_tensor_tile_id;

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"num_blocks", num_blocks_per_core},
                {"q_out_h_dim", q_out_h_dim},
                {"q_out_tensor_tile_id", q_out_tensor_tile_id},
                {"k_out_tensor_tile_id", k_out_tensor_tile_id},
                {"v_out_tensor_tile_id", v_out_tensor_tile_id},
            });
        num_blocks_written += num_blocks_per_core;
    }

    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));

    // One work unit per work-split core group: the reader and writer run on both (together, all_cores),
    // the per-group compute instance (transpose_k_heads only) on its own group.
    Group<WorkUnitSpec> work_units = {
        WorkUnitSpec{
            .name = "core_group_1",
            .kernels = std::move(work_unit_kernels_g1),
            .target_nodes = core_group_1,
        },
    };
    if (core_group_2.num_cores() > 0) {
        work_units.push_back(WorkUnitSpec{
            .name = "core_group_2",
            .kernels = std::move(work_unit_kernels_g2),
            .target_nodes = core_group_2,
        });
    }

    ProgramSpec spec{
        .name = "nlp_create_qkv_heads_vit",
        .kernels = std::move(kernels),
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
