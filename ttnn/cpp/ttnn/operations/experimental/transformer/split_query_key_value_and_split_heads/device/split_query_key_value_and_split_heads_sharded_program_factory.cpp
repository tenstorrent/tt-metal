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

ttnn::device_operation::ProgramArtifacts SplitFusedQKVAndSplitHeadsShardedProgramFactory::create_program_artifacts(
    const SplitQueryKeyValueAndSplitHeadsParams& /*operation_attributes*/,
    const SplitQueryKeyValueAndSplitHeadsInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName IN{"in"};
    const DFBSpecName K_PRE_TRANSPOSE{"k_pre_transpose"};
    const DFBSpecName OUT_Q{"out_q"};
    const DFBSpecName OUT_K{"out_k"};
    const DFBSpecName OUT_V{"out_v"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT_Q{"q"};
    const TensorParamName OUTPUT_K{"k"};
    const TensorParamName OUTPUT_V{"v"};

    const auto& a = tensor_args.input_tensor;
    auto& output = output_tensors;

    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt::tile_size(data_format);

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    auto all_cores = a.shard_spec().value().grid;
    auto bbox = all_cores.bounding_box();
    ShardOrientation shard_orientation = a.shard_spec().value().orientation;
    bool rm = shard_orientation == ShardOrientation::ROW_MAJOR;
    uint32_t num_h_cores = rm ? bbox.end_coord.y + 1 : bbox.end_coord.x + 1;
    uint32_t num_w_cores = rm ? bbox.end_coord.x + 1 : bbox.end_coord.y + 1;
    // tensor shape
    const auto& shape = a.padded_shape();
    uint32_t M = shape[2] * shape[0];  // 4608
    uint32_t K = shape[3];             // 3072
    uint32_t Mt = M / TILE_WIDTH;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t num_tensors = 3;
    uint32_t num_heads_per_tensor = 2;
    // block
    uint32_t block_w = K / num_w_cores;  // 384
    uint32_t block_h = M / num_h_cores;  // 384
    uint32_t block_wt = block_w / TILE_WIDTH;
    uint32_t block_ht = block_h / TILE_WIDTH;
    uint32_t out_block_w = block_w / num_tensors / num_heads_per_tensor;  // 64
    uint32_t out_block_wt = out_block_w / TILE_WIDTH;                     // 2
    uint32_t out_block_h = block_h * num_heads_per_tensor;                // 768
    uint32_t out_block_ht = out_block_h / TILE_WIDTH;                     // 24
    uint32_t per_core_tiles = block_ht * block_wt;
    uint32_t num_tiles_per_tensor = per_core_tiles / num_tensors;
    // check dims
    TT_ASSERT(M % TILE_WIDTH == 0 && "M must be divisible by tile width.");
    TT_ASSERT(K % TILE_WIDTH == 0 && "K must be divisible by tile width.");
    TT_ASSERT(Kt / num_w_cores == block_wt && "block_w must equal to K / num_cores_w.");
    TT_ASSERT(Mt / num_h_cores == block_ht && "block_h must equal to M / num_cores_h.");
    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // block size for in0 (tensor a)
    uint32_t in0_DFB_size = block_wt * block_ht * single_tile_size;
    // uint32_t im0_DFB_size = 2 * single_tile_size;
    uint32_t im0_DFB_size = 2 * block_ht * single_tile_size;
    uint32_t out_DFB_size = out_block_wt * out_block_ht * single_tile_size;

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    ProgramSpec spec;
    spec.name = "split_query_key_value_and_split_heads_sharded";

    // All four tensors are sharded and reached only through the dataflow buffers borrowed from
    // their shards below. No kernel builds a TensorAccessor, so none is bound to a kernel.
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = INPUT,
        .spec = a.mesh_tensor().tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT_Q,
        .spec = output[0].mesh_tensor().tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT_K,
        .spec = output[1].mesh_tensor().tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT_V,
        .spec = output[2].mesh_tensor().tensor_spec(),
    });

    // The reader and the writer take the same compile-time args.
    const KernelSpec::CompileTimeArgs dm_compile_time_args = {
        {"num_heads_per_tensor", num_heads_per_tensor},
        {"block_ht", block_ht},
        {"block_wt", block_wt},
        {"out_block_wt", out_block_wt},
        {"block_wt_size_bytes", block_wt * single_tile_size},
        {"out_block_wt_size_bytes", out_block_wt * single_tile_size},
        {"num_tiles_per_tensor", num_tiles_per_tensor},
        {"tensor_stride_size_bytes", block_wt * single_tile_size / num_tensors},
    };

    // reader compile arg
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
            "dataflow/reader_tm_tile_layout_create_qkv_heads_sharded.cpp",
        .dfb_bindings =
            {
                // The reader and the writer both read the input shard by base pointer, with no FIFO
                // ops; the reader takes the producer side and the writer the consumer side.
                DFBBinding{
                    .dfb_spec_name = IN,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                // The reader fills the Q and V output shards in place; nothing drains them.
                DFBBinding{
                    .dfb_spec_name = OUT_Q,
                    .accessor_name = "out_q",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT_Q,
                    .accessor_name = "out_q",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT_V,
                    .accessor_name = "out_v",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT_V,
                    .accessor_name = "out_v",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .compile_time_args = dm_compile_time_args,
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    // writer
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
            "dataflow/writer_tm_tile_layout_create_qkv_heads_sharded.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = K_PRE_TRANSPOSE,
                    .accessor_name = "k_pre_transpose",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args = dm_compile_time_args,
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    // compute kernel
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
            "compute/transpose_wh_sharded_metal2.cpp",
        // The legacy compute descriptor resolved to O3; Metal 2.0 defaults every kernel to O2.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = K_PRE_TRANSPOSE,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                // The packer fills the K output shard in place; nothing drains it.
                DFBBinding{
                    .dfb_spec_name = OUT_K,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT_K,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .compile_time_args = {{"num_tiles", num_tiles_per_tensor}},
        .hw_config = ComputeHardwareConfig{},
    };

    // Create dataflow buffers
    // The op has no runtime args at all: every dataflow buffer below except the K intermediate is
    // borrowed from a sharded tensor, so the tensor bindings are the entire per-dispatch state. The
    // framework refreshes them on a cache hit.
    // in0 sharded
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN,
        .entry_size = single_tile_size,
        .num_entries = in0_DFB_size / single_tile_size,
        .data_format_metadata = data_format,
        .borrowed_from = INPUT,
    });
    // im
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = K_PRE_TRANSPOSE,
        .entry_size = single_tile_size,
        .num_entries = im0_DFB_size / single_tile_size,
        .data_format_metadata = data_format,
    });
    // q sharded
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_Q,
        .entry_size = single_tile_size,
        .num_entries = out_DFB_size / single_tile_size,
        .data_format_metadata = data_format,
        .borrowed_from = OUTPUT_Q,
    });
    // k sharded
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_K,
        .entry_size = single_tile_size,
        .num_entries = out_DFB_size / single_tile_size,
        .data_format_metadata = data_format,
        .borrowed_from = OUTPUT_K,
    });
    // v sharded
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_V,
        .entry_size = single_tile_size,
        .num_entries = out_DFB_size / single_tile_size,
        .data_format_metadata = data_format,
        .borrowed_from = OUTPUT_V,
    });

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.kernels.push_back(std::move(compute));

    spec.work_units.push_back(WorkUnitSpec{
        .name = "split_query_key_value_and_split_heads_sharded",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = all_cores,
    });

    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {INPUT, a.mesh_tensor()},
        {OUTPUT_Q, output[0].mesh_tensor()},
        {OUTPUT_K, output[1].mesh_tensor()},
        {OUTPUT_V, output[2].mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
