// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "create_qkv_heads_from_separate_tensors_device_operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {
const KernelSpecName READER{"reader"};
const KernelSpecName COMPUTE{"compute"};
const DFBSpecName IN_Q{"in_q"};
const DFBSpecName IN_KV{"in_kv"};
const DFBSpecName OUT_Q{"out_q"};
const DFBSpecName OUT_K{"out_k"};
const DFBSpecName OUT_V{"out_v"};
const DFBSpecName K_PRE_TRANSPOSE{"k_pre_transpose"};
const TensorParamName INPUT_Q{"input_q"};
const TensorParamName INPUT_KV{"input_kv"};
const TensorParamName OUTPUT_Q{"output_q"};
const TensorParamName OUTPUT_K{"output_k"};
const TensorParamName OUTPUT_V{"output_v"};
}  // namespace

ttnn::device_operation::ProgramArtifacts
CreateQKVHeadsSeparateTensorsDeviceOperation::CreateQKVHeadsSeparateTensorsProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor_q = tensor_args.input_tensor.mesh_tensor();
    const auto& input_tensor_kv = tensor_args.input_tensor_kv.mesh_tensor();
    const auto& output_q = std::get<0>(tensor_return_value).mesh_tensor();
    const auto& output_k = std::get<1>(tensor_return_value).mesh_tensor();
    const auto& output_v = std::get<2>(tensor_return_value).mesh_tensor();

    const uint32_t num_q_heads = operation_attributes.num_q_heads;
    const uint32_t num_kv_heads = operation_attributes.num_kv_heads;
    const uint32_t head_dim = operation_attributes.head_dim;
    const bool transpose_k = operation_attributes.transpose_k_heads;
    const auto& q_shape = input_tensor_q.padded_shape();
    const auto& kv_shape = input_tensor_kv.padded_shape();
    auto shard_spec = input_tensor_q.shard_spec().value();
    auto all_cores = shard_spec.grid;
    auto bbox = all_cores.bounding_box();
    ShardOrientation shard_orientation = shard_spec.orientation;
    bool rm = shard_orientation == ShardOrientation::ROW_MAJOR;
    uint32_t num_h_cores = rm ? bbox.end_coord.y + 1 : bbox.end_coord.x + 1;
    uint32_t num_w_cores = rm ? bbox.end_coord.x + 1 : bbox.end_coord.y + 1;

    uint32_t q_shard_wt =
        (q_shape[3]) /
        (num_w_cores * TILE_WIDTH);  // number of tiles in width dimension  - multiple tiles per head, multiple heads
                                     // per group, multiple tensors in group, multiple groups per cores
    uint32_t q_shard_ht = (q_shape[0] * q_shape[2]) / (num_h_cores * TILE_HEIGHT);

    uint32_t k_shard_wt = (kv_shape[3] / (2 * num_w_cores * TILE_WIDTH));
    uint32_t k_shard_ht = (kv_shape[0] * kv_shape[2]) / (num_h_cores * TILE_HEIGHT);

    uint32_t per_core_q_tiles = q_shard_ht * q_shard_wt;
    uint32_t per_core_k_tiles = k_shard_ht * k_shard_wt;

    const auto q_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    const auto kv_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_kv.dtype());
    uint32_t single_tile_size = tile_size(q_data_format);

    uint32_t q_heads_per_core = num_q_heads / num_w_cores;
    uint32_t k_heads_per_core = num_kv_heads / num_w_cores;

    ProgramSpec spec;
    spec.name = "create_qkv_heads_from_separate_tensors";

    // All five tensors are sharded and reached only through the dataflow buffers borrowed from
    // their shards below. No kernel builds a TensorAccessor, so none is bound to a kernel.
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = INPUT_Q,
        .spec = input_tensor_q.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = INPUT_KV,
        .spec = input_tensor_kv.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT_Q,
        .spec = output_q.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT_K,
        .spec = output_k.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT_V,
        .spec = output_v.tensor_spec(),
    });

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors/device/kernels/"
            "reader_create_qkv_heads_sharded_separate.cpp",
        // The reader is the only kernel touching the input shards (read by base pointer, no FIFO
        // ops) and the Q / V output shards (filled in place, never drained), so it binds each of
        // them as both producer and consumer.
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN_Q,
                    .accessor_name = "in_q",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = IN_Q,
                    .accessor_name = "in_q",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = IN_KV,
                    .accessor_name = "in_kv",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = IN_KV,
                    .accessor_name = "in_kv",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
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
        .compile_time_args =
            {
                {"q_shard_ht", q_shard_ht},
                {"q_shard_wt", q_shard_wt},
                {"k_shard_ht", k_shard_ht},
                {"k_shard_wt", k_shard_wt},  // shard width for k and v individually, times two for entire kv tensor
                {"q_num_heads_per_core", q_heads_per_core},
                {"k_num_heads_per_core", k_heads_per_core},
                {"tiles_per_head", head_dim / TILE_WIDTH},  // tiles per head
            },
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    std::optional<KernelSpec> compute;
    if (transpose_k) {
        // Under transpose the reader writes K heads into an intermediate buffer, and the compute
        // kernel transposes them into the K output shard.
        reader.compiler_options.defines.emplace("TRANSPOSE_K_HEADS", "1");
        reader.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = K_PRE_TRANSPOSE,
            .accessor_name = "k_pre_transpose",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });

        // For FLOAT32 input, enable fp32 dest accumulation so the JIT data-format selection
        // resolves the unpack-dst buffer to Tf32 (10-bit mantissa) instead of Float16_b (7-bit
        // mantissa). Mirrors the per-dtype promotion in eltwise unary/binary primitives.
        const bool fp32_dest_acc_en = input_tensor_kv.dtype() == tt_metal::DataType::FLOAT32;
        ComputeHardwareConfig compute_hw{
            .enable_32_bit_dest = fp32_dest_acc_en,
        };
        if (fp32_dest_acc_en) {
            // The intermediate is Float32 whenever 32-bit Dest is on; it unpacks to Src as before.
            compute_hw.unpack_modes = {
                {K_PRE_TRANSPOSE, UnpackMode::UnpackToSrc},
            };
        }

        compute = KernelSpec{
            .unique_id = COMPUTE,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/"
                "kernels/compute/transpose_wh_sharded_metal2.cpp",
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
            .compile_time_args =
                {
                    {"num_tiles", per_core_k_tiles},  // number of K tiles
                },
            .hw_config = compute_hw,
        };
    } else {
        // Without transpose the reader fills the K output shard in place, like Q and V.
        reader.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = OUT_K,
            .accessor_name = "out_k",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = OUT_K,
            .accessor_name = "out_k",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    uint32_t q_size = per_core_q_tiles * single_tile_size;
    uint32_t k_size = per_core_k_tiles * single_tile_size;
    uint32_t v_size = k_size;
    uint32_t kv_size = 2 * k_size;

    // The op has no runtime args at all: every dataflow buffer below except the transpose
    // intermediate is borrowed from a sharded tensor, so the tensor bindings are the entire
    // per-dispatch state. The framework refreshes them on a cache hit.

    // qkv tensor
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN_Q,
        .entry_size = single_tile_size,
        .num_entries = q_size / single_tile_size,
        .data_format_metadata = q_data_format,
        .borrowed_from = INPUT_Q,
    });

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN_KV,
        .entry_size = single_tile_size,
        .num_entries = kv_size / single_tile_size,
        .data_format_metadata = kv_data_format,
        .borrowed_from = INPUT_KV,
    });

    // q sharded
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_Q,
        .entry_size = single_tile_size,
        .num_entries = q_size / single_tile_size,
        .data_format_metadata = q_data_format,
        .borrowed_from = OUTPUT_Q,
    });
    // k sharded
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_K,
        .entry_size = single_tile_size,
        .num_entries = k_size / single_tile_size,
        .data_format_metadata = kv_data_format,
        .borrowed_from = OUTPUT_K,
    });
    // v sharded
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_V,
        .entry_size = single_tile_size,
        .num_entries = v_size / single_tile_size,
        .data_format_metadata = kv_data_format,
        .borrowed_from = OUTPUT_V,
    });

    if (transpose_k) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = K_PRE_TRANSPOSE,
            .entry_size = single_tile_size,
            .num_entries = k_size / single_tile_size,
            .data_format_metadata = kv_data_format,
        });
    }

    WorkUnitSpec work_unit{
        .name = "create_qkv_heads_from_separate_tensors",
        .kernels = {READER},
        .target_nodes = all_cores,
    };
    spec.kernels.push_back(std::move(reader));
    if (compute.has_value()) {
        work_unit.kernels.push_back(COMPUTE);
        spec.kernels.push_back(std::move(*compute));
    }
    spec.work_units.push_back(std::move(work_unit));

    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {INPUT_Q, input_tensor_q},
        {INPUT_KV, input_tensor_kv},
        {OUTPUT_Q, output_q},
        {OUTPUT_K, output_k},
        {OUTPUT_V, output_v},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
