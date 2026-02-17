// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "create_qkv_heads_from_separate_tensors_program_factory.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;

CreateQKVHeadsSeparateTensorsProgramFactory::cached_program_t CreateQKVHeadsSeparateTensorsProgramFactory::create(
    const CreateQKVHeadsFromSeparateTensorsParams& operation_attributes,
    const CreateQKVHeadsFromSeparateTensorsInputs& tensor_args,
    CreateQKVHeadsFromSeparateTensorsResult& tensor_return_value) {
    const auto& input_tensor_q = tensor_args.input_tensor;
    const auto& input_tensor_kv = tensor_args.input_tensor_kv;
    auto& output_q = std::get<0>(tensor_return_value);
    auto& output_k = std::get<1>(tensor_return_value);
    auto& output_v = std::get<2>(tensor_return_value);

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

    Program program = tt::tt_metal::CreateProgram();
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)q_shard_ht,
        (std::uint32_t)q_shard_wt,
        (std::uint32_t)k_shard_ht,
        (std::uint32_t)k_shard_wt,  // shard width for k and v individually, times two for entire kv tensor
        (std::uint32_t)q_heads_per_core,
        (std::uint32_t)k_heads_per_core,
        (std::uint32_t)head_dim / TILE_WIDTH,  // tiles per head
    };

    std::map<std::string, std::string> reader_defines;
    if (transpose_k) {
        reader_defines["TRANSPOSE_K_HEADS"] = "1";
    }
    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors/device/kernels/"
        "reader_create_qkv_heads_sharded_separate.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    if (transpose_k) {
        std::vector<uint32_t> compute_args = {
            (std::uint32_t)(per_core_k_tiles),  // number of K tiles
        };
        tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
            "compute/transpose_wh_sharded.cpp",
            all_cores,
            tt_metal::ComputeConfig{.compile_args = compute_args});
    }

    uint32_t q_size = per_core_q_tiles * single_tile_size;
    uint32_t k_size = per_core_k_tiles * single_tile_size;
    uint32_t v_size = k_size;
    uint32_t kv_size = 2 * k_size;

    // qkv tensor
    auto c_in0_config = tt::tt_metal::CircularBufferConfig(q_size, {{CBIndex::c_0, q_data_format}})
                            .set_page_size(CBIndex::c_0, single_tile_size)
                            .set_globally_allocated_address(*input_tensor_q.buffer());
    auto cb_in0_id = CreateCircularBuffer(program, all_cores, c_in0_config);

    auto c_in1_config = tt::tt_metal::CircularBufferConfig(kv_size, {{CBIndex::c_1, kv_data_format}})
                            .set_page_size(CBIndex::c_1, single_tile_size)
                            .set_globally_allocated_address(*input_tensor_kv.buffer());
    auto cb_in1_id = CreateCircularBuffer(program, all_cores, c_in1_config);

    // q sharded
    auto c_out0_config = tt::tt_metal::CircularBufferConfig(q_size, {{CBIndex::c_16, q_data_format}})
                             .set_page_size(CBIndex::c_16, single_tile_size)
                             .set_globally_allocated_address(*output_q.buffer());
    auto cb_out0_id = CreateCircularBuffer(program, all_cores, c_out0_config);
    // k sharded
    auto c_out1_config = tt::tt_metal::CircularBufferConfig(k_size, {{CBIndex::c_17, kv_data_format}})
                             .set_page_size(CBIndex::c_17, single_tile_size)
                             .set_globally_allocated_address(*output_k.buffer());
    auto cb_out1_id = CreateCircularBuffer(program, all_cores, c_out1_config);
    // v sharded
    auto c_out2_config = tt::tt_metal::CircularBufferConfig(v_size, {{CBIndex::c_18, kv_data_format}})
                             .set_page_size(CBIndex::c_18, single_tile_size)
                             .set_globally_allocated_address(*output_v.buffer());
    auto cb_out2_id = CreateCircularBuffer(program, all_cores, c_out2_config);

    if (transpose_k) {
        auto c_im0_config = tt::tt_metal::CircularBufferConfig(k_size, {{CBIndex::c_24, kv_data_format}})
                                .set_page_size(CBIndex::c_24, single_tile_size);
        CreateCircularBuffer(program, all_cores, c_im0_config);
    }

    return {std::move(program), {cb_in0_id, cb_in1_id, cb_out0_id, cb_out1_id, cb_out2_id}};
}

void CreateQKVHeadsSeparateTensorsProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const CreateQKVHeadsFromSeparateTensorsParams& /*operation_attributes*/,
    const CreateQKVHeadsFromSeparateTensorsInputs& tensor_args,
    CreateQKVHeadsFromSeparateTensorsResult& tensor_return_value) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    const auto& input_tensor_q = tensor_args.input_tensor;
    const auto& input_tensor_kv = tensor_args.input_tensor_kv;
    auto& output_q = std::get<0>(tensor_return_value);
    auto& output_k = std::get<1>(tensor_return_value);
    auto& output_v = std::get<2>(tensor_return_value);

    auto* in0_buffer = input_tensor_q.buffer();
    auto* in1_buffer = input_tensor_kv.buffer();
    auto* out0_buffer = output_q.buffer();
    auto* out1_buffer = output_k.buffer();
    auto* out2_buffer = output_v.buffer();

    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_in0_id, *in0_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_in1_id, *in1_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_out0_id, *out0_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_out1_id, *out1_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_out2_id, *out2_buffer);
}

}  // namespace ttnn::experimental::prim
