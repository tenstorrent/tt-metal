// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "split_query_key_value_and_split_heads_sharded_program_factory.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt_metal;

SplitFusedQKVAndSplitHeadsShardedProgramFactory::cached_program_t
SplitFusedQKVAndSplitHeadsShardedProgramFactory::create(
    const SplitQueryKeyValueAndSplitHeadsParams& /*operation_attributes*/,
    const SplitQueryKeyValueAndSplitHeadsInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    const auto& a = tensor_args.input_tensor;
    auto& output = output_tensors;

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

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
    uint32_t in0_CB_size = block_wt * block_ht * single_tile_size;
    // uint32_t im0_CB_size = 2 * single_tile_size;
    uint32_t im0_CB_size = 2 * block_ht * single_tile_size;
    uint32_t out_CB_size = out_block_wt * out_block_ht * single_tile_size;

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = CreateProgram();
    // reader compile arg
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)num_heads_per_tensor,
        (std::uint32_t)block_ht,
        (std::uint32_t)block_wt,
        (std::uint32_t)out_block_wt,
        (std::uint32_t)block_wt * single_tile_size,
        (std::uint32_t)out_block_wt * single_tile_size,
        (std::uint32_t)num_tiles_per_tensor,
        (std::uint32_t)block_wt * single_tile_size / num_tensors};
    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
        "dataflow/reader_tm_tile_layout_create_qkv_heads_sharded.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    // writer
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)num_heads_per_tensor,
        (std::uint32_t)block_ht,
        (std::uint32_t)block_wt,
        (std::uint32_t)out_block_wt,
        (std::uint32_t)block_wt * single_tile_size,
        (std::uint32_t)out_block_wt * single_tile_size,
        (std::uint32_t)num_tiles_per_tensor,
        (std::uint32_t)block_wt * single_tile_size / num_tensors};
    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
        "dataflow/writer_tm_tile_layout_create_qkv_heads_sharded.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    // compute kernel
    std::vector<uint32_t> compute_args = {num_tiles_per_tensor};
    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
        "compute/transpose_wh_sharded.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = compute_args});

    // Create circular buffers
    // in0 sharded
    auto c_in0_config = CircularBufferConfig(in0_CB_size, {{CBIndex::c_0, cb_data_format}})
                            .set_page_size(CBIndex::c_0, single_tile_size)
                            .set_globally_allocated_address(*a.buffer());
    auto cb_in0_id = CreateCircularBuffer(program, all_cores, c_in0_config);
    // im
    auto c_im0_config = CircularBufferConfig(im0_CB_size, {{CBIndex::c_24, cb_data_format}})
                            .set_page_size(CBIndex::c_24, single_tile_size);
    CreateCircularBuffer(program, all_cores, c_im0_config);
    // q sharded
    auto c_out0_config = CircularBufferConfig(out_CB_size, {{CBIndex::c_16, cb_data_format}})
                             .set_page_size(CBIndex::c_16, single_tile_size)
                             .set_globally_allocated_address(*output[0].buffer());

    auto cb_out0_id = CreateCircularBuffer(program, all_cores, c_out0_config);
    // k sharded
    auto c_out1_config = CircularBufferConfig(out_CB_size, {{CBIndex::c_17, cb_data_format}})
                             .set_page_size(CBIndex::c_17, single_tile_size)
                             .set_globally_allocated_address(*output[1].buffer());

    auto cb_out1_id = CreateCircularBuffer(program, all_cores, c_out1_config);
    // v sharded
    auto c_out2_config = CircularBufferConfig(out_CB_size, {{CBIndex::c_18, cb_data_format}})
                             .set_page_size(CBIndex::c_18, single_tile_size)
                             .set_globally_allocated_address(*output[2].buffer());

    auto cb_out2_id = CreateCircularBuffer(program, all_cores, c_out2_config);

    return {std::move(program), {cb_in0_id, cb_out0_id, cb_out1_id, cb_out2_id}};
}

void SplitFusedQKVAndSplitHeadsShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const SplitQueryKeyValueAndSplitHeadsParams& /*operation_attributes*/,
    const SplitQueryKeyValueAndSplitHeadsInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;

    auto cb_in0_id = shared.cb_in0_id;
    auto cb_out0_id = shared.cb_out0_id;
    auto cb_out1_id = shared.cb_out1_id;
    auto cb_out2_id = shared.cb_out2_id;

    auto* in0_buffer = tensor_args.input_tensor.buffer();
    auto* out0_buffer = output_tensors.at(0).buffer();
    auto* out1_buffer = output_tensors.at(1).buffer();
    auto* out2_buffer = output_tensors.at(2).buffer();

    UpdateDynamicCircularBufferAddress(program, cb_in0_id, *in0_buffer);
    UpdateDynamicCircularBufferAddress(program, cb_out0_id, *out0_buffer);
    UpdateDynamicCircularBufferAddress(program, cb_out1_id, *out1_buffer);
    UpdateDynamicCircularBufferAddress(program, cb_out2_id, *out2_buffer);
}

}  // namespace ttnn::experimental::prim
