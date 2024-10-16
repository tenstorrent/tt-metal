// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "create_qkv_heads_device_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;
using namespace tt;

namespace ttnn::operations::experimental::transformer {

static inline operation::ProgramWithCallbacks create_heads_combined_qkv_sharded(
    const Tensor& input_tensor,
    const std::vector<uint32_t>&& heads_per_group,
    const uint32_t head_dim,
    const uint32_t groups,
    std::vector<Tensor>& output,
    bool transpose_k) {
    // groups = kv_heads usually
    // heads_per_group = [x 1 1] if qkv since q_heads >= kv_heads and k=v heads but this should be generic
    TT_FATAL(head_dim % TILE_WIDTH == 0, "head dim {} needs to be a multiple of tile width {}", head_dim, TILE_WIDTH);
    TT_FATAL(heads_per_group.size() == output.size() && output.size() == 3, "Error");

    const uint32_t total_heads_per_group =
        std::accumulate(heads_per_group.begin(), heads_per_group.end(), 0);  // num q heads + 2 * num_kv_heads
    const uint32_t elements_per_group =
        head_dim * total_heads_per_group;  // head_dim * (num q heads + 2 * num kv heads)
    const uint32_t tiles_per_group =
        elements_per_group / TILE_WIDTH;  // head_dim % TILE_WIDTH == 0 so guaranteed to fit evenly

    const auto& input_shape = input_tensor.get_legacy_shape();

    TT_FATAL(
        input_shape[3] % elements_per_group == 0,
        "flattened inner tensor dimension {} does not divide evenly into head dim {}, heads per group, and groups {}",
        input_shape[3],
        head_dim,
        groups);
    TT_FATAL(input_shape[2] % TILE_HEIGHT == 0,
             "Sequence length {} must divide evenly into Tiles of Tile Height {}",
             input_shape[2],
             TILE_HEIGHT);
    TT_FATAL(input_tensor.shard_spec().has_value() == true, "Unsharded input is invalid for create_qkv_heads");

    auto shard_spec = input_tensor.shard_spec().value();
    auto all_cores = shard_spec.grid;
    auto bbox = all_cores.bounding_box();
    ShardOrientation shard_orientation = shard_spec.orientation;
    bool rm = shard_orientation == ShardOrientation::ROW_MAJOR;
    uint32_t num_h_cores = rm ? bbox.end_coord.y + 1 : bbox.end_coord.x + 1;
    uint32_t num_w_cores = rm ? bbox.end_coord.x + 1 : bbox.end_coord.y + 1;

    TT_FATAL(input_shape[3] % (num_w_cores * TILE_WIDTH) == 0,
             "Flattened hidden dimensions of QKV {} must be a multiple of width cores {} times tile width {}",
             input_shape[3],
             num_w_cores,
             TILE_WIDTH);
    TT_FATAL(groups % num_w_cores == 0,
             "number of groups {} must be a multiple of the number of width cores {}",
             groups,
             num_w_cores);

    uint32_t groups_per_block = groups / num_w_cores;
    uint32_t M = input_shape[2] * input_shape[0];
    uint32_t K = input_shape[3];
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t block_wt = Kt / num_w_cores;  // number of tiles in width dimension  - multiple tiles per head, multiple
                                           // heads per group, multiple tensors in group, multiple groups per cores

    TT_FATAL(Mt % num_h_cores == 0,
             "Outer dimension of batch {} times sequence length {} must divide evenly across cores {}",
             input_shape[0],
             input_shape[2],
             num_h_cores);
    uint32_t block_ht = Mt / num_h_cores;  // number of tiles in each each batch*seq_len dimension
    TT_FATAL(
        input_shape[2] % (block_ht * TILE_HEIGHT) == 0,
        "Per core work load must be within a batch. The sequence length {} and elements in each height shard {} must "
        "divide evenly",
        input_shape[2],
        block_ht * TILE_HEIGHT);
    uint32_t per_core_tiles = block_ht * block_wt;

    auto data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t single_tile_size = tile_size(data_format);
    TT_FATAL(L1_SIZE >= 2 * per_core_tiles * single_tile_size,
             "Workload of Tiles {} at Tile Size {} (times 2 for output) exceeds L1 capacity {}",
             per_core_tiles,
             single_tile_size,
             L1_SIZE);

    std::vector<uint32_t> num_tiles_per_group;
    num_tiles_per_group.reserve(output.size());
    for (uint32_t heads : heads_per_group) {
        num_tiles_per_group.push_back(heads * head_dim / TILE_WIDTH);
    }

    Program program = CreateProgram();
    std::vector<uint32_t> reader_compile_time_args = {

        (std::uint32_t)heads_per_group[0],  // q heads in group
        (std::uint32_t)heads_per_group[1],  // k heads in group
        (std::uint32_t)heads_per_group[2],  // v heads in group

        (std::uint32_t)head_dim / TILE_WIDTH * single_tile_size,  // size of a q head
        (std::uint32_t)head_dim / TILE_WIDTH * single_tile_size,  // size of a k head
        (std::uint32_t)head_dim / TILE_WIDTH * single_tile_size,  // size of a v head

        (std::uint32_t)tiles_per_group *
            single_tile_size,             // group size, used to skip past group to the rest of the three tensors
        (std::uint32_t)block_ht,          // how many tiles to read along sequence dimension
        (std::uint32_t)groups_per_block,  // groups per shard (kv heads per core)

        (std::uint32_t)block_ht * num_tiles_per_group[0] * groups_per_block,  // number of pages in q output tensor
        (std::uint32_t)block_ht * num_tiles_per_group[1] * groups_per_block,  // number of pages in k output tensor
        (std::uint32_t)block_ht * num_tiles_per_group[2] * groups_per_block,  // number of pages in v output tensor

        (std::uint32_t)num_tiles_per_group[0] * single_tile_size,  // size of n*Q tiles in each group, in bytes
        (std::uint32_t)num_tiles_per_group[1] * single_tile_size,  // size of K tiles in each group, in bytes
        (std::uint32_t)num_tiles_per_group[2] * single_tile_size,  // size of V tiles in each group, in bytes
    };

    std::map<string, string> reader_defines;
    if (transpose_k) {
        reader_defines["TRANSPOSE_K_HEADS"] = "1";
    }
    auto reader_kernel_id =
        tt_metal::CreateKernel(program,
                               "ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads/device/kernels/"
                               "reader_create_qkv_heads_sharded.cpp",
                               all_cores,
                               tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    if (transpose_k) {
        std::vector<uint32_t> compute_args = {
            (std::uint32_t)block_ht * num_tiles_per_group[1] * groups_per_block,  // number of K tiles
        };
        auto compute_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
            "compute/transpose_wh_sharded.cpp",
            all_cores,
            tt_metal::ComputeConfig{.compile_args = compute_args});
    }

    uint32_t input_size = block_ht * block_wt * single_tile_size;
    uint32_t q_size = block_ht * num_tiles_per_group[0] * single_tile_size * groups_per_block;
    uint32_t k_size = block_ht * num_tiles_per_group[1] * single_tile_size * groups_per_block;
    uint32_t v_size = block_ht * num_tiles_per_group[2] * single_tile_size * groups_per_block;

    // qkv tensor
    auto c_in0_config = CircularBufferConfig(input_size, {{CB::c_in0, data_format}})
                            .set_page_size(CB::c_in0, single_tile_size)
                            .set_globally_allocated_address(*input_tensor.buffer());
    auto cb_in0_id = CreateCircularBuffer(program, all_cores, c_in0_config);

    // q sharded
    auto c_out0_config = CircularBufferConfig(q_size, {{CB::c_out0, data_format}})
                             .set_page_size(CB::c_out0, single_tile_size)
                             .set_globally_allocated_address(*output[0].buffer());
    auto cb_out0_id = CreateCircularBuffer(program, all_cores, c_out0_config);
    // k sharded
    auto c_out1_config = CircularBufferConfig(k_size, {{CB::c_out1, data_format}})
                             .set_page_size(CB::c_out1, single_tile_size)
                             .set_globally_allocated_address(*output[1].buffer());
    auto cb_out1_id = CreateCircularBuffer(program, all_cores, c_out1_config);
    // v sharded
    auto c_out2_config = CircularBufferConfig(v_size, {{CB::c_out2, data_format}})
                             .set_page_size(CB::c_out2, single_tile_size)
                             .set_globally_allocated_address(*output[2].buffer());
    auto cb_out2_id = CreateCircularBuffer(program, all_cores, c_out2_config);

    if (transpose_k) {
        auto c_im0_config = CircularBufferConfig(k_size, {{CB::c_intermed0, data_format}})
                                .set_page_size(CB::c_intermed0, single_tile_size);
        auto cb_im0_id = CreateCircularBuffer(program, all_cores, c_im0_config);
    }

    auto override_runtime_args_callback = [cb_in0_id, cb_out0_id, cb_out1_id, cb_out2_id](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                              const std::vector<Tensor>& output_tensors) {
        auto in0_buffer = input_tensors.at(0).buffer();
        auto out0_buffer = output_tensors.at(0).buffer();
        auto out1_buffer = output_tensors.at(1).buffer();
        auto out2_buffer = output_tensors.at(2).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_in0_id, *in0_buffer);
        UpdateDynamicCircularBufferAddress(program, cb_out0_id, *out0_buffer);
        UpdateDynamicCircularBufferAddress(program, cb_out1_id, *out1_buffer);
        UpdateDynamicCircularBufferAddress(program, cb_out2_id, *out2_buffer);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

/**
 * Combined QKV
 *
 * m = num_KV_heads
 * p = num_Q_heads
 * n = p/m
 * i = head index for K/V, corresponding Q group for K/V
 *
 * input: [nQi Ki Vi for i in [0, m)] repeated for the sequence length
 * dims: [B, 1, S, H] Hi = [nQi Ki Vi] for all i kv heads = ((n + 2)*head_dim)
 *
 * output: 3 separate tensors organized by heads
 * [Qs,j for j in [0, p] and s in [0,S)]
 * dims: [B, p, S, head_dim] (all nQi in the KVi group are stacked together, p = n*m)
 *
 * [Ks,i for i in [0, m) and s in [0,S)]
 * dims: [B, m, S, head_dim]
 *
 * [Vs,i for i in [0, m) and s in [0,S)]
 * dims: [B, m, S, head_dim]
 *
 * Tiles stay the same Vi,s[x:x+32] to Vi+32,s[x:x+32] stays in one tile, but now instead of nQi,s and Ki,s tiles there
 * is Vi,s-1 and Vi,s+1
 *
 * Shard across each i kv head group (width sharding) and then shard across each token s (height sharding)
 * Each block: B x [nQi Ki Vi]s (shard across flattened heads and sequence length)
 *
 * Combined batch/sequence sharding is possible too...that may best be left as an extension
 */
operation::ProgramWithCallbacks multi_core_create_qkv_heads_sharded(const Tensor& input_tensor_qkv,
                                                                    const uint32_t num_q_heads,
                                                                    const uint32_t num_kv_heads,
                                                                    const uint32_t head_dim,
                                                                    const bool transpose_k_heads,
                                                                    std::vector<Tensor>& output,
                                                                    CoreCoord compute_with_storage_grid_size) {
    TT_FATAL(num_q_heads % num_kv_heads == 0,
             "num q heads {} / num kv heads {} needs to be a whole number",
             num_q_heads,
             num_kv_heads);
    return create_heads_combined_qkv_sharded(
        input_tensor_qkv, {num_q_heads / num_kv_heads, 1, 1}, head_dim, num_kv_heads, output, transpose_k_heads);
}
}  // namespace ttnn::operations::experimental::transformer
