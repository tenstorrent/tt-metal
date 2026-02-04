// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "create_qkv_heads_program_factory.hpp"
#include "create_qkv_heads_device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>

using namespace tt::constants;
using namespace tt;

namespace ttnn::experimental::prim {

CreateQKVHeadsProgramFactory::cached_program_t CreateQKVHeadsProgramFactory::create(
    const CreateQKVHeadsParams& operation_attributes,
    const CreateQKVHeadsInputs& tensor_args,
    CreateQKVHeadsResult& output) {
    const auto& input_tensor = tensor_args.input;
    auto& [output_q, output_k, output_v] = output;

    const uint32_t num_q_heads = operation_attributes.num_q_heads;
    const uint32_t num_kv_heads = operation_attributes.num_kv_heads;
    const uint32_t head_dim = operation_attributes.head_dim;
    const bool transpose_k = operation_attributes.transpose_k_heads;

    // Compute heads_per_group
    std::vector<uint32_t> heads_per_group = {num_q_heads / num_kv_heads, 1, 1};
    const uint32_t groups = num_kv_heads;

    // Validation
    TT_FATAL(head_dim % TILE_WIDTH == 0, "head dim {} needs to be a multiple of tile width {}", head_dim, TILE_WIDTH);
    TT_FATAL(heads_per_group.size() == 3, "heads_per_group size ({}) must equal 3", heads_per_group.size());

    const uint32_t total_heads_per_group =
        std::accumulate(heads_per_group.begin(), heads_per_group.end(), 0u);  // num q heads + 2 * num_kv_heads
    const uint32_t elements_per_group =
        head_dim * total_heads_per_group;  // head_dim * (num q heads + 2 * num kv heads)
    const uint32_t tiles_per_group =
        elements_per_group / TILE_WIDTH;  // head_dim % TILE_WIDTH == 0 so guaranteed to fit evenly

    const auto& input_shape = input_tensor.padded_shape();

    TT_FATAL(
        input_shape[3] % elements_per_group == 0,
        "flattened inner tensor dimension {} does not divide evenly into head dim {}, heads per group, and groups {}",
        input_shape[3],
        head_dim,
        groups);
    TT_FATAL(
        input_shape[2] % TILE_HEIGHT == 0,
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

    TT_FATAL(
        input_shape[3] % (num_w_cores * TILE_WIDTH) == 0,
        "Flattened hidden dimensions of QKV {} must be a multiple of width cores {} times tile width {}",
        input_shape[3],
        num_w_cores,
        TILE_WIDTH);
    TT_FATAL(
        groups % num_w_cores == 0,
        "number of groups {} must be a multiple of the number of width cores {}",
        groups,
        num_w_cores);

    uint32_t groups_per_block = groups / num_w_cores;
    uint32_t M = input_shape[2] * input_shape[0];
    uint32_t K = input_shape[3];
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t block_wt = Kt / num_w_cores;

    TT_FATAL(
        Mt % num_h_cores == 0,
        "Outer dimension of batch {} times sequence length {} must divide evenly across cores {}",
        input_shape[0],
        input_shape[2],
        num_h_cores);
    uint32_t block_ht = Mt / num_h_cores;
    TT_FATAL(
        input_shape[2] % (block_ht * TILE_HEIGHT) == 0,
        "Per core work load must be within a batch. The sequence length {} and elements in each height shard {} must "
        "divide evenly",
        input_shape[2],
        block_ht * TILE_HEIGHT);
    uint32_t per_core_tiles = block_ht * block_wt;

    const uint32_t l1_size = input_tensor.device()->l1_size_per_core();
    auto data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tile_size(data_format);
    TT_FATAL(
        l1_size >= 2 * per_core_tiles * single_tile_size,
        "Workload of Tiles {} at Tile Size {} (times 2 for output) exceeds L1 capacity {}",
        per_core_tiles,
        single_tile_size,
        l1_size);

    std::vector<uint32_t> num_tiles_per_group;
    num_tiles_per_group.reserve(3);
    for (uint32_t heads : heads_per_group) {
        num_tiles_per_group.push_back(heads * head_dim / TILE_WIDTH);
    }

    Program program = tt::tt_metal::CreateProgram();
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

    std::map<std::string, std::string> reader_defines;
    if (transpose_k) {
        reader_defines["TRANSPOSE_K_HEADS"] = "1";
    }
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads/device/kernels/"
        "reader_create_qkv_heads_sharded.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    tt_metal::KernelHandle compute_kernel_id = 0;
    bool has_compute_kernel = false;
    if (transpose_k) {
        has_compute_kernel = true;
        std::vector<uint32_t> compute_args = {
            (std::uint32_t)block_ht * num_tiles_per_group[1] * groups_per_block,  // number of K tiles
        };
        compute_kernel_id = tt_metal::CreateKernel(
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
    auto c_in0_config = tt::tt_metal::CircularBufferConfig(input_size, {{CBIndex::c_0, data_format}})
                            .set_page_size(CBIndex::c_0, single_tile_size)
                            .set_globally_allocated_address(*input_tensor.buffer());
    auto cb_in0_id = CreateCircularBuffer(program, all_cores, c_in0_config);

    // q sharded
    auto c_out0_config = tt::tt_metal::CircularBufferConfig(q_size, {{CBIndex::c_16, data_format}})
                             .set_page_size(CBIndex::c_16, single_tile_size)
                             .set_globally_allocated_address(*output_q.buffer());
    auto cb_out0_id = CreateCircularBuffer(program, all_cores, c_out0_config);
    // k sharded
    auto c_out1_config = tt::tt_metal::CircularBufferConfig(k_size, {{CBIndex::c_17, data_format}})
                             .set_page_size(CBIndex::c_17, single_tile_size)
                             .set_globally_allocated_address(*output_k.buffer());
    auto cb_out1_id = CreateCircularBuffer(program, all_cores, c_out1_config);
    // v sharded
    auto c_out2_config = tt::tt_metal::CircularBufferConfig(v_size, {{CBIndex::c_18, data_format}})
                             .set_page_size(CBIndex::c_18, single_tile_size)
                             .set_globally_allocated_address(*output_v.buffer());
    auto cb_out2_id = CreateCircularBuffer(program, all_cores, c_out2_config);

    if (transpose_k) {
        auto c_im0_config = tt::tt_metal::CircularBufferConfig(k_size, {{CBIndex::c_24, data_format}})
                                .set_page_size(CBIndex::c_24, single_tile_size);
        CreateCircularBuffer(program, all_cores, c_im0_config);
    }

    return cached_program_t{
        std::move(program),
        CreateQKVHeadsSharedVariables{
            .reader_kernel_id = reader_kernel_id,
            .compute_kernel_id = compute_kernel_id,
            .cb_in0_id = cb_in0_id,
            .cb_out0_id = cb_out0_id,
            .cb_out1_id = cb_out1_id,
            .cb_out2_id = cb_out2_id,
            .all_cores = all_cores,
            .has_compute_kernel = has_compute_kernel,
        }};
}

void CreateQKVHeadsProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const CreateQKVHeadsParams& /*operation_attributes*/,
    const CreateQKVHeadsInputs& tensor_args,
    CreateQKVHeadsResult& output) {
    const auto& input_tensor = tensor_args.input;
    auto& [output_q, output_k, output_v] = output;

    auto& program = cached_program.program;
    auto& cb_in0_id = cached_program.shared_variables.cb_in0_id;
    auto& cb_out0_id = cached_program.shared_variables.cb_out0_id;
    auto& cb_out1_id = cached_program.shared_variables.cb_out1_id;
    auto& cb_out2_id = cached_program.shared_variables.cb_out2_id;

    auto* in0_buffer = input_tensor.buffer();
    auto* out0_buffer = output_q.buffer();
    auto* out1_buffer = output_k.buffer();
    auto* out2_buffer = output_v.buffer();

    UpdateDynamicCircularBufferAddress(program, cb_in0_id, *in0_buffer);
    UpdateDynamicCircularBufferAddress(program, cb_out0_id, *out0_buffer);
    UpdateDynamicCircularBufferAddress(program, cb_out1_id, *out1_buffer);
    UpdateDynamicCircularBufferAddress(program, cb_out2_id, *out2_buffer);
}

}  // namespace ttnn::experimental::prim
